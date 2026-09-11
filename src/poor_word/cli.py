import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import typer

from poor_word.config import PathsConfig
from poor_word.data.download import fetch_locked_source, lock_source
from poor_word.data.manifest import SourceSpec, load_source_lock, load_source_specs
from poor_word.evaluation.glyph_diagnostics import DiagnosticConfig, diagnose_glyph
from poor_word.evaluation.glyph_v2 import evaluate_glyph_v2
from poor_word.evaluation.oof import collect_oof_scores
from poor_word.evaluation.prototype_diagnostics import diagnose_prototypes
from poor_word.evaluation.real_report import RealSeedReportConfig, evaluate_real_seed
from poor_word.evaluation.report import evaluate_glyph_artifacts
from poor_word.glyphs.catalog import load_common_chars
from poor_word.glyphs.corrupt import OPERATORS
from poor_word.glyphs.generate import GenerationConfig, generate_dataset
from poor_word.glyphs.generate_v2 import V2GenerationConfig, generate_v2_dataset
from poor_word.glyphs.preview_v2 import PreviewConfig, generate_preview
from poor_word.glyphs.stroke_preview import StrokePreviewConfig, generate_stroke_preview
from poor_word.ocr.paddle_v5 import PaddleV5Adapter
from poor_word.real_data.crops import extract_character_crops
from poor_word.real_data.ingest import import_real_dataset
from poor_word.real_data.mining import MiningPolicy, mine_candidates, record_mining_yield
from poor_word.real_data.review import (
    build_review_queue,
    export_review_queue,
    import_review_labels,
)
from poor_word.real_data.split import assign_group_folds
from poor_word.training.adapt_real import AdaptConfig, adapt_real_encoder
from poor_word.training.finetune_real import RealFineTuneConfig, finetune_real_fold
from poor_word.training.nested_oof import NestedOofConfig, train_nested_real_oof
from poor_word.training.train_glyph import TrainConfig, train_glyph
from poor_word.training.train_mil import MilOofConfig, MilTrainConfig, train_mil_fold, train_mil_oof

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
glyphs_app = typer.Typer(no_args_is_help=True)
ocr_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
evaluate_app = typer.Typer(no_args_is_help=True)
real_data_app = typer.Typer(no_args_is_help=True)
review_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(glyphs_app, name="glyphs")
app.add_typer(ocr_app, name="ocr")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(real_data_app, name="real-data")
app.add_typer(review_app, name="review")


@app.callback()
def main() -> None:
    """AIGC malformed-Chinese detection tools."""


@app.command()
def doctor() -> None:
    """Report the active Python runtime and local test requirements."""
    minor = ".".join(platform.python_version_tuple()[:2])
    typer.echo(f"python={minor}")
    typer.echo("gpu=not-required-for-unit-tests")


def _get_source(source_id: str, sources_file: Path) -> SourceSpec:
    sources = {source.source_id: source for source in load_source_specs(sources_file)}
    try:
        return sources[source_id]
    except KeyError as error:
        raise typer.BadParameter(f"unknown source_id: {source_id}") from error


@data_app.command("lock")
def data_lock(
    source_id: Annotated[str, typer.Option("--source-id")],
    sources_file: Annotated[Path, typer.Option("--sources-file")] = Path("data/sources.toml"),
) -> None:
    """Download a declared source once and create its reviewed lock."""
    config = PathsConfig()
    spec = _get_source(source_id, sources_file)
    lock = lock_source(spec, config.raw_path, lock_dir=config.repo_root / "data/locks")
    typer.echo(f"sha256={lock.sha256}")
    typer.echo(f"size_bytes={lock.size_bytes}")
    typer.echo(f"license={lock.license_id}")
    typer.echo(f"resolved_url={lock.resolved_url}")


@data_app.command("fetch")
def data_fetch(
    source_id: Annotated[str, typer.Option("--source-id")],
    lock_dir: Annotated[Path, typer.Option("--lock-dir")] = Path("data/locks"),
) -> None:
    """Fetch a source only when its bytes match the reviewed lock."""
    config = PathsConfig()
    lock = load_source_lock(lock_dir / f"{source_id}.lock.json")
    path = fetch_locked_source(lock, config.raw_path)
    typer.echo(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_run_metadata(
    config: PathsConfig,
    generation: GenerationConfig,
    profile: str,
    manifest: Path,
) -> Path:
    actual_rows = pq.read_table(manifest).num_rows
    expected_rows = (
        len(generation.characters)
        * len(generation.font_paths)
        * (
            generation.normal_per_char
            + len(generation.operators) * generation.abnormal_per_operator
        )
    )
    lock_paths = sorted((config.repo_root / "data/locks").glob("*.lock.json"))
    payload = {
        "profile": profile,
        "seed": generation.seed,
        "output_dir": str(generation.output_dir),
        "git_commit": _git_commit(config.repo_root),
        "source_lock_sha256": {path.name: _file_sha256(path) for path in lock_paths},
        "dependency_lock_sha256": _file_sha256(config.repo_root / "uv.lock"),
        "row_count": actual_rows,
        "skipped_count": expected_rows - actual_rows,
        "created_at": datetime.now(UTC).isoformat(),
    }
    destination = generation.output_dir / "run.json"
    part_path = destination.with_name(f"{destination.name}.part")
    try:
        part_path.write_text(
            f"{json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)}\n",
            encoding="utf-8",
        )
        part_path.replace(destination)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    return destination


@glyphs_app.command("generate")
def glyphs_generate(
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    profile: Annotated[str, typer.Option("--profile")] = "smoke",
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
) -> None:
    """Generate deterministic legal and malformed glyph samples."""
    if profile not in {"smoke", "mvp"}:
        raise typer.BadParameter("profile must be smoke or mvp", param_hint="--profile")
    config = PathsConfig()
    characters = load_common_chars(config.raw_path / "common_chars_3500.txt")
    selected_characters = characters[:10] if profile == "smoke" else characters
    generation = GenerationConfig(
        output_dir=output_dir,
        characters=selected_characters,
        font_paths=(config.raw_path / "NotoSansCJKsc-Regular.otf",),
        normal_per_char=1 if profile == "smoke" else 4,
        abnormal_per_operator=1 if profile == "smoke" else 2,
        operators=tuple(sorted(OPERATORS)),
        seed=seed,
        source_asset_ids=("noto_sans_sc_regular",),
    )
    manifest = generate_dataset(generation)
    run_path = _write_run_metadata(config, generation, profile, manifest)
    typer.echo(f"manifest={manifest}")
    typer.echo(f"run={run_path}")


@glyphs_app.command("generate-v2")
def glyphs_generate_v2(
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    profile: Annotated[str, typer.Option("--profile")] = "mvp",
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260911,
    allow_experimental: Annotated[bool, typer.Option("--allow-experimental")] = False,
    graphics: Annotated[Path, typer.Option("--graphics")] = Path(
        "data/raw/makemeahanzi_graphics.txt"
    ),
    source_lock: Annotated[Path, typer.Option("--source-lock")] = Path(
        "data/locks/makemeahanzi_graphics.lock.json"
    ),
    license_path: Annotated[Path, typer.Option("--license")] = Path(
        "data/licenses/makemeahanzi.ARPHICPL.txt"
    ),
) -> None:
    """Generate separate train/calibration/test sets with the frozen native stroke rules."""
    if not allow_experimental:
        raise typer.BadParameter(
            "V2 is an offline experiment, not production-cleared data; use --allow-experimental"
        )
    if profile not in {"smoke", "mvp"}:
        raise typer.BadParameter("profile must be smoke or mvp", param_hint="--profile")
    characters = (
        tuple("永明林国春田合口")
        if profile == "smoke"
        else load_common_chars(PathsConfig().raw_path / "common_chars_3500.txt")
    )
    try:
        result = generate_v2_dataset(
            V2GenerationConfig(
                output_dir=output_dir,
                graphics_path=graphics,
                source_lock_path=source_lock,
                license_path=license_path,
                characters=characters,
                seed=seed,
                train_normal_per_char=4 if profile == "smoke" else 8,
                eval_normal_per_char=2 if profile == "smoke" else 4,
                train_abnormal_per_operator=1 if profile == "smoke" else 2,
                eval_abnormal_per_operator=1,
                allow_experimental=allow_experimental,
            ),
            progress=typer.echo,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"train_manifest={result.train_manifest}")
    typer.echo(f"calibration_manifest={result.calibration_manifest}")
    typer.echo(f"test_manifest={result.test_manifest}")
    typer.echo(f"run={result.run_path}")


@glyphs_app.command("preview-v2")
def glyphs_preview_v2(
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    fonts: Annotated[list[Path] | None, typer.Option("--font")] = None,
    characters: Annotated[str | None, typer.Option("--characters")] = None,
    per_operator: Annotated[int, typer.Option("--per-operator", min=1, max=100)] = 10,
    max_attempts_per_slot: Annotated[
        int, typer.Option("--max-attempts-per-slot", min=1, max=1000)
    ] = 64,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260910,
) -> None:
    """Preview scale-aware glyph candidates; REVIEW only, no training or model needed."""
    paths = PathsConfig()
    selected_fonts = (
        tuple(fonts)
        if fonts
        else (
            paths.raw_path / "NotoSansCJKsc-Regular.otf",
            paths.raw_path / "NotoSerifCJKsc-Regular.otf",
        )
    )
    selected_characters = (
        tuple(dict.fromkeys(characters))
        if characters is not None
        else load_common_chars(paths.raw_path / "common_chars_3500.txt")
    )
    try:
        config = PreviewConfig(
            output_dir=output_dir,
            characters=selected_characters,
            font_paths=selected_fonts,
            per_operator=per_operator,
            max_attempts_per_slot=max_attempts_per_slot,
            seed=seed,
        )
        artifacts = generate_preview(config, progress=typer.echo)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"preview={artifacts.html_path}")
    typer.echo(f"overview={artifacts.overview_path}")
    typer.echo(f"candidates={artifacts.candidates_path}")
    if not artifacts.complete:
        raise typer.Exit(code=2)


@glyphs_app.command("preview-strokes")
def glyphs_preview_strokes(
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    graphics: Annotated[Path, typer.Option("--graphics")] = Path(
        "data/raw/makemeahanzi_graphics.txt"
    ),
    source_lock: Annotated[Path, typer.Option("--source-lock")] = Path(
        "data/locks/makemeahanzi_graphics.lock.json"
    ),
    license_path: Annotated[Path, typer.Option("--license")] = Path(
        "data/licenses/makemeahanzi.ARPHICPL.txt"
    ),
    characters: Annotated[str | None, typer.Option("--characters")] = None,
    per_operator: Annotated[int, typer.Option("--per-operator", min=1, max=100)] = 10,
    bridges_only: Annotated[
        bool, typer.Option("--bridges-only", help="Only bridge candidates; quota per subtype.")
    ] = False,
    breaks_only: Annotated[
        bool, typer.Option("--breaks-only", help="Only interior-stroke break candidates.")
    ] = False,
    bridge_mode: Annotated[
        str | None,
        typer.Option("--bridge-mode", help="Bridge subtype: close_opening or block_gap."),
    ] = None,
    max_attempts_per_slot: Annotated[
        int, typer.Option("--max-attempts-per-slot", min=1, max=1000)
    ] = 48,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260910,
) -> None:
    """Preview intact-stroke edits; all REVIEW, no OCR, GPU or training."""
    paths = PathsConfig()
    try:
        selected = (
            tuple(dict.fromkeys(characters))
            if characters is not None
            else load_common_chars(paths.raw_path / "common_chars_3500.txt")
        )
        config = StrokePreviewConfig(
            output_dir=output_dir,
            graphics_path=paths.resolve(graphics),
            source_lock_path=paths.resolve(source_lock),
            license_path=paths.resolve(license_path),
            characters=selected,
            per_operator=per_operator,
            bridges_only=bridges_only,
            breaks_only=breaks_only,
            bridge_mode=cast(Literal["close_opening", "block_gap"] | None, bridge_mode),
            max_attempts_per_slot=max_attempts_per_slot,
            seed=seed,
        )
        artifacts = generate_stroke_preview(config, progress=typer.echo)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"preview={artifacts.html_path}")
    typer.echo(f"overview={artifacts.overview_path}")
    typer.echo(f"candidates={artifacts.candidates_path}")
    if not artifacts.complete:
        raise typer.Exit(code=2)


@ocr_app.command("audit")
def ocr_audit(
    image_dir: Annotated[Path, typer.Option("--image-dir")],
    output: Annotated[Path, typer.Option("--output")],
    endpoint: Annotated[str, typer.Option("--endpoint")] = "http://127.0.0.1:8765",
    warmup: Annotated[int, typer.Option("--warmup", min=0)] = 10,
    runs: Annotated[int, typer.Option("--runs", min=1)] = 30,
) -> None:
    """Audit the isolated PP-OCRv5 server on representative images."""
    images = tuple(str(path) for path in sorted(image_dir.rglob("*.png")))
    if not images:
        raise typer.BadParameter("image directory contains no PNG files", param_hint="--image-dir")
    audit = PaddleV5Adapter(endpoint=endpoint).audit(images, warmup=warmup, runs=runs)
    output.parent.mkdir(parents=True, exist_ok=True)
    part_path = output.with_name(f"{output.name}.part")
    part_path.write_text(f"{audit.model_dump_json(indent=2)}\n", encoding="utf-8")
    part_path.replace(output)
    typer.echo(output)

    server_models = (
        audit.detection_model_name == "PP-OCRv5_server_det"
        and audit.recognition_model_name == "PP-OCRv5_server_rec"
    )
    if not server_models or not audit.character_boxes_available:
        raise typer.Exit(code=2)


@train_app.command("glyph")
def train_glyph_command(
    manifest: Annotated[Path, typer.Option("--manifest")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 20,
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 256,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    pretrained: Annotated[bool, typer.Option("--pretrained/--no-pretrained")] = False,
    sampler: Annotated[str, typer.Option("--sampler")] = "random",
    allow_experimental: Annotated[bool, typer.Option("--allow-experimental")] = False,
    log_every: Annotated[int, typer.Option("--log-every", min=1)] = 25,
) -> None:
    """Train the glyph encoder and build its legal-character prototype bank."""
    artifacts = train_glyph(
        TrainConfig(
            manifest=manifest,
            output_dir=output_dir,
            epochs=epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            seed=seed,
            pretrained=pretrained,
            device=device,
            sampler=cast(Literal["random", "paired"], sampler),
            allow_experimental=allow_experimental,
            log_every=log_every,
        ),
        progress=typer.echo,
    )
    typer.echo(f"checkpoint={artifacts.checkpoint}")
    typer.echo(f"prototypes={artifacts.prototype_bank}")
    typer.echo(f"metrics={artifacts.metrics}")


@train_app.command("adapt-real")
def train_adapt_real_command(
    crop_manifest: Annotated[Path, typer.Option("--crop-manifest")],
    real_manifest: Annotated[Path, typer.Option("--real-manifest")],
    prior_checkpoint: Annotated[Path, typer.Option("--prior-checkpoint")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 20,
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=2)] = 64,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=0.0000001)] = 3e-5,
) -> None:
    """Adapt a prior encoder with eligible unlabeled real crops only."""
    artifacts = adapt_real_encoder(
        AdaptConfig(
            crop_manifest=crop_manifest,
            real_manifest=real_manifest,
            prior_checkpoint=prior_checkpoint,
            output_dir=output_dir,
            epochs=epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            seed=seed,
            device=device,
            learning_rate=learning_rate,
        )
    )
    typer.echo(f"checkpoint={artifacts.checkpoint}")
    typer.echo(f"metrics={artifacts.metrics}")


@train_app.command("real-oof")
def train_real_oof_command(
    real_manifest: Annotated[Path, typer.Option("--real-manifest")],
    crop_manifest: Annotated[Path, typer.Option("--crop-manifest")],
    gold_manifest: Annotated[Path, typer.Option("--gold-manifest")],
    fold_manifest: Annotated[Path, typer.Option("--fold-manifest")],
    synthetic_manifest: Annotated[Path, typer.Option("--synthetic-manifest")],
    adapted_checkpoint: Annotated[Path, typer.Option("--adapted-checkpoint")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 10,
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=2)] = 64,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=0.0000001)] = 1e-4,
) -> None:
    """Fine-tune five leakage-safe models and publish development OOF scores."""
    artifacts = []
    for held_out_fold in range(5):
        artifacts.append(
            finetune_real_fold(
                RealFineTuneConfig(
                    real_manifest=real_manifest,
                    crop_manifest=crop_manifest,
                    gold_manifest=gold_manifest,
                    fold_manifest=fold_manifest,
                    synthetic_manifest=synthetic_manifest,
                    adapted_checkpoint=adapted_checkpoint,
                    output_dir=output_dir / f"fold-{held_out_fold}",
                    epochs=epochs,
                    max_steps=max_steps,
                    batch_size=batch_size,
                    seed=seed,
                    device=device,
                    learning_rate=learning_rate,
                ),
                held_out_fold,
            )
        )
    oof = collect_oof_scores(
        artifacts,
        fold_manifest,
        real_manifest,
        crop_manifest,
        gold_manifest,
        adapted_checkpoint,
        synthetic_manifest,
        output_dir / "oof",
    )
    typer.echo(f"oof={oof}")
    typer.echo(f"metrics={oof.parent / 'metrics.json'}")


@train_app.command("real-nested-oof")
def train_real_nested_oof_command(
    real_manifest: Annotated[Path, typer.Option("--real-manifest")],
    crop_manifest: Annotated[Path, typer.Option("--crop-manifest")],
    gold_manifest: Annotated[Path, typer.Option("--gold-manifest")],
    fold_manifest: Annotated[Path, typer.Option("--fold-manifest")],
    synthetic_manifest: Annotated[Path, typer.Option("--synthetic-manifest")],
    adapted_checkpoint: Annotated[Path, typer.Option("--adapted-checkpoint")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 10,
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=2)] = 64,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=0.0000001)] = 1e-4,
) -> None:
    """Publish nested character features for leakage-safe five-fold MIL OOF."""
    artifacts = train_nested_real_oof(
        NestedOofConfig(
            real_manifest=real_manifest,
            crop_manifest=crop_manifest,
            gold_manifest=gold_manifest,
            fold_manifest=fold_manifest,
            synthetic_manifest=synthetic_manifest,
            adapted_checkpoint=adapted_checkpoint,
            output_dir=output_dir,
            epochs=epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            seed=seed,
            device=device,
            learning_rate=learning_rate,
        )
    )
    typer.echo(f"nested_manifest={artifacts.manifest}")


@train_app.command("mil")
def train_mil_command(
    real_manifest: Annotated[Path, typer.Option("--real-manifest")],
    fold_manifest: Annotated[Path, typer.Option("--fold-manifest")],
    feature_manifest: Annotated[Path, typer.Option("--feature-manifest")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    held_out_fold: Annotated[int, typer.Option("--held-out-fold", min=0)],
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 30,
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 32,
    patience: Annotated[int, typer.Option("--patience", min=1)] = 5,
    min_delta: Annotated[float, typer.Option("--min-delta", min=0)] = 1e-4,
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=0.0000001)] = 1e-3,
    normal_instance_weight: Annotated[
        float, typer.Option("--normal-instance-weight", min=0)
    ] = 0.25,
    hidden_dim: Annotated[int, typer.Option("--hidden-dim", min=1)] = 32,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
    device: Annotated[str, typer.Option("--device")] = "cuda",
) -> None:
    """Train a leakage-safe image-level MIL head from held-out character evidence."""
    artifacts = train_mil_fold(
        MilTrainConfig(
            real_manifest=real_manifest,
            fold_manifest=fold_manifest,
            feature_manifest=feature_manifest,
            output_dir=output_dir,
            held_out_fold=held_out_fold,
            epochs=epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            patience=patience,
            min_delta=min_delta,
            learning_rate=learning_rate,
            normal_instance_weight=normal_instance_weight,
            hidden_dim=hidden_dim,
            seed=seed,
            device=device,
        )
    )
    typer.echo(f"checkpoint={artifacts.checkpoint}")
    typer.echo(f"metrics={artifacts.metrics}")
    typer.echo(f"attention_candidates={artifacts.attention_candidates}")
    typer.echo(f"image_scores={artifacts.image_scores}")


@train_app.command("mil-oof")
def train_mil_oof_command(
    real_manifest: Annotated[Path, typer.Option("--real-manifest")],
    fold_manifest: Annotated[Path, typer.Option("--fold-manifest")],
    nested_feature_dir: Annotated[Path, typer.Option("--nested-feature-dir")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 30,
    max_steps: Annotated[int | None, typer.Option("--max-steps", min=1)] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 32,
    patience: Annotated[int, typer.Option("--patience", min=1)] = 5,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
    device: Annotated[str, typer.Option("--device")] = "cuda",
) -> None:
    """Train MIL OOF from outer-fold-aware nested character features only."""
    artifacts = train_mil_oof(
        MilOofConfig(
            real_manifest=real_manifest,
            fold_manifest=fold_manifest,
            nested_feature_dir=nested_feature_dir,
            output_dir=output_dir,
            epochs=epochs,
            max_steps=max_steps,
            batch_size=batch_size,
            patience=patience,
            seed=seed,
            device=device,
        )
    )
    typer.echo(f"image_oof={artifacts.image_oof}")
    typer.echo(f"inventory={artifacts.inventory}")
    typer.echo(f"metrics={artifacts.metrics}")


@evaluate_app.command("glyph-v2")
def evaluate_glyph_v2_command(
    calibration_manifest: Annotated[Path, typer.Option("--calibration-manifest")],
    test_manifest: Annotated[Path, typer.Option("--test-manifest")],
    artifacts_dir: Annotated[Path, typer.Option("--artifacts")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    allow_experimental: Annotated[bool, typer.Option("--allow-experimental")] = False,
    prevalence: Annotated[float, typer.Option("--prevalence", min=0.000001, max=0.999999)] = 0.001,
    max_fpr: Annotated[float, typer.Option("--max-fpr", min=0, max=1)] = 0.0001,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 64,
) -> None:
    """Choose a calibration-normal threshold, then evaluate it unchanged on V2 test data."""
    try:
        report = evaluate_glyph_v2(
            calibration_manifest,
            test_manifest,
            artifacts_dir,
            output_dir,
            allow_experimental=allow_experimental,
            prevalence=prevalence,
            max_fpr=max_fpr,
            device=device,
            batch_size=batch_size,
            progress=typer.echo,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"json={report.json_path}")
    typer.echo(f"markdown={report.markdown_path}")


@evaluate_app.command("glyph")
def evaluate_glyph_command(
    manifest: Annotated[Path, typer.Option("--manifest")],
    artifacts_dir: Annotated[Path, typer.Option("--artifacts")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    prevalence: Annotated[float, typer.Option("--prevalence", min=0.000001, max=0.999999)] = 0.001,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 64,
    ocr_audit: Annotated[Path | None, typer.Option("--ocr-audit")] = None,
) -> None:
    """Evaluate glyph scores with deployment base-rate math and provenance."""
    report = evaluate_glyph_artifacts(
        manifest,
        artifacts_dir,
        output_dir,
        prevalence=prevalence,
        device=device,
        batch_size=batch_size,
        ocr_audit_path=ocr_audit,
    )
    typer.echo(f"json={report.json_path}")
    typer.echo(f"markdown={report.markdown_path}")


@evaluate_app.command("diagnose-prototypes")
def diagnose_prototypes_command(
    train_manifest: Annotated[Path, typer.Option("--train-manifest")],
    calibration_manifest: Annotated[Path, typer.Option("--calibration-manifest")],
    artifacts_dir: Annotated[Path, typer.Option("--artifacts")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    characters: Annotated[str, typer.Option("--characters")],
    device: Annotated[str, typer.Option("--device")] = "cpu",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 64,
    allow_experimental: Annotated[bool, typer.Option("--allow-experimental")] = False,
) -> None:
    """Compare frozen prototypes with same-character training normals; no test or tuning."""
    try:
        artifacts = diagnose_prototypes(
            train_manifest=train_manifest,
            calibration_manifest=calibration_manifest,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            characters=characters,
            device_name=device,
            batch_size=batch_size,
            allow_experimental=allow_experimental,
            progress=typer.echo,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise typer.BadParameter(str(error)) from error
    for line in artifacts.summary_lines:
        typer.echo(line)
    typer.echo(f"json={artifacts.json_path}")
    typer.echo(f"markdown={artifacts.markdown_path}")


@evaluate_app.command("diagnose-glyph")
def diagnose_glyph_command(
    manifest: Annotated[Path, typer.Option("--manifest")],
    train_manifest: Annotated[Path, typer.Option("--train-manifest")],
    artifacts_dir: Annotated[Path, typer.Option("--artifacts")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    device: Annotated[str, typer.Option("--device")] = "cpu",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 64,
    threshold: Annotated[float | None, typer.Option("--threshold")] = None,
    max_fpr: Annotated[float, typer.Option("--max-fpr", min=0.000000001, max=1)] = 0.0001,
    examples_per_kind: Annotated[int, typer.Option("--examples-per-kind", min=0)] = 10,
    max_false_positives: Annotated[int, typer.Option("--max-false-positives", min=0)] = 50,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260910,
) -> None:
    """Diagnose synthetic errors and replay sampler coverage without retraining."""
    artifacts = diagnose_glyph(
        DiagnosticConfig(
            manifest=manifest,
            train_manifest=train_manifest,
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
            max_fpr=max_fpr,
            examples_per_kind=examples_per_kind,
            max_false_positives=max_false_positives,
            seed=seed,
        ),
        progress=lambda message: typer.echo(message),
    )
    typer.echo(f"json={artifacts.json_path}")
    typer.echo(f"markdown={artifacts.markdown_path}")
    typer.echo(f"scores={artifacts.scores_path}")
    typer.echo(f"examples={artifacts.examples_path}")


@evaluate_app.command("real-seed")
def evaluate_real_seed_command(
    real_manifest: Annotated[Path, typer.Option("--real-manifest")],
    fold_manifest: Annotated[Path, typer.Option("--fold-manifest")],
    crop_manifest: Annotated[Path, typer.Option("--crop-manifest")],
    gold_manifest: Annotated[Path, typer.Option("--gold-manifest")],
    character_oof: Annotated[Path, typer.Option("--character-oof")],
    image_oof: Annotated[Path, typer.Option("--image-oof")],
    ocr_manifest: Annotated[Path, typer.Option("--ocr-manifest")],
    ocr_audit: Annotated[Path, typer.Option("--ocr-audit")],
    common_chars: Annotated[Path, typer.Option("--common-chars")],
    source_lock: Annotated[Path, typer.Option("--source-lock")],
    dependency_lock: Annotated[Path, typer.Option("--dependency-lock")],
    character_model_inventory: Annotated[Path, typer.Option("--character-inventory")],
    image_model_inventory: Annotated[Path, typer.Option("--image-inventory")],
    nested_character_manifest: Annotated[Path, typer.Option("--nested-manifest")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    prevalence: Annotated[float, typer.Option("--prevalence", min=0.000001, max=0.999999)] = 0.001,
    threshold: Annotated[float, typer.Option("--threshold", min=0, max=1)] = 0.5,
) -> None:
    """Compare aligned real development OOF scores without opening locked-test data."""
    artifacts = evaluate_real_seed(
        RealSeedReportConfig(
            real_manifest=real_manifest,
            fold_manifest=fold_manifest,
            crop_manifest=crop_manifest,
            gold_manifest=gold_manifest,
            character_oof=character_oof,
            image_oof=image_oof,
            ocr_manifest=ocr_manifest,
            ocr_audit=ocr_audit,
            common_chars=common_chars,
            source_lock=source_lock,
            dependency_lock=dependency_lock,
            character_model_inventory=character_model_inventory,
            image_model_inventory=image_model_inventory,
            nested_character_manifest=nested_character_manifest,
            output_dir=output_dir,
            prevalence=prevalence,
            threshold=threshold,
        )
    )
    typer.echo(f"json={artifacts.json_path}")
    typer.echo(f"markdown={artifacts.markdown_path}")
    typer.echo(f"provenance={artifacts.provenance_path}")


@real_data_app.command("import")
def real_data_import_command(
    input_jsonl: Annotated[Path, typer.Option("--input")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Validate real seed labels and create an immutable Parquet dataset."""
    artifacts = import_real_dataset(input_jsonl, output_dir)
    typer.echo(f"manifest={artifacts.manifest}")
    typer.echo(f"dataset={artifacts.dataset_metadata}")
    typer.echo(f"validation={artifacts.validation_report}")


@real_data_app.command("split")
def real_data_split_command(
    manifest: Annotated[Path, typer.Option("--manifest")],
    image_root: Annotated[Path, typer.Option("--image-root")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    folds: Annotated[int, typer.Option("--folds", min=2)] = 5,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
) -> None:
    """Assign leakage-safe group folds with exact and perceptual duplicate closure."""
    artifacts = assign_group_folds(manifest, image_root, output_dir, folds=folds, seed=seed)
    typer.echo(f"folds={artifacts.folds}")
    typer.echo(f"audit={artifacts.audit}")


@real_data_app.command("crops")
def real_data_crops_command(
    manifest: Annotated[Path, typer.Option("--manifest")],
    image_root: Annotated[Path, typer.Option("--image-root")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    padding: Annotated[int, typer.Option("--padding", min=0)] = 2,
    ocr_results: Annotated[Path | None, typer.Option("--ocr-results")] = None,
    ocr_audit: Annotated[Path | None, typer.Option("--ocr-audit")] = None,
) -> None:
    """Extract immutable character crops from reviewed or audited OCR boxes."""
    artifacts = extract_character_crops(
        manifest,
        image_root,
        output_dir,
        padding=padding,
        ocr_results=ocr_results,
        ocr_audit=ocr_audit,
    )
    typer.echo(f"crops={artifacts.manifest}")
    typer.echo(f"audit={artifacts.audit}")


@real_data_app.command("mine")
def real_data_mine_command(
    scores: Annotated[Path, typer.Option("--scores")],
    real_manifest: Annotated[Path, typer.Option("--real-manifest")],
    fold_manifest: Annotated[Path, typer.Option("--fold-manifest")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    normal_false_positive_threshold: Annotated[
        float, typer.Option("--normal-fp-threshold", min=0, max=1)
    ] = 0.8,
    disagreement_threshold: Annotated[
        float, typer.Option("--disagreement-threshold", min=0, max=1)
    ] = 0.25,
    threshold_band_low: Annotated[float, typer.Option("--threshold-band-low", min=0, max=1)] = 0.45,
    threshold_band_high: Annotated[
        float, typer.Option("--threshold-band-high", min=0, max=1)
    ] = 0.55,
    abnormal_attention_threshold: Annotated[
        float, typer.Option("--abnormal-attention-threshold", min=0, max=1)
    ] = 0.8,
    style_novelty_threshold: Annotated[
        float, typer.Option("--style-novelty-threshold", min=0, max=1)
    ] = 0.8,
    overall_limit: Annotated[int, typer.Option("--overall-limit", min=1)] = 500,
    per_product_cap: Annotated[int, typer.Option("--per-product-cap", min=1)] = 25,
    per_template_cap: Annotated[int, typer.Option("--per-template-cap", min=1)] = 10,
    per_source_cap: Annotated[int, typer.Option("--per-source-cap", min=1)] = 50,
    fold_count: Annotated[int, typer.Option("--fold-count", min=2)] = 5,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
) -> None:
    """Mine a provenance-checked candidate queue for human review only."""
    artifacts = mine_candidates(
        scores,
        MiningPolicy(
            normal_false_positive_threshold=normal_false_positive_threshold,
            disagreement_threshold=disagreement_threshold,
            threshold_band_low=threshold_band_low,
            threshold_band_high=threshold_band_high,
            abnormal_attention_threshold=abnormal_attention_threshold,
            style_novelty_threshold=style_novelty_threshold,
            overall_limit=overall_limit,
            per_product_cap=per_product_cap,
            per_template_cap=per_template_cap,
            per_source_cap=per_source_cap,
            fold_count=fold_count,
            seed=seed,
        ),
        real_manifest=real_manifest,
        fold_manifest=fold_manifest,
        output_dir=output_dir,
    )
    typer.echo(f"queue={artifacts.queue}")
    typer.echo(f"metadata={artifacts.metadata}")
    typer.echo(f"review_scores={artifacts.review_scores}")
    typer.echo(f"review_disagreements={artifacts.review_disagreements}")


@real_data_app.command("mining-yield")
def real_data_mining_yield_command(
    candidate_queue: Annotated[Path, typer.Option("--candidate-queue")],
    review_queue: Annotated[Path, typer.Option("--review-queue")],
    review_labels: Annotated[Path, typer.Option("--review-labels")],
    reviewed_gold: Annotated[Path, typer.Option("--reviewed-gold")],
    import_audit: Annotated[Path, typer.Option("--import-audit")],
    base_real_manifest: Annotated[Path, typer.Option("--base-real-manifest")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    previous_gold: Annotated[Path | None, typer.Option("--previous-gold")] = None,
) -> None:
    """Record trusted review yield and publish the next dataset lineage version."""
    artifacts = record_mining_yield(
        candidate_queue,
        review_queue,
        review_labels,
        reviewed_gold,
        import_audit,
        base_real_manifest=base_real_manifest,
        output_dir=output_dir,
        previous_gold_manifest=previous_gold,
    )
    typer.echo(f"dataset_version={artifacts.dataset_version}")
    typer.echo(f"audit={artifacts.audit}")


def _read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise typer.BadParameter(f"invalid JSONL at line {number}: {error}") from error
        if not isinstance(row, dict):
            raise typer.BadParameter(f"JSONL row must be an object at line {number}")
        rows.append(cast(dict[str, object], row))
    return rows


@review_app.command("export")
def review_export_command(
    crops: Annotated[Path, typer.Option("--crops")],
    crop_root: Annotated[Path, typer.Option("--crop-root")],
    scores: Annotated[Path, typer.Option("--scores")],
    disagreements: Annotated[Path, typer.Option("--disagreements")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    limit: Annotated[int, typer.Option("--limit", min=1)] = 500,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 20260804,
) -> None:
    """Export a versioned CSV/JSONL quick-review queue with contact thumbnails."""
    score_rows = _read_jsonl_objects(scores)
    disagreement_values: dict[str, dict[str, object]] = {}
    for row in _read_jsonl_objects(disagreements):
        crop_id = row.get("crop_id")
        value = row.get("disagreement")
        model_id = row.get("disagreement_model_id")
        artifact_sha256 = row.get("disagreement_artifact_sha256")
        if (
            not isinstance(crop_id, str)
            or not isinstance(value, (int, float))
            or not isinstance(model_id, str)
            or not model_id
            or not isinstance(artifact_sha256, str)
            or not artifact_sha256
        ):
            raise typer.BadParameter(
                "disagreement rows require crop_id, numeric disagreement, model ID, "
                "and artifact SHA-256"
            )
        if crop_id in disagreement_values:
            raise typer.BadParameter(f"duplicate disagreement crop_id: {crop_id}")
        disagreement_values[crop_id] = {
            "disagreement": float(value),
            "disagreement_model_id": model_id,
            "disagreement_artifact_sha256": artifact_sha256,
        }
    queue = build_review_queue(score_rows, disagreement_values, limit, seed)
    artifacts = export_review_queue(queue, crops, crop_root, output_dir)
    typer.echo(f"queue={artifacts.queue}")
    typer.echo(f"queue_version={artifacts.queue_version}")
    typer.echo(f"csv={artifacts.csv}")
    typer.echo(f"jsonl={artifacts.jsonl}")
    typer.echo(f"contact_sheet={artifacts.contact_sheet}")


@review_app.command("import")
def review_import_command(
    labels: Annotated[Path, typer.Option("--labels")],
    queue: Annotated[Path, typer.Option("--queue")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    existing_gold: Annotated[Path | None, typer.Option("--existing-gold")] = None,
) -> None:
    """Import human labels after verifying queue version and gold-label ownership."""
    artifacts = import_review_labels(labels, queue, output_dir, existing_gold=existing_gold)
    typer.echo(f"gold={artifacts.gold_crops}")
    typer.echo(f"audit={artifacts.audit}")
