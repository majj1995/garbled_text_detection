import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import typer

from poor_word.config import PathsConfig
from poor_word.data.download import fetch_locked_source, lock_source
from poor_word.data.manifest import SourceSpec, load_source_lock, load_source_specs
from poor_word.glyphs.catalog import load_common_chars
from poor_word.glyphs.corrupt import OPERATORS
from poor_word.glyphs.generate import GenerationConfig, generate_dataset

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
glyphs_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(glyphs_app, name="glyphs")


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
    expected_rows = len(generation.characters) * len(generation.font_paths) * (
        generation.normal_per_char
        + len(generation.operators) * generation.abnormal_per_operator
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
