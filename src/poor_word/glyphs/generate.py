import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from poor_word.domain import AnomalyKind, BoundingBox, Decision, GeneratedSample
from poor_word.glyphs.corrupt import OPERATORS, CorruptionNotApplicable, corrupt_glyph
from poor_word.glyphs.render import render_glyph

_OPERATOR_KINDS = {
    "erase_segment": AnomalyKind.MISSING_STROKE,
    "add_stroke": AnomalyKind.EXTRA_STROKE,
    "break_stroke": AnomalyKind.BROKEN_STROKE,
    "bridge": AnomalyKind.BRIDGE,
    "component_shift": AnomalyKind.COMPONENT_SHIFT,
}


class GenerationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: Path
    characters: tuple[str, ...]
    font_paths: tuple[Path | None, ...]
    normal_per_char: int = Field(ge=0)
    abnormal_per_operator: int = Field(ge=0)
    operators: tuple[str, ...]
    seed: int = Field(ge=0)
    source_asset_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_configuration(self) -> "GenerationConfig":
        if not self.characters:
            raise ValueError("characters must not be empty")
        if any(len(character) != 1 for character in self.characters):
            raise ValueError("every character must contain exactly one Unicode code point")
        if not self.font_paths or len(self.font_paths) != len(self.source_asset_ids):
            raise ValueError("font_paths and source_asset_ids must have equal non-zero length")
        unknown = sorted(set(self.operators) - OPERATORS)
        if unknown:
            raise ValueError(f"unknown corruption operators: {', '.join(unknown)}")
        return self


def _canonical_json(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sample_seed(base_seed: int, **identity: object) -> int:
    digest = hashlib.sha256(_canonical_json({"base_seed": base_seed, **identity})).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def _sample_id(char: str, font_asset_id: str, operator: str, seed: int) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "character": char,
                "font_asset_id": font_asset_id,
                "operator": operator,
                "seed": seed,
            }
        )
    ).hexdigest()


def _png_bytes(array: NDArray[np.uint8]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", compress_level=9, optimize=False)
    return buffer.getvalue()


def _write_reproducible_file(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing generated file has unexpected content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_name(f"{path.name}.part")
    try:
        part_path.write_bytes(payload)
        part_path.replace(path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise


def _tight_bbox(mask: NDArray[np.uint8]) -> BoundingBox:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("generated mask has no foreground")
    return BoundingBox(
        x0=int(columns.min()),
        y0=int(rows.min()),
        x1=int(columns.max()) + 1,
        y1=int(rows.max()) + 1,
    )


def _write_sample_assets(
    output_dir: Path,
    sample_id: str,
    image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
) -> tuple[str, str]:
    prefix = sample_id[:2]
    image_relative = Path("images") / prefix / f"{sample_id}.png"
    mask_relative = Path("masks") / prefix / f"{sample_id}.png"
    _write_reproducible_file(output_dir / image_relative, _png_bytes(image.astype(np.uint8)))
    mask_image = (mask.astype(bool).astype(np.uint8) * 255).astype(np.uint8)
    _write_reproducible_file(output_dir / mask_relative, _png_bytes(mask_image))
    return image_relative.as_posix(), mask_relative.as_posix()


def _normal_sample(
    config: GenerationConfig,
    char: str,
    font_path: Path | None,
    font_asset_id: str,
    index: int,
) -> GeneratedSample:
    seed = _sample_seed(
        config.seed,
        character=char,
        font_asset_id=font_asset_id,
        operator="identity",
        index=index,
    )
    sample_id = _sample_id(char, font_asset_id, "identity", seed)
    rendered = render_glyph(char, font_path, seed=seed)
    image_path, mask_path = _write_sample_assets(
        config.output_dir, sample_id, rendered.image, rendered.mask
    )
    return GeneratedSample(
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        base_char=char,
        rendered_char=char,
        decision=Decision.PASS,
        anomaly_kind=AnomalyKind.NONE,
        operator="identity",
        changed_pixels=0,
        seed=seed,
        bbox=rendered.bbox,
        source_asset_ids=(font_asset_id,),
    )


def _abnormal_sample(
    config: GenerationConfig,
    char: str,
    font_path: Path | None,
    font_asset_id: str,
    operator: str,
    index: int,
) -> GeneratedSample:
    seed = _sample_seed(
        config.seed,
        character=char,
        font_asset_id=font_asset_id,
        operator=operator,
        index=index,
    )
    sample_id = _sample_id(char, font_asset_id, operator, seed)
    rendered = render_glyph(char, font_path, seed=seed)
    corrupted = corrupt_glyph(rendered, operator=operator, seed=seed)
    image_path, mask_path = _write_sample_assets(
        config.output_dir, sample_id, corrupted.image, corrupted.changed_mask
    )
    return GeneratedSample(
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        base_char=char,
        rendered_char=char,
        decision=Decision.BLOCK,
        anomaly_kind=_OPERATOR_KINDS[operator],
        operator=operator,
        changed_pixels=corrupted.changed_pixels,
        seed=seed,
        bbox=_tight_bbox(corrupted.mask),
        source_asset_ids=(font_asset_id,),
    )


def generate_dataset(config: GenerationConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[GeneratedSample] = []
    for char in config.characters:
        for font_path, font_asset_id in zip(
            config.font_paths, config.source_asset_ids, strict=True
        ):
            for index in range(config.normal_per_char):
                samples.append(_normal_sample(config, char, font_path, font_asset_id, index))
            for operator in config.operators:
                for index in range(config.abnormal_per_operator):
                    try:
                        sample = _abnormal_sample(
                            config, char, font_path, font_asset_id, operator, index
                        )
                    except CorruptionNotApplicable:
                        continue
                    samples.append(sample)

    rows = [sample.model_dump(mode="json") for sample in samples]
    table = pa.Table.from_pylist(rows)
    manifest = config.output_dir / "manifest.parquet"
    part_path = manifest.with_name(f"{manifest.name}.part")
    try:
        pq.write_table(table, part_path, compression="zstd", version="2.6")
        verified = pq.read_table(part_path)
        if verified.num_rows != len(samples):
            raise ValueError("generated manifest row count failed verification")
        part_path.replace(manifest)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    return manifest
