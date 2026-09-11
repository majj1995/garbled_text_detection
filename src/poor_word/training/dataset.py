import hashlib
import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from poor_word.glyphs.v2_manifest import V2_SCHEMA, safe_asset_path, validate_v2_manifest
from poor_word.training.augmentation import AugmentationTrace, augment_affine, augmentation_seed


@dataclass(frozen=True)
class GlyphItem:
    views: Tensor
    decision: str
    base_char: str
    label_id: int
    sample_id: str


class GlyphDataset(Dataset[GlyphItem]):
    def __init__(self, manifest: Path) -> None:
        self.manifest = manifest
        self.root = manifest.parent
        self.rows = cast(list[dict[str, Any]], pq.read_table(manifest).to_pylist())
        if not self.rows:
            raise ValueError("glyph manifest contains no rows")
        self.schema_version: str | None = None
        self.dataset_id: str | None = None
        self.split_role: str | None = None
        self.experimental_only = False
        self.run_metadata: dict[str, Any] = {}
        run_path = manifest.parent / "run.json"
        run = json.loads(run_path.read_text()) if run_path.is_file() else {}
        has_version = any("schema_version" in row for row in self.rows)
        if has_version or (isinstance(run, dict) and run.get("schema_version") == V2_SCHEMA):
            self.run_metadata = validate_v2_manifest(manifest, self.rows)
            self.schema_version = V2_SCHEMA
            self.dataset_id = str(self.run_metadata["dataset_id"])
            self.split_role = str(self.rows[0]["split_role"])
            self.experimental_only = True
        characters = sorted({str(row["base_char"]) for row in self.rows})
        self.char_to_id = {character: index for index, character in enumerate(characters)}
        self._source_alphas: OrderedDict[
            tuple[str, str, str, tuple[int, ...]], NDArray[np.uint8]
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_pixels(self, index: int) -> tuple[NDArray[np.uint8], bytes]:
        row = self.rows[index]
        sample_id = str(row["sample_id"])
        image_path = self.root / str(row["image_path"])
        mask_path = self.root / str(row["mask_path"])
        try:
            if self.schema_version == V2_SCHEMA:
                image_path = safe_asset_path(self.root, str(row["image_path"]))
                mask_path = safe_asset_path(self.root, str(row["mask_path"]))
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError("image or mask file is missing")
            image_bytes = image_path.read_bytes()
            mask_bytes = b""
            if self.schema_version == V2_SCHEMA:
                if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
                    raise ValueError("V2 image hash mismatch")
                mask_bytes = mask_path.read_bytes()
                if hashlib.sha256(mask_bytes).hexdigest() != row["mask_sha256"]:
                    raise ValueError("V2 mask hash mismatch")
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if self.schema_version == V2_SCHEMA:
                if hashlib.sha256(rgb.tobytes()).hexdigest() != row["pixel_sha256"]:
                    raise ValueError("V2 decoded pixel hash mismatch")
        except Exception as error:
            raise ValueError(f"could not load generated sample {sample_id}: {error}") from error
        return rgb, mask_bytes

    def _item(self, index: int, grayscale: NDArray[np.uint8]) -> GlyphItem:
        row = self.rows[index]
        grayscale = np.asarray(
            cv2.resize(grayscale, (96, 96), interpolation=cv2.INTER_AREA), dtype=np.uint8
        )
        glyph_mask = (grayscale > 8).astype(np.float32)
        edges = cv2.Canny(grayscale, threshold1=50, threshold2=150).astype(np.float32) / 255.0
        views = np.stack((grayscale.astype(np.float32) / 255.0, glyph_mask, edges), axis=0)
        base_char = str(row["base_char"])
        return GlyphItem(
            views=torch.from_numpy(views),
            decision=str(row["decision"]),
            base_char=base_char,
            label_id=self.char_to_id[base_char],
            sample_id=str(row["sample_id"]),
        )

    def __getitem__(self, index: int) -> GlyphItem:
        rgb, _ = self._load_pixels(index)
        return self._item(index, np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), dtype=np.uint8))

    def _normal_reference(
        self, index: int, gray: NDArray[np.uint8], mask_bytes: bytes
    ) -> NDArray[np.uint8]:
        row = self.rows[index]
        try:
            relative, expected_hash = row.get("source_layers_path"), row.get("source_layers_sha256")
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise ValueError("affine BLOCK views require source_layers_path and hash")
            path = safe_asset_path(self.root, relative)
            source = path.read_bytes()
            if hashlib.sha256(source).hexdigest() != expected_hash:
                raise ValueError("source layers hash mismatch")
            key = (str(path), expected_hash, str(row["base_char"]), gray.shape)
            if key not in self._source_alphas:
                with np.load(io.BytesIO(source), allow_pickle=False) as payload:
                    if str(payload["source_char"]) != row["base_char"]:
                        raise ValueError("source layers character mismatch")
                    names = sorted(name for name in payload.files if name.startswith("stroke_"))
                    if not names or len(names) > 64:
                        raise ValueError("invalid source stroke inventory")
                    layers = [payload[name] for name in names]
                    if any(
                        layer.dtype != np.uint8 or layer.shape != gray.shape for layer in layers
                    ):
                        raise ValueError("source layers dimensions or dtype mismatch")
                    original = np.maximum.reduce(layers)
                # At 128x128, the full 3500-character cache occupies about 55 MiB.
                if len(self._source_alphas) >= 4096:
                    self._source_alphas.popitem(last=False)
                self._source_alphas[key] = original
            self._source_alphas.move_to_end(key)
            original = self._source_alphas[key]
            raw_metrics = row.get("metrics")
            if not isinstance(raw_metrics, str):
                raise ValueError("affine BLOCK views require recorded appearance metrics")
            metrics = json.loads(raw_metrics)
            scale, angle, tx, ty = [
                float(metrics[name])
                for name in (
                    "appearance_scale",
                    "appearance_rotation_degrees",
                    "appearance_translate_x",
                    "appearance_translate_y",
                )
            ]
            if not np.all(np.isfinite([scale, angle, tx, ty])) or scale <= 0:
                raise ValueError("invalid recorded appearance transform")
            height, width = gray.shape
            matrix = np.asarray(
                cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, scale),
                dtype=np.float64,
            )
            matrix[:, 2] += (tx, ty)
            reference = np.asarray(
                cv2.warpAffine(
                    original,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0.0,),
                ),
                dtype=np.uint8,
            )
            with Image.open(io.BytesIO(mask_bytes)) as mask_image:
                edit_mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
            actual_edit = reference != gray
            if (
                edit_mask.shape != gray.shape
                or not np.array_equal(actual_edit, edit_mask)
                or int(actual_edit.sum()) != row["changed_pixels"]
            ):
                raise ValueError("reconstructed normal reference does not match the recorded edit")
        except Exception as error:
            raise ValueError(
                f"invalid augmentation reference for {row['sample_id']}: {error}"
            ) from error
        return reference

    def augmented_item(
        self, index: int, *, seed: int, epoch: int, step: int, draw: int
    ) -> tuple[GlyphItem, AugmentationTrace]:
        if self.schema_version != V2_SCHEMA or self.split_role != "train":
            raise ValueError("affine augmentation requires a V2 train manifest")
        rgb, mask_bytes = self._load_pixels(index)
        gray = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
        reference = (
            self._normal_reference(index, gray, mask_bytes)
            if self.rows[index]["decision"] == "BLOCK"
            else None
        )
        draw_seed = augmentation_seed(seed, epoch, step, draw, str(self.rows[index]["sample_id"]))
        augmented, trace = augment_affine(gray, seed=draw_seed, reference=reference)
        return self._item(index, augmented), trace

    def split_indices(self, train_fraction: float = 0.8) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between zero and one")
        if self.schema_version == V2_SCHEMA:
            indices = tuple(range(len(self.rows)))
            return (indices, ()) if self.split_role == "train" else ((), indices)
        train: list[int] = []
        validation: list[int] = []
        for index, row in enumerate(self.rows):
            source_ids = row.get("source_asset_ids") or ["unknown"]
            group = f"{row['base_char']}\0{source_ids[0]}".encode()
            bucket = int.from_bytes(hashlib.sha256(group).digest()[:8], "big") / (2**64 - 1)
            (train if bucket < train_fraction else validation).append(index)
        return tuple(train), tuple(validation)
