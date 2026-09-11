import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from poor_word.glyphs.v2_manifest import V2_SCHEMA, safe_asset_path, validate_v2_manifest


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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> GlyphItem:
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
            if self.schema_version == V2_SCHEMA:
                if hashlib.sha256(image_bytes).hexdigest() != row["image_sha256"]:
                    raise ValueError("V2 image hash mismatch")
                if hashlib.sha256(mask_path.read_bytes()).hexdigest() != row["mask_sha256"]:
                    raise ValueError("V2 mask hash mismatch")
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if self.schema_version == V2_SCHEMA:
                if hashlib.sha256(rgb.tobytes()).hexdigest() != row["pixel_sha256"]:
                    raise ValueError("V2 decoded pixel hash mismatch")
            grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            grayscale = cv2.resize(grayscale, (96, 96), interpolation=cv2.INTER_AREA)
            glyph_mask = (grayscale > 8).astype(np.float32)
            edges = cv2.Canny(grayscale, threshold1=50, threshold2=150).astype(np.float32) / 255.0
            views = np.stack((grayscale.astype(np.float32) / 255.0, glyph_mask, edges), axis=0)
        except Exception as error:
            raise ValueError(f"could not load generated sample {sample_id}: {error}") from error
        base_char = str(row["base_char"])
        return GlyphItem(
            views=torch.from_numpy(views),
            decision=str(row["decision"]),
            base_char=base_char,
            label_id=self.char_to_id[base_char],
            sample_id=sample_id,
        )

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
