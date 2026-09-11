"""Shared integrity boundary for explicitly experimental V2 glyph manifests."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictBool, model_validator

from poor_word.domain import AnomalyKind, Decision, GeneratedSample

V2_SCHEMA = "glyph-dataset-v2"


class V2Sample(GeneratedSample):
    schema_version: Literal["glyph-dataset-v2"]
    dataset_id: str = Field(min_length=1)
    split_role: Literal["train", "calibration", "test"]
    source_group_id: str = Field(min_length=1)
    pixel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_eligible: StrictBool
    production_allowed: StrictBool
    label_provenance: Literal["synthetic_rule_v2"]

    @model_validator(mode="after")
    def experimental_split_contract(self) -> "V2Sample":
        if self.decision not in {Decision.PASS, Decision.BLOCK}:
            raise ValueError("V2 experiments require PASS/BLOCK rule labels, not REVIEW")
        if self.production_allowed:
            raise ValueError("V2 stroke resources have no production clearance")
        if self.training_eligible != (self.split_role == "train"):
            raise ValueError("training_eligible must agree with the explicit split_role")
        if self.decision == Decision.PASS and (
            self.changed_pixels != 0 or self.anomaly_kind != AnomalyKind.NONE
        ):
            raise ValueError("normal V2 rows must describe unchanged legal glyph structure")
        if not self.sample_id or not self.source_asset_ids:
            raise ValueError("V2 sample identity and source assets must not be empty")
        return self


def safe_asset_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError("V2 asset path must stay inside its dataset directory")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("V2 asset path escapes its dataset directory")
    return resolved


def validate_v2_manifest(manifest: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the immutable manifest and row contract without decoding every image."""
    run_path = manifest.parent / "run.json"
    if not run_path.is_file():
        raise ValueError("V2 manifest requires its generation run.json")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if not isinstance(run, dict) or run.get("schema_version") != V2_SCHEMA:
        raise ValueError("V2 run metadata has an invalid schema_version")
    if run.get("experimental_only") is not True or run.get("production_allowed") is not False:
        raise ValueError("V2 run must retain experimental-only, non-production scope")
    inventory = run.get("manifests")
    entry = inventory.get(manifest.name) if isinstance(inventory, dict) else None
    if not isinstance(entry, dict):
        raise ValueError("V2 run does not register this manifest")
    if entry.get("sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("V2 manifest hash differs from its generation run")
    if entry.get("row_count") != len(rows) or not rows:
        raise ValueError("V2 manifest row count differs from its generation run")
    identities: set[str] = set()
    pixels: set[str] = set()
    roles: set[str] = set()
    groups: dict[str, str] = {}
    for row in rows:
        sample = V2Sample.model_validate(row)
        if sample.dataset_id != run.get("dataset_id"):
            raise ValueError("V2 row dataset_id differs from its generation run")
        if sample.sample_id in identities or sample.pixel_sha256 in pixels:
            raise ValueError("V2 manifest has duplicate sample identities or pixels")
        identities.add(sample.sample_id)
        pixels.add(sample.pixel_sha256)
        roles.add(sample.split_role)
        previous = groups.setdefault(sample.source_group_id, sample.base_char)
        if previous != sample.base_char:
            raise ValueError("V2 source group cannot contain different base characters")
        safe_asset_path(manifest.parent, sample.image_path)
        safe_asset_path(manifest.parent, sample.mask_path)
    if len(roles) != 1:
        raise ValueError("V2 manifests must have exactly one split_role")
    return run
