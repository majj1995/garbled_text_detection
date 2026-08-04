from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from poor_word.domain import AnomalyKind, BoundingBox, Decision


class ImageLabel(StrEnum):
    NORMAL = "NORMAL"
    ABNORMAL = "ABNORMAL"


class SplitRole(StrEnum):
    DEV = "DEV"
    LOCKED_TEST = "LOCKED_TEST"
    IMAGE_ONLY = "IMAGE_ONLY"
    NORMAL_REPLAY = "NORMAL_REPLAY"


class CharacterAnnotation(BaseModel):
    model_config = ConfigDict(frozen=True)

    annotation_id: str = Field(min_length=1, pattern=r"^[^/\\]+$")
    box: BoundingBox
    decision: Decision
    annotator_id: str | None = Field(default=None, min_length=1)
    text: str | None = Field(default=None, min_length=1, max_length=2)
    anomaly_kind: AnomalyKind = AnomalyKind.NONE

    @model_validator(mode="after")
    def gold_requires_annotator(self) -> "CharacterAnnotation":
        if self.decision in {Decision.PASS, Decision.BLOCK} and self.annotator_id is None:
            raise ValueError("PASS/BLOCK gold character requires annotator_id")
        return self


class RealImageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_id: str = Field(min_length=1, pattern=r"^[^/\\]+$")
    image_path: Path
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    image_label: ImageLabel
    split_role: SplitRole
    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source_group_id: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    production_allowed: bool
    product_id: str = Field(min_length=1)
    campaign_id: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    label_provenance: str = Field(min_length=1)
    training_eligible: bool
    characters: tuple[CharacterAnnotation, ...] = ()

    @field_validator("image_path")
    @classmethod
    def image_path_is_relative_file(cls, value: Path) -> Path:
        if value.is_absolute() or not value.name or value in {Path("."), Path("..")}:
            raise ValueError("image_path must be a relative file path")
        return value

    @model_validator(mode="after")
    def validate_label_and_training_contract(self) -> "RealImageRecord":
        annotation_ids = [annotation.annotation_id for annotation in self.characters]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("character annotation_id values must be unique within an image")
        if self.image_label is ImageLabel.NORMAL and any(
            annotation.decision is Decision.BLOCK for annotation in self.characters
        ):
            raise ValueError("normal image cannot contain BLOCK character")
        if self.split_role is SplitRole.LOCKED_TEST and self.training_eligible:
            raise ValueError("locked-test image cannot be training eligible")
        if not self.production_allowed and self.training_eligible:
            raise ValueError("unapproved source cannot be training eligible")
        return self
