from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Decision(StrEnum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


class AnomalyKind(StrEnum):
    NONE = "none"
    MISSING_STROKE = "missing_stroke"
    EXTRA_STROKE = "extra_stroke"
    BROKEN_STROKE = "broken_stroke"
    BRIDGE = "bridge"
    COMPONENT_SHIFT = "component_shift"
    FUSION = "fusion"


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    x1: int = Field(gt=0)
    y1: int = Field(gt=0)

    @model_validator(mode="after")
    def positive_area(self) -> "BoundingBox":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have positive area")
        return self


class GeneratedSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    image_path: str
    mask_path: str
    base_char: str = Field(min_length=1, max_length=1)
    rendered_char: str = Field(min_length=1, max_length=2)
    decision: Decision
    anomaly_kind: AnomalyKind
    operator: str
    changed_pixels: int = Field(ge=0)
    seed: int = Field(ge=0)
    bbox: BoundingBox
    source_asset_ids: tuple[str, ...]

    @model_validator(mode="after")
    def anomaly_changes_pixels(self) -> "GeneratedSample":
        if self.decision is Decision.BLOCK and self.changed_pixels == 0:
            raise ValueError("anomalous samples must change at least one pixel")
        return self
