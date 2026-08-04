from pydantic import BaseModel, ConfigDict, Field

from poor_word.domain import BoundingBox


class OcrCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class OcrCharacter(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    box: BoundingBox
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: tuple[OcrCandidate, ...] = ()
    logits: tuple[float, ...] | None = None

    @property
    def logits_available(self) -> bool:
        return self.logits is not None


class OcrLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    polygon: tuple[tuple[float, float], ...]
    characters: tuple[OcrCharacter, ...] = ()


class OcrResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    lines: tuple[OcrLine, ...]
    stage_ms: dict[str, float] = Field(default_factory=dict)


class OcrAudit(BaseModel):
    model_config = ConfigDict(frozen=True)

    paddleocr_version: str
    paddlepaddle_version: str
    cuda_version: str | None
    gpu_name: str | None
    detection_model_name: str
    recognition_model_name: str
    character_boxes_available: bool
    logits_available: bool
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0.0)
    required_capability_gaps: tuple[str, ...] = ()
