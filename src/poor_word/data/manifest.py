import tomllib
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validate_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source URL must use HTTP or HTTPS")
    return value


def _validate_output_name(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError("output_name must be a plain file name")
    return value


class SourceSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    url: str
    license_id: str = Field(min_length=1)
    production_allowed: bool
    output_name: str

    _http_url = field_validator("url")(_validate_http_url)
    _safe_output_name = field_validator("output_name")(_validate_output_name)


class SourceLock(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    declared_url: str
    resolved_url: str
    output_name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    license_id: str = Field(min_length=1)
    production_allowed: bool

    _declared_http_url = field_validator("declared_url")(_validate_http_url)
    _resolved_http_url = field_validator("resolved_url")(_validate_http_url)
    _safe_output_name = field_validator("output_name")(_validate_output_name)


def load_source_specs(path: Path) -> tuple[SourceSpec, ...]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_sources = payload.get("source")
    if not isinstance(raw_sources, list):
        raise ValueError("source manifest must contain one or more [[source]] entries")

    sources = tuple(SourceSpec.model_validate(item) for item in raw_sources)
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source manifest contains duplicate source_id values")
    return sources


def load_source_lock(path: Path) -> SourceLock:
    return SourceLock.model_validate_json(path.read_text(encoding="utf-8"))


def validate_production_sources(sources: Iterable[SourceSpec | SourceLock]) -> None:
    rejected = sorted(source.source_id for source in sources if not source.production_allowed)
    if rejected:
        joined = ", ".join(rejected)
        raise ValueError(f"production profile contains unapproved sources: {joined}")
