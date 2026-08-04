"""Auditable OCR and Unicode baselines with a fail-safe rare-character policy."""

import math
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

UnicodeClass = Literal["CJK", "OTHER", "PRIVATE_USE", "REPLACEMENT", "NONCHARACTER"]


class BaselineInput(BaseModel):
    """One OCR character observation aligned to a character OOF row."""

    model_config = ConfigDict(frozen=True)

    crop_id: str = Field(min_length=1)
    image_id: str = Field(min_length=1)
    fold: int = Field(ge=0)
    text: str = Field(min_length=1)
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    ocr_model_id: str = Field(min_length=1)
    ocr_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    visual_anomaly_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("text")
    @classmethod
    def one_codepoint(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("baseline OCR text must contain exactly one Unicode code point")
        return value


class BaselineScores(BaseModel):
    """Scores and policy flags kept separate so text rules cannot impersonate vision."""

    model_config = ConfigDict(frozen=True)

    ocr_confidence_risk: float = Field(ge=0.0, le=1.0)
    membership_risk: float = Field(ge=0.0, le=1.0)
    combined_risk: float = Field(ge=0.0, le=1.0)
    unicode_class: UnicodeClass
    out_of_catalog: bool
    text_rule_anomaly: bool
    visual_anomaly_evidence: bool
    needs_review: bool
    auto_block: bool
    block_score_semantics: str


def _is_cjk(codepoint: int) -> bool:
    ranges = (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B73F),
        (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x30000, 0x323AF),
        (0xF900, 0xFAFF),
        (0x2F800, 0x2FA1F),
    )
    return any(start <= codepoint <= end for start, end in ranges)


def _unicode_class(text: str) -> UnicodeClass:
    codepoint = ord(text)
    if text == "\ufffd":
        return "REPLACEMENT"
    if unicodedata.category(text) == "Co":
        return "PRIVATE_USE"
    if 0xFDD0 <= codepoint <= 0xFDEF or codepoint & 0xFFFF in {0xFFFE, 0xFFFF}:
        return "NONCHARACTER"
    if _is_cjk(codepoint):
        return "CJK"
    return "OTHER"


def score_baselines(
    item: BaselineInput,
    common_chars: frozenset[str] | set[str],
    *,
    block_threshold: float = 0.5,
) -> BaselineScores:
    """Score one OCR observation without treating valid rare CJK as malformed.

    Membership is deliberately a review rule, not visual proof. Valid CJK outside the
    catalog receives a score strictly below the configured BLOCK threshold.
    """
    if not math.isfinite(block_threshold) or not 0.0 < block_threshold <= 1.0:
        raise ValueError("block_threshold must be finite and within (0,1]")
    unicode_class = _unicode_class(item.text)
    out_of_catalog = item.text not in common_chars
    unsafe_text = unicode_class in {"PRIVATE_USE", "REPLACEMENT", "NONCHARACTER"}
    other_text = unicode_class == "OTHER"
    if unsafe_text:
        membership_risk = 1.0
    elif unicode_class == "CJK" and out_of_catalog:
        membership_risk = max(0.0, min(0.49, math.nextafter(block_threshold, 0.0)))
    elif other_text:
        membership_risk = min(0.49, math.nextafter(block_threshold, 0.0))
    else:
        membership_risk = 0.0
    ocr_risk = 1.0 - item.ocr_confidence
    visual_score = item.visual_anomaly_score or 0.0
    visual_evidence = item.visual_anomaly_score is not None and visual_score >= block_threshold
    auto_block = unsafe_text or visual_evidence
    needs_review = (
        out_of_catalog
        or unsafe_text
        or other_text
        or item.ocr_confidence < block_threshold
        or item.visual_anomaly_score is not None
    )
    return BaselineScores(
        ocr_confidence_risk=ocr_risk,
        membership_risk=membership_risk,
        combined_risk=max(ocr_risk, membership_risk, visual_score),
        unicode_class=unicode_class,
        out_of_catalog=out_of_catalog,
        text_rule_anomaly=unsafe_text or other_text,
        visual_anomaly_evidence=visual_evidence,
        needs_review=needs_review,
        auto_block=auto_block,
        block_score_semantics=(
            "membership is review-only for valid out-of-catalog CJK; automatic BLOCK "
            "requires structurally invalid Unicode or independent visual evidence"
        ),
    )
