import pytest
from pydantic import ValidationError

from poor_word.domain import BoundingBox, Decision
from poor_word.real_data.schema import (
    CharacterAnnotation,
    ImageLabel,
    RealImageRecord,
    SplitRole,
)


def _character(decision: Decision = Decision.PASS) -> CharacterAnnotation:
    return CharacterAnnotation(
        annotation_id="char-1",
        box=BoundingBox(x0=1, y0=2, x1=20, y1=30),
        decision=decision,
        annotator_id="reviewer-1",
    )


def _image(**changes: object) -> RealImageRecord:
    payload: dict[str, object] = {
        "image_id": "image-1",
        "image_path": "images/image-1.png",
        "image_label": ImageLabel.NORMAL,
        "split_role": SplitRole.DEV,
        "source_id": "business_seed",
        "source_group_id": "upload-batch-1",
        "license_id": "LicenseRef-Proprietary",
        "production_allowed": True,
        "product_id": "product-1",
        "campaign_id": "campaign-1",
        "template_id": "template-1",
        "label_provenance": "human-image-review-v1",
        "training_eligible": True,
        "characters": (_character(),),
    }
    payload.update(changes)
    return RealImageRecord.model_validate(payload)


def test_normal_image_cannot_contain_block_character() -> None:
    with pytest.raises(ValidationError, match="normal image cannot contain BLOCK"):
        _image(characters=(_character(Decision.BLOCK),))


def test_locked_test_cannot_be_training_eligible() -> None:
    with pytest.raises(ValidationError, match="locked-test image cannot be training eligible"):
        _image(split_role=SplitRole.LOCKED_TEST)


def test_unapproved_source_cannot_be_training_eligible() -> None:
    with pytest.raises(ValidationError, match="unapproved source cannot be training eligible"):
        _image(production_allowed=False)


def test_gold_character_requires_human_annotator() -> None:
    with pytest.raises(ValidationError, match="annotator_id"):
        CharacterAnnotation(
            annotation_id="char-1",
            box=BoundingBox(x0=1, y0=2, x1=20, y1=30),
            decision=Decision.BLOCK,
        )


def test_review_character_may_remain_unassigned() -> None:
    annotation = CharacterAnnotation(
        annotation_id="char-1",
        box=BoundingBox(x0=1, y0=2, x1=20, y1=30),
        decision=Decision.REVIEW,
    )
    assert annotation.annotator_id is None
