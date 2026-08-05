from poor_word.evaluation.baselines import BaselineInput, score_baselines

CATALOG = frozenset("常用天地")


def _score(text: str, *, confidence: float = 0.99, visual: float | None = None):
    return score_baselines(
        BaselineInput(
            crop_id="crop-1",
            image_id="image-1",
            fold=0,
            text=text,
            ocr_confidence=confidence,
            ocr_model_id="PP-OCRv5_server_rec",
            ocr_audit_sha256="a" * 64,
            visual_anomaly_score=visual,
        ),
        CATALOG,
        block_threshold=0.5,
    )


def test_rare_and_extension_cjk_are_review_only_without_visual_evidence() -> None:
    for text in ("龘", "𠀀"):
        result = _score(text)
        assert result.unicode_class == "CJK"
        assert result.out_of_catalog is True
        assert result.needs_review is True
        assert result.membership_risk < 0.5
        assert result.auto_block is False
        assert result.text_rule_anomaly is False


def test_text_anomalies_and_visual_evidence_are_distinguished() -> None:
    non_han = _score("A")
    replacement = _score("�")
    visual = _score("龘", visual=0.9)

    assert non_han.text_rule_anomaly is True
    assert non_han.visual_anomaly_evidence is False
    assert replacement.unicode_class == "REPLACEMENT"
    assert replacement.auto_block is True
    assert visual.out_of_catalog is True
    assert visual.visual_anomaly_evidence is True
    assert visual.auto_block is True


def test_low_ocr_confidence_requests_review_but_does_not_auto_block() -> None:
    result = _score("常", confidence=0.1)
    assert result.ocr_confidence_risk == 0.9
    assert result.needs_review is True
    assert result.auto_block is False
    assert result.auto_block_score == 0.0


def test_visual_or_structurally_invalid_evidence_sets_block_policy_score() -> None:
    assert _score("常", visual=0.9).auto_block_score == 1.0
    assert _score("�").auto_block_score == 1.0
