from pathlib import Path

import pytest
from pydantic import ValidationError

from poor_word.config import PathsConfig
from poor_word.domain import AnomalyKind, BoundingBox, Decision, GeneratedSample


def test_generated_sample_requires_changed_pixels_for_anomaly() -> None:
    with pytest.raises(ValidationError, match="at least one pixel"):
        GeneratedSample(
            sample_id="abc",
            image_path="images/abc.png",
            mask_path="masks/abc.png",
            base_char="文",
            rendered_char="文",
            decision=Decision.BLOCK,
            anomaly_kind=AnomalyKind.MISSING_STROKE,
            operator="erase_segment",
            changed_pixels=0,
            seed=7,
            bbox=BoundingBox(x0=1, y0=1, x1=30, y1=30),
            source_asset_ids=("noto_sans_sc",),
        )


def test_bounding_box_has_positive_area() -> None:
    with pytest.raises(ValidationError, match="positive area"):
        BoundingBox(x0=4, y0=3, x1=4, y1=9)


def test_paths_config_creates_only_generated_runtime_directories(tmp_path: Path) -> None:
    config = PathsConfig(repo_root=tmp_path)

    created = config.create_runtime_dirs()

    assert created == (
        tmp_path / "data/generated",
        tmp_path / "artifacts",
        tmp_path / "models",
    )
    assert all(path.is_dir() for path in created)
    assert not (tmp_path / "data/raw").exists()
