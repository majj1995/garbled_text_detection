from pathlib import Path

import pytest

from poor_word.glyphs.generate import GenerationConfig, generate_dataset


@pytest.fixture
def generated_manifest(tmp_path: Path) -> Path:
    return generate_dataset(
        GenerationConfig(
            output_dir=tmp_path / "generated",
            characters=("A", "B"),
            font_paths=(None,),
            normal_per_char=1,
            abnormal_per_operator=1,
            operators=("erase_segment", "add_stroke"),
            seed=31,
            source_asset_ids=("fixture_font",),
        )
    )
