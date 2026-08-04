import hashlib
from pathlib import Path

import pyarrow.parquet as pq

from poor_word.glyphs.generate import GenerationConfig, generate_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_writes_deterministic_manifest(tmp_path: Path) -> None:
    first_config = GenerationConfig(
        output_dir=tmp_path / "run-a",
        characters=("文", "字"),
        font_paths=(Path("data/raw/NotoSansCJKsc-Regular.otf"),),
        normal_per_char=1,
        abnormal_per_operator=1,
        operators=("erase_segment", "add_stroke"),
        seed=23,
        source_asset_ids=("noto_sans_sc_regular",),
    )
    second_config = first_config.model_copy(update={"output_dir": tmp_path / "run-b"})

    first_manifest = generate_dataset(first_config)
    second_manifest = generate_dataset(second_config)
    table = pq.read_table(first_manifest)

    assert table.num_rows == 6
    assert set(table.column("decision").to_pylist()) == {"PASS", "BLOCK"}
    assert len(set(table.column("sample_id").to_pylist())) == 6
    assert _sha256(first_manifest) == _sha256(second_manifest)
    for relative_path in table.column("image_path").to_pylist():
        assert _sha256(first_config.output_dir / relative_path) == _sha256(
            second_config.output_dir / relative_path
        )
