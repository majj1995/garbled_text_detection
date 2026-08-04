from pathlib import Path

import pytest

from poor_word.glyphs.catalog import load_common_chars


def test_catalog_has_expected_unique_characters(tmp_path: Path) -> None:
    path = tmp_path / "chars.txt"
    path.write_text("甲乙丙", encoding="utf-8")

    chars = load_common_chars(path, expected_count=3)

    assert chars == ("甲", "乙", "丙")


def test_catalog_parses_numbered_tabular_source(tmp_path: Path) -> None:
    path = tmp_path / "chars.txt"
    path.write_text("0001\t一\tyī\n0002\t乙\tyǐ\n", encoding="utf-8")

    assert load_common_chars(path, expected_count=2) == ("一", "乙")


def test_catalog_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "chars.txt"
    path.write_text("甲乙甲", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_common_chars(path, expected_count=3)
