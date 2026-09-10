import json
from pathlib import Path

import numpy as np
import pytest


def _write_record(tmp_path: Path, **updates: object) -> Path:
    record: dict[str, object] = {
        "character": "一",
        "strokes": ["M 100 400 L 900 400 L 900 500 L 100 500 Z"],
        "medians": [[[100, 450], [900, 450]]],
    }
    record.update(updates)
    source = tmp_path / "graphics.txt"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return source


def test_loader_preserves_original_paths_and_medians(tmp_path: Path) -> None:
    from poor_word.glyphs.stroke_source import load_stroke_records

    source = tmp_path / "graphics.txt"
    source.write_text(
        json.dumps(
            {
                "character": "一",
                "strokes": ["M 100 400 L 900 400 L 900 500 Z"],
                "medians": [[[100, 450], [900, 450]]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = load_stroke_records(source)["一"]
    assert record.character == "一"
    assert record.strokes == ("M 100 400 L 900 400 L 900 500 Z",)
    assert record.medians == (((100.0, 450.0), (900.0, 450.0)),)


@pytest.mark.parametrize(
    "path",
    [
        "M 00100 100 L 900 100 L 900 200 L 100 200 Z",
        "M 01e2 1E+2 L 9e2 1e2 L 9e2 2e2 L 1e2 2e2 Z",
        "M00100+00100l+00800-0+0+00100-00800-0Z",
        "M1e2+1E+2l+8e2-0+0+1e2-8e2-0Z",
    ],
)
def test_equivalent_svg_numbers_preserve_geometry_and_original_path(
    tmp_path: Path, path: str
) -> None:
    from poor_word.glyphs.stroke_source import (
        StrokeRecord,
        load_stroke_records,
        render_stroke_layers,
    )

    canonical = StrokeRecord(
        "一", ("M 100 100 L 900 100 L 900 200 L 100 200 Z",), (((100, 150), (900, 150)),)
    )
    (expected,) = render_stroke_layers(canonical)
    loaded = load_stroke_records(_write_record(tmp_path, strokes=[path]))["一"]
    (actual,) = render_stroke_layers(loaded)
    assert loaded.strokes == (path,)
    np.testing.assert_array_equal(actual, expected)
    rows, columns = np.nonzero(actual > 127)
    assert (int(rows.min()), int(rows.max())) == (58, 69)
    assert (int(columns.min()), int(columns.max())) == (16, 111)


@pytest.mark.parametrize(
    "path",
    [
        "M100 500A400 400 0 1 1 900 500A400 400 0 1 1 100 500Z",
        "M100 500A400 400 0 11900 500A400 400 0 11100 500Z",
        "M100 500a400 400 0 1 1 800 0a400 400 0 1 1-800 0Z",
    ],
)
def test_loader_rejects_unsupported_arc_commands_explicitly(tmp_path: Path, path: str) -> None:
    from poor_word.glyphs.stroke_source import load_stroke_records

    with pytest.raises(ValueError, match="arc commands are not supported"):
        load_stroke_records(_write_record(tmp_path, strokes=[path]))


@pytest.mark.parametrize(
    "updates",
    [
        {"character": ""},
        {"character": "两个"},
        {"character": 1},
        {"strokes": []},
        {"strokes": [""]},
        {"strokes": [123]},
        {"strokes": ["M 0 0 L 1 1"]},
        {"strokes": ["M 0 0 L 1 1 Z junk"]},
        {"strokes": ["<svg><use href='https://example.com'/></svg>"]},
        {"strokes": ["M 0 0 L 1e309 1 Z"]},
        {"strokes": ["M 0 0 L 999999 1 Z"]},
        {"strokes": ["M 0 0 Z"]},
        {"medians": []},
        {"medians": [[]]},
        {"medians": [[[1, 2]]]},
        {"medians": [[[1, 2, 3], [2, 3]]]},
        {"medians": [[[True, 2], [2, 3]]]},
        {"medians": [[[float("nan"), 2], [2, 3]]]},
        {"medians": [[[1, float("inf")], [2, 3]]]},
        {"medians": [[[999999, 2], [2, 3]]]},
    ],
)
def test_loader_rejects_invalid_records(tmp_path: Path, updates: dict[str, object]) -> None:
    from poor_word.glyphs.stroke_source import load_stroke_records

    with pytest.raises(ValueError):
        load_stroke_records(_write_record(tmp_path, **updates))


def test_loader_rejects_duplicate_characters_even_with_filter(tmp_path: Path) -> None:
    from poor_word.glyphs.stroke_source import load_stroke_records

    source = _write_record(tmp_path)
    source.write_text(source.read_text() * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_stroke_records(source, characters=("二",))


def test_loader_rejects_missing_requested_character_instead_of_tofu(tmp_path: Path) -> None:
    from poor_word.glyphs.stroke_source import load_stroke_records

    with pytest.raises(ValueError, match="二"):
        load_stroke_records(_write_record(tmp_path), characters=("二",))


def test_loader_filters_requested_characters(tmp_path: Path) -> None:
    from poor_word.glyphs.stroke_source import load_stroke_records

    source = _write_record(tmp_path)
    text = source.read_text()
    source.write_text(text + text.replace("\\u4e00", "\\u4e8c"), encoding="utf-8")
    assert tuple(load_stroke_records(source, characters=("二",))) == ("二",)


@pytest.mark.parametrize("total, expected", [(1, [1]), (250, [250]), (501, [250, 500, 501])])
def test_loader_reports_actual_processed_rows_without_duplicate_final_count(
    tmp_path: Path, total: int, expected: list[int]
) -> None:
    from poor_word.glyphs.stroke_source import load_stroke_records

    source = _write_record(tmp_path)
    template = json.loads(source.read_text())
    source.write_text(
        "\n".join(
            json.dumps({**template, "character": chr(0x4E00 + index)}) for index in range(total)
        )
        + "\n",
        encoding="utf-8",
    )
    counts: list[int] = []
    records = load_stroke_records(source, characters=("一",), progress=counts.append)
    assert counts == expected
    assert tuple(records) == ("一",)


def test_render_keeps_shared_geometry_y_up_and_independent_overlap() -> None:
    from poor_word.glyphs.stroke_source import StrokeRecord, render_stroke_layers

    # Both strokes cross at (500, 500); the short upper arm must stay above the cross.
    record = StrokeRecord(
        "十",
        (
            "M 100 450 L 900 450 L 900 550 L 100 550 Z",
            "M 450 100 L 550 100 L 550 700 L 450 700 Z",
        ),
        (((100, 500), (900, 500)), ((500, 100), (500, 700))),
    )
    horizontal, vertical = render_stroke_layers(record)
    assert horizontal.shape == vertical.shape == (128, 128)
    assert horizontal.dtype == vertical.dtype == np.uint8
    assert horizontal[52, 64] == vertical[52, 64] == 255
    assert vertical[20, 64] == 0
    assert vertical[85, 64] == 255
    assert horizontal[52, 20] == 255
    assert vertical[52, 20] == 0
    rows, cols = np.nonzero(np.maximum(horizontal, vertical) > 127)
    assert (int(cols.min()), int(cols.max())) == (16, 111)
    assert (int(rows.min()), int(rows.max())) == (28, 99)
    assert np.any((horizontal > 0) & (horizontal < 255))
    assert np.array_equal(horizontal, render_stroke_layers(record)[0])


def test_render_preserves_curved_outline_not_just_median() -> None:
    from poor_word.glyphs.stroke_source import StrokeRecord, render_stroke_layers

    # Quadratic arch bounds: x=[100,900], y=[100,500], not control-point y=900.
    record = StrokeRecord("一", ("M 100 100 Q 500 900 900 100 Z",), (((100, 100), (900, 100)),))
    (layer,) = render_stroke_layers(record)
    assert layer[45, 64] == 255
    assert layer[45, 20] == 0
    assert layer[80, 64] == 255
    rows, _ = np.nonzero(layer > 127)
    assert (int(rows.min()), int(rows.max())) == (40, 87)


def test_render_holes_follow_svg_nonzero_winding() -> None:
    from poor_word.glyphs.stroke_source import StrokeRecord, render_stroke_layers

    outer = "M 100 100 L 900 100 L 900 900 L 100 900 Z "
    inner_opposite = "M 300 300 L 300 700 L 700 700 L 700 300 Z"
    inner_same = "M 300 300 L 700 300 L 700 700 L 300 700 Z"
    (hole,) = render_stroke_layers(
        StrokeRecord("口", (outer + inner_opposite,), (((100, 100), (900, 900)),))
    )
    (solid,) = render_stroke_layers(
        StrokeRecord("口", (outer + inner_same,), (((100, 100), (900, 900)),))
    )
    assert hole[64, 64] == 0
    assert hole[64, 25] == 255
    assert solid[64, 64] == 255


@pytest.mark.parametrize("size", [0, -1, 32, 96, True, 128.5])
def test_render_rejects_canvas_without_room_for_96px_glyph(size: int) -> None:
    from poor_word.glyphs.stroke_source import StrokeRecord, render_stroke_layers

    record = StrokeRecord("一", ("M 100 400 L 900 400 L 900 500 Z",), (((100, 450), (900, 450)),))
    with pytest.raises(ValueError):
        render_stroke_layers(record, canvas_size=size)
