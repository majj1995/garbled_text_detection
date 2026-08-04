import hashlib
import json
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from PIL import Image, ImageDraw

from poor_word.real_data.split import assign_group_folds, perceptual_hash


def _image(path: Path, offset: int) -> str:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8 + offset, 10, 28 + offset, 52), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    image_id: str,
    image_sha256: str,
    *,
    product: str,
    campaign: str,
    template: str,
    source_group: str | None = None,
    role: str = "DEV",
    label: str = "NORMAL",
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "image_path": f"images/{image_id}.png",
        "image_sha256": image_sha256,
        "image_label": label,
        "split_role": role,
        "source_id": "business_seed",
        "source_group_id": source_group or f"upload-{image_id}",
        "product_id": product,
        "campaign_id": campaign,
        "template_id": template,
    }


def _manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_perceptual_hash_keeps_near_duplicate_shape_close(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _image(first, 0)
    _image(second, 1)

    first_hash = perceptual_hash(first)
    second_hash = perceptual_hash(second)

    assert (first_hash ^ second_hash).bit_count() <= 10


def test_group_relationships_are_transitively_closed_and_deterministic(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "root"
    hashes = {
        name: _image(image_root / f"images/{name}.png", offset)
        for name, offset in (("a", 0), ("b", 10), ("c", 20), ("d", 30))
    }
    rows = [
        _row("a", hashes["a"], product="p-shared", campaign="c-a", template="t-a"),
        _row("b", hashes["b"], product="p-shared", campaign="c-shared", template="t-b"),
        _row("c", hashes["c"], product="p-c", campaign="c-shared", template="t-c"),
        _row(
            "d",
            hashes["d"],
            product="p-d",
            campaign="c-d",
            template="t-d",
            label="ABNORMAL",
        ),
    ]
    manifest = _manifest(tmp_path / "manifest.parquet", rows)

    first = assign_group_folds(manifest, image_root, tmp_path / "split-a", folds=2, seed=9)
    second = assign_group_folds(manifest, image_root, tmp_path / "split-b", folds=2, seed=9)

    first_rows = {row["image_id"]: row for row in pq.read_table(first.folds).to_pylist()}
    second_rows = pq.read_table(second.folds).to_pylist()
    assert first_rows["a"]["component_id"] == first_rows["b"]["component_id"]
    assert first_rows["b"]["component_id"] == first_rows["c"]["component_id"]
    assert first_rows["a"]["fold"] == first_rows["c"]["fold"]
    assert first_rows["a"]["component_id"] != first_rows["d"]["component_id"]
    assert pq.read_table(first.folds).to_pylist() == second_rows
    audit = json.loads(first.audit.read_text(encoding="utf-8"))
    assert audit["leakage_violations"] == []


def test_exact_duplicates_join_components_even_with_different_groups(tmp_path: Path) -> None:
    image_root = tmp_path / "root"
    first_hash = _image(image_root / "images/a.png", 0)
    (image_root / "images/b.png").write_bytes((image_root / "images/a.png").read_bytes())
    rows = [
        _row("a", first_hash, product="p-a", campaign="c-a", template="t-a"),
        _row("b", first_hash, product="p-b", campaign="c-b", template="t-b"),
    ]
    manifest = _manifest(tmp_path / "manifest.parquet", rows)

    artifacts = assign_group_folds(manifest, image_root, tmp_path / "split", folds=2)
    split_rows = pq.read_table(artifacts.folds).to_pylist()

    assert split_rows[0]["component_id"] == split_rows[1]["component_id"]


def test_locked_test_component_cannot_overlap_development(tmp_path: Path) -> None:
    image_root = tmp_path / "root"
    hashes = {
        name: _image(image_root / f"images/{name}.png", offset)
        for name, offset in (("dev", 0), ("locked", 20))
    }
    rows = [
        _row("dev", hashes["dev"], product="same", campaign="c-a", template="t-a"),
        _row(
            "locked",
            hashes["locked"],
            product="same",
            campaign="c-b",
            template="t-b",
            role="LOCKED_TEST",
            label="ABNORMAL",
        ),
    ]
    manifest = _manifest(tmp_path / "manifest.parquet", rows)

    with pytest.raises(ValueError, match="locked-test leakage"):
        assign_group_folds(manifest, image_root, tmp_path / "split", folds=2)
