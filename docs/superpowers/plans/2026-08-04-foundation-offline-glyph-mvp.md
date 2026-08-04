# Foundation and Offline Glyph MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Python/uv project that downloads license-tracked seed assets, creates deterministically labeled normal and malformed Chinese glyphs, audits PP-OCRv5 server output, trains a first open-set glyph encoder, and reports offline metrics at the 0.1% production base rate.

**Architecture:** This plan delivers the first independently testable subsystem from the approved design. Data acquisition, glyph generation, OCR adaptation, embedding/prototype scoring, and evaluation communicate through typed domain models and Parquet manifests. GPU-specific PP-OCRv5 checks run on the L20 experiment host; all unit tests and a reduced synthetic smoke run remain CPU-compatible.

**Tech Stack:** Python 3.11, uv, PyTorch/torchvision, PaddleOCR 3.2+ with PP-OCRv5 server, OpenCV, Pillow, scikit-image, NumPy, PyArrow/Parquet, Pydantic, Typer, httpx, scikit-learn, pytest, Ruff, mypy.

## Global Constraints

- Python is exactly the 3.11 minor line; `uv.lock` is the dependency source of truth.
- The OCR models are `PP-OCRv5_server_det` and `PP-OCRv5_server_rec`; mobile models are out of scope.
- The L20 experiment GPU has 48 GB memory, but CPU tests must not import PaddlePaddle or initialize CUDA.
- Legal characters default to the 3,500 characters in the 2013 first-level General Standard Chinese Character table.
- A source without a recorded URL, resolved URL, SHA-256, license identifier, and production-use decision cannot be consumed by generators or training.
- Data whose commercial training permission is unclear must have `production_allowed = false` and cannot enter the production profile.
- Generated samples record source character, operator, seed, mask, character box, and source asset IDs.
- Automatic BLOCK metrics are reported at a 0.1% anomaly base rate; balanced-set precision is never presented as production precision.
- Reference input limits are long edge ≤ 2,048 pixels and ≤ 100 characters.
- The service/API, weakly supervised MIL, string anomaly model, fusion calibrator, and TensorRT optimization are separate implementation plans after this offline glyph MVP.

## File Structure

```text
pyproject.toml                         # uv metadata, dependencies, tools, CLI entry point
.python-version                       # pins Python 3.11
README.md                             # setup and reproducible MVP commands
data/sources.toml                     # reviewed source declarations
src/poor_word/cli.py                  # Typer command tree
src/poor_word/config.py               # filesystem and run settings
src/poor_word/domain.py               # shared enums and Pydantic records
src/poor_word/data/manifest.py        # source declaration/lock models
src/poor_word/data/download.py        # streaming, locking, checksum verification
src/poor_word/glyphs/catalog.py       # validates and loads the 3,500-character catalog
src/poor_word/glyphs/render.py        # deterministic glyph rendering
src/poor_word/glyphs/corrupt.py       # deterministic malformed-glyph operators
src/poor_word/glyphs/generate.py      # PNG/mask generation and Parquet manifest writing
src/poor_word/ocr/types.py             # OCR-neutral typed outputs
src/poor_word/ocr/paddle_v5.py         # lazy PP-OCRv5 server adapter and audit
src/poor_word/models/glyph_encoder.py  # ConvNeXt-Tiny based 3-view encoder
src/poor_word/models/prototypes.py     # per-character prototype bank and OOD scores
src/poor_word/training/dataset.py      # Parquet-backed generated glyph dataset
src/poor_word/training/train_glyph.py  # supervised contrastive/classification smoke trainer
src/poor_word/evaluation/metrics.py    # base-rate-aware metrics
src/poor_word/evaluation/report.py     # JSON/Markdown offline report
tests/                                # unit and integration tests mirroring src modules
```

---

### Task 1: Initialize the Python 3.11 uv project and quality gates

**Files:**
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/poor_word/__init__.py`
- Create: `src/poor_word/cli.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Consumes: no project code.
- Produces: `poor-word` CLI entry point and a locked Python 3.11 development environment used by every later task.

- [ ] **Step 1: Write the package and CLI smoke test**

```python
from typer.testing import CliRunner

from poor_word import __version__
from poor_word.cli import app


def test_package_version_and_doctor() -> None:
    assert __version__ == "0.1.0"
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "python=3.11" in result.stdout
    assert "gpu=not-required-for-unit-tests" in result.stdout
```

- [ ] **Step 2: Run the test before scaffolding to verify failure**

Run: `uv run pytest tests/test_package.py -q`

Expected: FAIL because `pyproject.toml` and the `poor_word` package do not exist.

- [ ] **Step 3: Create the uv project metadata and package**

Use this dependency layout in `pyproject.toml`:

```toml
[project]
name = "poor-word"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
  "httpx>=0.28,<1",
  "numpy>=2,<3",
  "opencv-python-headless>=4.10,<5",
  "pillow>=11,<12",
  "pyarrow>=19,<20",
  "pydantic>=2.10,<3",
  "pydantic-settings>=2.7,<3",
  "scikit-image>=0.25,<1",
  "scikit-learn>=1.6,<2",
  "torch>=2.6,<3",
  "torchvision>=0.21,<1",
  "typer>=0.15,<1",
]

[project.optional-dependencies]
ocr = [
  "paddleocr>=3.2,<4",
  "paddlepaddle-gpu>=3.1,<4; sys_platform == 'linux' and platform_machine == 'x86_64'",
]

[dependency-groups]
dev = ["mypy>=1.15,<2", "pytest>=8.3,<9", "pytest-cov>=6,<7", "ruff>=0.11,<1"]

[project.scripts]
poor-word = "poor_word.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers"
markers = ["gpu: requires the NVIDIA L20 experiment host"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "RUF"]

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["poor_word"]
```

Set `.python-version` to `3.11`. Implement `doctor` without importing torch or Paddle:

```python
import platform

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def doctor() -> None:
    minor = ".".join(platform.python_version_tuple()[:2])
    typer.echo(f"python={minor}")
    typer.echo("gpu=not-required-for-unit-tests")
```

- [ ] **Step 4: Install Python 3.11 and lock dependencies**

Run: `uv python install 3.11 && uv sync --all-groups && uv lock --check`

Expected: Python 3.11 is available under uv, `.venv` and `uv.lock` are created, and the lock check succeeds.

- [ ] **Step 5: Run package, formatting, and type checks**

Run: `uv run pytest tests/test_package.py -q && uv run ruff check . && uv run mypy src`

Expected: all commands exit 0.

- [ ] **Step 6: Commit the project scaffold**

```bash
git add .python-version pyproject.toml uv.lock README.md src/poor_word tests/test_package.py
git commit -m "build: initialize Python uv project"
```

### Task 2: Define shared configuration and domain records

**Files:**
- Create: `src/poor_word/config.py`
- Create: `src/poor_word/domain.py`
- Create: `tests/test_domain.py`

**Interfaces:**
- Consumes: Pydantic from Task 1.
- Produces: `Decision`, `AnomalyKind`, `BoundingBox`, `GeneratedSample`, and `PathsConfig`; every Parquet row and evaluator uses these exact names.

- [ ] **Step 1: Write validation tests for immutable records**

```python
import pytest
from pydantic import ValidationError

from poor_word.domain import AnomalyKind, BoundingBox, Decision, GeneratedSample


def test_generated_sample_requires_changed_pixels_for_anomaly() -> None:
    with pytest.raises(ValidationError):
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
    with pytest.raises(ValidationError):
        BoundingBox(x0=4, y0=3, x1=4, y1=9)
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_domain.py -q`

Expected: FAIL with `ModuleNotFoundError` for `poor_word.domain`.

- [ ] **Step 3: Implement the exact domain interfaces**

```python
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Decision(StrEnum):
    PASS = "PASS"
    BLOCK = "BLOCK"
    REVIEW = "REVIEW"


class AnomalyKind(StrEnum):
    NONE = "none"
    MISSING_STROKE = "missing_stroke"
    EXTRA_STROKE = "extra_stroke"
    BROKEN_STROKE = "broken_stroke"
    BRIDGE = "bridge"
    COMPONENT_SHIFT = "component_shift"
    FUSION = "fusion"


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)
    x0: int = Field(ge=0)
    y0: int = Field(ge=0)
    x1: int = Field(gt=0)
    y1: int = Field(gt=0)

    @model_validator(mode="after")
    def positive_area(self) -> "BoundingBox":
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have positive area")
        return self


class GeneratedSample(BaseModel):
    model_config = ConfigDict(frozen=True)
    sample_id: str
    image_path: str
    mask_path: str
    base_char: str = Field(min_length=1, max_length=1)
    rendered_char: str = Field(min_length=1, max_length=2)
    decision: Decision
    anomaly_kind: AnomalyKind
    operator: str
    changed_pixels: int = Field(ge=0)
    seed: int = Field(ge=0)
    bbox: BoundingBox
    source_asset_ids: tuple[str, ...]

    @model_validator(mode="after")
    def anomaly_changes_pixels(self) -> "GeneratedSample":
        if self.decision is Decision.BLOCK and self.changed_pixels == 0:
            raise ValueError("anomalous samples must change at least one pixel")
        return self
```

Implement `PathsConfig` with repository-relative defaults for `data/raw`, `data/generated`, `artifacts`, and `models`, and a `create_runtime_dirs()` method that creates only generated/runtime directories.

- [ ] **Step 4: Run domain tests and static checks**

Run: `uv run pytest tests/test_domain.py -q && uv run ruff check src tests && uv run mypy src`

Expected: all commands exit 0.

- [ ] **Step 5: Commit the domain layer**

```bash
git add src/poor_word/config.py src/poor_word/domain.py tests/test_domain.py
git commit -m "feat: add typed domain records"
```

### Task 3: Add license-aware source declarations, locking, and downloads

**Files:**
- Create: `data/sources.toml`
- Create: `src/poor_word/data/__init__.py`
- Create: `src/poor_word/data/manifest.py`
- Create: `src/poor_word/data/download.py`
- Modify: `src/poor_word/cli.py`
- Create: `tests/data/test_download.py`

**Interfaces:**
- Consumes: `PathsConfig` from Task 2.
- Produces: `SourceSpec`, `SourceLock`, `lock_source(spec, target_dir) -> SourceLock`, and `fetch_locked_source(lock, target_dir) -> Path`.

- [ ] **Step 1: Write downloader tests with an in-process HTTP fixture**

```python
import hashlib
from pathlib import Path

from poor_word.data.download import fetch_locked_source
from poor_word.data.manifest import SourceLock


def test_fetch_locked_source_is_idempotent(httpserver, tmp_path: Path) -> None:
    payload = b"licensed-asset"
    httpserver.expect_request("/asset.bin").respond_with_data(payload)
    lock = SourceLock(
        source_id="fixture",
        declared_url=httpserver.url_for("/asset.bin"),
        resolved_url=httpserver.url_for("/asset.bin"),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="Apache-2.0",
        production_allowed=True,
    )
    first = fetch_locked_source(lock, tmp_path)
    second = fetch_locked_source(lock, tmp_path)
    assert first == second
    assert first.read_bytes() == payload
```

Add `pytest-httpserver>=1.1,<2` to the dev dependency group.

- [ ] **Step 2: Verify the downloader test fails**

Run: `uv sync && uv run pytest tests/data/test_download.py -q`

Expected: FAIL because `poor_word.data.download` does not exist.

- [ ] **Step 3: Implement source models and safe download behavior**

`SourceSpec` fields are `source_id`, `url`, `license_id`, `production_allowed`, and `output_name`. `SourceLock` fields match the test. `lock_source` streams the source once, records the final redirected URL, byte count, and SHA-256, writes `<source_id>.lock.json` atomically, and retains the downloaded file. `fetch_locked_source` writes to `.part`, verifies size and SHA-256, then uses `Path.replace()`; an existing valid file is returned without another request.

Reject a production profile when any requested source has `production_allowed = false`.

- [ ] **Step 4: Declare the first reviewed seed sources**

Add these source IDs to `data/sources.toml`:

```toml
[[source]]
source_id = "common_chars_3500"
url = "https://raw.githubusercontent.com/lqfeng/ChineseCharacters/master/%E9%80%9A%E7%94%A8%E8%A7%84%E8%8C%83%E6%B1%89%E5%AD%97%E8%A1%A8%282013%29%E4%B8%80%E7%BA%A7%E5%AD%97%E8%A1%A8%283500%E5%AD%97%29.txt"
license_id = "Apache-2.0"
production_allowed = true
output_name = "common_chars_3500.txt"

[[source]]
source_id = "noto_sans_sc_regular"
url = "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
license_id = "OFL-1.1"
production_allowed = true
output_name = "NotoSansCJKsc-Regular.otf"
```

Add `poor-word data lock --source-id ...` and `poor-word data fetch --source-id ...` commands. The lock command prints SHA-256, size, license, and resolved URL for review.

- [ ] **Step 5: Run tests and lock the real sources**

Run:

```bash
uv run pytest tests/data/test_download.py -q
uv run poor-word data lock --source-id common_chars_3500
uv run poor-word data lock --source-id noto_sans_sc_regular
uv run poor-word data fetch --source-id common_chars_3500
uv run poor-word data fetch --source-id noto_sans_sc_regular
```

Expected: tests pass; two lock JSON files are created under `data/locks`; downloaded files pass checksum verification. Downloaded binaries remain ignored, while `data/sources.toml` and `data/locks/*.json` are committed.

- [ ] **Step 6: Commit source acquisition code and locks**

```bash
git add pyproject.toml uv.lock data/sources.toml data/locks src/poor_word/data src/poor_word/cli.py tests/data
git commit -m "feat: add license-aware asset downloads"
```

### Task 4: Validate the common-character catalog and render legal glyphs

**Files:**
- Create: `src/poor_word/glyphs/__init__.py`
- Create: `src/poor_word/glyphs/catalog.py`
- Create: `src/poor_word/glyphs/render.py`
- Create: `tests/glyphs/test_catalog.py`
- Create: `tests/glyphs/test_render.py`

**Interfaces:**
- Consumes: locked `common_chars_3500` and `noto_sans_sc_regular` assets from Task 3.
- Produces: `load_common_chars(path) -> tuple[str, ...]`, `RenderedGlyph`, and `render_glyph(char, font_path, seed, canvas_size=128) -> RenderedGlyph`; `font_path=None` uses Pillow's default font only in unit tests.

- [ ] **Step 1: Write catalog and renderer tests**

```python
from pathlib import Path

import numpy as np

from poor_word.glyphs.catalog import load_common_chars
from poor_word.glyphs.render import render_glyph


def test_catalog_has_exactly_3500_unique_characters(tmp_path: Path) -> None:
    path = tmp_path / "chars.txt"
    path.write_text("甲乙丙", encoding="utf-8")
    chars = load_common_chars(path, expected_count=3)
    assert chars == ("甲", "乙", "丙")


def test_render_is_seed_deterministic() -> None:
    a = render_glyph("A", None, seed=9)
    b = render_glyph("A", None, seed=9)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.mask, b.mask)
    assert a.bbox == b.bbox
    assert int(a.mask.sum()) > 0
```

- [ ] **Step 2: Verify tests fail before implementation**

Run: `uv run pytest tests/glyphs/test_catalog.py tests/glyphs/test_render.py -q`

Expected: FAIL because the glyph modules do not exist.

- [ ] **Step 3: Implement strict catalog validation**

Normalize line endings and whitespace only; do not apply Unicode compatibility normalization that could replace a character. Reject duplicate characters, non-single-code-point entries, and counts other than `expected_count`. Production calls use `expected_count=3500`.

- [ ] **Step 4: Implement deterministic rendering**

`RenderedGlyph` is a frozen dataclass with `image: np.ndarray`, `mask: np.ndarray`, and `bbox: BoundingBox`. `render_glyph` accepts `font_path: Path | None`; `None` selects Pillow's default font for ASCII-only tests. Render grayscale foreground with Pillow at a seed-selected font size from 78 to 104 pixels for real font assets, center by the font bounding box, then produce a 3-channel RGB image, binary mask, and tight nonzero bounding box. The same character/font/seed must be byte-identical.

- [ ] **Step 5: Run catalog/render tests against both fixtures and downloaded assets**

Run:

```bash
uv run pytest tests/glyphs/test_catalog.py tests/glyphs/test_render.py -q
uv run python -c "from poor_word.glyphs.catalog import load_common_chars; from pathlib import Path; assert len(load_common_chars(Path('data/raw/common_chars_3500.txt'))) == 3500"
```

Expected: all assertions pass.

- [ ] **Step 6: Commit catalog and legal rendering**

```bash
git add src/poor_word/glyphs tests/glyphs
git commit -m "feat: render validated common Chinese glyphs"
```

### Task 5: Implement deterministic malformed-glyph operators

**Files:**
- Create: `src/poor_word/glyphs/corrupt.py`
- Create: `tests/glyphs/test_corrupt.py`

**Interfaces:**
- Consumes: `RenderedGlyph` from Task 4.
- Produces: `CorruptionResult` and `corrupt_glyph(rendered, operator, seed) -> CorruptionResult` for `erase_segment`, `add_stroke`, `break_stroke`, `bridge`, and `component_shift`.

- [ ] **Step 1: Write parameterized invariants for every operator**

```python
import numpy as np
import pytest

from poor_word.glyphs.corrupt import OPERATORS, corrupt_glyph
from poor_word.glyphs.render import RenderedGlyph
from poor_word.domain import BoundingBox


@pytest.fixture
def rendered_glyph() -> RenderedGlyph:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[6:26, 13:18] = 255
    mask[13:18, 6:26] = 255
    mask[2:5, 2:5] = 255
    image = np.repeat(mask[:, :, None], 3, axis=2)
    return RenderedGlyph(
        image=image,
        mask=mask,
        bbox=BoundingBox(x0=2, y0=2, x1=26, y1=26),
    )


@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_corruption_is_deterministic_and_changes_foreground(rendered_glyph, operator: str) -> None:
    a = corrupt_glyph(rendered_glyph, operator=operator, seed=17)
    b = corrupt_glyph(rendered_glyph, operator=operator, seed=17)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.changed_mask, b.changed_mask)
    assert a.changed_pixels > 0
    assert a.operator == operator
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/glyphs/test_corrupt.py -q`

Expected: FAIL because `corrupt.py` does not exist.

- [ ] **Step 3: Implement operators with postconditions**

Use `numpy.random.Generator(np.random.PCG64(seed))`. Skeletonize the mask with `skimage.morphology.skeletonize`. Implement:

- `erase_segment`: erase a disk centered on a sampled skeleton pixel.
- `add_stroke`: draw a 1–3 pixel line between two sampled foreground-near points.
- `break_stroke`: erase a short line perpendicular to the local skeleton direction.
- `bridge`: connect the nearest pixels of two connected components; when only one component exists, connect two distant skeleton points across a background gap.
- `component_shift`: move the smallest connected component by 4–12 pixels without clipping the canvas.

Retry up to 20 seed-derived candidates. If no operator candidate changes at least 8 and at most 35% of foreground pixels, raise `CorruptionNotApplicable`; the dataset generator records and skips that attempt rather than emitting a mislabeled sample.

- [ ] **Step 4: Add topology-focused tests**

Assert that `bridge` does not increase connected-component count, `component_shift` preserves foreground pixel count within 2%, and erase operators reduce foreground pixels. Use hand-built masks so these tests do not depend on a font.

- [ ] **Step 5: Run corruption tests and quality checks**

Run: `uv run pytest tests/glyphs/test_corrupt.py -q && uv run ruff check src tests && uv run mypy src`

Expected: all commands exit 0.

- [ ] **Step 6: Commit malformed-glyph operators**

```bash
git add src/poor_word/glyphs/corrupt.py tests/glyphs/test_corrupt.py
git commit -m "feat: generate deterministic malformed glyphs"
```

### Task 6: Generate versioned PNG/mask datasets with Parquet manifests

**Files:**
- Create: `src/poor_word/glyphs/generate.py`
- Modify: `src/poor_word/cli.py`
- Create: `tests/glyphs/test_generate.py`

**Interfaces:**
- Consumes: Tasks 2–5 domain records, catalog, renderer, corruptors, and source locks.
- Produces: `GenerationConfig`, `generate_dataset(config) -> Path`, and CLI `poor-word glyphs generate`.

- [ ] **Step 1: Write an end-to-end deterministic generation test**

```python
from pathlib import Path

import pyarrow.parquet as pq

from poor_word.glyphs.generate import GenerationConfig, generate_dataset


def test_generation_writes_deterministic_manifest(tmp_path: Path) -> None:
    config = GenerationConfig(
        output_dir=tmp_path / "run",
        characters=("文", "字"),
        font_paths=(None,),
        normal_per_char=1,
        abnormal_per_operator=1,
        operators=("erase_segment", "add_stroke"),
        seed=23,
        source_asset_ids=("fixture_font",),
    )
    manifest = generate_dataset(config)
    table = pq.read_table(manifest)
    assert table.num_rows == 6
    assert set(table.column("decision").to_pylist()) == {"PASS", "BLOCK"}
    assert len(set(table.column("sample_id").to_pylist())) == 6
```

- [ ] **Step 2: Verify the generation test fails**

Run: `uv run pytest tests/glyphs/test_generate.py -q`

Expected: FAIL because `generate.py` does not exist.

- [ ] **Step 3: Implement atomic dataset generation**

Derive `sample_id` as SHA-256 of canonical JSON containing character, font asset ID, operator, and seed. Write RGB PNGs to `images/<prefix>/<sample_id>.png`, masks to `masks/<prefix>/<sample_id>.png`, and manifest rows using `GeneratedSample.model_dump(mode="json")`. Write `manifest.parquet.part`, close and verify it, then rename to `manifest.parquet`. Existing sample IDs are checksum-verified and reused.

- [ ] **Step 4: Add the CLI command and run manifest**

`poor-word glyphs generate` accepts `--profile smoke|mvp`, `--seed`, and `--output-dir`. `smoke` uses 10 characters and all five operators. `mvp` uses all 3,500 characters, 4 normal renders per font, and 2 abnormal attempts per operator and render. Write `run.json` with command arguments, git commit, source lock hashes, dependency lock hash, row counts, skipped counts, and timestamp.

- [ ] **Step 5: Run the smoke generator twice and compare manifests**

Run:

```bash
uv run pytest tests/glyphs/test_generate.py -q
uv run poor-word glyphs generate --profile smoke --seed 20260804 --output-dir data/generated/smoke-a
uv run poor-word glyphs generate --profile smoke --seed 20260804 --output-dir data/generated/smoke-b
uv run python -c "from pathlib import Path; import hashlib; p=lambda x: hashlib.sha256(Path(x).read_bytes()).hexdigest(); assert p('data/generated/smoke-a/manifest.parquet') == p('data/generated/smoke-b/manifest.parquet')"
```

Expected: tests pass and both manifests are byte-identical.

- [ ] **Step 6: Commit dataset generation**

```bash
git add src/poor_word/glyphs/generate.py src/poor_word/cli.py tests/glyphs/test_generate.py
git commit -m "feat: build versioned synthetic glyph datasets"
```

### Task 7: Add the PP-OCRv5 server adapter and L20 capability audit

**Files:**
- Create: `src/poor_word/ocr/__init__.py`
- Create: `src/poor_word/ocr/types.py`
- Create: `src/poor_word/ocr/paddle_v5.py`
- Modify: `src/poor_word/cli.py`
- Create: `tests/ocr/test_paddle_v5.py`

**Interfaces:**
- Consumes: image paths or NumPy images.
- Produces: `OcrCharacter`, `OcrLine`, `OcrResult`, `PaddleV5Adapter.recognize(image) -> OcrResult`, and `PaddleV5Adapter.audit(image) -> OcrAudit`.

- [ ] **Step 1: Write adapter tests against a fake Paddle backend**

```python
from poor_word.ocr.paddle_v5 import PaddleV5Adapter


class FakePaddleBackend:
    def predict(self, image: str) -> dict[str, object]:
        return {
            "model_name": "PP-OCRv5_server",
            "lines": [{
                "text": "优惠",
                "polygon": [[0, 0], [80, 0], [80, 32], [0, 32]],
                "characters": [
                    {"text": "优", "box": [0, 0, 40, 32], "top_k": [["优", 0.8]], "logits": [1.0]},
                    {"text": "惠", "box": [40, 0, 80, 32], "top_k": [["惠", 0.9]], "logits": [1.0]},
                ],
            }],
        }


def test_adapter_preserves_character_boxes_and_raw_candidates() -> None:
    result = PaddleV5Adapter(backend=FakePaddleBackend()).recognize("poster.png")
    assert result.model_name == "PP-OCRv5_server"
    assert result.lines[0].text == "优惠"
    assert [char.text for char in result.lines[0].characters] == ["优", "惠"]
    assert result.lines[0].characters[0].top_k[0].text == "优"
    assert result.lines[0].characters[0].logits_available is True
```

- [ ] **Step 2: Verify tests fail without importing Paddle at collection time**

Run: `uv run pytest tests/ocr/test_paddle_v5.py -q`

Expected: FAIL because the OCR modules do not exist; the failure output must not contain a Paddle import error.

- [ ] **Step 3: Implement OCR-neutral records and lazy adapter construction**

Define frozen Pydantic records for polygons, text, confidence, character boxes, Top-K candidates, logits availability, model names, and per-stage milliseconds. `PaddleV5Adapter.__init__` accepts an injected backend implementing `predict(image: str | np.ndarray) -> dict[str, object]` for tests. If no backend is supplied, import PaddleOCR inside `__init__` and instantiate exactly `PP-OCRv5_server_det` and `PP-OCRv5_server_rec` with character coordinates enabled.

- [ ] **Step 4: Implement explicit capability audit behavior**

`audit` returns PaddleOCR/PaddlePaddle/CUDA versions, GPU name, model names, character-box support, logits support, P50/P95 over 30 warm requests, and peak GPU memory. The command exits nonzero when model names are not server variants or character coordinates are unavailable. Missing logits are reported as `logits_available=false` without falsifying candidates; the audit report then names `raw_logits_adapter` as a required capability gap.

- [ ] **Step 5: Run CPU tests and the L20 audit**

CPU run:

```bash
uv run pytest tests/ocr/test_paddle_v5.py -q
```

L20 experiment host run:

```bash
uv sync --extra ocr
uv run poor-word ocr audit --image-dir data/generated/smoke-a/images --warmup 10 --runs 30 --output artifacts/ocr-audit-l20.json
```

Expected: CPU tests pass. The L20 JSON reports both server model names, `character_boxes_available=true`, measured latency percentiles, and an explicit boolean for logits availability.

- [ ] **Step 6: Commit the OCR adapter and audit**

```bash
git add src/poor_word/ocr src/poor_word/cli.py tests/ocr
git commit -m "feat: audit PP-OCRv5 server outputs"
```

### Task 8: Implement the glyph encoder, prototype bank, and OOD scoring

**Files:**
- Create: `src/poor_word/models/__init__.py`
- Create: `src/poor_word/models/glyph_encoder.py`
- Create: `src/poor_word/models/prototypes.py`
- Create: `tests/models/test_glyph_encoder.py`
- Create: `tests/models/test_prototypes.py`

**Interfaces:**
- Consumes: batches shaped `[N, 3, 96, 96]` where channels are grayscale, mask, and edge.
- Produces: `GlyphEncoder.forward(x) -> Tensor[N, 256]`, `PrototypeBank.fit(embeddings, labels)`, and `PrototypeBank.score(embeddings) -> GlyphScores`.

- [ ] **Step 1: Write encoder shape and normalization tests**

```python
import torch

from poor_word.models.glyph_encoder import GlyphEncoder


def test_encoder_returns_unit_normalized_embeddings() -> None:
    model = GlyphEncoder(embedding_dim=256, pretrained=False).eval()
    output = model(torch.rand(2, 3, 96, 96))
    assert output.shape == (2, 256)
    assert torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-5)
```

- [ ] **Step 2: Write prototype scoring tests**

```python
import numpy as np

from poor_word.models.prototypes import PrototypeBank


def test_nearest_prototype_and_margin() -> None:
    bank = PrototypeBank(max_prototypes_per_char=2, random_state=5)
    bank.fit(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]), ["甲", "甲", "乙"])
    scores = bank.score(np.array([[1.0, 0.0]]))
    assert scores.nearest_chars == ("甲",)
    assert scores.nearest_distance[0] < scores.second_distance[0]
    assert scores.margin[0] > 0
```

- [ ] **Step 3: Verify both test modules fail**

Run: `uv run pytest tests/models/test_glyph_encoder.py tests/models/test_prototypes.py -q`

Expected: FAIL because the model modules do not exist.

- [ ] **Step 4: Implement the ConvNeXt-Tiny encoder**

Instantiate `torchvision.models.convnext_tiny(weights=None)` for `pretrained=False` and the default ImageNet weights for `pretrained=True`. Replace the classifier with `LayerNorm -> Linear(768, 256)`, resize input to 96×96, and L2-normalize output. Pretrained weight download is explicit and cached under `models/cache`; unit tests never download weights.

- [ ] **Step 5: Implement the prototype bank**

For each character, use normalized mean when it has fewer than four embeddings; otherwise use `MiniBatchKMeans` with at most eight clusters and a fixed random state. Normalize centers. Score with cosine distance, return nearest character, nearest/second distance, and margin. Persist arrays to `.npz` and metadata to adjacent JSON containing catalog hash, encoder checkpoint hash, source manifest hash, and creation command.

- [ ] **Step 6: Run model tests and commit**

Run: `uv run pytest tests/models -q && uv run ruff check src tests && uv run mypy src`

Expected: all commands exit 0.

```bash
git add src/poor_word/models tests/models
git commit -m "feat: add glyph embeddings and prototype scoring"
```

### Task 9: Train a reproducible synthetic glyph baseline

**Files:**
- Create: `src/poor_word/training/__init__.py`
- Create: `src/poor_word/training/dataset.py`
- Create: `src/poor_word/training/train_glyph.py`
- Modify: `src/poor_word/cli.py`
- Create: `tests/training/conftest.py`
- Create: `tests/training/test_dataset.py`
- Create: `tests/training/test_train_smoke.py`

**Interfaces:**
- Consumes: Task 6 Parquet manifest and Task 8 `GlyphEncoder`.
- Produces: `GlyphDataset`, `TrainConfig`, `train_glyph(config) -> TrainArtifacts`, encoder checkpoint, prototype bank, and metrics JSON.

- [ ] **Step 1: Write dataset channel-construction tests**

In `tests/training/conftest.py`, create `generated_manifest` by calling Task 6's generator with characters `("A", "B")`, `font_paths=(None,)`, one normal sample, operators `("erase_segment", "add_stroke")`, and seed `31`. The fixture returns the generated `manifest.parquet` path.

```python
import torch

from poor_word.training.dataset import GlyphDataset


def test_dataset_returns_three_views_and_label(generated_manifest) -> None:
    item = GlyphDataset(generated_manifest)[0]
    assert item.views.shape == (3, 96, 96)
    assert item.views.dtype == torch.float32
    assert item.decision in {"PASS", "BLOCK"}
    assert len(item.base_char) == 1
```

- [ ] **Step 2: Write a two-batch trainer smoke test**

```python
from poor_word.training.train_glyph import TrainConfig, train_glyph


def test_train_smoke_writes_checkpoint_and_metrics(generated_manifest, tmp_path) -> None:
    artifacts = train_glyph(
        TrainConfig(
            manifest=generated_manifest,
            output_dir=tmp_path,
            epochs=1,
            max_steps=2,
            batch_size=4,
            seed=11,
            pretrained=False,
            device="cpu",
        )
    )
    assert artifacts.checkpoint.exists()
    assert artifacts.metrics.exists()
    assert artifacts.prototype_bank.exists()
```

- [ ] **Step 3: Verify training tests fail**

Run: `uv run pytest tests/training -q`

Expected: FAIL because the training package does not exist.

- [ ] **Step 4: Implement the dataset and deterministic split**

Build grayscale, binary mask, and Canny edge channels. Split by the SHA-256 prefix of `base_char + source_asset_ids[0]`, not by row, so the same rendered source group cannot cross train/validation. Return typed samples; corrupt or missing files raise an error that includes `sample_id`.

- [ ] **Step 5: Implement the baseline loss and artifact recording**

Use cross-entropy over legal character IDs for PASS samples, supervised contrastive loss over all legal samples, and an energy-margin loss that pushes BLOCK samples above the legal energy threshold. Fixed weights are `1.0`, `0.5`, and `0.5`. Record seed, device, git commit, uv lock hash, manifest hash, per-loss history, validation nearest-prototype accuracy, synthetic OOD AUCPR, and elapsed time.

- [ ] **Step 6: Run CPU smoke training and the L20 MVP training command**

CPU:

```bash
uv run pytest tests/training -q
uv run poor-word train glyph --manifest data/generated/smoke-a/manifest.parquet --epochs 1 --max-steps 2 --device cpu --output-dir artifacts/train-smoke
```

L20:

```bash
uv run poor-word glyphs generate --profile mvp --seed 20260804 --output-dir data/generated/mvp-v1
uv run poor-word train glyph --manifest data/generated/mvp-v1/manifest.parquet --epochs 20 --batch-size 256 --device cuda --pretrained --output-dir artifacts/glyph-mvp-v1
```

Expected: CPU smoke artifacts exist; the L20 run writes checkpoint, prototype bank, and metrics without OOM.

- [ ] **Step 7: Commit training code**

```bash
git add src/poor_word/training src/poor_word/cli.py tests/training
git commit -m "feat: train synthetic glyph baseline"
```

### Task 10: Add base-rate-aware evaluation and the reproducible MVP report

**Files:**
- Create: `src/poor_word/evaluation/__init__.py`
- Create: `src/poor_word/evaluation/metrics.py`
- Create: `src/poor_word/evaluation/report.py`
- Modify: `src/poor_word/cli.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/evaluation/test_report.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: label/score arrays, prototype artifacts, OCR audit JSON, and run manifests.
- Produces: `base_rate_precision(prevalence, recall, fpr) -> float`, threshold table, `MvpReport`, CLI `poor-word evaluate glyph`, JSON report, and Markdown summary.

- [ ] **Step 1: Write metric tests for the approved production math**

```python
import pytest

from poor_word.evaluation.metrics import base_rate_precision


def test_trial_gate_precision_at_point_one_percent_base_rate() -> None:
    precision = base_rate_precision(prevalence=0.001, recall=0.50, fpr=0.000025)
    assert precision == pytest.approx(0.9524263, rel=1e-6)


def test_zero_fpr_has_perfect_precision_when_recall_is_positive() -> None:
    assert base_rate_precision(prevalence=0.001, recall=0.4, fpr=0.0) == 1.0
```

- [ ] **Step 2: Write report gate tests**

Create synthetic labels with exactly 40% recall and 0.01% FPR. Assert the report marks the offline MVP gate as passing, prints raw counts, reports synthetic AUCPR separately, and never substitutes balanced precision for production precision.

- [ ] **Step 3: Verify evaluation tests fail**

Run: `uv run pytest tests/evaluation -q`

Expected: FAIL because the evaluation package does not exist.

- [ ] **Step 4: Implement metrics and threshold selection**

Compute recall, FPR, AUROC, AUCPR, raw confusion counts, and production precision over a sorted threshold grid. The MVP gate is `FPR <= 0.0001` and `Recall >= 0.40`; the trial gate is `FPR <= 0.000025` and `Recall >= 0.50`. When no threshold satisfies a gate, report the closest threshold by smallest normalized constraint violation and keep `passed=false`.

- [ ] **Step 5: Implement JSON and Markdown reporting**

The report includes dataset/asset/model hashes, source license decisions, PP-OCRv5 L20 capabilities, synthetic split metrics, real-data sections only when provided, production-base-rate calculations, latency, failures, and exact reproduction commands. A synthetic-only report must contain the heading `Not a production claim`.

- [ ] **Step 6: Run the complete CPU MVP workflow**

Run:

```bash
uv run pytest -m "not gpu" --cov=poor_word --cov-report=term-missing
uv run ruff check .
uv run mypy src
uv run poor-word evaluate glyph --manifest data/generated/smoke-a/manifest.parquet --artifacts artifacts/train-smoke --prevalence 0.001 --output-dir artifacts/report-smoke
```

Expected: all checks pass; `artifacts/report-smoke/report.json` and `report.md` exist and label the result as synthetic-only.

- [ ] **Step 7: Update README with exact local and L20 commands**

Document `uv python install 3.11`, `uv sync`, source locking/fetching, smoke generation, CPU tests, L20 OCR audit, L20 training, and evaluation. State that the current development machine has no NVIDIA runtime and that L20 commands must run on the experiment host.

- [ ] **Step 8: Commit evaluation and MVP documentation**

```bash
git add src/poor_word/evaluation src/poor_word/cli.py tests/evaluation README.md
git commit -m "feat: report base-rate-aware glyph MVP metrics"
```

## Plan Completion Gate

Before starting the second implementation plan, run:

```bash
uv lock --check
uv run pytest -m "not gpu" --cov=poor_word --cov-fail-under=85
uv run ruff check .
uv run mypy src
git status --short
```

The phase is complete only when all CPU checks pass, the working tree is clean, both real seed assets have reviewed lock files, the synthetic smoke workflow is reproducible, and the L20 audit report explicitly records character-coordinate and logits availability. A missing raw-logits capability is an observed PP-OCRv5 integration result and becomes an input to the second plan; it must not be represented as available.

The remaining approved system scope is intentionally split into three independently reviewable plans:

1. Real-data adaptation, character quick-review workflow, weakly supervised MIL, and hard-negative mining.
2. Lexicon-free sequence recognition, n-gram/Transformer string anomaly scoring, whitelist policy, and fusion calibration.
3. FastAPI serving, two-worker L20 orchestration, degradation semantics, load testing, large-scale normal replay, and pilot rollout.
