import hashlib
from pathlib import Path

import pytest
from pytest_httpserver import HTTPServer

from poor_word.data.download import fetch_locked_source
from poor_word.data.manifest import SourceLock, SourceSpec, validate_production_sources


def test_fetch_locked_source_is_idempotent(httpserver: HTTPServer, tmp_path: Path) -> None:
    payload = b"licensed-asset"
    httpserver.expect_request("/asset.bin").respond_with_data(payload)
    url = httpserver.url_for("/asset.bin")
    lock = SourceLock(
        source_id="fixture",
        declared_url=url,
        resolved_url=url,
        output_name="asset.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="Apache-2.0",
        production_allowed=True,
    )

    first = fetch_locked_source(lock, tmp_path)
    second = fetch_locked_source(lock, tmp_path)

    assert first == second
    assert first.read_bytes() == payload


def test_fetch_locked_source_rejects_checksum_mismatch(
    httpserver: HTTPServer, tmp_path: Path
) -> None:
    payload = b"tampered"
    httpserver.expect_request("/asset.bin").respond_with_data(payload)
    url = httpserver.url_for("/asset.bin")
    lock = SourceLock(
        source_id="fixture",
        declared_url=url,
        resolved_url=url,
        output_name="asset.bin",
        sha256=hashlib.sha256(b"expected").hexdigest(),
        size_bytes=len(payload),
        license_id="Apache-2.0",
        production_allowed=True,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        fetch_locked_source(lock, tmp_path)

    assert not (tmp_path / "asset.bin").exists()
    assert not (tmp_path / "asset.bin.part").exists()


def test_production_profile_rejects_unapproved_sources() -> None:
    source = SourceSpec(
        source_id="research_only",
        url="https://example.invalid/research.bin",
        license_id="LicenseRef-Unknown",
        production_allowed=False,
        output_name="research.bin",
    )

    with pytest.raises(ValueError, match="research_only"):
        validate_production_sources((source,))
