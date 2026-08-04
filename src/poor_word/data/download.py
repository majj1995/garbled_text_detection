import hashlib
from pathlib import Path

import httpx

from poor_word.data.manifest import SourceLock, SourceSpec

_CHUNK_SIZE = 1024 * 1024
_TIMEOUT = httpx.Timeout(60.0, connect=20.0)


def _digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            size_bytes += len(chunk)
            digest.update(chunk)
    return size_bytes, digest.hexdigest()


def _valid_existing_file(path: Path, lock: SourceLock) -> bool:
    if not path.is_file():
        return False
    size_bytes, sha256 = _digest_file(path)
    return size_bytes == lock.size_bytes and sha256 == lock.sha256


def _stream_download(url: str, part_path: Path) -> tuple[int, str, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with httpx.Client(follow_redirects=True, timeout=_TIMEOUT) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with part_path.open("wb") as stream:
                    for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                        size_bytes += len(chunk)
                        digest.update(chunk)
                        stream.write(chunk)
                resolved_url = str(response.url)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    return size_bytes, digest.hexdigest(), resolved_url


def _write_lock(lock: SourceLock, lock_dir: Path) -> Path:
    lock_dir.mkdir(parents=True, exist_ok=True)
    destination = lock_dir / f"{lock.source_id}.lock.json"
    part_path = destination.with_name(f"{destination.name}.part")
    try:
        part_path.write_text(f"{lock.model_dump_json(indent=2)}\n", encoding="utf-8")
        part_path.replace(destination)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    return destination


def lock_source(
    spec: SourceSpec,
    target_dir: Path,
    *,
    lock_dir: Path | None = None,
) -> SourceLock:
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / spec.output_name
    part_path = destination.with_name(f"{destination.name}.part")
    size_bytes, sha256, resolved_url = _stream_download(spec.url, part_path)
    part_path.replace(destination)

    lock = SourceLock(
        source_id=spec.source_id,
        declared_url=spec.url,
        resolved_url=resolved_url,
        output_name=spec.output_name,
        sha256=sha256,
        size_bytes=size_bytes,
        license_id=spec.license_id,
        production_allowed=spec.production_allowed,
    )
    _write_lock(lock, target_dir if lock_dir is None else lock_dir)
    return lock


def fetch_locked_source(lock: SourceLock, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / lock.output_name
    if _valid_existing_file(destination, lock):
        return destination

    part_path = destination.with_name(f"{destination.name}.part")
    size_bytes, sha256, _ = _stream_download(lock.resolved_url, part_path)
    if size_bytes != lock.size_bytes:
        part_path.unlink(missing_ok=True)
        raise ValueError(
            f"size mismatch for {lock.source_id}: expected {lock.size_bytes}, got {size_bytes}"
        )
    if sha256 != lock.sha256:
        part_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {lock.source_id}: expected {lock.sha256}, got {sha256}"
        )

    part_path.replace(destination)
    return destination
