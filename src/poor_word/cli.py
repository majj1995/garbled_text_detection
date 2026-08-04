import platform
from pathlib import Path
from typing import Annotated

import typer

from poor_word.config import PathsConfig
from poor_word.data.download import fetch_locked_source, lock_source
from poor_word.data.manifest import SourceSpec, load_source_lock, load_source_specs

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """AIGC malformed-Chinese detection tools."""


@app.command()
def doctor() -> None:
    """Report the active Python runtime and local test requirements."""
    minor = ".".join(platform.python_version_tuple()[:2])
    typer.echo(f"python={minor}")
    typer.echo("gpu=not-required-for-unit-tests")


def _get_source(source_id: str, sources_file: Path) -> SourceSpec:
    sources = {source.source_id: source for source in load_source_specs(sources_file)}
    try:
        return sources[source_id]
    except KeyError as error:
        raise typer.BadParameter(f"unknown source_id: {source_id}") from error


@data_app.command("lock")
def data_lock(
    source_id: Annotated[str, typer.Option("--source-id")],
    sources_file: Annotated[Path, typer.Option("--sources-file")] = Path("data/sources.toml"),
) -> None:
    """Download a declared source once and create its reviewed lock."""
    config = PathsConfig()
    spec = _get_source(source_id, sources_file)
    lock = lock_source(spec, config.raw_path, lock_dir=config.repo_root / "data/locks")
    typer.echo(f"sha256={lock.sha256}")
    typer.echo(f"size_bytes={lock.size_bytes}")
    typer.echo(f"license={lock.license_id}")
    typer.echo(f"resolved_url={lock.resolved_url}")


@data_app.command("fetch")
def data_fetch(
    source_id: Annotated[str, typer.Option("--source-id")],
    lock_dir: Annotated[Path, typer.Option("--lock-dir")] = Path("data/locks"),
) -> None:
    """Fetch a source only when its bytes match the reviewed lock."""
    config = PathsConfig()
    lock = load_source_lock(lock_dir / f"{source_id}.lock.json")
    path = fetch_locked_source(lock, config.raw_path)
    typer.echo(path)
