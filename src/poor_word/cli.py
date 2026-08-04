import platform

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """AIGC malformed-Chinese detection tools."""


@app.command()
def doctor() -> None:
    """Report the active Python runtime and local test requirements."""
    minor = ".".join(platform.python_version_tuple()[:2])
    typer.echo(f"python={minor}")
    typer.echo("gpu=not-required-for-unit-tests")
