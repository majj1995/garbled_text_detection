from typer.testing import CliRunner

from poor_word import __version__
from poor_word.cli import app


def test_package_exposes_release_version() -> None:
    assert __version__ == "0.1.0"


def test_doctor_reports_active_python_and_unit_test_gpu_policy() -> None:
    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "python=3.12" in result.stdout
    assert "gpu=not-required-for-unit-tests" in result.stdout
