"""CLI smoke tests."""

from __future__ import annotations

from typer.testing import CliRunner

from roxy import __version__
from roxy.cli.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "roxy" in result.output.lower()


def test_list_empty_registry():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0


def test_aaindex_bundled():
    """AAIndex data ships with the package — no download needed."""
    from roxy.core.aaindex import load_aaindex

    df = load_aaindex()
    assert df.shape == (566, 21)  # 20 AA columns + "index" column
