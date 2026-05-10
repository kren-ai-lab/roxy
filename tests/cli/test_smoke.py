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


def test_list():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "composition" in result.output


def test_list_filter_family():
    result = runner.invoke(app, ["list", "--family", "composition"])
    assert result.exit_code == 0
    assert "aac" in result.output


def test_describe_known():
    result = runner.invoke(app, ["describe", "aac"])
    assert result.exit_code == 0
    assert "composition" in result.output


def test_describe_unknown():
    result = runner.invoke(app, ["describe", "not_a_descriptor"])
    assert result.exit_code == 1


def test_init_config_stdout():
    result = runner.invoke(app, ["init-config", "-d", "aac"])
    assert result.exit_code == 0
    assert "aac:" in result.output


def test_init_config_no_selector():
    result = runner.invoke(app, ["init-config"])
    assert result.exit_code == 1


def test_compute_help():
    result = runner.invoke(app, ["compute", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output


def test_init_config_help():
    result = runner.invoke(app, ["init-config", "--help"])
    assert result.exit_code == 0
    assert "--all" in result.output


def test_aaindex_bundled():
    """AAIndex data ships with the package — no download needed."""
    from roxy.core.aaindex import load_aaindex

    df = load_aaindex()
    assert df.shape == (566, 21)  # 20 AA columns + "index" column
