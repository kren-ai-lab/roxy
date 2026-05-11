"""Tests for descriptor config loading."""

from __future__ import annotations

import pytest
import typer

from roxy.cli.compute import _load_config


def test_load_config_valid(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n  include_counts: true\n")
    result = _load_config(cfg)
    assert result == {"aac": {"include_counts": True}}


def test_load_config_empty_params(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n")
    result = _load_config(cfg)
    assert result == {"aac": {}}


def test_load_config_unknown_descriptor(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("nonexistent_descriptor:\n")
    with pytest.raises(typer.BadParameter, match="Unknown descriptor"):
        _load_config(cfg)


def test_load_config_unknown_param(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n  totally_fake_param: 42\n")
    with pytest.raises(typer.BadParameter, match="Unknown param"):
        _load_config(cfg)


def test_load_config_invalid_yaml(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("key: [unclosed\n")
    with pytest.raises(typer.BadParameter, match="Invalid YAML"):
        _load_config(cfg)
