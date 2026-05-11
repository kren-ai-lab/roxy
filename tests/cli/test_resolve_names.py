"""Tests for CLI descriptor name resolution."""

from __future__ import annotations

from roxy.cli._utils import _resolve_names
from roxy.descriptors import DESCRIPTOR_REGISTRY


def test_resolve_names_all():
    result = _resolve_names(True, [], [])
    assert result == sorted(DESCRIPTOR_REGISTRY)


def test_resolve_names_descriptors():
    result = _resolve_names(False, ["charge", "aac"], [])
    assert result == ["aac", "charge"]


def test_resolve_names_family():
    result = _resolve_names(False, [], ["composition"])
    composition = sorted(n for n, c in DESCRIPTOR_REGISTRY.items() if c.family == "composition")
    assert result == composition


def test_resolve_names_dedup():
    result = _resolve_names(False, ["aac"], ["composition"])
    assert result.count("aac") == 1
