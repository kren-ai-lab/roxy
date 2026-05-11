"""Shared helpers for CLI tests."""

from __future__ import annotations

from typer.testing import CliRunner

FASTA = ">seq1\nACDEFGHIKLMNPQRSTVWY\n>seq2\nAAAACCCC\n"
runner = CliRunner()
