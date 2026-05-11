"""CLI example that writes a config and computes features from FASTA."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import polars as pl

from roxy import read_fasta

FASTA_TEXT = """>seq1
MKWVTFISLLFLFSSAYSRGVFRR
>seq2
GAVLKVLTTGLPALISWIKRKRQQ
>seq3
DDDEEEGGGSSSNNNQQQ
"""


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    """Exercise `roxy init-config` and `roxy compute` end to end."""
    repo_root = Path(__file__).resolve().parents[1]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fasta_path = tmp_path / "toy_sequences.fasta"
        config_path = tmp_path / "toy_config.yaml"
        output_path = tmp_path / "toy_features.csv"

        fasta_path.write_text(FASTA_TEXT, encoding="utf-8")

        _run(
            "uv",
            "run",
            "roxy",
            "init-config",
            "-d",
            "aac",
            "-d",
            "charge",
            "-o",
            str(config_path),
            cwd=repo_root,
        )

        config_text = config_path.read_text(encoding="utf-8")
        config_text = config_text.replace("include_counts: true", "include_counts: false")
        config_path.write_text(config_text, encoding="utf-8")

        _run(
            "uv",
            "run",
            "roxy",
            "compute",
            str(fasta_path),
            "--config",
            str(config_path),
            "-o",
            str(output_path),
            "--no-progress",
            cwd=repo_root,
        )

        records = read_fasta(fasta_path)
        df = pl.read_csv(output_path)

        assert len(records) == 3
        assert df.height == 3
        assert "length" in df.columns
        assert "aac_freq_A" in df.columns
        assert "aac_count_A" not in df.columns
        assert "charge_net_charge_ph5p0" in df.columns

        print("cli_roundtrip.py")
        print(df.select(["id", "length", "aac_freq_A", "charge_net_charge_ph5p0"]))


if __name__ == "__main__":
    main()
