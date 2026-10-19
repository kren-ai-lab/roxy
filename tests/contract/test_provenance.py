"""Provenance sidecar written by ``roxy compute``."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from roxy.cli.init_config import _default_params
from roxy.cli.main import app
from roxy.descriptors import DESCRIPTOR_REGISTRY
from tests.cli._helpers import FASTA, runner

SCHEMA = json.loads((Path(__file__).parent / "provenance.schema.json").read_text())


def _run(tmp_path, name, *args):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(FASTA)
    out = tmp_path / name
    result = runner.invoke(app, ["compute", str(fasta), *args, "-o", str(out), "--no-progress"])
    assert result.exit_code == 0, result.output
    return json.loads(out.with_name(out.name + ".provenance.json").read_text())


def test_sidecar_conforms_to_schema(tmp_path):
    doc = _run(tmp_path, "out.csv", "-d", "aac")
    jsonschema.validate(doc, SCHEMA)
    assert doc["tool"]["name"] == "roxy"
    assert doc["outputs"][0]["n_rows"] == 2


def test_parameters_include_defaults(tmp_path):
    doc = _run(tmp_path, "out.csv", "-d", "kmer_full_alphabet")
    expected = json.loads(json.dumps(_default_params(DESCRIPTOR_REGISTRY["kmer_full_alphabet"])))
    assert doc["parameters"]["descriptors"]["kmer_full_alphabet"] == expected
    assert expected  # kmer has defaulted params
    assert doc["parameters"]["seq_col"] == "sequence"


def test_fingerprint_same_for_config_and_flags(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n")  # defaults filled in by compute
    a = _run(tmp_path, "a.csv", "--config", str(cfg))
    b = _run(tmp_path, "b.csv", "-d", "aac")
    assert a["config_fingerprint"] == b["config_fingerprint"]
