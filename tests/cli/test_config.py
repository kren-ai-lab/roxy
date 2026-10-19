"""Config shape and `roxy validate-config`."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from roxy.cli.main import app
from roxy.cli.validate import check_config, load_config

from ._helpers import FASTA, runner

SCHEMA = json.loads((Path(__file__).parents[1] / "contract" / "validate-errors.schema.json").read_text())


def _validate(tmp_path, text):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(text)
    result = runner.invoke(app, ["validate-config", str(cfg), "--json"])
    issues = json.loads(result.stdout)
    jsonschema.validate(issues, SCHEMA)
    return result.exit_code, {i["path"] for i in issues}


def test_valid_config(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("schema_version: 1\ndescriptors:\n  aac:\n    include_counts: true\n  charge:\n")
    assert load_config(cfg) == {"aac": {"include_counts": True}, "charge": {}}
    assert _validate(tmp_path, cfg.read_text()) == (0, set())


def test_validate_collects_every_problem(tmp_path):
    text = (
        "schema_version: 2\nextra: 1\ndescriptors:\n"
        "  nope: {}\n"
        "  aac: {bogus: 1, include_counts: maybe}\n"
        "  aaindex: {codes: [NOT_A_CODE]}\n"
    )
    code, paths = _validate(tmp_path, text)
    assert code == 2
    assert paths == {
        "schema_version",
        "extra",
        "descriptors.nope",
        "descriptors.aac.bogus",
        "descriptors.aac.include_counts",
        "descriptors.aaindex.codes",
    }


def test_old_flat_shape_rejected():
    issues = check_config({"aac": {}})
    assert {i.path for i in issues} == {"aac", "descriptors"}
    assert any("descriptors:" in i.message for i in issues if i.path == "aac")


def test_compute_rejects_invalid_config(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(FASTA)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n")
    out = tmp_path / "out.csv"
    result = runner.invoke(app, ["compute", str(fasta), "--config", str(cfg), "-o", str(out)])
    assert result.exit_code == 2
    assert not out.exists()


def test_invalid_yaml_is_an_issue(tmp_path):
    assert _validate(tmp_path, "key: [unclosed\n") == (2, {""})


def test_validate_config_python_api(tmp_path):
    from types import MappingProxyType

    from roxy import ConfigIssue, validate_config

    assert validate_config({"descriptors": {"aac": {}}}) == []
    bad = MappingProxyType({"descriptors": {"aac": {"include_counts": "x"}}})
    assert validate_config(bad) == [ConfigIssue("descriptors.aac.include_counts", "expected bool, got str")]
    cfg = tmp_path / "ok.yaml"
    cfg.write_text("descriptors:\n  charge:\n")
    assert validate_config(cfg) == []
    assert validate_config(str(cfg)) == []
    assert [i.path for i in validate_config(tmp_path / "missing.yaml")] == [""]
