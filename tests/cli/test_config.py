"""Config shape and `roxy validate-config`."""

from __future__ import annotations

import json

from roxy.cli.main import app
from roxy.cli.validate import check_config, load_config

from ._helpers import FASTA, runner


def _validate(tmp_path, text):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(text)
    result = runner.invoke(app, ["validate-config", str(cfg), "--json"])
    issues = json.loads(result.stdout)
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


def test_families_and_all_in_config(tmp_path):
    from roxy.descriptors import DESCRIPTOR_REGISTRY

    composition = sorted(n for n, c in DESCRIPTOR_REGISTRY.items() if c.family == "composition")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("families: [composition]\n")
    assert list(load_config(cfg)) == composition
    # with a family the union is sorted like `-d charge -f composition`; parameters still apply
    cfg.write_text("descriptors:\n  charge: {}\n  aac: {include_counts: true}\nfamilies: [composition]\n")
    resolved = load_config(cfg)
    assert list(resolved) == sorted(["charge", *composition])
    assert resolved["aac"] == {"include_counts": True}
    assert all(resolved[n] == {} for n in resolved if n != "aac")
    # descriptors alone keep the file order, as before
    cfg.write_text("descriptors:\n  charge: {}\n  aac: {}\n")
    assert list(load_config(cfg)) == ["charge", "aac"]
    cfg.write_text("all: true\n")
    assert list(load_config(cfg)) == sorted(DESCRIPTOR_REGISTRY)


def test_invalid_selection_keys(tmp_path):
    code, paths = _validate(tmp_path, "families: [composition, nope]\nall: yes please\n")
    assert code == 2
    assert paths == {"families.1", "all"}
    assert _validate(tmp_path, "families: composition\n") == (2, {"families"})
    assert _validate(tmp_path, "all: false\n") == (2, {"descriptors"})
    assert _validate(tmp_path, "families: []\n") == (2, {"descriptors"})


def test_config_families_match_family_flag(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(FASTA)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("families: [physicochemical]\n")
    by_flag, by_config = tmp_path / "flag.csv", tmp_path / "config.csv"
    flag = runner.invoke(
        app, ["compute", str(fasta), "-f", "physicochemical", "-o", str(by_flag), "--no-progress"]
    )
    conf = runner.invoke(
        app, ["compute", str(fasta), "--config", str(cfg), "-o", str(by_config), "--no-progress"]
    )
    assert (flag.exit_code, conf.exit_code) == (0, 0)
    assert by_flag.read_bytes() == by_config.read_bytes()
    docs = [json.loads(p.with_name(p.name + ".provenance.json").read_text()) for p in (by_flag, by_config)]
    assert docs[0]["parameters"] == docs[1]["parameters"]
    assert docs[0]["config_fingerprint"] == docs[1]["config_fingerprint"]
