"""Tests for descriptor registry names and families."""

from __future__ import annotations

from roxy.descriptors import DESCRIPTOR_REGISTRY


def test_documented_descriptor_names_are_registered():
    expected = {
        "grouped",
        "kmer_full_alphabet",
        "entropy_complexity",
        "ctd_classic",
        "pattern",
        "functional_residue_content",
        "normalized",
        "family_summary",
    }

    assert expected <= set(DESCRIPTOR_REGISTRY)


def test_removed_implementation_names_are_not_registered():
    removed = {
        "grouped_composition",
        "kmer",
        "functional_residue",
        "motif",
        "positional",
        "complexity",
        "ctd",
        "hybrid",
    }

    assert set(DESCRIPTOR_REGISTRY).isdisjoint(removed)


def test_descriptor_families_match_plan():
    expected_families = {
        "grouped": "composition",
        "kmer_full_alphabet": "composition",
        "entropy_complexity": "complexity",
        "ctd_classic": "ctd",
        "pattern": "motif",
        "user_regex": "motif",
        "spacing": "motif",
        "functional_residue_content": "motif",
        "distribution": "ctd",
        "normalized": "positional",
        "sliding_window": "positional",
        "terminal": "positional",
        "qso": "pseudo",
        "sequence_order": "pseudo",
        "local_repetition": "complexity",
        "run_blockiness": "complexity",
        "family_summary": "hybrid",
    }

    for name, family in expected_families.items():
        assert DESCRIPTOR_REGISTRY[name].family == family
