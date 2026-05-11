"""Basic Python API example for Roxy."""

from __future__ import annotations

from roxy import DESCRIPTOR_REGISTRY


def main() -> None:
    """Compute a simple descriptor block from Python."""
    sequences = [
        "MKWVTFISLLFLFSSAYSRGVFRR",
        "GAVLKVLTTGLPALISWIKRKRQQ",
        "DDDEEEGGGSSSNNNQQQ",
    ]
    ids = ["albumin_like", "signal_peptide_like", "acidic_low_complexity"]

    descriptor = DESCRIPTOR_REGISTRY["aac"]()
    features = descriptor.compute(sequences, ids=ids)

    assert features.height == 3
    assert "id" in features.columns
    assert "aac_length" in features.columns
    assert "aac_freq_A" in features.columns

    print("basic_api.py")
    print(features.select(["id", "aac_length", "aac_freq_A", "aac_freq_K"]))


if __name__ == "__main__":
    main()
