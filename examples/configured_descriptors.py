"""Configured descriptor example using the Python API."""

from __future__ import annotations

from roxy.descriptors.aaindex import AAIndexDescriptor
from roxy.descriptors.physicochemical import ChargeDescriptor


def main() -> None:
    """Run parameterized descriptors and inspect selected columns."""
    sequences = [
        "MKWVTFISLLFLFSSAYSRGVFRR",
        "GAVLKVLTTGLPALISWIKRKRQQ",
        "DDDEEEGGGSSSNNNQQQ",
    ]
    ids = ["albumin_like", "signal_peptide_like", "acidic_low_complexity"]

    aaindex = AAIndexDescriptor(
        codes=["ANDN920101", "KYTJ820101"],
        include_terminal=False,
    )
    charge = ChargeDescriptor(ph_values=(7.0,))

    aaindex_df = aaindex.compute(sequences, ids=ids)
    charge_df = charge.compute(sequences, ids=ids)

    assert aaindex_df.height == len(sequences)
    assert charge_df.height == len(sequences)
    assert "aaindex_ANDN920101_mean" in aaindex_df.columns
    assert "charge_net_charge_ph7p0" in charge_df.columns
    assert "charge_positive_fraction" in charge_df.columns

    joined = aaindex_df.join(charge_df, on="id", how="inner")

    print("configured_descriptors.py")
    print(
        joined.select(
            [
                "id",
                "aaindex_ANDN920101_mean",
                "aaindex_KYTJ820101_mean",
                "charge_net_charge_ph7p0",
                "charge_positive_fraction",
            ]
        )
    )


if __name__ == "__main__":
    main()
