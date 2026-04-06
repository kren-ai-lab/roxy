"""Grouped residue composition descriptor block.

Ratio features remain implemented here because they share the same
group-normalization and grouped-count infrastructure used by the main
grouped-composition block. The ratio-oriented public surface is exposed
through :mod:`roxy.sequence.descriptors.basic.ratios`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from roxy.core.constants import (
    DEFAULT_GROUPED_RESIDUE_ORDER,
    DEFAULT_GROUPED_RESIDUE_SETS,
)
from roxy.core.exceptions import SequenceInputError
from roxy.sequence.descriptors._base import (
    DescriptorBlock,
    DescriptorRecord,
    count_residues,
    prepare_descriptor_sequence,
    safe_divide,
)

GROUPED_RESIDUE_SETS: dict[str, frozenset[str]] = dict(DEFAULT_GROUPED_RESIDUE_SETS)
DEFAULT_GROUP_ORDER: tuple[str, ...] = DEFAULT_GROUPED_RESIDUE_ORDER
GROUPED_RATIO_SPECS: tuple[tuple[str, str, str], ...] = (
    ("acidic_basic", "negative", "positive"),
    ("polar_nonpolar", "polar", "nonpolar"),
    ("hydrophobic_hydrophilic", "hydrophobic", "hydrophilic"),
    ("disorder_order", "disorder_promoting", "order_promoting"),
    ("aromatic_aliphatic", "aromatic", "aliphatic"),
    ("charged_uncharged", "charged", "uncharged"),
)


def _normalize_groups(
    groups: Mapping[str, Iterable[str]] | None,
) -> dict[str, frozenset[str]]:
    """Return validated grouped-residue definitions in stable order."""
    if groups is None:
        return dict(GROUPED_RESIDUE_SETS)

    normalized: dict[str, frozenset[str]] = {}
    for group_name, residues in groups.items():
        if not isinstance(group_name, str):
            raise SequenceInputError("Grouped residue names must be strings.")

        normalized_name = group_name.strip()
        if not normalized_name:
            raise SequenceInputError("Grouped residue names cannot be empty.")
        if not normalized_name.replace("_", "").isalnum():
            raise SequenceInputError(
                "Grouped residue names must contain only letters, digits, or underscores."
            )

        residue_set = frozenset(str(residue).strip().upper() for residue in residues)
        if not residue_set:
            raise SequenceInputError(
                f"Grouped residue set {normalized_name!r} cannot be empty."
            )
        if any(len(residue) != 1 or not residue.isalpha() for residue in residue_set):
            raise SequenceInputError(
                f"Grouped residue set {normalized_name!r} must contain one-letter amino-acid codes."
            )
        normalized[normalized_name] = residue_set

    return normalized


def _ratio_or_none(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    """Return a ratio or ``None`` when the denominator is zero."""
    if denominator == 0:
        return None
    return safe_divide(numerator, denominator)


def _validate_ratio_group_requirements(
    groups: Mapping[str, Iterable[str]],
) -> None:
    """Ensure all ratio prerequisites exist in the grouped definition."""
    required_groups = {numerator for _, numerator, _ in GROUPED_RATIO_SPECS}
    required_groups.update(
        denominator
        for _, _, denominator in GROUPED_RATIO_SPECS
        if denominator != "uncharged"
    )
    missing_groups = sorted(
        group_name for group_name in required_groups if group_name not in groups
    )
    if missing_groups:
        raise SequenceInputError(
            "Grouped ratios require these groups to be defined: "
            + ", ".join(missing_groups)
            + "."
        )


def grouped_residue_counts(
    sequence: str,
    *,
    groups: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, int]:
    """Return grouped residue counts for a cleaned sequence."""
    cleaned = prepare_descriptor_sequence(sequence)
    normalized_groups = _normalize_groups(groups)
    return {
        group_name: count_residues(cleaned, residues)
        for group_name, residues in normalized_groups.items()
    }


def grouped_residue_fractions(
    sequence: str,
    *,
    groups: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, float]:
    """Return grouped residue fractions over cleaned-sequence length."""
    cleaned = prepare_descriptor_sequence(sequence)
    length = len(cleaned)
    counts = grouped_residue_counts(cleaned, groups=groups)
    if length == 0:
        return {group_name: 0.0 for group_name in counts}

    return {
        group_name: safe_divide(group_count, length)
        for group_name, group_count in counts.items()
    }


def grouped_fraction_descriptors(
    sequence: str,
    *,
    groups: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, float]:
    """Return grouped composition fractions with stable ``grp_*_frac`` names."""
    fractions = grouped_residue_fractions(sequence, groups=groups)
    return {
        f"grp_{group_name}_frac": group_fraction
        for group_name, group_fraction in fractions.items()
    }


def grouped_ratio_descriptors(
    sequence: str,
    *,
    groups: Mapping[str, Iterable[str]] | None = None,
) -> DescriptorRecord:
    """Return grouped composition ratios."""
    cleaned = prepare_descriptor_sequence(sequence)
    length = len(cleaned)
    normalized_groups = _normalize_groups(groups)
    _validate_ratio_group_requirements(normalized_groups)
    group_counts = grouped_residue_counts(cleaned, groups=normalized_groups)
    uncharged_count = max(length - group_counts.get("charged", 0), 0)

    ratios: DescriptorRecord = {}
    for ratio_name, numerator_group, denominator_group in GROUPED_RATIO_SPECS:
        numerator = group_counts[numerator_group]
        if denominator_group == "uncharged":
            denominator = uncharged_count
        else:
            denominator = group_counts[denominator_group]
        ratios[f"grp_ratio_{ratio_name}"] = _ratio_or_none(numerator, denominator)
    return ratios


def grouped_composition_descriptors(
    sequence: str,
    *,
    groups: Mapping[str, Iterable[str]] | None = None,
    include_ratios: bool = True,
) -> DescriptorRecord:
    """Return grouped composition descriptors for one cleaned sequence."""
    features: DescriptorRecord = dict(
        grouped_fraction_descriptors(sequence, groups=groups)
    )
    if include_ratios:
        features.update(grouped_ratio_descriptors(sequence, groups=groups))
    return features


class GroupedCompositionDescriptors(DescriptorBlock):
    """Compute grouped residue composition fractions and basic balances."""

    family_name = "grouped"
    column_prefix = "grp"

    def __init__(
        self,
        *,
        groups: Mapping[str, Iterable[str]] | None = None,
        include_ratios: bool = True,
    ) -> None:
        self.groups = _normalize_groups(groups)
        self.include_ratios = include_ratios
        self._validate_ratio_groups()
        super().__init__()

    def feature_names(self) -> tuple[str, ...]:
        """Return the stable grouped-composition feature order."""
        names = tuple(f"grp_{group_name}_frac" for group_name in self.groups)
        if not self.include_ratios:
            return names
        return names + tuple(
            f"grp_ratio_{ratio_name}" for ratio_name, _, _ in GROUPED_RATIO_SPECS
        )

    def _transform_sequence(self, sequence: str) -> DescriptorRecord:
        """Compute grouped composition features for one cleaned sequence."""
        features: DescriptorRecord = dict(
            grouped_fraction_descriptors(sequence, groups=self.groups)
        )
        if self.include_ratios:
            features.update(grouped_ratio_descriptors(sequence, groups=self.groups))
        return features

    def _validate_ratio_groups(self) -> None:
        """Ensure ratio features only run when the required groups are available."""
        if not self.include_ratios:
            return
        _validate_ratio_group_requirements(self.groups)


DEFAULT_GROUPED_FRACTION_GROUPS: dict[str, frozenset[str]] = {
    group_name: GROUPED_RESIDUE_SETS[group_name]
    for group_name in (
        "positive",
        "negative",
        "polar",
        "nonpolar",
        "aromatic",
    )
}


__all__ = [
    "DEFAULT_GROUPED_FRACTION_GROUPS",
    "DEFAULT_GROUP_ORDER",
    "GROUPED_RATIO_SPECS",
    "GROUPED_RESIDUE_SETS",
    "GroupedCompositionDescriptors",
    "grouped_composition_descriptors",
    "grouped_fraction_descriptors",
    "grouped_ratio_descriptors",
    "grouped_residue_counts",
    "grouped_residue_fractions",
]
