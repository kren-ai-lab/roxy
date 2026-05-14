"""Composition-Transition-Distribution (CTD) descriptors."""

from __future__ import annotations

import math
from collections import Counter

from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_CTD_GROUPS: dict[str, dict[str, frozenset[str]]] = {
    "hydrophobicity": {
        "1": frozenset("RKEDQN"),
        "2": frozenset("GASTPHY"),
        "3": frozenset("CLVIMFW"),
    },
    "polarity": {
        "1": frozenset("LIFWCMVY"),
        "2": frozenset("PATGS"),
        "3": frozenset("HQRKNED"),
    },
    "charge": {
        "1": frozenset("KR"),
        "2": frozenset("ANCQGHILMFPSTWYV"),
        "3": frozenset("DE"),
    },
}

_CLASS_IDS = ("1", "2", "3")
_TRANSITION_PAIRS = (("1", "2"), ("1", "3"), ("2", "3"))
_DIST_LABELS = ("001", "025", "050", "075", "100")
_DIST_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)


def _assign_classes(seq: str, groups: dict[str, frozenset[str]]) -> list[str]:
    labels = []
    for aa in seq:
        for class_id, members in groups.items():
            if aa in members:
                labels.append(class_id)
                break
    return labels


def _composition(labels: list[str], prefix: str) -> dict[str, float]:
    total = len(labels)
    if total == 0:
        return {f"{prefix}_comp_{c}": _NAN for c in _CLASS_IDS}
    counts = Counter(labels)
    return {f"{prefix}_comp_{c}": counts.get(c, 0) / total for c in _CLASS_IDS}


def _transition(labels: list[str], prefix: str) -> dict[str, float]:
    if len(labels) < 2:  # noqa: PLR2004
        return {f"{prefix}_trans_{a}{b}": _NAN for a, b in _TRANSITION_PAIRS}
    total = len(labels) - 1
    counts: dict[tuple[str, str], int] = dict.fromkeys(_TRANSITION_PAIRS, 0)
    for i in range(total):
        left = labels[i]
        right = labels[i + 1]
        pair = (left, right) if left <= right else (right, left)
        if pair in counts:
            counts[pair] += 1
    return {f"{prefix}_trans_{a}{b}": counts[(a, b)] / total for a, b in _TRANSITION_PAIRS}


def _distribution(labels: list[str], prefix: str, seq_len: int) -> dict[str, float]:
    result: dict[str, float] = {}
    if seq_len == 0:
        for c in _CLASS_IDS:
            for dlabel in _DIST_LABELS:
                result[f"{prefix}_dist_{c}_{dlabel}"] = _NAN
        return result
    for c in _CLASS_IDS:
        positions = [i + 1 for i, lbl in enumerate(labels) if lbl == c]
        if not positions:
            for dlabel in _DIST_LABELS:
                result[f"{prefix}_dist_{c}_{dlabel}"] = _NAN
            continue
        n = len(positions)
        for dlabel, frac in zip(_DIST_LABELS, _DIST_FRACTIONS, strict=True):
            idx = math.ceil(frac * n) - 1 if frac > 0 else 0
            idx = max(0, min(idx, n - 1))
            result[f"{prefix}_dist_{c}_{dlabel}"] = positions[idx] / seq_len
    return result


@register("ctd_classic", family="ctd")
class CTDClassicDescriptor(BaseDescriptor):
    r"""Classic Composition-Transition-Distribution (CTD) descriptors.

    For each physicochemical property, residues are assigned to one of
    three classes (Dubchak et al., 1995). CTD then computes:

    * **(C) Composition**: class fractions :math:`|\text{class}| / N`.
    * **(T) Transition**: fraction of adjacent pairs that switch between
      two classes :math:`\text{transitions}(a,b) / (N - 1)`.
    * **(D) Distribution**: normalized positions at 5 quantiles
      (0%, 25%, 50%, 75%, 100%) for each class.

    Args:
        properties: Property names to use. Must be keys in ``ctd_groups``.
        ctd_groups: Mapping of property to ``{class_id: residue_set}``.
            Defaults to built-in hydrophobicity, polarity, and charge
            partitions.

    Returns:
        ``compute()`` returns a DataFrame with columns prefixed ``ctd_classic_``.
        ``length``, ``valid_residue_count``,
        per property: ``{prop}_comp_{1/2/3}``,
        ``{prop}_trans_{12/13/23}``,
        ``{prop}_dist_{1/2/3}_{001/025/050/075/100}``.

    """

    def __init__(
        self,
        *,
        properties: tuple[str, ...] = ("hydrophobicity", "polarity", "charge"),
        ctd_groups: dict[str, dict[str, frozenset[str]]] | None = None,
    ) -> None:
        """Initialize CTDClassicDescriptor."""
        self.properties = tuple(properties)
        self.ctd_groups = ctd_groups if ctd_groups is not None else _CTD_GROUPS

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for prop in self.properties:
            prefix = prop
            for c in _CLASS_IDS:
                feats[f"{prefix}_comp_{c}"] = _NAN
            for a, b in _TRANSITION_PAIRS:
                feats[f"{prefix}_trans_{a}{b}"] = _NAN
            for c in _CLASS_IDS:
                for dlabel in _DIST_LABELS:
                    feats[f"{prefix}_dist_{c}_{dlabel}"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute CTD features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        for prop in self.properties:
            groups = self.ctd_groups[prop]
            labels = _assign_classes(seq, groups)
            prefix = prop
            feats.update(_composition(labels, prefix))
            feats.update(_transition(labels, prefix))
            feats.update(_distribution(labels, prefix, n))

        return feats
