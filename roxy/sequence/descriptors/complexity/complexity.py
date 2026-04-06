"""General sequence complexity descriptors."""

from __future__ import annotations

import math
from collections.abc import Iterable

from roxy.sequence.descriptors._base import prepare_descriptor_sequence


def shannon_entropy(sequence: str, alphabet: Iterable[str]) -> float:
    """Return Shannon entropy in bits over the provided alphabet."""
    cleaned = prepare_descriptor_sequence(sequence)
    length = len(cleaned)
    if length == 0:
        return math.nan

    probabilities = []
    for token in alphabet:
        count = cleaned.count(token)
        if count:
            probabilities.append(count / length)

    if not probabilities:
        return math.nan
    return float(
        -sum(probability * math.log2(probability) for probability in probabilities)
    )


def linguistic_complexity(
    sequence: str,
    *,
    max_k: int = 3,
    alphabet_size: int = 20,
) -> dict[str, float]:
    """Compute simple k-mer linguistic complexity for ``k = 1..max_k``."""
    cleaned = prepare_descriptor_sequence(sequence)
    length = len(cleaned)
    if length == 0:
        return {f"lc_k{k}": 0.0 for k in range(1, max_k + 1)}

    features: dict[str, float] = {}
    for k in range(1, max_k + 1):
        if length < k:
            features[f"lc_k{k}"] = 0.0
            continue
        observed = {cleaned[i : i + k] for i in range(length - k + 1)}
        max_possible = min(alphabet_size**k, length - k + 1)
        features[f"lc_k{k}"] = (
            len(observed) / max_possible if max_possible > 0 else 0.0
        )
    return features


__all__ = [
    "linguistic_complexity",
    "shannon_entropy",
]

__all__: tuple[str, ...] = ()
