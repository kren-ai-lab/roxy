"""Shared CLI utilities."""

from __future__ import annotations

from roxy.descriptors import DESCRIPTOR_REGISTRY


def _resolve_names(
    all_: bool,
    descriptors: list[str],
    family: list[str],
) -> list[str]:
    if all_:
        return sorted(DESCRIPTOR_REGISTRY)
    names: list[str] = list(descriptors)
    for fam in family:
        names += [n for n, c in DESCRIPTOR_REGISTRY.items() if c.family == fam]
    seen: set[str] = set()
    result: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return sorted(result)
