"""Central descriptor registry and @register decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base import BaseDescriptor

DESCRIPTOR_REGISTRY: dict[str, type[BaseDescriptor]] = {}


def register(name: str, *, family: str = "misc") -> Callable[[type[BaseDescriptor]], type[BaseDescriptor]]:
    """Class decorator that registers a descriptor family by name.

    Usage::

        @register("aac", family="composition")
        class AACDescriptor(BaseDescriptor):
            ...
    """

    def decorator(cls: type[BaseDescriptor]) -> type[BaseDescriptor]:
        cls.name = name
        cls.family = family
        DESCRIPTOR_REGISTRY[name] = cls
        return cls

    return decorator


__all__ = ["DESCRIPTOR_REGISTRY", "register"]
