"""Central descriptor registry and @register decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from .base import BaseDescriptor

DESCRIPTOR_REGISTRY: dict[str, type[BaseDescriptor]] = {}

_T = TypeVar("_T", bound="BaseDescriptor")


def register(name: str, *, family: str = "misc") -> Callable[[type[_T]], type[_T]]:
    """Class decorator that registers a descriptor family by name.

    Usage::

        @register("aac", family="composition")
        class AACDescriptor(BaseDescriptor):
            ...
    """

    def decorator(cls: type[_T]) -> type[_T]:
        cls.name = name
        cls.family = family
        DESCRIPTOR_REGISTRY[name] = cls
        return cls

    return decorator


__all__ = ["DESCRIPTOR_REGISTRY", "register"]
