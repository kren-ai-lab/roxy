"""``roxy validate-config`` — check a ``--config`` file without reading any input data."""

from __future__ import annotations

import inspect
import os
import re
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer

from roxy.cli._config_file import ConfigIssue, read_config, report_issues
from roxy.cli._utils import _resolve_names
from roxy.descriptors import DESCRIPTOR_REGISTRY

_ROOT_KEYS = {"schema_version", "descriptors", "families", "all"}
_SCHEMA_VERSION = 1
_SEQUENCE_ORIGINS = (tuple, list, set, frozenset)


def _type_ok(value: object, tp: object) -> bool:  # noqa: PLR0911
    """Check a YAML value against an annotation. Unrecognised annotations pass."""
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is typing.Union or origin is types.UnionType:
        return any(_type_ok(value, a) for a in args)
    if tp is type(None):
        return value is None
    if tp is bool:
        return isinstance(value, bool)
    if tp is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if tp is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tp is str:
        return isinstance(value, str)
    if origin in _SEQUENCE_ORIGINS:
        # YAML has no tuples or sets: a list stands in for all of them
        return isinstance(value, list) and all(_type_ok(v, args[0]) for v in value if args)
    if origin is dict:
        return isinstance(value, dict) and all(_type_ok(v, args[1]) for v in value.values() if args)
    return True


def _type_name(tp: object) -> str:
    return tp.__name__ if isinstance(tp, type) else re.sub(r"\btyping\.", "", str(tp))


def _check_descriptor(name: str, params: object) -> list[ConfigIssue]:
    path = f"descriptors.{name}"
    cls = DESCRIPTOR_REGISTRY.get(name)
    if cls is None:
        return [ConfigIssue(path, "unknown descriptor")]
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return [ConfigIssue(path, f"expected a mapping of parameters, got {type(params).__name__}")]

    valid = {k for k in inspect.signature(cls.__init__).parameters if k != "self"}
    issues = [ConfigIssue(f"{path}.{k}", "unknown parameter") for k in params if k not in valid]
    try:
        hints = typing.get_type_hints(cls.__init__)
    except Exception:  # noqa: BLE001  # unresolvable annotations only skip the type check
        hints = {}
    issues += [
        ConfigIssue(f"{path}.{k}", f"expected {_type_name(hints[k])}, got {type(v).__name__}")
        for k, v in params.items()
        if k in hints and not _type_ok(v, hints[k])
    ]
    if issues:
        return issues
    try:
        cls(**params)
    except Exception as exc:  # noqa: BLE001  # any constructor failure is a bad value
        msg = str(exc)
        named = [k for k in params if re.search(rf"\b{re.escape(k)}\b", msg)]
        return [ConfigIssue(f"{path}.{named[0]}" if len(named) == 1 else path, msg)]
    return []


def _check_selection(raw: dict[str, Any]) -> list[ConfigIssue]:
    """Check ``families`` and ``all``, the selectors that add descriptors with default parameters."""
    issues: list[ConfigIssue] = []
    if "all" in raw and not isinstance(raw["all"], bool):
        issues.append(ConfigIssue("all", f"expected bool, got {type(raw['all']).__name__}"))
    families = raw.get("families")
    if families is None:
        return issues
    if not isinstance(families, list) or not all(isinstance(f, str) for f in families):
        issues.append(ConfigIssue("families", "expected a list of family names"))
        return issues
    known = {cls.family for cls in DESCRIPTOR_REGISTRY.values()}
    issues += [
        ConfigIssue(f"families.{i}", f"unknown family {f!r}; expected one of: {', '.join(sorted(known))}")
        for i, f in enumerate(families)
        if f not in known
    ]
    return issues


def check_config(raw: dict[str, Any]) -> list[ConfigIssue]:
    """Collect every problem in a parsed config of shape ``{schema_version, descriptors, families, all}``."""
    issues: list[ConfigIssue] = []
    for key in raw:
        if key in _ROOT_KEYS:
            continue
        hint = "; nest descriptor settings under 'descriptors:'" if key in DESCRIPTOR_REGISTRY else ""
        issues.append(ConfigIssue(str(key), f"unknown key{hint}"))
    if "schema_version" in raw and (
        raw["schema_version"] != _SCHEMA_VERSION or isinstance(raw["schema_version"], bool)
    ):
        issues.append(
            ConfigIssue("schema_version", f"unsupported schema_version, expected {_SCHEMA_VERSION}")
        )
    issues += _check_selection(raw)
    descriptors = raw.get("descriptors", {})
    if descriptors is None:
        descriptors = {}
    if not isinstance(descriptors, dict):
        issues.append(ConfigIssue("descriptors", "expected a mapping of descriptor name to parameters"))
        return issues
    if not descriptors and not raw.get("families") and raw.get("all") is not True:
        issues.append(
            ConfigIssue(
                "descriptors",
                "expected a non-empty mapping of descriptor name to parameters, "
                "or descriptors selected with 'families' or 'all: true'",
            )
        )
        return issues
    for name, params in descriptors.items():
        issues += _check_descriptor(str(name), params)
    return issues


def resolve_config(raw: dict[str, Any]) -> dict[str, dict[str, object]]:
    """Return ``{descriptor: params}`` from a valid config.

    With only ``descriptors``, the file order is kept. ``families`` and ``all: true`` select like
    ``-f`` and ``--all``: the union with the named ``descriptors`` is sorted, so a config and the
    equivalent flags give the same columns in the same order. Parameters under ``descriptors``
    override the defaults of any descriptor they name, however it was selected.
    """
    params = raw.get("descriptors") or {}
    families = list(raw.get("families") or [])
    if not families and raw.get("all") is not True:
        return {n: dict(p or {}) for n, p in params.items()}
    names = _resolve_names(raw.get("all") is True, list(params), families)
    return {n: dict(params.get(n) or {}) for n in names}


def load_config(path: Path) -> dict[str, dict[str, object]]:
    """Return ``{descriptor: params}`` from a valid config, else raise ``typer.BadParameter`` (exit 2)."""
    raw, issues = read_config(path)
    issues = issues or check_config(raw)
    if issues:
        lines = "\n".join(f"  {i.path or '<root>'}: {i.message}" for i in issues)
        msg = f"{path}:\n{lines}"
        raise typer.BadParameter(msg, param_hint="'--config'")
    return resolve_config(raw)


def validate_config(config: Mapping[str, Any] | str | os.PathLike[str]) -> list[ConfigIssue]:
    """Check a ``roxy compute --config`` config (a mapping or a YAML/JSON file path).

    Returns every problem found, empty when valid. Never raises for config problems.
    """
    if isinstance(config, Mapping):
        raw, issues = dict(config), []
    else:
        raw, issues = read_config(Path(config))
    return issues or check_config(raw)


def validate(
    config: Path = typer.Argument(..., help="YAML/JSON config file."),
    *,
    as_json: bool = typer.Option(False, "--json", help="Print problems as a JSON array on stdout."),
) -> None:
    """Validate CONFIG as accepted by `roxy compute --config CONFIG`."""
    raise typer.Exit(report_issues(validate_config(config), as_json=as_json))
