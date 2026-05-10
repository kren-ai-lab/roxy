"""``roxy init-config`` — generate a YAML config with descriptor defaults."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import typer

from roxy.descriptors import DESCRIPTOR_REGISTRY

_YAML_SPECIAL = frozenset(': #{}[]|>&*!,\'"')


def _value_to_yaml(v: object) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return f'"{v}"' if any(c in v for c in _YAML_SPECIAL) else v
    if isinstance(v, (list, tuple)):
        return "[]" if not v else "[" + ", ".join(_value_to_yaml(i) for i in v) + "]"
    return str(v)


def _ann_str(ann: object) -> str:
    raw = str(ann)
    for old, new in (
        ("<class '", ""),
        ("'>", ""),
        ("typing.", ""),
        ("collections.abc.", ""),
        ("builtins.", ""),
    ):
        raw = raw.replace(old, new)
    return raw


def _descriptor_block(name: str, cls: type) -> str:
    sig = inspect.signature(cls.__init__)
    params = {k: v for k, v in sig.parameters.items() if k != "self"}
    lines = [f"{name}:"]
    if not params:
        lines.append("  {}  # no configurable parameters")
    for pname, p in params.items():
        val = _value_to_yaml(p.default) if p.default is not inspect.Parameter.empty else "~"
        ann = f"  # {_ann_str(p.annotation)}" if p.annotation is not inspect.Parameter.empty else ""
        lines.append(f"  {pname}: {val}{ann}")
    return "\n".join(lines)


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


def init_config(
    all_: bool = typer.Option(False, "--all", help="Include all registered descriptors."),
    descriptors: list[str] = typer.Option([], "--descriptor", "-d", help="Descriptor name (repeatable)."),
    family: list[str] = typer.Option([], "--family", "-f", help="Family name (repeatable)."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file (default: stdout)."),
) -> None:
    """Generate a YAML config file populated with descriptor defaults."""
    names = _resolve_names(all_, descriptors, family)

    if not names:
        typer.echo("Specify --all, -d <name>, or -f <family>.", err=True)
        raise typer.Exit(1)

    unknown = [n for n in names if n not in DESCRIPTOR_REGISTRY]
    if unknown:
        typer.echo(f"Unknown descriptors: {', '.join(unknown)}", err=True)
        raise typer.Exit(1)

    header = (
        "# Roxy descriptor configuration\n"
        "# Edit parameters, then run:\n"
        "#   roxy compute sequences.fasta --config <this-file> -o results.parquet\n"
        "# Use 'roxy describe <name>' for parameter documentation.\n"
    )
    blocks = "\n\n".join(
        _descriptor_block(n, DESCRIPTOR_REGISTRY[n]) for n in names
    )
    content = header + "\n" + blocks + "\n"

    if output is None:
        sys.stdout.write(content)
    else:
        output.write_text(content)
        typer.echo(f"Config written to {output}")
