"""``roxy init-config`` — generate a YAML config with descriptor defaults."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import typer
import yaml

from roxy.cli._utils import _resolve_names
from roxy.descriptors import DESCRIPTOR_REGISTRY


def _default_params(cls: type) -> dict[str, object]:
    sig = inspect.signature(cls.__init__)
    return {
        name: param.default
        for name, param in sig.parameters.items()
        if name != "self" and param.default is not inspect.Parameter.empty
    }


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
    config_map = {
        name: _default_params(DESCRIPTOR_REGISTRY[name]) for name in names
    }
    body = yaml.safe_dump(
        config_map,
        sort_keys=False,
        default_flow_style=False,
    )
    content = header + "\n" + body

    if output is None:
        sys.stdout.write(content)
    else:
        output.write_text(content)
        typer.echo(f"Config written to {output}")
