"""Roxy CLI entrypoint."""

from __future__ import annotations

import typer

from roxy import __version__
from roxy.cli._shared import HELP_CONTEXT_SETTINGS
from roxy.cli.compute import compute
from roxy.cli.describe_descriptor import describe_descriptor
from roxy.cli.init_config import init_config
from roxy.cli.list_descriptors import list_descriptors

app = typer.Typer(
    name="roxy",
    add_completion=False,
    context_settings=HELP_CONTEXT_SETTINGS,
    help="Roxy — protein sequence descriptors for ML.",
)


def _version_callback(value: bool | None) -> None:
    if value:
        typer.echo(f"roxy {__version__}")
        raise typer.Exit


@app.callback()
def main(
    _version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Roxy CLI main callback."""


app.command(
    name="list",
    help="List all registered descriptors grouped by family.",
)(list_descriptors)

app.command(
    name="describe",
    help="Show full details for a descriptor.",
)(describe_descriptor)

app.command(
    name="compute",
    help="Compute descriptors for sequences in a file.",
)(compute)

app.command(
    name="init-config",
    help="Generate a YAML config file populated with descriptor defaults.",
)(init_config)


if __name__ == "__main__":
    app()
