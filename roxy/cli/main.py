"""Roxy CLI entrypoint."""

from __future__ import annotations

import typer

from roxy import __version__
from roxy.cli._shared import HELP_CONTEXT_SETTINGS
from roxy.cli.list_descriptors import list_descriptors

app = typer.Typer(
    name="roxy",
    add_completion=False,
    context_settings=HELP_CONTEXT_SETTINGS,
    help="Roxy — protein sequence descriptors for ML.",
)


def _version_callback(value: bool | None) -> None:  # noqa: FBT001
    if value:
        typer.echo(f"roxy {__version__}")
        raise typer.Exit


@app.callback()
def main(
    _version: bool | None = typer.Option(  # noqa: FBT001
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
    help="List all registered descriptor families.",
)(list_descriptors)


if __name__ == "__main__":
    app()
