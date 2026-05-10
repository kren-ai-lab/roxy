"""``roxy cache`` — manage the Roxy file cache (AAIndex CSV, etc.)."""

from __future__ import annotations

import shutil

import typer

from roxy.core.config import get_cache_root

app = typer.Typer(
    name="cache",
    help="Inspect and manage the Roxy cache (AAIndex CSV and other downloads).",
    no_args_is_help=True,
)


@app.command("path")
def cache_path() -> None:
    """Print the current cache root directory."""
    typer.echo(str(get_cache_root()))


@app.command("list")
def cache_list() -> None:
    """List files present in the cache."""
    root = get_cache_root()
    files = sorted(root.rglob("*"))
    if not files:
        typer.echo("Cache is empty.")
        return
    for f in files:
        if f.is_file():
            size = f.stat().st_size
            typer.echo(f"{f.relative_to(root)}  ({size:,} bytes)")


@app.command("clear")
def cache_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete all files in the cache directory."""
    root = get_cache_root()
    if not yes:
        typer.confirm(
            f"Delete all cached files in {root}?",
            abort=True,
        )
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    typer.echo("Cache cleared.")
