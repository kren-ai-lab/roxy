"""``roxy list`` — display all registered descriptor families."""

from __future__ import annotations

import typer

from roxy.descriptors import DESCRIPTOR_REGISTRY


def list_descriptors(
    family: str | None = typer.Option(None, "--family", "-f", help="Filter by family name."),
) -> None:
    """List all registered descriptor families."""
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415

        console = Console()
        table = Table(title="Roxy Descriptor Registry", show_lines=False)
        table.add_column("Name", style="bold cyan")
        table.add_column("Family", style="green")

        entries = sorted(DESCRIPTOR_REGISTRY.items())
        if family:
            entries = [(n, c) for n, c in entries if c.family == family]

        if not entries:
            console.print("[yellow]No descriptors registered.[/yellow]")
            raise typer.Exit

        for name, cls in entries:
            table.add_row(name, cls.family)

        console.print(table)

    except ImportError:
        # Fallback without rich
        entries = sorted(DESCRIPTOR_REGISTRY.items())
        if family:
            entries = [(n, c) for n, c in entries if c.family == family]
        for name, cls in entries:
            typer.echo(f"{name:<30} {cls.family}")
