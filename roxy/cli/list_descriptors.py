"""``roxy list`` — display all registered descriptor families."""

from __future__ import annotations

import inspect

import typer

from roxy.descriptors import DESCRIPTOR_REGISTRY


def list_descriptors(
    family: str | None = typer.Option(None, "--family", "-f", help="Filter by family name."),
) -> None:
    """List all registered descriptors grouped by family."""
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.tree import Tree  # noqa: PLC0415

        entries = sorted(DESCRIPTOR_REGISTRY.items())
        if family:
            entries = [(n, c) for n, c in entries if c.family == family]

        if not entries:
            Console().print("[yellow]No descriptors registered.[/yellow]")
            raise typer.Exit

        families: dict[str, list[tuple[str, type]]] = {}
        for name, cls in entries:
            families.setdefault(cls.family, []).append((name, cls))

        total = sum(len(v) for v in families.values())
        tree = Tree(f"[bold]Roxy Descriptor Registry[/bold] ({total})")

        for fam in sorted(families):
            descriptors = families[fam]
            branch = tree.add(f"[green]{fam}[/green] ({len(descriptors)})")
            for name, cls in descriptors:
                doc = inspect.getdoc(cls) or ""
                first_line = doc.splitlines()[0] if doc else ""
                branch.add(f"[cyan]{name}[/cyan]  [dim]{first_line}[/dim]")

        console = Console()
        console.print(tree)
        console.print()
        console.print("[dim]Use [bold]roxy describe <name>[/bold] for full details.[/dim]")

    except ImportError:
        entries = sorted(DESCRIPTOR_REGISTRY.items())
        if family:
            entries = [(n, c) for n, c in entries if c.family == family]
        for name, cls in entries:
            typer.echo(f"{name:<30} {cls.family}")
