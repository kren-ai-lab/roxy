"""``roxy list`` — display all registered descriptor families."""

from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING

import typer

from roxy.descriptors import DESCRIPTOR_REGISTRY

if TYPE_CHECKING:
    from rich.text import Text


def _descriptor_lines(
    name: str,
    summary: str,
    *,
    name_width: int,
    desc_width: int,
) -> list[Text]:
    from rich.text import Text  # noqa: PLC0415

    desc_lines = textwrap.wrap(summary, width=desc_width) or [""]
    lines: list[Text] = []
    for idx, desc_line in enumerate(desc_lines):
        line = Text()
        if idx == 0:
            line.append(f"{name:<{name_width}}", style="cyan")
        else:
            line.append(" " * name_width)
        line.append("  ")
        line.append(desc_line, style="dim")
        lines.append(line)
    return lines


def list_descriptors(
    family: str | None = typer.Option(None, "--family", "-f", help="Filter by family name."),
) -> None:
    """List all registered descriptors grouped by family."""
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

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
        max_name = max(len(n) for n, _ in entries)
        console = Console()
        console.print(f"[bold]Roxy descriptors[/bold] ({total})")

        for fam in sorted(families):
            descriptors = families[fam]
            title = Text()
            title.append(fam, style="bold green")
            title.append(f" ({len(descriptors)})", style="dim")
            console.print()
            console.print(title)

            for name, cls in descriptors:
                doc = inspect.getdoc(cls) or ""
                first_line = doc.splitlines()[0] if doc else ""
                desc_width = max(24, console.width - max_name - 2)
                for line in _descriptor_lines(
                    name,
                    first_line,
                    name_width=max_name,
                    desc_width=desc_width,
                ):
                    console.print(line)

        console.print()
        console.print("[dim]Use [bold]roxy describe <name>[/bold] for full details.[/dim]")

    except ImportError:
        entries = sorted(DESCRIPTOR_REGISTRY.items())
        if family:
            entries = [(n, c) for n, c in entries if c.family == family]
        for name, cls in entries:
            typer.echo(f"{name:<30} {cls.family}")
