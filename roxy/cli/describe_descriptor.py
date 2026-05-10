"""``roxy describe`` — show full details for a registered descriptor."""

from __future__ import annotations

import inspect

import typer

from roxy.descriptors import DESCRIPTOR_REGISTRY

_SEQ = "ACDEFGHIKLMNPQRSTVWY"


def _parse_google_args(doc: str) -> dict[str, str]:
    """Extract param descriptions from a Google-style Args section."""
    args: dict[str, str] = {}
    in_args = False
    current: str | None = None
    parts: list[str] = []

    for line in doc.splitlines():
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if not in_args:
            continue
        # New top-level section ends Args
        if stripped and not line.startswith(" "):
            break
        # Param line: exactly 4-space indent with ": "
        if line.startswith("    ") and not line.startswith("        ") and ": " in stripped:
            if current is not None:
                args[current] = " ".join(parts)
            param, _, desc = stripped.partition(": ")
            current, parts = param.strip(), [desc.strip()]
        elif line.startswith("        ") and current is not None:
            parts.append(stripped)

    if current is not None:
        args[current] = " ".join(parts)
    return args


def _summary(doc: str) -> str:
    return doc.splitlines()[0] if doc else ""


def describe_descriptor(name: str = typer.Argument(..., help="Descriptor name.")) -> None:
    """Show family, parameters, column count, and column names for a descriptor."""
    cls = DESCRIPTOR_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(DESCRIPTOR_REGISTRY))
        typer.echo(f"Unknown descriptor: {name!r}\nAvailable: {available}", err=True)
        raise typer.Exit(1)

    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.panel import Panel  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
        from rich.text import Text  # noqa: PLC0415

        console = Console()

        doc = inspect.getdoc(cls) or ""
        sig = inspect.signature(cls.__init__)
        params = {k: v for k, v in sig.parameters.items() if k != "self"}
        arg_descs = _parse_google_args(doc)

        inst = cls()
        columns = list(inst.compute_one(_SEQ))

        header = Text()
        header.append(name, style="bold cyan")
        header.append(f"  [{cls.family}]", style="green")
        console.print(header)
        console.print()
        if doc:
            console.print(_summary(doc))
            console.print()

        if params:
            ptable = Table(show_header=True, header_style="bold", box=None, padding=(0, 2, 0, 0))
            ptable.add_column("Parameter")
            ptable.add_column("Default")
            ptable.add_column("Annotation")
            ptable.add_column("Description")
            for pname, p in params.items():
                default = "" if p.default is inspect.Parameter.empty else repr(p.default)
                ann = "" if p.annotation is inspect.Parameter.empty else str(p.annotation)
                desc = arg_descs.get(pname, "")
                ptable.add_row(pname, default, ann, f"[dim]{desc}[/dim]")
            console.print(Panel(ptable, title="Parameters", title_align="left"))

        col_text = "  ".join(f"[dim]{c}[/dim]" for c in columns)
        console.print(
            Panel(
                f"[bold]{len(columns)} columns[/bold] (default parameters)\n\n{col_text}",
                title="Output columns",
                title_align="left",
            )
        )

    except ImportError:
        inst = cls()
        columns = list(inst.compute_one(_SEQ))
        sig = inspect.signature(cls.__init__)
        params = {k: v for k, v in sig.parameters.items() if k != "self"}
        typer.echo(f"name:   {name}")
        typer.echo(f"family: {cls.family}")
        typer.echo(f"params: {', '.join(params) or 'none'}")
        typer.echo(f"cols ({len(columns)}): {', '.join(columns)}")
