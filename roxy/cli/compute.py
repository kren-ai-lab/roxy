"""``roxy compute`` — run descriptors on a sequence file."""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import typer
import yaml

from roxy.cli._utils import _resolve_names
from roxy.core.exceptions import RoxyIOError
from roxy.core.io import read_sequences, write_table
from roxy.descriptors import DESCRIPTOR_REGISTRY

if TYPE_CHECKING:
    from roxy.descriptors.base import BaseDescriptor

_BASE_COLS = {"length", "valid_residue_count"}


def _load_config(path: Path) -> dict[str, dict[str, object]]:
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in {path}: {exc}"
        raise typer.BadParameter(msg) from exc

    if not isinstance(raw, dict):
        msg = "Config must be a YAML mapping of descriptor names to params."
        raise typer.BadParameter(msg)

    validated: dict[str, dict[str, object]] = {}
    for name, raw_params in raw.items():
        if name not in DESCRIPTOR_REGISTRY:
            available = ", ".join(sorted(DESCRIPTOR_REGISTRY))
            msg = f"Unknown descriptor {name!r}.\nAvailable: {available}"
            raise typer.BadParameter(msg)
        norm_params: dict[str, object] = raw_params or {}
        if not isinstance(norm_params, dict):
            msg = f"Params for {name!r} must be a mapping, got {type(norm_params).__name__}."
            raise typer.BadParameter(msg)
        sig = inspect.signature(DESCRIPTOR_REGISTRY[name].__init__)
        valid = {k for k in sig.parameters if k != "self"}
        unknown = set(norm_params) - valid
        if unknown:
            msg = (
                f"Unknown param(s) for {name!r}: {', '.join(sorted(unknown))}.\n"
                f"Valid: {', '.join(sorted(valid))}"
            )
            raise typer.BadParameter(msg)
        validated[name] = dict(norm_params)
    return validated


def _build_instances(
    config: dict[str, dict[str, object]] | None,
    names: list[str],
) -> list[tuple[str, BaseDescriptor]]:
    if config is not None:
        return [(n, DESCRIPTOR_REGISTRY[n](**p)) for n, p in config.items()]
    return [(n, DESCRIPTOR_REGISTRY[n]()) for n in names]


def _run_descriptors(
    instances: list[tuple[str, BaseDescriptor]],
    seqs: list[str],
    ids: list[str],
    *,
    no_progress: bool,
) -> tuple[list[pl.DataFrame], list[str]]:
    frames: list[pl.DataFrame] = []
    desc_names: list[str] = []

    if no_progress:
        for name, inst in instances:
            frames.append(inst.compute(seqs, ids=ids))
            desc_names.append(name)
        return frames, desc_names

    from rich.progress import Progress  # noqa: PLC0415

    with Progress() as progress:
        task = progress.add_task("Computing descriptors...", total=len(instances))
        for name, inst in instances:
            progress.update(task, description=f"[cyan]{name}[/cyan]")
            frames.append(inst.compute(seqs, ids=ids))
            desc_names.append(name)
            progress.advance(task)
    return frames, desc_names


def _concat_frames(frames: list[pl.DataFrame], desc_names: list[str]) -> pl.DataFrame:
    """Horizontally concat descriptor frames, keeping id and base cols only from first."""
    _drop_from_rest = _BASE_COLS | {"id"}
    parts: list[pl.DataFrame] = []
    for i, (df, name) in enumerate(zip(frames, desc_names, strict=True)):
        if i == 0:
            rename = {f"{name}_{col}": col for col in _BASE_COLS if f"{name}_{col}" in df.columns}
            parts.append(df.rename(rename))
        else:
            drop = [c for col in _drop_from_rest for c in (f"{name}_{col}", col) if c in df.columns]
            parts.append(df.drop(drop))
    return functools.reduce(pl.DataFrame.hstack, parts)


def compute(
    input_path: Path = typer.Argument(..., metavar="INPUT", help="FASTA, CSV, or Parquet file."),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="YAML config file with descriptor params."
    ),
    all_: bool = typer.Option(False, "--all", help="Run all registered descriptors."),
    descriptors: list[str] = typer.Option([], "--descriptor", "-d", help="Descriptor name (repeatable)."),
    family: list[str] = typer.Option(
        [], "--family", "-f", help="Family name — runs all descriptors in it (repeatable)."
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output file (.csv or .parquet)."),
    seq_col: str = typer.Option("sequence", "--seq-col", help="Sequence column name (CSV/Parquet)."),
    id_col: str | None = typer.Option(None, "--id-col", help="ID column to preserve (CSV/Parquet)."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bar."),
) -> None:
    """Compute descriptors for sequences in INPUT and write to OUTPUT."""
    if config is not None and (all_ or descriptors or family):
        typer.echo("--config is mutually exclusive with --all / -d / -f.", err=True)
        raise typer.Exit(1)

    if config is not None:
        cfg = _load_config(config)
        instances = _build_instances(cfg, [])
    else:
        names = _resolve_names(all_, descriptors, family)
        if not names:
            typer.echo("Specify --config, --all, -d <name>, or -f <family>.", err=True)
            raise typer.Exit(1)
        unknown = [n for n in names if n not in DESCRIPTOR_REGISTRY]
        if unknown:
            typer.echo(f"Unknown descriptors: {', '.join(unknown)}", err=True)
            raise typer.Exit(1)
        instances = _build_instances(None, names)

    try:
        records = read_sequences(input_path, seq_col=seq_col, id_col=id_col)
    except RoxyIOError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    ids = [r[0] for r in records]
    seqs = [r[1] for r in records]

    try:
        frames, desc_names = _run_descriptors(instances, seqs, ids, no_progress=no_progress)
    except Exception as exc:
        typer.echo(f"Error computing descriptors: {exc}", err=True)
        raise typer.Exit(1) from exc

    result = _concat_frames(frames, desc_names)

    try:
        write_table(result, output)
    except RoxyIOError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Wrote {result.shape[0]} sequences x {result.shape[1]} features → {output}")
