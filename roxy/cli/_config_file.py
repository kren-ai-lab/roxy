"""Config-file helpers from the shared ecosystem CLI contract (copied, not imported)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import typer
import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ConfigIssue:
    """One problem found in a config file; ``path`` is ``""`` for the root."""

    path: str
    message: str


def read_config(path: Path) -> tuple[dict[str, Any], list[ConfigIssue]]:
    """Parse a YAML (or JSON) config whose root must be a mapping. An empty file is ``{}``."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return {}, [ConfigIssue("", f"cannot read config: {exc}")]
    if raw is None:
        return {}, []
    if not isinstance(raw, dict):
        return {}, [ConfigIssue("", f"config root must be a mapping, got {type(raw).__name__}")]
    return raw, []


def report_issues(issues: list[ConfigIssue], *, as_json: bool) -> int:
    """Print issues in the contract's format and return the exit code (0 valid, 2 invalid)."""
    if as_json:
        typer.echo(json.dumps([asdict(issue) for issue in issues]))
    else:
        for issue in issues:
            typer.echo(f"{issue.path or '<root>'}: {issue.message}", err=True)
    if not issues and not as_json:
        typer.echo("config is valid", err=True)
    return 2 if issues else 0
