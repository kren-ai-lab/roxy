from __future__ import annotations

"""Markdown rendering utilities for dataset-level reports.

This module provides :class:`MarkdownReportBuilder`, which converts a
:class:`roxy.core.report.DatasetReport` into a human-readable Markdown
document, and a convenience wrapper
:func:`dataset_report_to_markdown`.
"""

from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from roxy.core.report import DatasetReport, FeatureSummary
from roxy.core.logging_utils import get_logger

logger = get_logger(__name__)


class MarkdownReportBuilder:
    """Build a human-readable Markdown report from a :class:`DatasetReport`.

    The goal is to provide a lightweight textual summary that can be
    embedded in notebooks, saved to disk, or converted to HTML/PDF by
    downstream tools.

    Parameters
    ----------
    include_feature_table :
        Whether to include a table with per-feature summary statistics.
    max_features :
        Maximum number of features to display in the summary table.
        If ``None``, all features are included.
    include_correlation :
        Whether to include a textual representation of the correlation
        matrix, if available in the report.
    float_fmt :
        Format string for floating-point values (e.g. ``".3f"``).
    """

    def __init__(
        self,
        *,
        include_feature_table: bool = True,
        max_features: Optional[int] = None,
        include_correlation: bool = True,
        float_fmt: str = ".3f",
    ) -> None:
        self.include_feature_table = include_feature_table
        self.max_features = max_features
        self.include_correlation = include_correlation
        self.float_fmt = float_fmt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, report: DatasetReport) -> str:
        """Build a Markdown document from a :class:`DatasetReport`.

        Parameters
        ----------
        report :
            Dataset-level report produced by the EDA utilities.

        Returns
        -------
        str
            Markdown representation of the report.
        """
        logger.info(
            "Building Markdown report for dataset=%r (n_samples=%d, n_features=%d).",
            report.dataset_name,
            report.n_samples,
            report.n_features,
        )

        lines: List[str] = []

        title = report.dataset_name or "Dataset report"
        lines.append(f"# Roxy dataset report — {title}")
        lines.append("")

        # Overview
        lines.extend(self._build_overview_section(report))

        # Feature summary table
        if self.include_feature_table:
            lines.append("## Feature summary")
            lines.append("")
            lines.extend(self._build_feature_table(report))
            lines.append("")

        # Correlation section
        if self.include_correlation and report.correlation_matrix is not None:
            lines.append("## Correlation matrix")
            lines.append("")
            lines.extend(self._build_correlation_section(report))
            lines.append("")

        # Notes, if any
        if getattr(report, "notes", None):
            lines.append("## Notes")
            lines.append("")
            for note in report.notes:
                lines.append(f"- {note}")
            lines.append("")

        md = "\n".join(lines)
        logger.debug(
            "Markdown report built with %d lines for dataset %r.",
            len(lines),
            report.dataset_name,
        )
        return md

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_overview_section(self, report: DatasetReport) -> List[str]:
        lines: List[str] = []
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- **Samples:** `{report.n_samples}`")
        lines.append(f"- **Features:** `{report.n_features}`")

        if report.task_type is not None:
            lines.append(f"- **Task type:** `{report.task_type}`")

        if report.class_distribution:
            lines.append("- **Class distribution:**")
            for cls, count in report.class_distribution.items():
                lines.append(f"  - `{cls}`: `{count}`")
        lines.append("")
        return lines

    def _build_feature_table(self, report: DatasetReport) -> List[str]:
        # Convert feature summaries into a DataFrame for easier formatting
        feature_dicts: List[dict] = []
        for name, fs in report.feature_summaries.items():
            # asdict handles optional fields gracefully
            d = asdict(fs)
            d["name"] = name  # ensure consistency with dict key
            feature_dicts.append(d)

        if not feature_dicts:
            logger.debug("MarkdownReportBuilder: no feature summaries available.")
            return ["_No feature summaries available._", ""]

        df = pd.DataFrame(feature_dicts)

        # Reorder and rename columns to something user-friendly
        column_order: List[str] = [
            "name",
            "dtype",
            "n_missing",
            "missing_ratio",
            "n_unique",
            "mean",
            "std",
            "min",
            "max",
            "skewness",
            "kurtosis",
        ]
        df = df[[c for c in column_order if c in df.columns]]

        # Apply max_features limit if requested
        if self.max_features is not None and self.max_features < len(df):
            logger.debug(
                "MarkdownReportBuilder: truncating feature table from %d to %d rows.",
                len(df),
                self.max_features,
            )
            df = df.head(self.max_features)

        # Format floats
        float_cols = [
            c
            for c in [
                "missing_ratio",
                "mean",
                "std",
                "min",
                "max",
                "skewness",
                "kurtosis",
            ]
            if c in df.columns
        ]
        df[float_cols] = df[float_cols].applymap(
            lambda x: f"{x:{self.float_fmt}}" if pd.notnull(x) else ""
        )

        # Build Markdown table manually (no external deps)
        header = "| " + " | ".join(df.columns) + " |"
        separator = "| " + " | ".join("---" for _ in df.columns) + " |"

        rows = [header, separator]
        for _, row in df.iterrows():
            cells = [str(row[c]) for c in df.columns]
            rows.append("| " + " | ".join(cells) + " |")

        return rows

    def _build_correlation_section(self, report: DatasetReport) -> List[str]:
        corr = report.correlation_matrix
        if corr is None or corr.empty:
            logger.debug("MarkdownReportBuilder: no correlation matrix available.")
            return ["_No correlations available._", ""]

        # Limit size in case of very wide matrices
        max_dim = 25
        if corr.shape[0] > max_dim:
            logger.debug(
                "MarkdownReportBuilder: truncating correlation matrix from %d to %d.",
                corr.shape[0],
                max_dim,
            )
            corr = corr.iloc[:max_dim, :max_dim]

        # Round for readability
        corr = corr.round(3)

        lines: List[str] = []
        lines.append(
            "_Shown below is a truncated correlation matrix for numeric features "
            f"(up to {max_dim}×{max_dim})._"
        )
        lines.append("")
        lines.append("```text")
        lines.append(corr.to_string())
        lines.append("```")
        return lines


def dataset_report_to_markdown(
    report: DatasetReport,
    *,
    include_feature_table: bool = True,
    max_features: Optional[int] = None,
    include_correlation: bool = True,
    float_fmt: str = ".3f",
) -> str:
    """Convenience wrapper to build a Markdown report.

    Parameters
    ----------
    report :
        Dataset-level report produced by the EDA utilities.
    include_feature_table :
        Whether to include the feature summary table.
    max_features :
        Maximum number of features to display in the feature table.
    include_correlation :
        Whether to include the correlation matrix (if present).
    float_fmt :
        Format string for floating-point values.

    Returns
    -------
    str
        A complete Markdown document as a single string.
    """
    logger.info(
        "dataset_report_to_markdown: rendering dataset=%r with %d features.",
        report.dataset_name,
        report.n_features,
    )
    builder = MarkdownReportBuilder(
        include_feature_table=include_feature_table,
        max_features=max_features,
        include_correlation=include_correlation,
        float_fmt=float_fmt,
    )
    return builder.build(report)
