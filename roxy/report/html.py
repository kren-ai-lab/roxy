from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from roxy.core.report import DatasetReport, FeatureSummary


class HTMLReportBuilder:
    """Build a minimal HTML representation of a :class:`DatasetReport`.

    The generated HTML is intentionally lightweight and self-contained,
    so it can be saved as a standalone file or embedded into other
    front-ends (e.g. dashboards or static websites).

    Parameters
    ----------
    include_feature_table :
        Whether to include a table with per-feature summary statistics.
    max_features :
        Maximum number of features to display in the summary table.
        If ``None``, all features are included.
    include_correlation :
        Whether to include an HTML representation of the correlation
        matrix, if available in the report.
    """

    def __init__(
        self,
        *,
        include_feature_table: bool = True,
        max_features: Optional[int] = None,
        include_correlation: bool = True,
    ) -> None:
        self.include_feature_table = include_feature_table
        self.max_features = max_features
        self.include_correlation = include_correlation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, report: DatasetReport) -> str:
        """Build a full HTML document from a :class:`DatasetReport`."""
        title = report.dataset_name or "Roxy dataset report"

        parts: List[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append("<html lang='en'>")
        parts.append("<head>")
        parts.append("<meta charset='utf-8' />")
        parts.append(f"<title>{self._escape(title)}</title>")
        parts.append(self._default_styles())
        parts.append("</head>")
        parts.append("<body>")
        parts.append(f"<h1>Roxy dataset report — {self._escape(title)}</h1>")

        # Overview
        parts.append(self._build_overview_section(report))

        # Feature table
        if self.include_feature_table:
            parts.append("<h2>Feature summary</h2>")
            parts.append(self._build_feature_table(report))

        # Correlation matrix
        if self.include_correlation and report.correlation_matrix is not None:
            parts.append("<h2>Correlation matrix</h2>")
            parts.append(self._build_correlation_section(report))

        # Notes
        if getattr(report, "notes", None):
            parts.append("<h2>Notes</h2>")
            parts.append("<ul>")
            for note in report.notes:
                parts.append(f"<li>{self._escape(str(note))}</li>")
            parts.append("</ul>")

        parts.append("</body>")
        parts.append("</html>")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_overview_section(self, report: DatasetReport) -> str:
        items: List[str] = []
        items.append("<section class='overview'>")
        items.append("<h2>Overview</h2>")
        items.append("<ul>")
        items.append(f"<li><strong>Samples:</strong> {report.n_samples}</li>")
        items.append(f"<li><strong>Features:</strong> {report.n_features}</li>")
        if report.task_type is not None:
            items.append(f"<li><strong>Task type:</strong> {self._escape(report.task_type)}</li>")
        if report.class_distribution:
            items.append("<li><strong>Class distribution:</strong>")
            items.append("<ul>")
            for cls, count in report.class_distribution.items():
                items.append(
                    f"<li><code>{self._escape(str(cls))}</code>: {int(count)}</li>"
                )
            items.append("</ul>")
            items.append("</li>")
        items.append("</ul>")
        items.append("</section>")
        return "\n".join(items)

    def _build_feature_table(self, report: DatasetReport) -> str:
        feature_dicts: List[dict] = []
        for name, fs in report.feature_summaries.items():
            d = asdict(fs)
            d["name"] = name
            feature_dicts.append(d)

        if not feature_dicts:
            return "<p><em>No feature summaries available.</em></p>"

        df = pd.DataFrame(feature_dicts)

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

        if self.max_features is not None and self.max_features < len(df):
            df = df.head(self.max_features)

        # Nice formatting for floats
        float_cols = [
            c
            for c in ["missing_ratio", "mean", "std", "min", "max", "skewness", "kurtosis"]
            if c in df.columns
        ]
        df[float_cols] = df[float_cols].round(3)

        html_table = df.to_html(
            index=False,
            border=0,
            classes=["table", "table-striped", "table-sm"],
            escape=False,
        )
        return f"<div class='feature-table'>{html_table}</div>"

    def _build_correlation_section(self, report: DatasetReport) -> str:
        corr = report.correlation_matrix
        if corr is None or corr.empty:
            return "<p><em>No correlations available.</em></p>"

        max_dim = 25
        if corr.shape[0] > max_dim:
            corr = corr.iloc[:max_dim, :max_dim]

        corr = corr.round(3)
        html_table = corr.to_html(
            border=0,
            classes=["table", "table-sm", "corr-table"],
            escape=False,
        )

        caption = (
            f"<p class='caption'>Shown below is a truncated correlation matrix "
            f"for numeric features (up to {max_dim}&times;{max_dim}).</p>"
        )
        return f"<div class='correlation-section'>{caption}{html_table}</div>"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _escape(text: str) -> str:
        """Simple HTML escaping."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    @staticmethod
    def _default_styles() -> str:
        """Return a small CSS block for basic styling."""
        return """
<style>
body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 2rem;
  line-height: 1.5;
}
h1, h2 {
  color: #222;
}
code {
  font-family: "Fira Code", Menlo, Consolas, monospace;
  background-color: #f5f5f5;
  padding: 0.1rem 0.25rem;
  border-radius: 3px;
}
.table {
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 1.5rem;
}
.table th,
.table td {
  border: 1px solid #ddd;
  padding: 0.35rem 0.5rem;
  font-size: 0.9rem;
}
.table-striped tbody tr:nth-child(odd) {
  background-color: #fafafa;
}
.caption {
  font-size: 0.9rem;
  color: #555;
}
.overview ul {
  list-style: disc;
}
</style>
        """.strip()


def dataset_report_to_html(
    report: DatasetReport,
    *,
    include_feature_table: bool = True,
    max_features: Optional[int] = None,
    include_correlation: bool = True,
) -> str:
    """Convenience wrapper to build an HTML report.

    Parameters
    ----------
    report:
        Dataset-level report produced by the EDA utilities.
    include_feature_table:
        Whether to include the feature summary table.
    max_features:
        Maximum number of features to display in the feature table.
    include_correlation:
        Whether to include the correlation matrix (if present).

    Returns
    -------
    str
        A complete HTML document as a single string.
    """
    builder = HTMLReportBuilder(
        include_feature_table=include_feature_table,
        max_features=max_features,
        include_correlation=include_correlation,
    )
    return builder.build(report)
