"""Roxy report package.

This package contains utilities to turn structured report objects
(e.g. :class:`~roxy.core.report.DatasetReport`) into human-readable
representations such as Markdown and HTML.

Typical usage
-------------

>>> from roxy.eda.summary import build_report
>>> from roxy.report import dataset_report_to_markdown
>>>
>>> dataset_report = build_report(
...     X,
...     y,
...     dataset_name="My dataset",
...     task_type="classification",
... )
>>> md = dataset_report_to_markdown(dataset_report)
>>> print(md)

The same :class:`DatasetReport` instance can be rendered as HTML:

>>> from roxy.report import dataset_report_to_html
>>> html = dataset_report_to_html(dataset_report)
>>> Path("report.html").write_text(html)
"""

from __future__ import annotations

from .markdown import MarkdownReportBuilder, dataset_report_to_markdown
from .html import HTMLReportBuilder, dataset_report_to_html

__all__ = [
    "MarkdownReportBuilder",
    "dataset_report_to_markdown",
    "HTMLReportBuilder",
    "dataset_report_to_html",
]
