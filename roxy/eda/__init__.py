"""Roxy EDA package.

This subpackage provides utilities for exploratory data analysis:

- :mod:`roxy.eda.basic`:
    General-purpose EDA engine (:class:`BasicEDA`) and report container
    (:class:`BasicEDAReport`).

- :mod:`roxy.eda.summary`:
    High-level dataset summary via :func:`build_report`, returning a
    :class:`~roxy.core.report.DatasetReport`.

- :mod:`roxy.eda.target_relations`:
    Feature–target association tests for classification tasks
    (ANOVA / Kruskal–Wallis + post hoc), via
    :class:`FeatureTargetAssociation` and
    :func:`compute_feature_target_associations`.

- :mod:`roxy.eda.missing`:
    Missingness diagnostics (global and stratified by label).
"""

from .basic import BasicEDA, BasicEDAReport
from .summary import build_report
from .target_relations import (
    FeatureTargetAssociation,
    compute_feature_target_associations,
)
from .missing import (
    missingness_summary,
    missingness_by_label,
)

from roxy.core.exceptions import EDAError

__all__ = [
    "BasicEDA",
    "BasicEDAReport",
    "build_report",
    "FeatureTargetAssociation",
    "compute_feature_target_associations",
    "missingness_summary",
    "missingness_by_label",
    "EDAError",
]
