"""Roxy

A lightweight feature engineering and exploratory data analysis (EDA)
library for protein, structure and small-molecule datasets.

Roxy sits between data curation (PRISM) and representation learning
(Sylphy). It focuses on:

- computing classical descriptors for sequences, structures and molecules,
- organising feature blocks through `RoxyDataset`,
- running EDA and feature–target association analyses,
- projecting features into low-dimensional spaces,
- visualising and reporting dataset characteristics.

The symbols re-exported here form the *high-level* public API.
"""

from __future__ import annotations

from .core.dataset import RoxyDataset
from .descriptors import (
    GlobalSequenceDescriptors,
    BasicStructureDescriptors,
    BasicMoleculeDescriptors,
    ProteinDescriptorError
)

from .eda.summary import build_report
from .projection import project
from .viz import (
    plot_feature_distribution,
    plot_embedding,
)
from .report import (
    dataset_report_to_markdown,
    dataset_report_to_html,
)

from .helpers import RoxyHelpers

__all__ = [
    # Core container
    "RoxyDataset",
    # Descriptor engines
    "GlobalSequenceDescriptors",
    "BasicStructureDescriptors",
    "BasicMoleculeDescriptors",
    "ProteinDescriptorError",
    # EDA / summary
    "build_report",
    # Projection
    "project",
    # Visualisation (high-level)
    "plot_feature_distribution",
    "plot_embedding",
    # Reporting
    "dataset_report_to_markdown",
    "dataset_report_to_html",
    # Helpers
    "RoxyHelpers",
]
