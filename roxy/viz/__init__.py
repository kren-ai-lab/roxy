"""Lightweight static visualization helpers for descriptor inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .plots import (
        plot_correlation_heatmap,
        plot_embedding,
        plot_feature_boxplot,
        plot_feature_distribution,
        plot_scatter_features,
    )


def __getattr__(name: str) -> Any:
    """Resolve plotting helpers lazily."""
    if name in {
        "plot_feature_distribution",
        "plot_feature_boxplot",
        "plot_scatter_features",
        "plot_embedding",
        "plot_correlation_heatmap",
    }:
        from .plots import (
            plot_correlation_heatmap,
            plot_embedding,
            plot_feature_boxplot,
            plot_feature_distribution,
            plot_scatter_features,
        )

        return {
            "plot_feature_distribution": plot_feature_distribution,
            "plot_feature_boxplot": plot_feature_boxplot,
            "plot_scatter_features": plot_scatter_features,
            "plot_embedding": plot_embedding,
            "plot_correlation_heatmap": plot_correlation_heatmap,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "plot_feature_distribution",
    "plot_feature_boxplot",
    "plot_scatter_features",
    "plot_embedding",
    "plot_correlation_heatmap",
]
