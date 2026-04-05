"""Lightweight static visualization helpers for descriptor inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_PLOT_EXPORTS = {
    "plot_feature_distribution",
    "plot_feature_boxplot",
    "plot_scatter_features",
    "plot_embedding",
    "plot_correlation_heatmap",
}
_OPTIONAL_PLOTTING_DEPENDENCIES = {
    "numpy",
    "pandas",
    "matplotlib",
}

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .plots import (
        plot_correlation_heatmap,
        plot_embedding,
        plot_feature_boxplot,
        plot_feature_distribution,
        plot_scatter_features,
    )


def _load_plot_export(name: str) -> Any:
    """Load a plotting helper from ``roxy.viz.plots``."""
    from . import plots

    return getattr(plots, name)


def _make_lazy_plot_proxy(name: str) -> Any:
    """Return a callable proxy that imports the plotting helper on demand."""

    def _lazy_plot(*args: Any, **kwargs: Any) -> Any:
        return _load_plot_export(name)(*args, **kwargs)

    _lazy_plot.__name__ = name
    _lazy_plot.__qualname__ = name
    _lazy_plot.__module__ = __name__
    _lazy_plot.__doc__ = f"Lazy proxy for ``roxy.viz.plots.{name}``."
    return _lazy_plot


def __getattr__(name: str) -> Any:
    """Resolve plotting helpers lazily."""
    if name in _PLOT_EXPORTS:
        try:
            value = _load_plot_export(name)
        except ModuleNotFoundError as exc:
            if exc.name not in _OPTIONAL_PLOTTING_DEPENDENCIES:
                raise
            value = _make_lazy_plot_proxy(name)

        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "plot_feature_distribution",
    "plot_feature_boxplot",
    "plot_scatter_features",
    "plot_embedding",
    "plot_correlation_heatmap",
]
