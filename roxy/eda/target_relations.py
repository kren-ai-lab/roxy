from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None  # type: ignore[assignment]


@dataclass
class FeatureTargetAssociation:
    """Result of a univariate feature–target association test.

    This structure summarises how a single numeric feature relates to a
    categorical target (e.g. class labels) through an omnibus test
    (ANOVA or Kruskal–Wallis) and optional post hoc pairwise tests.

    Attributes
    ----------
    feature_name :
        Name of the feature.
    target_name :
        Name of the target variable.
    test_name :
        Name of the omnibus test (e.g. ``"anova"`` or ``"kruskal"``).
    statistic :
        Test statistic from the omnibus test.
    p_value :
        P-value from the omnibus test (uncorrected).
    n_groups :
        Number of groups (unique target levels) included in the analysis.
    group_sizes :
        Mapping from target level to group size.
    effect_size :
        Optional effect-size estimate (e.g. eta-squared for ANOVA or
        epsilon-squared for Kruskal–Wallis).
    posthoc_pvalues :
        Optional DataFrame with adjusted p-values for all pairwise
        comparisons between groups. Indices and columns are group labels,
        and the matrix is symmetric with NaNs on the diagonal.
    notes :
        Optional list of textual notes (e.g. method choices, warnings).
    """

    feature_name: str
    target_name: str
    test_name: str
    statistic: float
    p_value: float
    n_groups: int
    group_sizes: Dict[Any, int]
    effect_size: Optional[float] = None
    posthoc_pvalues: Optional[pd.DataFrame] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a JSON-serialisable dictionary."""
        return {
            "feature_name": self.feature_name,
            "target_name": self.target_name,
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_groups": self.n_groups,
            "group_sizes": self.group_sizes,
            "effect_size": self.effect_size,
            "posthoc_pvalues": self.posthoc_pvalues.to_dict()
            if self.posthoc_pvalues is not None
            else None,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bh_correction(p_values: List[float]) -> List[float]:
    """Benjamini–Hochberg correction for multiple testing.

    Parameters
    ----------
    p_values :
        List of unadjusted p-values.

    Returns
    -------
    list of float
        List of adjusted p-values in the same order as the input.
    """
    m = len(p_values)
    if m == 0:
        return []

    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]

    adjusted = np.empty_like(ranked)
    cummin = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        cummin = min(cummin, val)
        adjusted[i] = cummin

    # Map back to original order
    adjusted_full = np.empty_like(adjusted)
    adjusted_full[order] = adjusted
    return adjusted_full.tolist()


def _group_numeric_by_target(
    feature: pd.Series,
    y: pd.Series,
    min_group_size: int = 2,
) -> Tuple[Dict[Any, np.ndarray], Dict[Any, int]]:
    """Group numeric feature values by target label.

    Returns only groups with at least ``min_group_size`` non-NaN
    observations.
    """
    df = pd.DataFrame({"feature": feature, "target": y})
    df = df.dropna(subset=["feature", "target"])

    groups: Dict[Any, np.ndarray] = {}
    sizes: Dict[Any, int] = {}
    for level, sub in df.groupby("target"):
        vals = sub["feature"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) >= min_group_size:
            groups[level] = vals
            sizes[level] = len(vals)

    return groups, sizes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_numeric_feature_across_groups(
    feature: pd.Series,
    y: pd.Series,
    *,
    feature_name: Optional[str] = None,
    target_name: str = "target",
    omnibus: str = "kruskal",
    min_group_size: int = 2,
    compute_posthoc: bool = True,
) -> Optional[FeatureTargetAssociation]:
    """Compare distributions of a numeric feature across target groups.

    This function performs an omnibus test (ANOVA or Kruskal–Wallis)
    to evaluate whether the mean/median of a numeric feature differs
    across groups defined by the categorical target. Optionally, it
    runs post hoc pairwise tests with Benjamini–Hochberg correction.

    Parameters
    ----------
    feature :
        Numeric feature values.
    y :
        Target labels (categorical).
    feature_name :
        Optional name of the feature for reporting. If ``None``, will use
        ``feature.name`` or a generic placeholder.
    target_name :
        Name of the target variable.
    omnibus :
        Omnibus test to use. One of ``"anova"`` or ``"kruskal"``.
    min_group_size :
        Minimum number of observations per group required to include the
        group in the analysis.
    compute_posthoc :
        If ``True`` and at least three groups are available, perform
        pairwise post hoc tests between all groups with BH correction.

    Returns
    -------
    FeatureTargetAssociation or None
        The result of the association test, or ``None`` if fewer than two
        valid groups are available or if ``scipy`` is not installed.
    """
    if stats is None:
        raise ImportError(
            "scipy is required for statistical tests in `target_relations`. "
            "Please install scipy to use this functionality."
        )

    if feature_name is None:
        feature_name = feature.name or "<unnamed_feature>"

    groups, sizes = _group_numeric_by_target(feature, y, min_group_size=min_group_size)
    if len(groups) < 2:
        # Not enough groups for a meaningful comparison
        return None

    group_labels = list(groups.keys())
    group_arrays = [groups[g] for g in group_labels]
    notes: List[str] = []

    omnibus = omnibus.lower()

    # ------------------------------------------------------------------
    # Omnibus test with robust error handling
    # ------------------------------------------------------------------
    try:
        if omnibus == "anova":
            stat, p = stats.f_oneway(*group_arrays)
            test_name = "anova"
            # Effect size: eta-squared
            all_vals = np.concatenate(group_arrays)
            grand_mean = float(all_vals.mean())
            ss_between = sum(len(g) * (float(g.mean()) - grand_mean) ** 2 for g in group_arrays)
            ss_within = sum(((g - float(g.mean())) ** 2).sum() for g in group_arrays)
            ss_total = ss_between + ss_within
            effect_size = ss_between / ss_total if ss_total > 0 else np.nan
            notes.append("Effect size is eta-squared for one-way ANOVA.")
        elif omnibus == "kruskal":
            stat, p = stats.kruskal(*group_arrays)
            test_name = "kruskal"
            # Effect size: epsilon-squared approximation
            n_total = sum(len(g) for g in group_arrays)
            k = len(group_arrays)
            if n_total > k:
                effect_size = (stat - (k - 1)) / (n_total - k)
            else:
                effect_size = np.nan
            notes.append("Effect size is epsilon-squared for Kruskal–Wallis.")
        else:
            raise ValueError("`omnibus` must be 'anova' or 'kruskal'.")
    except ValueError as e:
        # Typical case: all numbers identical in Kruskal, or other
        # pathologies in the input data. We return a valid association
        # object with NaN statistics and a note instead of raising.
        notes.append(f"Omnibus test failed: {e}")
        return FeatureTargetAssociation(
            feature_name=feature_name,
            target_name=target_name,
            test_name=omnibus,
            statistic=np.nan,
            p_value=np.nan,
            n_groups=len(groups),
            group_sizes=sizes,
            effect_size=np.nan,
            posthoc_pvalues=None,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Post hoc pairwise tests (only if omnibus succeeded)
    # ------------------------------------------------------------------
    posthoc_df: Optional[pd.DataFrame] = None
    if compute_posthoc and len(groups) >= 3:
        pair_labels: List[Tuple[Any, Any]] = []
        unadj_pvals: List[float] = []

        for i, gi in enumerate(group_labels):
            for j in range(i + 1, len(group_labels)):
                gj = group_labels[j]
                g1 = groups[gi]
                g2 = groups[gj]
                if omnibus == "anova":
                    # Welch's t-test (does not assume equal variances)
                    _, p_pair = stats.ttest_ind(
                        g1, g2, equal_var=False, nan_policy="omit"
                    )
                else:
                    # Non-parametric: Mann–Whitney U test
                    _, p_pair = stats.mannwhitneyu(
                        g1, g2, alternative="two-sided"
                    )
                pair_labels.append((gi, gj))
                unadj_pvals.append(p_pair)

        adj_pvals = _bh_correction(unadj_pvals)
        posthoc_df = pd.DataFrame(
            np.nan,
            index=group_labels,
            columns=group_labels,
            dtype=float,
        )

        for (gi, gj), p_adj in zip(pair_labels, adj_pvals):
            posthoc_df.loc[gi, gj] = p_adj
            posthoc_df.loc[gj, gi] = p_adj

        notes.append(
            "Post hoc pairwise tests with Benjamini–Hochberg correction "
            f"({omnibus}-consistent tests)."
        )

    return FeatureTargetAssociation(
        feature_name=feature_name,
        target_name=target_name,
        test_name=test_name,
        statistic=float(stat),
        p_value=float(p),
        n_groups=len(groups),
        group_sizes=sizes,
        effect_size=float(effect_size) if effect_size is not None else None,
        posthoc_pvalues=posthoc_df,
        notes=notes,
    )


def compute_feature_target_associations(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task_type: str = "classification",
    numeric_only: bool = True,
    omnibus: str = "kruskal",
    min_group_size: int = 2,
    compute_posthoc: bool = True,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Compute univariate feature–target association tests for all features.

    This function is primarily intended for classification tasks, where
    the target is categorical and features are numeric.

    Parameters
    ----------
    X :
        Feature matrix.
    y :
        Target labels. Should be categorical for classification tasks.
    task_type :
        Task type. Currently only ``"classification"`` triggers tests;
        other values return an empty or ``None`` mapping.
    numeric_only :
        If ``True``, only numeric columns in ``X`` are tested.
    omnibus :
        Omnibus test to use (``"anova"`` or ``"kruskal"``).
    min_group_size :
        Minimum number of observations per group required.
    compute_posthoc :
        Whether to compute post hoc pairwise tests.

    Returns
    -------
    dict
        Mapping from feature name to a dictionary representation of
        :class:`FeatureTargetAssociation`, or ``None`` if the test could
        not be performed (e.g. insufficient groups).
    """
    associations: Dict[str, Optional[Dict[str, Any]]] = {}

    if task_type != "classification":
        # For now we only support classification-style targets.
        for col in X.columns:
            associations[col] = None
        return associations

    if numeric_only:
        cols = X.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = list(X.columns)

    for col in X.columns:
        if col not in cols:
            associations[col] = None
            continue

        assoc = compare_numeric_feature_across_groups(
            feature=X[col],
            y=y,
            feature_name=col,
            target_name=getattr(y, "name", "target") or "target",
            omnibus=omnibus,
            min_group_size=min_group_size,
            compute_posthoc=compute_posthoc,
        )
        associations[col] = assoc.to_dict() if assoc is not None else None

    return associations
