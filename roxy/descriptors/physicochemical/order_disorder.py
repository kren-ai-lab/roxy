"""Order/disorder propensity descriptors."""

from __future__ import annotations

import math

import numpy as np

from roxy.core.constants import AA_GROUPS
from roxy.descriptors._utils import (
    fraction_above_threshold,
    fraction_from_group,
    longest_run,
    terminal_segment,
    transition_fraction,
    windows,
)
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_DISORDER_PROMOTING = AA_GROUPS["disorder_promoting"]
_ORDER_PROMOTING = AA_GROUPS["order_promoting"]
_FLEXIBILITY_RELATED = frozenset("GSP")
_RIGIDITY_RELATED = frozenset("ILVFYW")
_PRO_GLY_RICH = frozenset("PG")
_AROMATIC_ALIPHATIC_ORDER = frozenset("FWYILV")
_POLAR_DISORDER_SUPPORT = frozenset("STNQDEKR")

_PATCH_THRESHOLD = 0.5


@register("order_disorder", family="physicochemical")
class OrderDisorderDescriptor(BaseDescriptor):
    r"""Order/disorder propensity features.

    Residue groups follow the disorder/order classification from
    ``AA_GROUPS`` (disorder-promoting: S, A, P, R, E, K, G;
    order-promoting: C, W, Y, F, I, L, V).

    * **Balance**: :math:`f_{\text{disorder}} - f_{\text{order}}`
    * **Transition fraction**: fraction of adjacent pairs switching
      between order and disorder groups.
    * **Patch fraction**: fraction of windows with group fraction
      above threshold (0.5).

    Args:
        window_sizes: Window sizes for local patch profiles.
        terminal_window: Residue count for N/C-terminal segments.

    Returns:
        ``compute()`` returns a DataFrame with columns prefixed ``order_disorder_``.
        ``length``, ``valid_residue_count``,
        ``disorder/order_fraction``,
        ``disorder_order/order_disorder_balance/ratio``,
        ``flexibility/rigidity/pro_gly/
        aromatic_aliphatic_order/
        polar_disorder_support_fraction``,
        ``longest_disorder/order_run``,
        ``order_disorder_transition_fraction``,
        ``nterm/cterm_disorder/order_fraction``,
        ``terminal_disorder/order_asymmetry``,
        per-window: ``w{N}_disorder/order_patch_fraction``,
        ``w{N}_disorder/order_mean``.

    """

    def __init__(
        self,
        *,
        window_sizes: tuple[int, ...] = (5, 7),
        terminal_window: int = 10,
    ) -> None:
        """Initialize OrderDisorderDescriptor."""
        self.window_sizes = tuple(window_sizes)
        self.terminal_window = terminal_window

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for key in (
            "disorder_fraction",
            "order_fraction",
            "disorder_order_balance",
            "order_disorder_balance",
            "disorder_order_ratio",
            "order_disorder_ratio",
            "flexibility_fraction",
            "rigidity_fraction",
            "pro_gly_fraction",
            "aromatic_aliphatic_order_fraction",
            "polar_disorder_support_fraction",
            "longest_disorder_run",
            "longest_order_run",
            "order_disorder_transition_fraction",
            "nterm_disorder_fraction",
            "cterm_disorder_fraction",
            "nterm_order_fraction",
            "cterm_order_fraction",
            "terminal_disorder_asymmetry",
            "terminal_order_asymmetry",
        ):
            feats[key] = _NAN
        for ws in self.window_sizes:
            feats[f"w{ws}_disorder_patch_fraction"] = _NAN
            feats[f"w{ws}_order_patch_fraction"] = _NAN
            feats[f"w{ws}_disorder_mean"] = _NAN
            feats[f"w{ws}_order_mean"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute order/disorder features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        dis_frac = fraction_from_group(seq, _DISORDER_PROMOTING)
        ord_frac = fraction_from_group(seq, _ORDER_PROMOTING)

        feats["disorder_fraction"] = dis_frac
        feats["order_fraction"] = ord_frac
        feats["disorder_order_balance"] = dis_frac - ord_frac
        feats["order_disorder_balance"] = ord_frac - dis_frac
        feats["disorder_order_ratio"] = dis_frac / ord_frac if ord_frac else _NAN
        feats["order_disorder_ratio"] = ord_frac / dis_frac if dis_frac else _NAN

        feats["flexibility_fraction"] = fraction_from_group(seq, _FLEXIBILITY_RELATED)
        feats["rigidity_fraction"] = fraction_from_group(seq, _RIGIDITY_RELATED)
        feats["pro_gly_fraction"] = fraction_from_group(seq, _PRO_GLY_RICH)
        feats["aromatic_aliphatic_order_fraction"] = fraction_from_group(seq, _AROMATIC_ALIPHATIC_ORDER)
        feats["polar_disorder_support_fraction"] = fraction_from_group(seq, _POLAR_DISORDER_SUPPORT)

        feats["longest_disorder_run"] = float(longest_run(seq, _DISORDER_PROMOTING))
        feats["longest_order_run"] = float(longest_run(seq, _ORDER_PROMOTING))
        feats["order_disorder_transition_fraction"] = transition_fraction(
            seq, _ORDER_PROMOTING, _DISORDER_PROMOTING
        )

        nterm = terminal_segment(seq, "N", self.terminal_window)
        cterm = terminal_segment(seq, "C", self.terminal_window)
        nterm_dis = fraction_from_group(nterm, _DISORDER_PROMOTING)
        cterm_dis = fraction_from_group(cterm, _DISORDER_PROMOTING)
        nterm_ord = fraction_from_group(nterm, _ORDER_PROMOTING)
        cterm_ord = fraction_from_group(cterm, _ORDER_PROMOTING)

        feats["nterm_disorder_fraction"] = nterm_dis
        feats["cterm_disorder_fraction"] = cterm_dis
        feats["nterm_order_fraction"] = nterm_ord
        feats["cterm_order_fraction"] = cterm_ord
        feats["terminal_disorder_asymmetry"] = nterm_dis - cterm_dis
        feats["terminal_order_asymmetry"] = nterm_ord - cterm_ord

        for ws in self.window_sizes:
            ws_list = windows(seq, ws)
            disorder_profile = [sum(aa in _DISORDER_PROMOTING for aa in w) / ws for w in ws_list]
            order_profile = [sum(aa in _ORDER_PROMOTING for aa in w) / ws for w in ws_list]

            feats[f"w{ws}_disorder_patch_fraction"] = fraction_above_threshold(
                disorder_profile, _PATCH_THRESHOLD
            )
            feats[f"w{ws}_order_patch_fraction"] = fraction_above_threshold(order_profile, _PATCH_THRESHOLD)
            feats[f"w{ws}_disorder_mean"] = float(np.mean(disorder_profile)) if disorder_profile else _NAN
            feats[f"w{ws}_order_mean"] = float(np.mean(order_profile)) if order_profile else _NAN

        return feats
