"""Plot helpers for the new ConSim score tables.

Adapted from :mod:`utils.old_plot` but without the implicit prompt-type rename:
callers pass an explicit ``baseline_for`` mapping from each explanation prompt
type to its baseline prompt type (e.g. ``{"C1": "B1", "C2": "B2", "AC1":
"AB1", "AC2": "AB2", ...}``), so no string mangling happens inside the plot.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_accuracies_violins(
    accuracies: pd.Series,
    baselines: pd.Series,
    baseline_for: Mapping[str, str],
    *,
    compared_index: str = "method",
    method_order: Optional[list[str]] = None,
    split_index: Optional[str] = None,
    split_order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_dir: Optional[str] = None,
    file_name: Optional[str] = None,
) -> plt.Axes:
    """Violin plot of accuracies by ``prompt_type`` and ``compared_index``.

    Parameters
    ----------
    accuracies, baselines
        Series of scores indexed by a MultiIndex that must contain at least
        ``prompt_type`` and ``compared_index`` (and ``split_index`` when set).
        ``accuracies`` holds the explanation prompt types (e.g. ``C*`` /
        ``AC*``); ``baselines`` holds the baseline prompt types
        (e.g. ``B*`` / ``AB*``).
    baseline_for
        Mapping from each explanation prompt_type to its baseline prompt_type.
        Example: ``{"C1": "B1", "C2": "B2", "C3": "B2", "AC1": "AB1",
        "AC2": "AB2", "AC3": "AB2"}``. Prompt types not in this dict are
        plotted without a NoExplanation comparator.
    compared_index
        Index level whose values become side-by-side violins inside each
        prompt-type group (default ``"method"``).
    method_order
        Optional explicit ordering of ``compared_index`` values. If ``None``,
        sorted unique values are used.
    split_index
        Optional index level used to split every method slot into adjacent
        sub-violins. Typical use: ``split_index="specification"`` to draw
        ``old_consim`` and ``new_consim`` side by side for each method so they
        alternate across the figure.
    split_order
        Optional explicit ordering of ``split_index`` values (e.g.
        ``["old_consim", "new_consim"]``). If ``None``, sorted unique values
        are used.
    title
        Figure / axes title.
    ax
        Existing axes to draw on. If ``None`` a fresh figure is created.
    save_dir, file_name
        If both provided, save the figure as ``{save_dir}/{file_name}``.

    Returns
    -------
    The axes the violins were drawn on.
    """
    # Prompt types: keep explanation order from the data, longest names last to
    # match the old plot's "2-char then 4-char" grouping (e.g. C1, C2, C3 then
    # AC1, AC2, AC3).
    explanation_prompt_types = list(
        accuracies.index.get_level_values("prompt_type").unique()
    )
    explanation_prompt_types = (
        [pt for pt in explanation_prompt_types if len(pt) == 2]
        + [pt for pt in explanation_prompt_types if len(pt) != 2]
    )
    if not explanation_prompt_types:
        raise ValueError("No prompt_types found in `accuracies`.")

    # Methods order.
    methods = (
        list(method_order)
        if method_order is not None
        else sorted(accuracies.index.get_level_values(compared_index).unique())
    )
    if not methods:
        raise ValueError(f"No values found at index level {compared_index!r}.")

    # Split values (e.g. specifications). When None we use a single "All" value
    # so the rest of the function can treat split-mode and no-split-mode the
    # same way.
    if split_index is not None:
        if split_order is not None:
            split_values = list(split_order)
        else:
            split_values = sorted(
                set(accuracies.index.get_level_values(split_index).unique())
                | set(baselines.index.get_level_values(split_index).unique())
            )
        if not split_values:
            raise ValueError(f"No values found at index level {split_index!r}.")
    else:
        split_values = [None]  # sentinel: no split

    n_splits = len(split_values)

    def _values_for(series: pd.Series, level: str, key: str) -> np.ndarray:
        try:
            return series.xs(key, level=level).to_numpy()
        except KeyError:
            return np.array([])

    # Collect per-(prompt_type, method, split) distributions and the matching
    # NoExplanation distribution for each (prompt_type, split).
    # distributions[prompt_type][method][split_value] = np.ndarray
    distributions: dict[str, dict[str, dict[Optional[str], np.ndarray]]] = {}
    baseline_means: dict[str, dict[Optional[str], float]] = {}
    for prompt_type in explanation_prompt_types:
        pt_acc = accuracies.xs(prompt_type, level="prompt_type").dropna()
        if pt_acc.empty:
            continue

        per_method: dict[str, dict[Optional[str], np.ndarray]] = {}
        for method in methods:
            try:
                method_acc = pt_acc.xs(method, level=compared_index)
            except KeyError:
                method_acc = pt_acc.iloc[0:0]  # empty Series with same index type

            per_split: dict[Optional[str], np.ndarray] = {}
            if split_index is None:
                per_split[None] = method_acc.to_numpy()
            else:
                for sv in split_values:
                    per_split[sv] = _values_for(method_acc, split_index, sv)
            per_method[method] = per_split

        baseline_pt = baseline_for.get(prompt_type)
        if baseline_pt is not None:
            try:
                bl_acc = baselines.xs(baseline_pt, level="prompt_type").dropna()
            except KeyError:
                bl_acc = baselines.iloc[0:0]

            per_split_bl: dict[Optional[str], np.ndarray] = {}
            if split_index is None:
                per_split_bl[None] = bl_acc.to_numpy()
            else:
                for sv in split_values:
                    per_split_bl[sv] = _values_for(bl_acc, split_index, sv)
            per_method["NoExplanation"] = per_split_bl

            # Per-split mean for the horizontal baseline lines.
            means: dict[Optional[str], float] = {}
            for sv, arr in per_split_bl.items():
                if arr.size:
                    means[sv] = float(np.mean(arr))
            if means:
                baseline_means[prompt_type] = means

        distributions[prompt_type] = per_method

    if not distributions:
        raise ValueError("No data to plot after filtering by prompt_type.")

    # Keep only prompt_types that ended up with data.
    plotted_prompt_types = [pt for pt in explanation_prompt_types if pt in distributions]

    # Violin slots: methods (in order) + NoExplanation last (if any group has one).
    has_baseline = any("NoExplanation" in d for d in distributions.values())
    slot_keys = list(methods) + (["NoExplanation"] if has_baseline else [])
    num_slots = len(slot_keys)
    num_groups = len(plotted_prompt_types)

    # Geometry: copied from old_plot for visual continuity. When splitting, the
    # method slot is divided into `n_splits` adjacent sub-violins.
    slot_width = 0.12
    sub_width = slot_width / n_splits
    bars_index = np.arange(num_groups)

    # Colors per method.
    tab10 = mpl.colormaps["tab10"].colors
    colors = {key: tab10[i % len(tab10)] for i, key in enumerate(methods)}
    colors["NoExplanation"] = "black"

    # Per-split visual style. With 2 splits we want clear alternation: the
    # first split is drawn with a lighter fill + hatching, the second with a
    # solid darker fill. With other split counts we vary alpha smoothly.
    if n_splits == 1:
        split_styles = {None: {"alpha": 0.75, "hatch": None}}
    elif n_splits == 2:
        split_styles = {
            split_values[0]: {"alpha": 0.45, "hatch": "//"},
            split_values[1]: {"alpha": 0.95, "hatch": None},
        }
    else:
        # Evenly spaced alphas in [0.4, 0.95].
        alphas = np.linspace(0.4, 0.95, n_splits)
        split_styles = {sv: {"alpha": float(a), "hatch": None} for sv, a in zip(split_values, alphas)}

    # Figure.
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(18, 2.2 * num_groups * (1 + 0.15 * num_slots * n_splits)), 6))
    else:
        fig = ax.figure

    method_legend_handles: list[patches.Patch] = []
    method_legend_labels: list[str] = []
    seen_legend_keys: set[str] = set()

    for slot_idx, slot_key in enumerate(slot_keys):
        slot_center_offset = slot_idx * slot_width

        for split_idx, split_value in enumerate(split_values):
            # Sub-slot positions: offset within the slot so the n_splits
            # sub-violins are adjacent and centered around the slot center.
            sub_offset = (split_idx - (n_splits - 1) / 2) * sub_width

            values_per_group = [
                distributions[pt].get(slot_key, {}).get(split_value, np.array([]))
                for pt in plotted_prompt_types
            ]
            positions = bars_index + slot_center_offset + sub_offset

            style = split_styles[split_value]

            first_body = None
            for pos, values in zip(positions, values_per_group):
                if values.size == 0:
                    continue
                parts = ax.violinplot(
                    [values],
                    positions=[pos],
                    widths=sub_width * 0.95,
                    showmeans=True,
                    showmedians=False,
                )
                for body in parts["bodies"]:
                    body.set_facecolor(colors[slot_key])
                    body.set_edgecolor("black")
                    body.set_alpha(style["alpha"])
                    if style["hatch"] is not None:
                        body.set_hatch(style["hatch"])
                edge = "black" if slot_key == "NoExplanation" else colors[slot_key]
                for line_key in ("cmeans", "cmaxes", "cmins", "cbars"):
                    if line_key in parts:
                        parts[line_key].set_edgecolor(edge)
                if first_body is None:
                    first_body = parts["bodies"][0]

            # Method legend: one entry per method (use the darker split style so
            # the color is recognisable).
            if first_body is not None and slot_key not in seen_legend_keys:
                # Use the most-opaque style for the legend swatch.
                legend_style = max(split_styles.values(), key=lambda s: s["alpha"])
                method_legend_handles.append(
                    patches.Patch(
                        facecolor=colors[slot_key],
                        edgecolor="black",
                        alpha=legend_style["alpha"],
                    )
                )
                method_legend_labels.append(slot_key)
                seen_legend_keys.add(slot_key)

    # Horizontal NoExplanation baseline line(s) per prompt-type group. With
    # split_index, draw one dashed line per split value so the comparison is
    # visible at a glance.
    for group_idx, prompt_type in enumerate(plotted_prompt_types):
        if prompt_type not in baseline_means:
            continue
        means = baseline_means[prompt_type]
        xmin = bars_index[group_idx] - slot_width / 2
        xmax = bars_index[group_idx] + slot_width * (num_slots - 1) + slot_width / 2
        for split_value, baseline in means.items():
            style = split_styles[split_value]
            linestyle = "--" if style.get("hatch") else "-"
            ax.plot(
                [xmin, xmax],
                [baseline, baseline],
                color="black",
                linewidth=1,
                linestyle=linestyle,
                alpha=max(0.5, style["alpha"]),
            )

    # Axes cosmetics.
    ax.set_xlabel("Prompt types")
    ax.set_ylabel("Simulatability score")
    ax.set_xticks(bars_index + slot_width * (num_slots - 1) / 2)
    ax.set_xticklabels(plotted_prompt_types)
    ax.set_ylim(-0.05, 1.05)
    if title is not None:
        ax.set_title(title)

    # Two legends: methods (colors) and splits (alpha/hatch).
    method_legend = ax.legend(
        method_legend_handles,
        method_legend_labels,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        title=compared_index,
    )
    if split_index is not None:
        split_handles = [
            patches.Patch(
                facecolor="grey",
                edgecolor="black",
                alpha=split_styles[sv]["alpha"],
                hatch=split_styles[sv]["hatch"],
            )
            for sv in split_values
        ]
        ax.add_artist(method_legend)
        ax.legend(
            split_handles,
            list(split_values),
            bbox_to_anchor=(1.01, 0.0),
            loc="lower left",
            title=split_index,
        )

    fig.tight_layout()

    if save_dir is not None and file_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, file_name))

    return ax


# Default baseline mapping for the current prompt-type naming convention.
# C1 / AC1 are paired with B1 / AB1; C2, C3, AC2, AC3 with B2 / AB2.
DEFAULT_BASELINE_FOR: dict[str, str] = {
    "C1": "B1",
    "C2": "B2",
    "C3": "B2",
    "AC1": "AB1",
    "AC2": "AB2",
    "AC3": "AB2",
}
