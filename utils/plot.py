"""Plot helpers for the new ConSim score tables.

Adapted from :mod:`utils.old_plot` but without the implicit prompt-type rename:
callers pass an explicit ``baseline_for`` mapping from each explanation prompt
type to its baseline prompt type (e.g. ``{"C1": "B1", "C2": "B2", "AC1":
"AB1", "AC2": "AB2", ...}``), so no string mangling happens inside the plot.
"""

from __future__ import annotations

import os
from itertools import combinations, product
from typing import Mapping, Optional, Sequence

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from utils.analysis import filter_keep


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
    random_chance: Optional[float] = None,
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

    # Exact old_plot-style visual for the common case used in the notebook:
    # one violin per method and prompt type, plus a black NoExplanation violin.
    # In particular, keep Matplotlib's default violin body styling for methods
    # (the old plot did not manually recolor those bodies), and only force
    # NoExplanation to black.
    if split_index is None:
        has_baseline = any("NoExplanation" in d for d in distributions.values())
        slot_keys = list(methods) + (["NoExplanation"] if has_baseline else [])
        num_slots = len(slot_keys)
        bar_width = 0.12
        # The old plot used unit-spaced prompt-type clusters. With the new
        # grids we can have more method slots, so make the group spacing just
        # wide enough to keep neighbouring prompt-type clusters from touching
        # while preserving the old within-cluster violin geometry.
        group_spacing = max(1.0, bar_width * (num_slots + 2))
        bars_index = np.arange(len(plotted_prompt_types)) * group_spacing

        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))
        else:
            fig = ax.figure

        handles, labels = [], []
        for i, slot_key in enumerate(slot_keys):
            values_per_group = [
                distributions[pt].get(slot_key, {}).get(None, np.array([]))
                for pt in plotted_prompt_types
            ]
            non_empty = [values.size > 0 for values in values_per_group]
            if not any(non_empty):
                continue

            # Matplotlib cannot draw empty distributions. Draw the available
            # prompt-type groups in one call so the visual matches old_plot as
            # closely as possible.
            values = [v for v, keep in zip(values_per_group, non_empty) if keep]
            positions = bars_index[non_empty] + i * bar_width
            violins = ax.violinplot(
                values,
                positions=positions,
                widths=bar_width,
                showmeans=True,
                showmedians=False,
            )
            if slot_key == "NoExplanation":
                for body in violins["bodies"]:
                    body.set_facecolor("black")
                    body.set_edgecolor("black")
                for line_key in ("cmeans", "cmaxes", "cmins", "cbars"):
                    if line_key in violins:
                        violins[line_key].set_edgecolor("black")

            handles.append(
                patches.Patch(color=violins["bodies"][0].get_facecolor().flatten())
            )
            labels.append(slot_key)

        for i, prompt_type in enumerate(plotted_prompt_types):
            if prompt_type not in baseline_means:
                continue
            baseline = baseline_means[prompt_type].get(None)
            if baseline is None:
                continue
            xmin = bars_index[i] - bar_width
            xmax = bars_index[i] + bar_width * num_slots
            ax.plot([xmin, xmax], [baseline, baseline], color="black", linewidth=1)

        if random_chance is not None:
            ax.axhline(random_chance, color="dimgray", linestyle=":", linewidth=1.2, label="Random choice")
            handles.append(Line2D([0], [0], color="dimgray", linestyle=":"))
            labels.append("Random choice")

        ax.set_xlabel("Prompt types")
        ax.set_ylabel("Simulatability score")
        ax.set_xticks(bars_index + bar_width * (num_slots - 1) / 2)
        ax.set_xticklabels(plotted_prompt_types)
        if title is not None:
            ax.set_title(title)
        ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc="upper left")
        fig.tight_layout()

        if save_dir is not None and file_name is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, file_name))

        return ax

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
        split_styles = {split_values[0]: {"alpha": 0.65, "hatch": None}}
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

    if random_chance is not None:
        ax.axhline(random_chance, color="dimgray", linestyle=":", linewidth=1.2, label="Random choice")
        method_legend_handles.append(Line2D([0], [0], color="dimgray", linestyle=":"))
        method_legend_labels.append("Random choice")

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

def plot_difference_bars(
    differences: pd.Series,
    *,
    ax: plt.Axes | None = None,
    comparison_col: str = "prompt_type",
    ylabel: str = "Score difference",
    title: str | None = None,
    positive_color: str = "#4c72b0",
    negative_color: str = "#c44e52",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot bucketed score differences as simple signed bar plots.

    ``differences`` is expected to already contain one row per displayed bar. This
    helper intentionally does not compute statistics: notebooks keep filtering
    and aggregation explicit, while this function centralizes the visual style.
    """
    means = differences.groupby(comparison_col).mean().sort_values(ascending=False)
    stds = differences.groupby(comparison_col).std().reindex(means.index)  # ensure stds aligns with means

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(means) + 3), 4))
    else:
        fig = ax.figure

    x = np.arange(len(means))
    colors = [positive_color if value >= 0 else negative_color for value in means]

    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    return fig, ax

def panel_difference_bars(
    differences: pd.Series,
    *,
    panel_col: str | None = None,
    comparison_col: str = "prompt_type",
    ylabel: str = "Score difference",
    title: str | None = None,
    positive_color: str = "#4c72b0",
    negative_color: str = "#c44e52",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot bucketed score differences as simple signed bar plots.

    ``differences`` is expected to already contain one row per displayed bar. This
    helper intentionally does not compute statistics: notebooks keep filtering
    and aggregation explicit, while this function centralizes the visual style.
    """
    if differences.empty:
        raise ValueError("Cannot plot an empty differences dataframe.")

    panels = differences.index.get_level_values(panel_col).unique()
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(max(7, 0.7 * differences.index.get_level_values(comparison_col).nunique() + 3), 4 * len(panels)),
        sharey=True,
        squeeze=False,
    )

    for ax, panel_value in zip(axes[:, 0], panels):
        plot_difference_bars(
            filter_keep(differences, keep={panel_col: panel_value}),
            ax=ax,
            comparison_col=comparison_col,
            ylabel=ylabel,
            title=f"{panel_col}={panel_value}",
            positive_color=positive_color,
            negative_color=negative_color
        )

    # if title is not None:
    #     fig.suptitle(title, fontsize=10)
    #     fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.tight_layout()

    return fig, axes


def plot_pairwise_comparison_matrices(
    scores: pd.Series,
    *,
    compared_index: str = "method",
    ttest_rel_alpha: float = 0.05,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Figure, pd.DataFrame, pd.DataFrame]:
    """Plot old-ConSim-style pairwise win-rate and difference matrices.

    ``scores`` must be a Series indexed by a MultiIndex containing
    ``compared_index``. Every other index level defines a paired experimental
    setting. The function intersects settings before each pairwise comparison,
    so methods are compared only on common cells.
    """
    if not isinstance(scores.index, pd.MultiIndex):
        raise TypeError("scores must use a MultiIndex.")
    if compared_index not in scores.index.names:
        raise ValueError(f"Missing index level {compared_index!r}.")

    # Average exact duplicate rows first. This handles rescoring duplicates and
    # repeated baseline rows without changing the experimental key names.
    scores = scores.groupby(level=scores.index.names).mean().dropna()
    contenders = list(scores.index.get_level_values(compared_index).unique())
    if len(contenders) < 2:
        raise ValueError("Need at least two contenders for a pairwise matrix.")

    n_contenders = len(contenders)
    percentage = pd.DataFrame(0.0, index=contenders, columns=contenders)
    diff_mean = pd.DataFrame(0.0, index=contenders, columns=contenders)
    diff_std = pd.DataFrame(0.0, index=contenders, columns=contenders)
    pvalues = pd.DataFrame(1.0, index=contenders, columns=contenders)

    for contender in contenders:
        percentage.loc[contender, contender] = 50

    for c1, c2 in combinations(contenders, 2):
        sc1 = scores.xs(c1, level=compared_index).dropna()
        sc2 = scores.xs(c2, level=compared_index).dropna()
        common = sc1.index.intersection(sc2.index)
        if len(common) == 0:
            continue

        sc1 = sc1.loc[common]
        sc2 = sc2.loc[common]
        wins = (sc1 > sc2).sum()
        draws = (sc1 == sc2).sum()
        defeats = (sc1 < sc2).sum()

        percentage.loc[c1, c2] = round((wins + 0.5 * draws) / len(common) * 100)
        percentage.loc[c2, c1] = round((defeats + 0.5 * draws) / len(common) * 100)
        diff = sc1 - sc2
        diff_mean.loc[c1, c2] = diff.mean()
        diff_mean.loc[c2, c1] = -diff.mean()
        diff_std.loc[c1, c2] = diff_std.loc[c2, c1] = diff.std()
        if len(common) > 1:
            pvalue = ttest_rel(sc1, sc2).pvalue
            pvalues.loc[c1, c2] = pvalues.loc[c2, c1] = pvalue

    ranking = n_contenders + 1 - (percentage >= 50).sum(axis="columns")
    order = ranking.sort_values().index
    percentage = percentage.reindex(index=order, columns=order)
    diff_mean = diff_mean.reindex(index=order, columns=order)
    diff_std = diff_std.reindex(index=order, columns=order)
    pvalues = pvalues.reindex(index=order, columns=order)

    percentage_with_rank = percentage.copy()
    percentage_with_rank["rank"] = ranking.reindex(order)

    fig_pct, ax_pct = plt.subplots(figsize=figsize)
    pct_values = percentage_with_rank.to_numpy(dtype=float)
    pct_image = ax_pct.imshow(pct_values, cmap="coolwarm", vmin=0, vmax=100)
    for row in range(pct_values.shape[0]):
        for col in range(pct_values.shape[1]):
            text = f"{pct_values[row, col]:.0f}"
            weight = "bold" if col == pct_values.shape[1] - 1 else "normal"
            ax_pct.text(col, row, text, ha="center", va="center", weight=weight)
    ax_pct.set_xticks(np.arange(percentage_with_rank.shape[1]))
    ax_pct.set_yticks(np.arange(percentage_with_rank.shape[0]))
    ax_pct.set_xticklabels(percentage_with_rank.columns, rotation=45, ha="right")
    ax_pct.set_yticklabels(percentage_with_rank.index, rotation=0)
    ax_pct.set_xlabel("Methods 2")
    ax_pct.set_ylabel("Methods 1")
    fig_pct.colorbar(pct_image, ax=ax_pct, label="Win rate of method 1 over method 2 (%)")
    fig_pct.tight_layout()

    fig_diff, ax_diff = plt.subplots(figsize=figsize)
    diff_values = diff_mean.to_numpy(dtype=float)
    max_abs_diff = max(float(np.nanmax(np.abs(diff_values))), 1e-9)
    diff_image = ax_diff.imshow(diff_values, cmap="coolwarm", vmin=-max_abs_diff, vmax=max_abs_diff)
    for i, j in product(range(diff_mean.shape[0]), range(diff_mean.shape[1])):
        annot = f"{diff_mean.iloc[i, j]:.2f}\n±{diff_std.iloc[i, j]:.2f}"
        weight = "bold" if pvalues.iloc[i, j] < ttest_rel_alpha else "normal"
        ax_diff.text(j, i, annot, ha="center", va="center", weight=weight, fontsize="small")
    ax_diff.set_xticks(np.arange(diff_mean.shape[1]))
    ax_diff.set_yticks(np.arange(diff_mean.shape[0]))
    ax_diff.set_xticklabels(diff_mean.columns, rotation=45, ha="right")
    ax_diff.set_yticklabels(diff_mean.index, rotation=0)
    ax_diff.set_xlabel("Methods 2")
    ax_diff.set_ylabel("Methods 1")
    fig_diff.colorbar(diff_image, ax=ax_diff, label="Score difference mean")
    fig_diff.tight_layout()

    return fig_pct, fig_diff, percentage_with_rank, diff_mean
