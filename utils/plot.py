"""Plot helpers for the new ConSim score tables."""

from __future__ import annotations

from itertools import combinations, product
import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_rel

from utils.analysis import filter_keep


PLOT_FONT_SIZE = 10
plt.rcParams.update(
    {
        "font.size": PLOT_FONT_SIZE,
        "axes.labelsize": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_FONT_SIZE + 1,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
        "legend.fontsize": PLOT_FONT_SIZE,
        "legend.title_fontsize": PLOT_FONT_SIZE,
    }
)


def plot_accuracies_violins(
    scores: pd.Series,
    *,
    group_col: str = "prompt_type",
    atom_col: str = "method",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7.0, 3.2),
    font_size: float = PLOT_FONT_SIZE,
    legend_font_size: float | None = None,
    ylabel: str = "Simulatability score",
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot score distributions by group and atom using the old ConSim style."""
    if not isinstance(scores.index, pd.MultiIndex):
        raise TypeError("scores must use a MultiIndex.")
    for level in (group_col, atom_col):
        if level not in scores.index.names:
            raise ValueError(f"Missing index level {level!r}.")

    scores = scores.groupby(level=scores.index.names, dropna=False).mean().dropna()
    groups = list(scores.index.get_level_values(group_col).unique())
    atoms = list(scores.index.get_level_values(atom_col).unique())
    baseline_atoms = {"B", "B1", "B2", "NoExplanation"}
    atoms = [atom for atom in atoms if atom not in baseline_atoms] + [atom for atom in atoms if atom in baseline_atoms]
    if not groups or not atoms:
        raise ValueError("No data to plot.")

    bar_width = min(0.14, 0.9 / len(atoms))
    bars_index = np.arange(len(groups))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    handles, labels = [], []
    for i, atom in enumerate(atoms):
        values_per_group = [_values_at(scores, group_col, group, atom_col, atom) for group in groups]
        keep = [values.size > 0 for values in values_per_group]
        if not any(keep):
            continue

        violins = ax.violinplot(
            [values for values, ok in zip(values_per_group, keep) if ok],
            positions=bars_index[keep] + i * bar_width,
            widths=bar_width,
            showmeans=True,
            showmedians=False,
        )
        if atom in baseline_atoms:
            for body in violins["bodies"]:
                body.set_facecolor("#303030")
                body.set_edgecolor("black")
            for line_key in ("cmeans", "cmaxes", "cmins", "cbars"):
                violins[line_key].set_edgecolor("black")

        handles.append(patches.Patch(color=violins["bodies"][0].get_facecolor().flatten()))
        labels.append(_display_label(atom))

    for baseline_atom in [atom for atom in atoms if atom in baseline_atoms]:
        for i, group in enumerate(groups):
            values = _values_at(scores, group_col, group, atom_col, baseline_atom)
            if values.size == 0:
                continue
            ax.plot(
                [bars_index[i] - bar_width, bars_index[i] + bar_width * len(atoms)],
                [values.mean(), values.mean()],
                color="black",
                linewidth=1,
            )

    xlabel = "Prompt types" if group_col == "prompt_type" else group_col.replace("_", " ").title()
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_ylim(0, 1)
    ax.set_xlim(
        bars_index[0] - bar_width,
        bars_index[-1] + bar_width * len(atoms),
    )
    ax.set_xticks(bars_index + bar_width * (len(atoms) - 1) / 2)
    ax.set_xticklabels(groups, fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    if title is not None:
        ax.set_title(title, fontsize=font_size + 1)
    ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0, 0.01, 1, 0.01),
        mode="expand",
        ncol=min(4, len(labels)),
        fontsize=legend_font_size or (font_size - 3 if len(labels) > 7 else font_size - 1),
        framealpha=0.92,
        borderpad=0.35,
        handlelength=1.2,
    )
    fig.tight_layout()

    return fig, ax
def _values_at(
    scores: pd.Series,
    group_col: str,
    group: str,
    atom_col: str,
    atom: str,
) -> np.ndarray:
    try:
        return scores.xs(group, level=group_col).xs(atom, level=atom_col).to_numpy()
    except KeyError:
        return np.array([])

def plot_difference_bars(
    differences: pd.Series,
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (3.35, 3.0),
    comparison_col: str = "prompt_type",
    ylabel: str = "Simulatability difference",
    title: str | None = None,
    positive_color: str = "#4c72b0",
    negative_color: str = "#c44e52",
    pvalue_alpha: float = 0.05,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot bucketed score differences as simple signed bar plots.

    ``differences`` is expected to already contain one row per displayed bar. This
    helper intentionally does not compute statistics: notebooks keep filtering
    and aggregation explicit, while this function centralizes the visual style.
    """
    grouped = differences.groupby(comparison_col)
    means = grouped.mean().sort_values(ascending=False)
    stds = grouped.std().reindex(means.index)  # ensure stds aligns with means

    pvalues = pd.Series(np.nan, index=means.index, dtype=float)
    for key in means.index:
        values = grouped.get_group(key).dropna()
        pvalues.loc[key] = ttest_1samp(values, popmean=0).pvalue

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
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
    xticklabels = [
        f"{key} {_format_pvalue_tick(pvalues.loc[key], alpha=pvalue_alpha)}"
        for key in means.index
    ]
    tick_font_size = 6.5 if len(means) > 7 else PLOT_FONT_SIZE - 1
    tick_rotation = 65 if len(means) > 7 else 45
    ax.set_xticklabels(
        xticklabels,
        rotation=tick_rotation,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_font_size,
    )
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    return fig, ax


def _format_pvalue_tick(pvalue: float, *, alpha: float) -> str:
    if pd.isna(pvalue):
        text = "p=n/a"
    elif pvalue < 0.01:
        text = "p<.01"
    else:
        text = f"p={pvalue:.2f}".replace("=0.", "=.")

    if pd.notna(pvalue) and pvalue < alpha:
        return r"$\bf{" + text.replace("<", r"<") + "}$"
    return text

def panel_difference_bars(
    differences: pd.Series,
    *,
    panel_col: str | None = None,
    comparison_col: str = "prompt_type",
    ylabel: str = "Simulatability difference",
    title: str | None = None,
    positive_color: str = "#4c72b0",
    negative_color: str = "#c44e52",
    pvalue_alpha: float = 0.05,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot bucketed score differences as simple signed bar plots.

    ``differences`` is expected to already contain one row per displayed bar. This
    helper intentionally does not compute statistics: notebooks keep filtering
    and aggregation explicit, while this function centralizes the visual style.
    """
    if differences.empty:
        raise ValueError("Cannot plot an empty differences dataframe.")

    panels = list(differences.index.get_level_values(panel_col).unique())
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0, 2.7 * nrows),
        sharey=True,
        squeeze=False,
    )

    for panel_index, (ax, panel_value) in enumerate(zip(axes.flat, panels)):
        plot_difference_bars(
            filter_keep(differences, keep={panel_col: panel_value}),
            ax=ax,
            comparison_col=comparison_col,
            ylabel=ylabel if panel_index % ncols == 0 else "",
            title=str(panel_value),
            positive_color=positive_color,
            negative_color=negative_color,
            pvalue_alpha=pvalue_alpha,
        )

    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    if title is not None:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()

    return fig, axes


def plot_pairwise_comparison_matrices(
    scores: pd.Series,
    *,
    compared_index: str = "method",
    ttest_rel_alpha: float = 0.05,
    figsize: tuple[float, float] = (3.6, 3.4),
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
    scores = scores.groupby(level=scores.index.names, dropna=False).mean().dropna()
    contenders = list(scores.index.get_level_values(compared_index).unique())
    if len(contenders) < 2:
        raise ValueError("Need at least two contenders for a pairwise matrix.")

    n_contenders = len(contenders)
    percentage = pd.DataFrame(np.nan, index=contenders, columns=contenders)
    diff_mean = pd.DataFrame(np.nan, index=contenders, columns=contenders)
    diff_std = pd.DataFrame(np.nan, index=contenders, columns=contenders)
    pvalues = pd.DataFrame(1.0, index=contenders, columns=contenders)

    for contender in contenders:
        percentage.loc[contender, contender] = 50
        diff_mean.loc[contender, contender] = 0
        diff_std.loc[contender, contender] = 0

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

    fig_pct, ax_pct = plt.subplots(figsize=figsize)
    pct_values = percentage.to_numpy(dtype=float)
    ax_pct.imshow(pct_values, cmap="coolwarm", vmin=0, vmax=100)
    annotation_size = 6 if n_contenders >= 7 and figsize[0] < 5 else 8
    for row in range(pct_values.shape[0]):
        for col in range(pct_values.shape[1]):
            value = pct_values[row, col]
            if np.isnan(value):
                continue
            text = f"{value:.0f}%"
            ax_pct.text(col, row, text, ha="center", va="center", fontsize=annotation_size + 1, color="black")
    ax_pct.set_xticks(np.arange(percentage.shape[1]))
    ax_pct.set_yticks(np.arange(percentage.shape[0]))
    display_columns = [_display_label(value) for value in percentage.columns]
    display_index = [_display_label(value) for value in percentage.index]
    ax_pct.set_xticklabels(display_columns, rotation=45, ha="right", rotation_mode="anchor")
    ax_pct.set_yticklabels(display_index, rotation=0)
    fig_pct.tight_layout()

    fig_diff, ax_diff = plt.subplots(figsize=figsize)
    diff_values = diff_mean.to_numpy(dtype=float)
    max_abs_diff = max(float(np.nanmax(np.abs(diff_values))), 1e-9)
    ax_diff.imshow(diff_values, cmap="coolwarm", vmin=-max_abs_diff, vmax=max_abs_diff)
    for i, j in product(range(diff_mean.shape[0]), range(diff_mean.shape[1])):
        value = diff_mean.iloc[i, j]
        if pd.isna(value):
            continue
        annot = "—" if i == j else f"{value:.2f}\n±{diff_std.iloc[i, j]:.2f}"
        weight = "bold" if pvalues.iloc[i, j] < ttest_rel_alpha else "normal"
        ax_diff.text(j, i, annot, ha="center", va="center", weight=weight, fontsize=annotation_size, color="black")
    ax_diff.set_xticks(np.arange(diff_mean.shape[1]))
    ax_diff.set_yticks(np.arange(diff_mean.shape[0]))
    ax_diff.set_xticklabels(display_columns, rotation=45, ha="right", rotation_mode="anchor")
    ax_diff.set_yticklabels(display_index, rotation=0)
    fig_diff.tight_layout()

    return fig_pct, fig_diff, percentage, diff_mean


def _display_label(value: object) -> str:
    labels = {
        "NoExplanation": "No expl.",
        "integrated_gradients": "Int. gradients",
        "VanillaSAE": "Vanilla SAE",
        "NeuronsAs": "Neurons",
        "ClassesAs": "Classes",
        "Qwen/Qwen3.5-2B": "Qwen 3.5-2B",
        "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2-3B",
    }
    return labels.get(str(value), str(value))
