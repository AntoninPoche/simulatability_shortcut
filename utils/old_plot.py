

from itertools import combinations, product
import os
from typing import List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import seaborn as sns


def plot_pairwise_comparison_matrix(accuracies: pd.Series,
                                    baselines: pd.Series,
                                    compared_index: str = 'method',
                                    save_dir: Optional[str] = None,
                                    ttest_rel_alpha: float = 0.05,
                                    difference_mean_alpha: float = 0.5) -> None:
    contenders = list(accuracies.index.get_level_values(compared_index).unique())
    assert compared_index != "prompt_type"
    contenders.append("NoExplanation")
    n_contenders = len(contenders)

    # prepare results matrices
    # count_results_matrix = pd.DataFrame(0, index=contenders, columns=contenders, dtype=int)
    percentage_results_matrix = pd.DataFrame(0.0, index=contenders, columns=contenders)
    difference_mean_results_matrix = pd.DataFrame(0.0, index=contenders, columns=contenders)
    difference_std_results_matrix = pd.DataFrame(0.0, index=contenders, columns=contenders)
    ttest_rel_results_matrix = pd.DataFrame(1.0, index=contenders, columns=contenders)

    # iterate over pairs of contenders
    for c1, c2 in combinations(contenders, 2):
        if c1 == "NoExplanation":
            raise NotImplementedError

        # get accuracies for each contender
        sc1 = accuracies.xs(c1, level=compared_index).dropna()
        if c2 != "NoExplanation":
            sc2 = accuracies.xs(c2, level=compared_index).dropna()
        else:
            sc2 = baselines.xs(c1, level=compared_index).dropna()

        if percentage_results_matrix.loc[c1, c1] == 0:
            # count_results_matrix.loc[c1, c1] = len(sc1)
            percentage_results_matrix.loc[c1, c1] = 50

        # treat only common experiments
        common_indices = sc1.index.intersection(sc2.index)
        if len(common_indices) == 0:
            continue
        sc1 = sc1.loc[common_indices]
        sc2 = sc2.loc[common_indices]

        # count
        # count_results_matrix.loc[c1, c2] = count_results_matrix.loc[c2, c1] = len(common_indices)

        # percentage
        wins, draws, defeats = (sc1 > sc2).sum(), (sc1 == sc2).sum(), (sc1 < sc2).sum()
        percentage_results_matrix.loc[c1, c2] = round((wins + 0.5 * draws) / len(common_indices) * 100)
        percentage_results_matrix.loc[c2, c1] = round((defeats + 0.5 * draws) / len(common_indices) * 100)

        # difference mean
        difference_mean_results_matrix.loc[c1, c2] = (sc1 - sc2).mean()
        difference_mean_results_matrix.loc[c2, c1] = - difference_mean_results_matrix.loc[c1, c2]
        difference_std_results_matrix.loc[c1, c2] = difference_std_results_matrix.loc[c2, c1] = (sc1 - sc2).std()
        if np.abs(difference_mean_results_matrix.loc[c2, c1]) > difference_mean_alpha:
            ttest_rel_results_matrix.loc[c1, c2] = ttest_rel_results_matrix.loc[c2, c1] = ttest_rel(sc1, sc2).pvalue

    # fill diagonal
    if "NoExplanation" in contenders:
        # count_results_matrix.loc["NoExplanation", "NoExplanation"] = count_results_matrix.max().max()
        percentage_results_matrix.loc["NoExplanation", "NoExplanation"] = 50
    else:
        percentage_results_matrix.loc[c2, c2] = 50

    # add mean column
    ranking = n_contenders + 1 - (percentage_results_matrix >= 50).sum(axis="columns")

    sorted_ranking = ranking.sort_values(ascending=True)
    index_order = sorted_ranking.index

    # count_results_matrix = count_results_matrix.reindex(index=sorted_combined_means.index, columns=sorted_combined_means.index)
    percentage_results_matrix = percentage_results_matrix.reindex(index=index_order, columns=index_order)
    difference_mean_results_matrix = difference_mean_results_matrix.reindex(index=index_order, columns=index_order)
    difference_std_results_matrix = difference_std_results_matrix.reindex(index=index_order, columns=index_order)
    ttest_rel_results_matrix = ttest_rel_results_matrix.reindex(index=index_order, columns=index_order)
    percentage_results_matrix["rank"] = sorted_ranking

    mask = np.zeros((n_contenders, n_contenders + 1), dtype=bool)
    mask[:, n_contenders] = True
    
    # Create subplots

    sns.set(font_scale=1.5)
    # # Plot count matrix
    # sns.heatmap(count_results_matrix, annot=True, fmt='d', cmap='plasma', cbar_kws={'label': 'Count'}, vmin=0, ax=axes[0])
    # axes[0].set_title('Count comparison matrix')

    # Plot percentage matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(percentage_results_matrix, annot=True,  mask=~mask, ax=ax,  #  annot_kws={'size': 15},
                cmap='Greys', vmin=100, vmax=100, cbar=False)
    sns.heatmap(percentage_results_matrix, annot=True,  mask=mask, ax=ax,  #  annot_kws={'size': 15},
                cmap='coolwarm', vmin=0, vmax=100, cbar_kws={'label': 'Win rate of Method 1 over 2 (in %)'})
    
    # # Add a black box around rankings
    # rect = patches.Rectangle(
    #     (n_contenders, 0),  # (x, y) starting point (column index, row index)
    #     1,  # Width of the box (1 column)
    #     n_contenders,  # Height of the box (all rows)
    #     linewidth=2, edgecolor='black', facecolor='none'
    # )
    # ax.add_patch(rect)  # Add the rectangle to the heatmap

    # plt.title('Percentage of wins Method 1 over Method 2')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('Methods 2')
    plt.ylabel('Methods 1')
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{compared_index}_pairwise_percentage_comparison_matrices.pdf"
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

    # Plot difference matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(difference_mean_results_matrix, annot=False, fmt='', cmap='coolwarm', cbar_kws={'label': 'Accuracies difference mean',})
    for i, j in product(range(difference_mean_results_matrix.shape[0]), range(difference_mean_results_matrix.shape[1])):
        annot = f"{round(difference_mean_results_matrix.iloc[i, j], 1)}"
        annot += f"\n±{round(difference_std_results_matrix.iloc[i, j], 1)}"
        if ttest_rel_results_matrix.iloc[i, j] < ttest_rel_alpha:
            plt.text(j + 0.5, i + 0.5, annot, ha='center', va='center', weight='bold', fontsize='small')
        else:
            plt.text(j + 0.5, i + 0.5, annot, ha='center', va='center', fontsize='small')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('Methods 2')
    plt.ylabel('Methods 1')
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{compared_index}_pairwise_difference_comparison_matrices.pdf"
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()


def plot_prompt_type_pairwise_comparison_matrix(accuracies: pd.Series,
                                                baselines: pd.Series,
                                                save_dir: Optional[str] = None,
                                                ttest_rel_alpha: float = 0.05,
                                                difference_mean_alpha: float = 0.5) -> None:
    compared_index = "prompt_type"

    # reuse correct baseline prompt type names
    prompt_types_baselines = {
        "E1": "L1",
        "E2": "L2",
        "E3": "L2",
        "U1": "L2",
        "E1-a": "L1-a",
        "E2-a": "L2-a",
        "E3-a": "L2-a",
        "U1-a": "L2-a",
    }
    baselines = baselines.rename(index=prompt_types_baselines, level=compared_index)

    # fuse explanation and baseline prompt types
    accuracies = pd.concat([accuracies, baselines], axis="rows")

    # average accuracies on duplicated indices (it is the case for several baselines and might influence statistical tests)
    accuracies = accuracies.groupby(level=accuracies.index.names).mean()

    contenders = list(accuracies.index.get_level_values(compared_index).unique())
    n_contenders = len(contenders)

    # prepare results matrices
    # count_results_matrix = pd.DataFrame(0, index=contenders, columns=contenders, dtype=int)
    percentage_results_matrix = pd.DataFrame(0.0, index=contenders, columns=contenders)
    difference_mean_results_matrix = pd.DataFrame(0.0, index=contenders, columns=contenders)
    difference_std_results_matrix = pd.DataFrame(0.0, index=contenders, columns=contenders)
    ttest_rel_results_matrix = pd.DataFrame(1.0, index=contenders, columns=contenders)

    # iterate over pairs of contenders
    for c1, c2 in combinations(contenders, 2):

        # get accuracies for each contender
        sc1 = accuracies.xs(c1, level=compared_index).dropna()
        sc2 = accuracies.xs(c2, level=compared_index).dropna()

        if percentage_results_matrix.loc[c1, c1] == 0:
            # count_results_matrix.loc[c1, c1] = len(sc1)
            percentage_results_matrix.loc[c1, c1] = 50

        # treat only common experiments
        common_indices = sc1.index.intersection(sc2.index)
        sc1 = sc1.loc[common_indices]
        sc2 = sc2.loc[common_indices]

        if len(sc1) == 0 or len(sc2) == 0:
            continue

        # count
        # count_results_matrix.loc[c1, c2] = count_results_matrix.loc[c2, c1] = len(common_indices)

        # percentage
        wins, draws, defeats = (sc1 > sc2).sum(), (sc1 == sc2).sum(), (sc1 < sc2).sum()
        percentage_results_matrix.loc[c1, c2] = round((wins + 0.5 * draws) / len(common_indices) * 100)
        percentage_results_matrix.loc[c2, c1] = round((defeats + 0.5 * draws) / len(common_indices) * 100)

        # difference mean
        difference_mean_results_matrix.loc[c1, c2] = (sc1 - sc2).mean()
        difference_mean_results_matrix.loc[c2, c1] = - difference_mean_results_matrix.loc[c1, c2]
        difference_std_results_matrix.loc[c1, c2] = difference_std_results_matrix.loc[c2, c1] = (sc1 - sc2).std()
        if np.abs(difference_mean_results_matrix.loc[c2, c1]) > difference_mean_alpha:
            ttest_rel_results_matrix.loc[c1, c2] = ttest_rel_results_matrix.loc[c2, c1] = ttest_rel(sc1, sc2).pvalue

    # fill diagonal
    percentage_results_matrix.loc[c2, c2] = 50

    # add mean column
    ranking = n_contenders + 1 - (percentage_results_matrix >= 50).sum(axis="columns")

    sorted_ranking = ranking.sort_values(ascending=True)
    index_order = sorted_ranking.index

    # count_results_matrix = count_results_matrix.reindex(index=sorted_combined_means.index, columns=sorted_combined_means.index)
    percentage_results_matrix = percentage_results_matrix.reindex(index=index_order, columns=index_order)
    difference_mean_results_matrix = difference_mean_results_matrix.reindex(index=index_order, columns=index_order)
    difference_std_results_matrix = difference_std_results_matrix.reindex(index=index_order, columns=index_order)
    ttest_rel_results_matrix = ttest_rel_results_matrix.reindex(index=index_order, columns=index_order)
    percentage_results_matrix["rank"] = sorted_ranking

    mask = np.zeros((n_contenders, n_contenders + 1), dtype=bool)
    mask[:, -1] = True
    
    # Create subplots

    sns.set(font_scale=1.5)
    # # Plot count matrix
    # sns.heatmap(count_results_matrix, annot=True, fmt='d', cmap='plasma', cbar_kws={'label': 'Count'}, vmin=0, ax=axes[0])
    # axes[0].set_title('Count comparison matrix')

    # Plot percentage matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(percentage_results_matrix, annot=True,  mask=~mask, ax=ax,  #  annot_kws={'size': 15},
                cmap='Greys', vmin=100, vmax=100, cbar=False)
    sns.heatmap(percentage_results_matrix, annot=True,  mask=mask, ax=ax,  #  annot_kws={'size': 15},
                cmap='coolwarm', vmin=0, vmax=100, cbar_kws={'label': 'Win rate of Prompt type 1 over 2 (in %)'})
    
    # # Add a black box around rankings
    # rect = patches.Rectangle(
    #     (n_contenders, 0),  # (x, y) starting point (column index, row index)
    #     1,  # Width of the box (1 column)
    #     n_contenders,  # Height of the box (all rows)
    #     linewidth=2, edgecolor='black', facecolor='none'
    # )
    # ax.add_patch(rect)  # Add the rectangle to the heatmap

    # plt.title('Percentage of wins Method 1 over Method 2')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('Prompt types 2')
    plt.ylabel('Prompt types 1')
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{compared_index}_pairwise_percentage_comparison_matrices.pdf"
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

    # Plot difference matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(difference_mean_results_matrix, annot=False, fmt='', cmap='coolwarm', cbar_kws={'label': 'Accuracies difference mean',})
    for i, j in product(range(difference_mean_results_matrix.shape[0]), range(difference_mean_results_matrix.shape[1])):
        annot = f"{round(difference_mean_results_matrix.iloc[i, j], 1)}"
        annot += f"\n±{round(difference_std_results_matrix.iloc[i, j], 1)}"
        if ttest_rel_results_matrix.iloc[i, j] < ttest_rel_alpha:
            plt.text(j + 0.5, i + 0.5, annot, ha='center', va='center', weight='bold', fontsize='x-small')
        else:
            plt.text(j + 0.5, i + 0.5, annot, ha='center', va='center', fontsize='x-small')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('Prompt types 2')
    plt.ylabel('Prompt types 1')
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{compared_index}_pairwise_difference_comparison_matrices.pdf"
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()


def plot_accuracies_summaries(accuracies: pd.Series,
                              baselines: pd.Series,
                              compared_index1: str = 'method',
                              save_dir: Optional[str] = None) -> None:
    datasets = list(accuracies.index.get_level_values('dataset').unique())
    models = list(accuracies.index.get_level_values('model').unique())
    indices = list(accuracies.index.get_level_values(compared_index1).unique())
    if "NMF" in indices:
        # smooth ordering
        indices = ["NMF", "SAE", "ICA", "PCA", "SVD", "NoProjection"]
    if len(indices) <= 1:
        return
    prompt_types = list(accuracies.index.get_level_values('prompt_type').unique())
    prompt_types = [pt for pt in prompt_types if len(pt) == 2] +\
                   [pt for pt in prompt_types if len(pt) == 4]
    if len(prompt_types) <= 1:
        return

    counts = {}
    prompt_accuracies = {}
    prompt_mean_accuracies = {}
    prompt_std_accuracies = {}
    # baselines_accuracies = {}
    for prompt_type in prompt_types:
        pt_accuracies = accuracies.xs(prompt_type, level="prompt_type").dropna()
        pt_baselines = baselines.xs(prompt_type, level="prompt_type").dropna()

        if len(pt_accuracies) == 0 or len(pt_baselines) == 0:
            continue
        
        counts[prompt_type] = {}
        prompt_accuracies[prompt_type] = {}
        prompt_mean_accuracies[prompt_type] = {}
        prompt_std_accuracies[prompt_type] = {}
        # baselines_accuracies[prompt_type] = []
        for index in indices:
            pt_mt_accuracies = pt_accuracies.xs(index, level=compared_index1)
            pt_mt_baselines = pt_baselines.xs(index, level=compared_index1)

            # save the accuracies for the method
            prompt_accuracies[prompt_type][index] = pt_mt_accuracies
            counts[prompt_type][index] = len(pt_mt_accuracies)
            prompt_mean_accuracies[prompt_type][index] = np.mean(pt_mt_accuracies)
            prompt_std_accuracies[prompt_type][index] = np.std(pt_mt_accuracies)

            # save the accuracies for the baseline
            prompt_accuracies[prompt_type]["NoExplanation"] = pt_mt_baselines
            counts[prompt_type]["NoExplanation"] = len(pt_mt_baselines)
            prompt_mean_accuracies[prompt_type]["NoExplanation"] = np.mean(pt_mt_baselines)
            prompt_std_accuracies[prompt_type]["NoExplanation"] = np.std(pt_mt_baselines)
        #     baselines_accuracies[prompt_type] += pt_mt_baselines.tolist()
        # baselines_accuracies[prompt_type] = np.mean(baselines_accuracies[prompt_type])
    
    if len(counts) == 0:
        return
    
    indices.append("NoExplanation")
    num_types = len(prompt_types)
    num_indices = len(indices)
    prompt_types_ids = [pt.split(":")[0] for pt in prompt_types]
    colors = {
        index: mpl.colormaps['tab10'].colors[i]
        for i, index in enumerate(indices)
    }
    colors["NoExplanation"] = "black"

    # Calculate positions for the groups
    bar_width = 0.12
    bars_index = np.arange(num_types)

    # ----------------------------------------------------------------------------------------------
    # Bar plot showing the mean accuracy for each method and prompt type

    # # Determine the maximum count for normalization
    # max_count = max(max(counts[typ].values()) for typ in counts)

    # # Plotting
    # fig, ax = plt.subplots(figsize=(12, 6))
    # title = f"Mean accuracy by prompt type"
    # if len(datasets) == 1:
    #     title += f" - Dataset: {datasets[0]}"
    # if len(models) == 1:
    #     title += f" - Model: {models[0]}"
    # fig.suptitle(title)

    # for i, index in enumerate(indices):
    #     accuracies = [prompt_mean_accuracies[typ][index] for typ in prompt_types]
    #     stds = [prompt_std_accuracies[typ][index] for typ in prompt_types]
    #     element_counts = [counts[typ][index] for typ in prompt_types]
    #     if accuracies is None or any(acc is None for acc in accuracies):
    #         continue
    #     bars = ax.bar(
    #         bars_index + i * bar_width,
    #         accuracies,
    #         bar_width,
    #         yerr=stds,
    #         label=index,
    #         color=colors[index],
    #     )
            
    #     # Adding values on top of the bars and setting alpha
    #     # for bar, count, baseline_acc in zip(bars, element_counts, baselines_accuracies.values()):
    #     for bar, count in zip(bars, element_counts):
    #         height = bar.get_height()
    #         alpha_value = count / max_count
    #         bar.set_alpha(alpha_value)
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2.0,
    #             height,
    #             f'{height:.1f}',
    #             ha='center',
    #             va='bottom',
    #         )
    #         ax.text(
    #             bar.get_x() + bar.get_width() / 2.0,
    #             0.1,
    #             f'{count}',
    #             ha='center',
    #             va='bottom'
    #         )

    # # Adding labels
    # ax.set_xlabel('Prompt types', fontsize=12)
    # ax.set_ylabel('Simulatability score', fontsize=12)
    # ax.set_xticks(bars_index + bar_width * (num_indices - 1) / 2)
    # ax.set_xticklabels([f"{typ.split(':')[0]}" for typ in prompt_types], fontsize=8)

    # ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    # plt.tight_layout()
    
    # if save_dir is not None:
    #     os.makedirs(save_dir, exist_ok=True)
    #     file_name = f"{compared_index1}_barplot.pdf"
    #     plt.savefig(os.path.join(save_dir, file_name))
    #     plt.close()

    # -------------------------------------------------------------------------------
    # Violin plot showing the accuracies distribution for each method and prompt type

    # Plotting
    fig, ax = plt.subplots(figsize=(18, 6))
    # title = f"Accuracy distribution by method and prompt type"
    # if len(datasets) == 1:
    #     title += f" - Dataset: {datasets[0]}"
    # if len(models) == 1:
    #     title += f" - Model: {models[0]}"
    # fig.suptitle(title)

    handles, labels = [], []
    for i, index in enumerate(indices):
        accuracies = [prompt_accuracies[typ][index] for typ in prompt_types]
        if accuracies is None or any(acc is None for acc in accuracies):
            continue
        violins = ax.violinplot(
            accuracies,
            positions=bars_index + i * bar_width,
            widths=bar_width,
            showmeans=True,
            showmedians=False,
        )
        if index == "NoExplanation":
            for body in violins["bodies"]:
                body.set_facecolor("black")
                body.set_edgecolor("black")
            violins["cmeans"].set_edgecolor("black")
            violins["cmaxes"].set_edgecolor("black")
            violins["cmins"].set_edgecolor("black")
            violins["cbars"].set_edgecolor("black")
        handles.append(patches.Patch(color=violins["bodies"][0].get_facecolor().flatten()))
        labels.append(index)

    for i, typ in enumerate(prompt_types):
        baseline = prompt_mean_accuracies[typ]["NoExplanation"].mean()
        # Determine the x positions to plot the line
        xmin = bars_index[i] - bar_width
        xmax = bars_index[i] + bar_width * num_indices
        
        # Draw the horizontal baseline line across the entire group of methods
        ax.plot([xmin, xmax], [baseline, baseline], color='black', linewidth=1)

    # Adding labels
    ax.set_xlabel('Prompt types')
    ax.set_ylabel('Simulatability score')
    ax.set_xticks(bars_index + bar_width * (num_indices - 1) / 2)
    ax.set_xticklabels(prompt_types_ids)

    ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{compared_index1}_violinplot.pdf"
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()                           
