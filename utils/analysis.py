"""Small dataframe helpers for experiment notebooks.

The notebooks should show the analysis decisions, not reimplement common
filtering, baseline handling, bucketed summaries, or family inference.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import ast


def class_subset_len(value) -> float:
    try:
        return float(len(ast.literal_eval(str(value))))
    except (ValueError, SyntaxError, TypeError):
        return np.nan


def keep_ge_class_subset_len(df: pd.DataFrame, length: int = 3) -> pd.DataFrame:
    return df[(df["dataset"] != "GE") | (df["classes_subset"].map(class_subset_len) == length)].copy()


def filter_keep(df, keep=None, filters=None, drop_single_level=False):
    """take a subset of a multi-index dataframe by keeping or filtering out values at certain levels."""
    out = df.copy()

    # filter
    for level, values in (filters or {}).items():
        values = [values] if isinstance(values, str) else values
        out = out[~out.index.get_level_values(level).isin(values)]

    # keep
    for level, values in (keep or {}).items():
        values = [values] if isinstance(values, str) else values
        out = out[out.index.get_level_values(level).isin(values)]
    
    if drop_single_level:
        for level in out.index.names:
            if len(out.index.get_level_values(level).unique()) == 1:
                out = out.droplevel(level)

    return out

def preprocess(df, seeds_per_bucket=5):
    """Preprocess the dataframe for analysis.
    This includes filtering out certain columns, averaging duplicate rows, and aggregating seeds into buckets.
    """
    df = df.copy()
    score = pd.to_numeric(df["score"], errors="coerce")
    if "num_correct" in df.columns:
        fallback_score = pd.to_numeric(df["num_correct"], errors="coerce")
        score = score.fillna(fallback_score.where(fallback_score.between(0, 1)))
    df["score"] = score

    required = ["dataset", "model", "classes_subset", "seed", "method", "prompt_type", "specification", "score"]
    df = df.dropna(subset=[col for col in required if col in df.columns])

    df = keep_ge_class_subset_len(df, 3)
    df.drop(columns=["nb_concepts", "time", "num_correct", "num_valid", "num_expected"], inplace=True, errors="ignore")

    index_cols = df.columns.drop("score").tolist()

    # 1. Average duplicate rows
    df = (
        df.groupby(index_cols, dropna=False)["score"]
          .mean()
    )

    # 2. Aggregate seeds into buckets
    df = (
        df.reset_index()
          .assign(seed=lambda x: x["seed"] // seeds_per_bucket)
          .groupby(index_cols, dropna=False)["score"]
          .mean()
          .rename_axis(index={"seed": "group_seed"})
    )

    return df

def print_df_keys(df):
    """Print the unique values for each index level of a multi-index dataframe."""
    print(f"DataFrame has {len(df):,} rows.")
    for level in df.index.names:
        print(f"{level}: {df.index.get_level_values(level).unique().tolist()}")

def score_diff(reference, challenger):
    """Compute the difference in scores between a reference and a challenger DataFrame."""
    common = reference.index.intersection(challenger.index)

    return challenger[common] - reference[common]

def merge_concept_baseline(df):
    """Merge the concept and baseline prompt types into a single prompt type."""
    # extract B2 and AB2, duplicate then into B3 and AB3
    B2_AB2 = filter_keep(df, keep={"prompt_type": ["B2", "AB2"]})
    B2_AB2.rename(index={"B2": "B3", "AB2": "AB3"}, level="prompt_type", inplace=True)
    df = pd.concat([df, B2_AB2], axis=0)

    # map baseline prompt types to the concepts prompt types
    df.rename(
        index={"B1": "C1", "B2": "C2", "B3": "C3", "AB1": "AC1", "AB2": "AC2", "AB3": "AC3"},
        level="prompt_type", inplace=True
    )
    return df
