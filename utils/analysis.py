"""Small dataframe helpers for experiment notebooks.

The notebooks should show the analysis decisions, not reimplement common
filtering, baseline handling, bucketed summaries, or family inference.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from utils.registries import ATTRIBUTION_METHOD_NAMES, CONCEPT_METHOD_NAMES


BASELINE_PROMPT_TYPES = {"B1", "B2", "AB1", "AB2"}
CONCEPT_PROMPT_TYPES = {"C1", "C2", "C3", "AC1", "AC2", "AC3"}
RATIONALE_PROMPT_TYPES = {"R1", "AR1"}
ATTRIBUTION_PROMPT_TYPES = {"A1", "AA1"}

NON_ANON_PROMPT_TYPES_BY_FAMILY = {
    "concepts": {"C1", "C2", "C3"},
    "rationales": {"R1"},
    "attributions": {"A1"},
}
NON_ANON_BASELINES = {"B1", "B2"}
NON_ANON_PROMPT_TYPES = set().union(*NON_ANON_PROMPT_TYPES_BY_FAMILY.values()) | NON_ANON_BASELINES

FAMILY_ORDER = ["concepts", "rationales", "attributions"]
FAMILY_COLORS = {
    "concepts": "#4c72b0",
    "rationales": "#55a868",
    "attributions": "#c44e52",
    "baseline": "#333333",
}
EXCLUDED_BEST_METHODS = {"baseline", "classes", "ClassesAs"}

CONCEPT_METHODS = set(CONCEPT_METHOD_NAMES) | set(CONCEPT_METHOD_NAMES.values())
ATTRIBUTION_METHODS = set(ATTRIBUTION_METHOD_NAMES) | {
    "Saliency",
    "IntegratedGradients",
    "Integrated Gradients",
    "SmoothGrad",
    "SquareGrad",
    "VarGrad",
    "GradientShap",
    "LIME",
    "KernelShap",
    "Occlusion",
    "Sobol",
}


def _values(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def filter_keep(
    df: pd.DataFrame,
    filters: Mapping[str, Sequence[Any] | Any] | None = None,
    keep: Mapping[str, Sequence[Any] | Any] | None = None,
    *,
    drop_scalar_keep: bool = False,
) -> pd.DataFrame:
    """Drop values, then keep values, using column -> allowed/disallowed values.

    ``filters`` means "remove rows whose column is one of these values".
    ``keep`` means "keep rows whose column is one of these values". A string is
    accepted as a scalar value. When ``drop_scalar_keep`` is true, columns kept
    to one scalar value are dropped after filtering.
    """
    out = df.copy()
    for column, raw_values in (filters or {}).items():
        values = _values(raw_values)
        if values is None:
            continue
        out = out[~out[column].isin(values)].copy()

    for column, raw_values in (keep or {}).items():
        values = _values(raw_values)
        if values is None:
            continue
        out = out[out[column].isin(values)].copy()
        if drop_scalar_keep and len(values) == 1:
            out = out.drop(columns=[column])
    return out


def class_subset_len(value: Any) -> float:
    try:
        return float(len(ast.literal_eval(str(value))))
    except (ValueError, SyntaxError, TypeError):
        return np.nan


def random_chance_for(df: pd.DataFrame, column: str = "classes_subset") -> float | None:
    lengths = df[column].map(class_subset_len).dropna().unique()
    if len(lengths) != 1:
        return None
    return 1 / float(lengths[0])


def keep_ge_class_subset_len(df: pd.DataFrame, length: int = 3) -> pd.DataFrame:
    return df[(df["dataset"] != "GE") | (df["classes_subset"].map(class_subset_len) == length)].copy()


def infer_family(row: pd.Series) -> str:
    prompt_type = str(row["prompt_type"])
    method = str(row["method"])
    if prompt_type in BASELINE_PROMPT_TYPES or method == "baseline":
        return "baseline"
    if prompt_type in CONCEPT_PROMPT_TYPES or method in CONCEPT_METHODS:
        return "concepts"
    if prompt_type in RATIONALE_PROMPT_TYPES:
        return "rationales"
    if prompt_type in ATTRIBUTION_PROMPT_TYPES or method in ATTRIBUTION_METHODS:
        return "attributions"
    return "unknown"


def family_config(row: pd.Series) -> str:
    method = str(row["method"])
    if row.get("family") == "concepts":
        interp = row.get("interpretation")
        if pd.isna(interp) or str(interp) in {"None", "nan", "<NA>"}:
            return method
        return f"{method} / {interp}"
    return method


def add_family_columns(df: pd.DataFrame, *, raw_col: str = "family_raw") -> pd.DataFrame:
    out = df.copy()
    out[raw_col] = out.apply(infer_family, axis=1)
    out["family"] = out[raw_col]
    out["family_config"] = out.apply(family_config, axis=1)
    return out


def canonicalize_baselines(
    df: pd.DataFrame,
    *,
    row_key_cols: list[str],
    baseline_key_cols: list[str],
    baseline_mask: pd.Series | None = None,
    baseline_prompt_types: set[str] | None = None,
    family_raw_col: str | None = None,
) -> pd.DataFrame:
    """Average duplicated baseline rows and reject duplicated non-baselines."""
    if baseline_mask is None:
        prompt_types = baseline_prompt_types or BASELINE_PROMPT_TYPES
        baseline_mask = (df["method"] == "baseline") | df["prompt_type"].isin(prompt_types)

    baselines = df[baseline_mask].copy()
    non_baselines = df[~baseline_mask].copy()

    duplicates = (
        non_baselines.groupby(row_key_cols, dropna=False)
        .size()
        .reset_index(name="n")
        .query("n > 1")
    )
    if not duplicates.empty:
        raise ValueError(f"Duplicated non-baseline score rows found: {len(duplicates):,} keys")

    if baselines.empty:
        return non_baselines

    before = len(baselines)
    baseline_scores = baselines.groupby(baseline_key_cols, dropna=False)["score"].mean().reset_index()
    collapsed = pd.DataFrame(columns=df.columns)
    for column in baseline_scores.columns:
        collapsed[column] = baseline_scores[column]
    collapsed["method"] = "baseline"
    collapsed["nb_concepts"] = np.nan
    collapsed["interpretation"] = np.nan
    if "specification" in collapsed:
        collapsed["specification"] = collapsed["specification"].fillna("baseline")
    if family_raw_col is not None and family_raw_col in collapsed:
        collapsed[family_raw_col] = "baseline"
    if "family" in collapsed:
        collapsed["family"] = "baseline"
    if "family_config" in collapsed:
        collapsed["family_config"] = "baseline"

    after = len(collapsed)
    if before != after:
        print(f"Averaged duplicated baseline rows: {before:,} -> {after:,}")
    return pd.concat([non_baselines, collapsed], ignore_index=True)


def bucket_stats(
    group: pd.DataFrame,
    *,
    value_col: str = "score",
    seed_col: str = "seed",
    seeds_per_bucket: int = 10,
    expected_seeds: int | None = 50,
) -> pd.Series:
    per_seed = group.groupby(seed_col, as_index=False)[value_col].mean().sort_values(seed_col)
    n_seeds = len(per_seed)
    if expected_seeds is not None and n_seeds != expected_seeds:
        print(f"warning: group={getattr(group, 'name', None)!r} has {n_seeds} seeds, expected {expected_seeds}")
    n_buckets = n_seeds // seeds_per_bucket
    if n_buckets == 0:
        return pd.Series({"mean_score": np.nan, "std_score": np.nan, "n_buckets": 0, "n_seeds": n_seeds})
    vals = per_seed[value_col].to_numpy()[: n_buckets * seeds_per_bucket]
    bucket_means = vals.reshape(n_buckets, seeds_per_bucket).mean(axis=1)
    return pd.Series({
        "mean_score": float(bucket_means.mean()),
        "std_score": float(bucket_means.std(ddof=1)) if n_buckets > 1 else 0.0,
        "n_buckets": int(n_buckets),
        "n_seeds": int(n_seeds),
    })


def grouped_bucket_stats(df: pd.DataFrame, group_cols: list[str], **kwargs: Any) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)
        .apply(lambda g: bucket_stats(g, **kwargs), include_groups=False)
        .reset_index()
        .dropna(subset=["mean_score"])
    )


def common_value_table(
    df: pd.DataFrame,
    *,
    variable_col: str,
    value_a: Any,
    value_b: Any,
    key_cols: list[str],
    value_col: str = "score",
) -> pd.DataFrame:
    small = df[df[variable_col].isin([value_a, value_b])].copy()
    grouped = small.groupby(key_cols + [variable_col], dropna=False)[value_col].mean().reset_index()
    sentinel = "__MISSING_KEY__"
    grouped[key_cols] = grouped[key_cols].fillna(sentinel)
    wide = grouped.pivot_table(index=key_cols, columns=variable_col, values=value_col, aggfunc="mean")
    wide = wide.dropna(subset=[value_a, value_b]).reset_index().replace(sentinel, np.nan)
    wide["diff"] = wide[value_b] - wide[value_a]
    return wide


def bucket_difference_stats(
    wide: pd.DataFrame,
    *,
    value_a: Any,
    value_b: Any,
    group_cols: list[str],
    seed_col: str = "seed",
    seeds_per_bucket: int = 10,
    expected_seeds: int | None = 50,
) -> pd.DataFrame:
    def _one_group(group: pd.DataFrame) -> pd.Series:
        per_seed = group.groupby(seed_col, as_index=False)[[value_a, value_b]].mean().sort_values(seed_col)
        n_seeds = len(per_seed)
        if expected_seeds is not None and n_seeds != expected_seeds:
            print(f"warning: group={group.name!r} has {n_seeds} seeds, expected {expected_seeds}")
        n_buckets = n_seeds // seeds_per_bucket
        if n_buckets == 0:
            return pd.Series({"mean_diff": np.nan, "std_diff": np.nan, "n_buckets": 0, "n_seeds": n_seeds})
        vals_a = per_seed[value_a].to_numpy()[: n_buckets * seeds_per_bucket]
        vals_b = per_seed[value_b].to_numpy()[: n_buckets * seeds_per_bucket]
        diffs = vals_b.reshape(n_buckets, seeds_per_bucket).mean(axis=1) - vals_a.reshape(n_buckets, seeds_per_bucket).mean(axis=1)
        return pd.Series({
            "mean_diff": float(diffs.mean()),
            "std_diff": float(diffs.std(ddof=1)) if n_buckets > 1 else 0.0,
            "n_buckets": int(n_buckets),
            "n_seeds": int(n_seeds),
        })

    return (
        wide.groupby(group_cols, dropna=False)
        .apply(_one_group, include_groups=False)
        .reset_index()
        .sort_values("mean_diff", ascending=False)
        .reset_index(drop=True)
    )


def complete_candidate_configs(
    candidates: pd.DataFrame,
    *,
    prompt_types_by_family: Mapping[str, set[str]] = NON_ANON_PROMPT_TYPES_BY_FAMILY,
    expected_seeds: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates.copy(), pd.DataFrame()

    expected_parts = []
    for family, prompt_types in prompt_types_by_family.items():
        dataset_cells = candidates.loc[candidates["family"] == family, ["dataset", "classes_subset"]].drop_duplicates()
        for prompt_type in sorted(prompt_types):
            part = dataset_cells.copy()
            part["family"] = family
            part["prompt_type"] = prompt_type
            expected_parts.append(part)

    if not expected_parts:
        return candidates.iloc[0:0].copy(), pd.DataFrame()

    expected_cells = pd.concat(expected_parts, ignore_index=True).drop_duplicates()
    configs = candidates[["family", "family_config"]].drop_duplicates()
    expected = configs.merge(expected_cells, on="family", how="inner")
    observed = (
        candidates.dropna(subset=["score"])
        .groupby(["family", "family_config", "dataset", "classes_subset", "prompt_type"], dropna=False)["seed"]
        .nunique()
        .reset_index(name="n_seeds")
    )
    coverage = expected.merge(
        observed,
        on=["family", "family_config", "dataset", "classes_subset", "prompt_type"],
        how="left",
    )
    coverage["n_seeds"] = coverage["n_seeds"].fillna(0).astype(int)
    coverage["complete"] = coverage["n_seeds"] == expected_seeds
    complete_keys = (
        coverage.groupby(["family", "family_config"], dropna=False)["complete"]
        .all()
        .reset_index()
        .query("complete")[["family", "family_config"]]
    )
    return (
        candidates.merge(complete_keys, on=["family", "family_config"], how="inner"),
        coverage[~coverage["complete"]].sort_values(["family", "family_config", "dataset", "classes_subset", "prompt_type"]),
    )


def best_configs_by_family(
    candidates: pd.DataFrame,
    *,
    group_cols: list[str] = ["family", "family_config"],
    seeds_per_bucket: int = 10,
    expected_seeds: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        empty = pd.DataFrame(columns=group_cols + ["mean_score", "std_score", "n_buckets", "n_seeds"])
        return empty, empty
    stats = grouped_bucket_stats(
        candidates,
        group_cols,
        seeds_per_bucket=seeds_per_bucket,
        expected_seeds=expected_seeds,
    ).sort_values("mean_score", ascending=False)
    best = stats.groupby("family", as_index=False, sort=False).head(1).sort_values("family").reset_index(drop=True)
    return best, stats


def add_contender(
    df: pd.DataFrame,
    *,
    parts: list[str],
    contender_col: str = "contender",
    sep: str = " / ",
) -> pd.DataFrame:
    out = df.copy()
    out[contender_col] = out[parts].astype(str).agg(sep.join, axis=1)
    return out
