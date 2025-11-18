# Copyright 2025, Xenia Kuznetsova
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
from typing import List, Tuple


def pairwise_similarity(
    profile_df: pd.DataFrame,
    metric: str = "pearson",
) -> list:
    """
    Compute pairwise similarity between all rows in a DataFrame.

    Parameters
    ----------
    profile_df : pd.DataFrame
        Feature matrix (rows = wells, columns = features).
    metric : str, default "pearson"
        Similarity metric: "pearson", "spearman", or "cosine".

    Returns
    -------
    list of float
        Similarity values for all unique row pairs.
    """
    metric = metric.lower()

    if metric in ("pearson", "spearman"):
        # correlation across rows (wells)
        sim_matrix = profile_df.T.corr(method=metric)
    elif metric == "cosine":
        # cosine similarity across rows
        X = profile_df.to_numpy(dtype=float)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # avoid division by zero
        Xn = X / norms
        sim = Xn @ Xn.T
        sim_matrix = pd.DataFrame(sim, index=profile_df.index, columns=profile_df.index)
    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported: pearson, spearman, cosine."
        )

    # upper triangle (no diagonal)
    ones_matrix = np.ones(sim_matrix.shape)
    upper_tri_mask = np.triu(ones_matrix, k=1).astype(bool)
    upper_triangle = sim_matrix.where(upper_tri_mask)
    return upper_triangle.stack().tolist()


def median_pairwise_similarity(group: pd.DataFrame, metric: str = "pearson") -> float:
    """
    Median of pairwise similarities within a group.
    """
    sims = pairwise_similarity(group, metric=metric)
    if not sims:
        return float("nan")
    return float(np.median(sims))


def group_median_similarity(
    profile_df: pd.DataFrame,
    group_columns: List[str],
    morphology_features: List[str],
    metric: str = "pearson",
) -> pd.DataFrame:
    """
    Median pairwise similarity per group.

    Returns
    -------
    DataFrame with columns: group_columns + ['median_correlation']
    """
    grouped = profile_df.groupby(group_columns, sort=False)[morphology_features]
    median_series = grouped.apply(
        lambda df: median_pairwise_similarity(df, metric=metric)
    )
    group_median = median_series.reset_index(name="median_correlation")
    return group_median


def non_replicate_groups(
    profile_df: pd.DataFrame,
    replicate_group_cols: List[str],
    num_samples: int = 10000,
    group_size: int = None,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, int]:
    """
    Generate non-replicate / non-match groups for null distribution.

    Each sampled group contains 'group_size' rows, each from a different
    replicate group (defined by replicate_group_cols).
    """
    rng = np.random.default_rng(random_state)
    profile_df = profile_df.copy()

    # Build replicate group key
    if len(replicate_group_cols) == 1:
        profile_df["replicate_group_key"] = profile_df[replicate_group_cols[0]]
    else:
        profile_df["replicate_group_key"] = profile_df[replicate_group_cols].agg(
            "_".join, axis=1
        )

    replicate_groups = profile_df["replicate_group_key"].unique()
    group_counts = profile_df["replicate_group_key"].value_counts()

    # Default group size = max group size observed
    if group_size is None:
        group_size = int(group_counts.max())
        print(f"Group size automatically set to: {group_size}")

    if len(replicate_groups) < group_size:
        raise ValueError(
            f"Not enough distinct replicate groups ({len(replicate_groups)}) "
            f"for requested group size {group_size}."
        )

    # Precompute indices for each group key
    group_index_map = {
        key: profile_df.index[profile_df["replicate_group_key"] == key].to_numpy()
        for key in replicate_groups
    }

    data_rows = []

    for combo_id in range(1, num_samples + 1):
        # sample distinct replicate groups for this null group
        sampled_keys = rng.choice(replicate_groups, size=group_size, replace=False)
        for group_key in sampled_keys:
            idxs = group_index_map[group_key]
            chosen_idx = int(rng.integers(0, len(idxs)))
            row = profile_df.loc[idxs[chosen_idx]]
            row_dict = row.to_dict()
            row_dict["non_replicate_group"] = combo_id
            data_rows.append(row_dict)

    result_df = pd.DataFrame(data_rows)

    # put non_replicate_group first
    cols = ["non_replicate_group"] + [
        c for c in result_df.columns if c != "non_replicate_group"
    ]
    result_df = result_df[cols]

    if "replicate_group_key" in result_df.columns:
        result_df = result_df.drop(columns=["replicate_group_key"])

    return result_df, num_samples


def null_distr_similarity(
    non_replicate_groups_df: pd.DataFrame,
    morphology_features: List[str],
    metric: str = "pearson",
) -> pd.DataFrame:
    """
    Null distribution: median similarity per non_replicate_group.
    """
    return group_median_similarity(
        profile_df=non_replicate_groups_df,
        group_columns=["non_replicate_group"],
        morphology_features=morphology_features,
        metric=metric,
    )


def percent_replicating_matching(
    profile_df: pd.DataFrame,
    group_columns: List[str],
    features: List[str],
    threshold_quantile: float = 0.95,
    null_num_samples: int = 10000,
    null_group_size: int = None,
    random_state: int = 0,
    metric: str = "pearson",
) -> Tuple[
    float,
    pd.DataFrame,
    float,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    """
    Calculate percent replicating or percent matching with a configurable similarity metric.

    - group_columns = ['perturbation_id'] for PR
    - group_columns = ['target'] for PM
    """
    # 1) Median similarity per replicate/match group
    median_corr_df = group_median_similarity(
        profile_df, group_columns, features, metric=metric
    )

    # 2) Null: non-replicate / non-match groups
    non_rep_df, num_null = non_replicate_groups(
        profile_df=profile_df,
        replicate_group_cols=group_columns,
        num_samples=null_num_samples,
        group_size=null_group_size,
        random_state=random_state,
    )

    null_df = null_distr_similarity(
        non_replicate_groups_df=non_rep_df,
        morphology_features=features,
        metric=metric,
    )

    # 95th percentile threshold from null distribution
    null_threshold = np.percentile(
        null_df["median_correlation"].dropna(), threshold_quantile * 100
    )

    # Groups above threshold
    above = median_corr_df[median_corr_df["median_correlation"] > null_threshold]
    percent = len(above) / len(median_corr_df) * 100.0

    return percent, above, null_threshold, median_corr_df, null_df, non_rep_df, num_null

