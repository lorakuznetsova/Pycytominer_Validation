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

import logging
import warnings
import sys
import pathlib
import datetime
import pandas as pd
import argparse

from stat_helpers_metrics import percent_replicating_matching
from graph_helpers import plot_kde_with_threshold, plot_bar_with_threshold

# -------------------------------------------------------------------------
# Logging and warnings setup
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)
except Exception:
    pass


def main(profile_file: str, metric: str = "pearson") -> None:
    """
    Profile quality evaluation with configurable similarity metric.

    Assumes the CSV contains per-well, already post-processed embeddings
    (e.g., CellPaintSSL-style). Steps:

    - Identify emb* feature columns and minimal metadata.
    - Remove DMSO/control wells (no normalization).
    - Percent replicating (by perturbation_id).
    - Percent matching (by target), using only targets with >1 compound.
    - Null distributions built from 10,000 non-replicate / non-match groups.
    - Similarity metric = pearson / spearman / cosine.
    """
    metric = metric.lower()
    if metric not in ("pearson", "spearman", "cosine"):
        logger.error(
            f"Unsupported metric '{metric}'. Use one of: pearson, spearman, cosine."
        )
        sys.exit(1)

    logger.info("SCRIPT START: Profile Quality Evaluation Pipeline")
    logger.info(f"Similarity metric: {metric}")

    profile_path = pathlib.Path(profile_file)
    if not profile_path.is_file():
        logger.error(
            f"Invalid profile file path: '{profile_file}'. Please provide a valid CSV file."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info(f"Loading profiles from {profile_path}...")
    profile_df = pd.read_csv(profile_path)
    logger.info(
        f"Loaded {profile_df.shape[0]} rows and {profile_df.shape[1]} columns"
    )

    # ------------------------------------------------------------------
    # Output folders
    # ------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{metric}"  # e.g. 20251116_164533_pearson

    output_dir = pathlib.Path.cwd() / "reports" / run_id
    new_output_dir = profile_path.parent / "reports" / run_id

    for d in [output_dir, new_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    summary_report = output_dir / "summary_report.txt"
    new_summary_report = new_output_dir / "summary_report.txt"

    # ------------------------------------------------------------------
    # Identify features and metadata
    # ------------------------------------------------------------------
    morphology_features = [col for col in profile_df.columns if col.startswith("emb")]
    meta_features = ["batch", "plate", "well", "perturbation_id", "target"]

    logger.info(f"Morphology features identified: {len(morphology_features)}")
    logger.info(f"Metadata features identified: {len(meta_features)}")

    (output_dir / "morphology_features.txt").write_text(
        "\n".join(morphology_features)
    )
    (new_output_dir / "morphology_features.txt").write_text(
        "\n".join(morphology_features)
    )
    (output_dir / "metadata_features.txt").write_text("\n".join(meta_features))
    (new_output_dir / "metadata_features.txt").write_text("\n".join(meta_features))

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    profile_df = profile_df.dropna(subset=["perturbation_id", "target"])
    profile_df["perturbation_id"] = (
        profile_df["perturbation_id"].astype(str).str.strip().str.lower()
    )
    profile_df["target"] = profile_df["target"].astype(str).str.strip()

    # ------------------------------------------------------------------
    # Negative controls (no normalization)
    # ------------------------------------------------------------------
    neg_con_col = "perturbation_id"
    neg_con_val = "dmso"
    target_con_val = "control"

    logger.info(
        f"Negative control set: {neg_con_col}='{neg_con_val}' and target='{target_con_val}'"
    )

    neg_controls = profile_df.query(
        f'{neg_con_col} == "{neg_con_val}" and target == "{target_con_val}"'
    )
    logger.info(f"Found {neg_controls.shape[0]} DMSO/control rows")

    profile_df = profile_df[profile_df[neg_con_col] != neg_con_val]
    logger.info(
        f"Removed negative controls; remaining {profile_df.shape[0]} rows and {profile_df.shape[1]} columns"
    )

    profile_df.to_csv(output_dir / "filtered_profiles.csv", index=False)
    profile_df.to_csv(new_output_dir / "filtered_profiles.csv", index=False)

    # ------------------------------------------------------------------
    # STEP 2: Percent replicating (by perturbation_id)
    # ------------------------------------------------------------------
    logger.info("Calculating percent replicating")
    replicate_groups = ["perturbation_id"]

    rep_stats = percent_replicating_matching(
        profile_df=profile_df,
        group_columns=replicate_groups,
        features=morphology_features,
        threshold_quantile=0.95,
        null_num_samples=10000,
        null_group_size=None,
        random_state=0,
        metric=metric,
    )
    (
        percent_replicating,
        replicating_groups_df,
        replicate_threshold,
        replicate_median_corr_df,
        replicate_null_df,
        _,
        _,
    ) = rep_stats

    logger.info(f"Percent replicating: {percent_replicating:.2f}%")
    with open(summary_report, "a") as f:
        f.write(f"Percent replicating ({metric}): {percent_replicating:.2f}%\n")
    with open(new_summary_report, "a") as f:
        f.write(f"Percent replicating ({metric}): {percent_replicating:.2f}%\n")

    replicating_groups_df.to_csv(
        output_dir / "replicating_groups.csv",
        index=False,
        float_format="%.10f",
    )
    replicating_groups_df.to_csv(
        new_output_dir / "replicating_groups.csv",
        index=False,
        float_format="%.10f",
    )

    # ------------------------------------------------------------------
    # STEP 3: Percent matching (by target, only targets with >1 compound)
    # ------------------------------------------------------------------
    logger.info("Calculating percent matching")

    target_comp_counts = profile_df.groupby("target")["perturbation_id"].nunique()
    total_targets = target_comp_counts.shape[0]
    valid_targets = target_comp_counts[target_comp_counts > 1].index.tolist()

    if len(valid_targets) == 0:
        logger.warning(
            "No targets with > 1 compound found. Percent matching will be calculated using ALL targets."
        )
        match_df = profile_df.copy()
    else:
        match_df = profile_df[profile_df["target"].isin(valid_targets)].copy()
        logger.info(
            f"Percent matching will use {len(valid_targets)} targets with > 1 compound "
            f"(out of {total_targets} total targets)"
        )

    match_groups = ["target"]

    match_stats = percent_replicating_matching(
        profile_df=match_df,
        group_columns=match_groups,
        features=morphology_features,
        threshold_quantile=0.95,
        null_num_samples=10000,
        null_group_size=None,
        random_state=1,
        metric=metric,
    )
    (
        percent_matching,
        matching_groups_df,
        match_threshold,
        match_median_corr_df,
        match_null_df,
        _,
        _,
    ) = match_stats

    logger.info(f"Percent matching: {percent_matching:.2f}%")
    with open(summary_report, "a") as f:
        f.write(f"Percent matching ({metric}): {percent_matching:.2f}%\n")
    with open(new_summary_report, "a") as f:
        f.write(f"Percent matching ({metric}): {percent_matching:.2f}%\n")

    matching_groups_df.to_csv(
        output_dir / "matching_groups.csv",
        index=False,
        float_format="%.10f",
    )
    matching_groups_df.to_csv(
        new_output_dir / "matching_groups.csv",
        index=False,
        float_format="%.10f",
    )

    # ------------------------------------------------------------------
    # STEP 4: Visualizations
    # ------------------------------------------------------------------
    logger.info("Generating visualizations")

    # Replicate null distribution
    plot_kde_with_threshold(
        data=replicate_null_df["median_correlation"],
        threshold=replicate_threshold,
        output_file=output_dir / "replicate_null_distribution_plot.png",
        title=f"Null Distribution (Non-Replicates), metric={metric}",
        xlabel="Median Pairwise Similarity",
        ylabel="Density",
    )
    plot_kde_with_threshold(
        data=replicate_null_df["median_correlation"],
        threshold=replicate_threshold,
        output_file=new_output_dir / "replicate_null_distribution_plot.png",
        title=f"Null Distribution (Non-Replicates), metric={metric}",
        xlabel="Median Pairwise Similarity",
        ylabel="Density",
    )

    # Replicate correlation bar plot
    replicate_median_corr_df["replicate_group"] = replicate_median_corr_df[
        replicate_groups
    ].agg("_".join, axis=1)
    plot_bar_with_threshold(
        data=replicate_median_corr_df,
        group_col="replicate_group",
        value_col="median_correlation",
        threshold=replicate_threshold,
        output_file=output_dir / "replicate_correlation_plot.png",
        title=f"Median Similarity per Replicate Group (metric={metric})",
        xlabel="Replicate Groups",
        ylabel="Median Similarity",
    )
    plot_bar_with_threshold(
        data=replicate_median_corr_df,
        group_col="replicate_group",
        value_col="median_correlation",
        threshold=replicate_threshold,
        output_file=new_output_dir / "replicate_correlation_plot.png",
        title=f"Median Similarity per Replicate Group (metric={metric})",
        xlabel="Replicate Groups",
        ylabel="Median Similarity",
    )

    # Match null distribution
    plot_kde_with_threshold(
        data=match_null_df["median_correlation"],
        threshold=match_threshold,
        output_file=output_dir / "match_null_distribution_plot.png",
        title=f"Null Distribution (Non-Matches), metric={metric}",
        xlabel="Median Pairwise Similarity",
        ylabel="Density",
    )
    plot_kde_with_threshold(
        data=match_null_df["median_correlation"],
        threshold=match_threshold,
        output_file=new_output_dir / "match_null_distribution_plot.png",
        title=f"Null Distribution (Non-Matches), metric={metric}",
        xlabel="Median Pairwise Similarity",
        ylabel="Density",
    )

    # Match correlation bar plot
    match_median_corr_df["match_group"] = match_median_corr_df["target"]
    plot_bar_with_threshold(
        data=match_median_corr_df,
        group_col="match_group",
        value_col="median_correlation",
        threshold=match_threshold,
        output_file=output_dir / "match_correlation_plot.png",
        title=f"Median Similarity per Match Group (metric={metric})",
        xlabel="Match Groups",
        ylabel="Median Similarity",
    )
    plot_bar_with_threshold(
        data=match_median_corr_df,
        group_col="match_group",
        value_col="median_correlation",
        threshold=match_threshold,
        output_file=new_output_dir / "match_correlation_plot.png",
        title=f"Median Similarity per Match Group (metric={metric})",
        xlabel="Match Groups",
        ylabel="Median Similarity",
    )

    logger.info("END SCRIPT: Profile quality evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Profile quality evaluation with configurable similarity metric "
            "(assumes per-well embeddings are already post-processed)."
        )
    )
    parser.add_argument(
        "profile_file",
        type=str,
        help="Path to the input CSV file with embeddings and metadata",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="pearson",
        choices=["pearson", "spearman", "cosine"],
        help="Similarity metric to use (default: pearson)",
    )
    args = parser.parse_args()
    main(args.profile_file, metric=args.metric)

