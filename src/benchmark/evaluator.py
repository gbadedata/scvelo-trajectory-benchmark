"""Benchmark evaluator for trajectory inference.

Evaluates RNA velocity trajectory recovery against oracle diffusion
pseudotime using three independent tests:

Task 1 -- Global ordering (Spearman rho):
    Does the inferred velocity pseudotime correlate with the oracle?
    Measured across all cells simultaneously.

Task 2 -- Rank preservation (Mann-Whitney U):
    Are progenitor cells consistently ranked earlier than terminal cells?
    Tests whether the known biological ordering is respected at the
    population level, not just globally. Compares the pseudotime
    distribution of each cell-type pair where the ordering is known.

Task 3 -- Hidden branch recovery:
    Mask a terminal cell type (Epsilon by default) entirely from the
    dataset. Rerun the velocity pipeline on the masked data. Does the
    remaining trajectory still order correctly? This tests whether the
    velocity signal is global (detects transcriptional momentum throughout
    the trajectory) or local (depends on the presence of the terminal state
    to define the direction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryRecoveryResult:
    """Result from global trajectory ordering evaluation (Task 1).

    Attributes:
        spearman_rho: Spearman rank correlation between velocity
            pseudotime and oracle diffusion pseudotime.
        spearman_pvalue: Two-tailed p-value for the correlation.
        n_cells: Number of cells evaluated.
        n_nan_excluded: Cells excluded due to NaN pseudotime.
        passes_threshold: Whether rho exceeds the configured threshold.
    """

    spearman_rho: float
    spearman_pvalue: float
    n_cells: int
    n_nan_excluded: int
    passes_threshold: bool


@dataclass
class RankPreservationResult:
    """Result from pairwise rank preservation evaluation (Task 2).

    Attributes:
        pairwise_results: Dict mapping (earlier_type, later_type) pairs
            to their Mann-Whitney U statistics and p-values.
        n_pairs_tested: Number of cell-type pairs tested.
        n_pairs_passing: Pairs where velocity pseudotime correctly
            ranks the earlier type before the later type.
        pct_pairs_passing: Percentage of pairs passing.
        overall_passes: True if >= 75% of pairs pass.
    """

    pairwise_results: dict[tuple[str, str], dict]
    n_pairs_tested: int
    n_pairs_passing: int
    pct_pairs_passing: float
    overall_passes: bool


@dataclass
class HiddenBranchResult:
    """Result from hidden branch recovery evaluation (Task 3).

    Attributes:
        hidden_type: Cell type masked from the dataset.
        spearman_rho_masked: Spearman rho on masked dataset.
        spearman_rho_full: Spearman rho on full dataset (for comparison).
        rho_drop: Drop in rho due to masking (full - masked).
        n_cells_masked: Number of cells removed.
        n_cells_remaining: Cells in the masked dataset.
        remaining_ordering_preserved: Whether the remaining cell types
            still order correctly without the hidden branch.
    """

    hidden_type: str
    spearman_rho_masked: float
    spearman_rho_full: float
    rho_drop: float
    n_cells_masked: int
    n_cells_remaining: int
    remaining_ordering_preserved: bool


@dataclass
class BenchmarkResult:
    """Container for all three benchmark task results.

    Attributes:
        trajectory_recovery: Task 1 result (global Spearman rho).
        rank_preservation: Task 2 result (pairwise ordering).
        hidden_branch: Task 3 result (branch recovery under masking).
        summary: Human-readable summary dict for JSON serialisation.
    """

    trajectory_recovery: TrajectoryRecoveryResult
    rank_preservation: RankPreservationResult
    hidden_branch: HiddenBranchResult
    summary: dict = field(default_factory=dict)


def evaluate_trajectory_recovery(adata: ad.AnnData) -> TrajectoryRecoveryResult:
    """Task 1: Measure global correlation between velocity and oracle pseudotime.

    Computes Spearman rank correlation between velocity_pseudotime and
    dpt_pseudotime across all cells. NaN values in either column are
    excluded and the count is reported.

    Args:
        adata: AnnData with both dpt_pseudotime and velocity_pseudotime.

    Returns:
        TrajectoryRecoveryResult with Spearman rho and p-value.
    """
    _check_columns(adata, ["dpt_pseudotime", "velocity_pseudotime"])

    dpt = adata.obs["dpt_pseudotime"].values.astype(float)
    vpt = adata.obs["velocity_pseudotime"].values.astype(float)

    valid = ~(np.isnan(dpt) | np.isnan(vpt))
    n_nan = int((~valid).sum())

    rho, pval = spearmanr(dpt[valid], vpt[valid])
    rho = float(rho)
    pval = float(pval)
    passes = rho >= settings.trajectory_rho_threshold

    logger.info(
        "trajectory_recovery: rho=%.4f (p=%.2e), n=%d, nan_excluded=%d, passes=%s",
        rho, pval, valid.sum(), n_nan, passes,
    )
    return TrajectoryRecoveryResult(
        spearman_rho=round(rho, 4),
        spearman_pvalue=pval,
        n_cells=int(valid.sum()),
        n_nan_excluded=n_nan,
        passes_threshold=passes,
    )


def evaluate_rank_preservation(adata: ad.AnnData) -> RankPreservationResult:
    """Task 2: Test whether known cell-type ordering is preserved.

    Tests two components of the trajectory:

    A) Linear progenitor axis: consecutive pairs in known_ordering
       (Ductal -> Ngn3 low EP -> Ngn3 high EP -> Pre-endocrine -> Alpha)

    B) Terminal fan-out: each terminal fate vs Pre-endocrine from
       terminal_ordering_pairs. The four terminal cell types are parallel
       branches -- no ordering among them is biologically defined, so
       they are only tested against their shared progenitor (Pre-endocrine).

    Args:
        adata: AnnData with clusters and velocity_pseudotime.

    Returns:
        RankPreservationResult with per-pair statistics.
    """
    _check_columns(adata, ["clusters", "velocity_pseudotime"])

    vpt = adata.obs["velocity_pseudotime"].values.astype(float)
    clusters = adata.obs["clusters"].astype(str)
    available = set(clusters.unique())

    # Build all pairs to test
    ordering = settings.known_ordering
    consecutive_pairs = [
        (ordering[i], ordering[i + 1])
        for i in range(len(ordering) - 1)
    ]
    terminal_pairs = [
        (pair[0], pair[1])
        for pair in settings.terminal_ordering_pairs
    ]
    # Deduplicate and filter to available cell types
    all_pairs = list(dict.fromkeys(consecutive_pairs + terminal_pairs))
    all_pairs = [(a, b) for a, b in all_pairs if a in available and b in available]

    pairwise_results: dict[tuple[str, str], dict] = {}
    n_passing = 0

    for earlier, later in all_pairs:
        early_vpt = vpt[clusters == earlier]
        late_vpt = vpt[clusters == later]

        early_vpt = early_vpt[~np.isnan(early_vpt)]
        late_vpt = late_vpt[~np.isnan(late_vpt)]

        if len(early_vpt) == 0 or len(late_vpt) == 0:
            continue

        stat, pval = mannwhitneyu(early_vpt, late_vpt, alternative="less")
        early_mean = float(early_vpt.mean())
        late_mean = float(late_vpt.mean())
        direction_correct = early_mean < late_mean

        pairwise_results[(earlier, later)] = {
            "early_mean_vpt": round(early_mean, 4),
            "late_mean_vpt": round(late_mean, 4),
            "direction_correct": direction_correct,
            "mannwhitney_stat": round(float(stat), 2),
            "pvalue": float(pval),
            "significant": pval < 0.05,
        }

        if direction_correct and pval < 0.05:
            n_passing += 1

    n_tested = len(pairwise_results)
    pct_passing = round(n_passing / n_tested * 100, 1) if n_tested > 0 else 0.0
    overall = pct_passing >= 75.0

    logger.info(
        "rank_preservation: %d/%d pairs pass (%.1f%%), overall_passes=%s",
        n_passing, n_tested, pct_passing, overall,
    )
    return RankPreservationResult(
        pairwise_results=pairwise_results,
        n_pairs_tested=n_tested,
        n_pairs_passing=n_passing,
        pct_pairs_passing=pct_passing,
        overall_passes=overall,
    )


def evaluate_hidden_branch(
    adata: ad.AnnData,
    run_velocity_fn: "callable",
) -> HiddenBranchResult:
    """Task 3: Test trajectory recovery with a terminal branch masked.

    Removes all cells of the configured hidden_branch_type from the
    dataset, re-runs the velocity pipeline on the masked data, and
    evaluates whether the remaining trajectory is still correctly ordered.

    This tests whether RNA velocity is detecting global transcriptional
    momentum (robust to missing branches) or relying on the presence of
    the terminal state to define directionality.

    Args:
        adata: Full AnnData with velocity and pseudotime computed.
        run_velocity_fn: Callable that takes an AnnData and returns
            (AnnData, VelocityReport) -- the velocity pipeline function.

    Returns:
        HiddenBranchResult with masked vs full dataset comparison.
    """
    _check_columns(adata, ["clusters", "velocity_pseudotime", "dpt_pseudotime"])

    hidden_type = settings.hidden_branch_type
    clusters = adata.obs["clusters"].astype(str)
    mask = clusters != hidden_type
    n_masked = int((~mask).sum())

    logger.info(
        "hidden_branch_masking: removing %d %s cells (%d remaining)",
        n_masked, hidden_type, mask.sum(),
    )

    if n_masked == 0:
        logger.warning("hidden branch type '%s' not found; skipping", hidden_type)
        dpt = adata.obs["dpt_pseudotime"].values.astype(float)
        vpt = adata.obs["velocity_pseudotime"].values.astype(float)
        valid = ~(np.isnan(dpt) | np.isnan(vpt))
        rho_full, _ = spearmanr(dpt[valid], vpt[valid])
        return HiddenBranchResult(
            hidden_type=hidden_type,
            spearman_rho_masked=float(rho_full),
            spearman_rho_full=float(rho_full),
            rho_drop=0.0,
            n_cells_masked=0,
            n_cells_remaining=int(adata.n_obs),
            remaining_ordering_preserved=True,
        )

    # Compute rho on full dataset
    dpt_full = adata.obs["dpt_pseudotime"].values.astype(float)
    vpt_full = adata.obs["velocity_pseudotime"].values.astype(float)
    valid_full = ~(np.isnan(dpt_full) | np.isnan(vpt_full))
    rho_full, _ = spearmanr(dpt_full[valid_full], vpt_full[valid_full])

    # Create masked dataset
    adata_masked = adata[mask, :].copy()

    # Re-run velocity on masked dataset
    logger.info("hidden_branch: running velocity on masked dataset")
    adata_masked, _ = run_velocity_fn(adata_masked)

    # Evaluate masked trajectory
    dpt_masked = adata_masked.obs["dpt_pseudotime"].values.astype(float)
    vpt_masked = adata_masked.obs["velocity_pseudotime"].values.astype(float)
    valid_masked = ~(np.isnan(dpt_masked) | np.isnan(vpt_masked))
    rho_masked, _ = spearmanr(dpt_masked[valid_masked], vpt_masked[valid_masked])
    rho_masked = float(rho_masked)
    rho_drop = round(float(rho_full) - rho_masked, 4)

    # Check remaining ordering on the LINEAR progenitor axis only.
    # The terminal fates are parallel branches with no defined ordering
    # among themselves -- only test that Pre-endocrine precedes each terminal.
    linear_axis = [t for t in settings.known_ordering if t != hidden_type]
    clusters_masked = adata_masked.obs["clusters"].astype(str)
    vpt_m = adata_masked.obs["velocity_pseudotime"].values.astype(float)

    ordering_preserved = True
    for i in range(len(linear_axis) - 1):
        a, b = linear_axis[i], linear_axis[i + 1]
        if a not in clusters_masked.values or b not in clusters_masked.values:
            continue
        a_vals = vpt_m[clusters_masked == a]
        b_vals = vpt_m[clusters_masked == b]
        a_mean = float(a_vals[~np.isnan(a_vals)].mean())
        b_mean = float(b_vals[~np.isnan(b_vals)].mean())
        if a_mean >= b_mean:
            ordering_preserved = False
            logger.warning(
                "ordering violated: %s (%.3f) >= %s (%.3f) after masking",
                a, a_mean, b, b_mean,
            )

    logger.info(
        "hidden_branch: rho_full=%.4f, rho_masked=%.4f, drop=%.4f, "
        "ordering_preserved=%s",
        rho_full, rho_masked, rho_drop, ordering_preserved,
    )
    return HiddenBranchResult(
        hidden_type=hidden_type,
        spearman_rho_masked=round(rho_masked, 4),
        spearman_rho_full=round(float(rho_full), 4),
        rho_drop=rho_drop,
        n_cells_masked=n_masked,
        n_cells_remaining=int(mask.sum()),
        remaining_ordering_preserved=ordering_preserved,
    )


def run_benchmark(
    adata: ad.AnnData,
    run_velocity_fn: "callable",
) -> BenchmarkResult:
    """Run all three benchmark tasks and compile results.

    Args:
        adata: AnnData with velocity and oracle pseudotime computed.
        run_velocity_fn: Velocity pipeline function for Task 3.

    Returns:
        BenchmarkResult with all task results and a summary dict.
    """
    task1 = evaluate_trajectory_recovery(adata)
    task2 = evaluate_rank_preservation(adata)
    task3 = evaluate_hidden_branch(adata, run_velocity_fn)

    summary = {
        "task1_trajectory_recovery": {
            "spearman_rho": task1.spearman_rho,
            "pvalue": task1.spearman_pvalue,
            "n_cells": task1.n_cells,
            "passes": task1.passes_threshold,
        },
        "task2_rank_preservation": {
            "pairs_passing": f"{task2.n_pairs_passing}/{task2.n_pairs_tested}",
            "pct_passing": task2.pct_pairs_passing,
            "passes": task2.overall_passes,
        },
        "task3_hidden_branch": {
            "hidden_type": task3.hidden_type,
            "rho_full": task3.spearman_rho_full,
            "rho_masked": task3.spearman_rho_masked,
            "rho_drop": task3.rho_drop,
            "ordering_preserved": task3.remaining_ordering_preserved,
        },
    }

    logger.info(
        "benchmark_complete: rho=%.4f, rank_pct=%.1f%%, "
        "hidden_rho=%.4f (drop=%.4f)",
        task1.spearman_rho,
        task2.pct_pairs_passing,
        task3.spearman_rho_masked,
        task3.rho_drop,
    )
    return BenchmarkResult(
        trajectory_recovery=task1,
        rank_preservation=task2,
        hidden_branch=task3,
        summary=summary,
    )


def _check_columns(adata: ad.AnnData, required: list[str]) -> None:
    """Raise ValueError if any required column is missing from obs."""
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        raise ValueError(
            f"Missing required obs columns: {missing}. "
            "Run the full pipeline before benchmarking."
        )
