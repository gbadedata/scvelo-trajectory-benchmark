"""Biological constraint validators for trajectory inference.

These validators check necessary conditions for correct trajectory
inference that can be verified from expression data and cell-type
annotations, without needing the oracle pseudotime.

They are complementary to the benchmark evaluator: the evaluator
measures accuracy against the oracle; validators check biological
plausibility independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import anndata as ad
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a single biological constraint validator.

    Attributes:
        name: Validator name.
        passed: Whether the constraint was satisfied.
        score: Numeric score in [0, 1].
        details: Human-readable explanation.
        evidence: Supporting data.
    """

    name: str
    passed: bool
    score: float
    details: str
    evidence: dict = field(default_factory=dict)


def validate_root_cell_ordering(adata: ad.AnnData) -> ValidationResult:
    """Check that root cells have lower velocity pseudotime than terminal cells.

    This is a necessary condition for correct trajectory orientation.
    If Ductal cells do not have lower pseudotime than Alpha/Beta/Delta/Epsilon,
    the velocity direction is inverted.

    Args:
        adata: AnnData with clusters and velocity_pseudotime.

    Returns:
        ValidationResult with ordering check result.
    """
    if "velocity_pseudotime" not in adata.obs.columns:
        return ValidationResult(
            name="root_cell_ordering",
            passed=False,
            score=0.0,
            details="velocity_pseudotime not found in obs",
        )

    vpt = adata.obs["velocity_pseudotime"].values.astype(float)
    clusters = adata.obs["clusters"].astype(str)

    root_type = settings.root_cell_type
    terminal_types = settings.terminal_cell_types

    if root_type not in clusters.values:
        return ValidationResult(
            name="root_cell_ordering",
            passed=False,
            score=0.0,
            details=f"Root type '{root_type}' not found in dataset",
        )

    root_vpt = vpt[clusters == root_type]
    root_vpt = root_vpt[~np.isnan(root_vpt)]
    root_mean = float(root_vpt.mean()) if len(root_vpt) > 0 else np.nan

    terminal_means = {}
    n_correct = 0
    for tt in terminal_types:
        if tt not in clusters.values:
            continue
        t_vpt = vpt[clusters == tt]
        t_vpt = t_vpt[~np.isnan(t_vpt)]
        if len(t_vpt) == 0:
            continue
        t_mean = float(t_vpt.mean())
        terminal_means[tt] = t_mean
        if root_mean < t_mean:
            n_correct += 1

    n_checked = len(terminal_means)
    score = n_correct / n_checked if n_checked > 0 else 0.0
    passed = score >= 0.75

    details = (
        f"Root type '{root_type}' (mean_vpt={root_mean:.3f}) has lower "
        f"velocity pseudotime than {n_correct}/{n_checked} terminal types."
    )

    logger.info("validate_root_cell_ordering: score=%.4f, passed=%s", score, passed)
    return ValidationResult(
        name="root_cell_ordering",
        passed=passed,
        score=round(score, 4),
        details=details,
        evidence={
            "root_mean_vpt": root_mean,
            "terminal_means": terminal_means,
        },
    )


def validate_velocity_gene_coverage(adata: ad.AnnData) -> ValidationResult:
    """Check that a reasonable number of velocity genes were identified.

    A very low velocity gene count (<100) indicates the model could not
    find reliable splicing dynamics. A count of 0 means the pipeline
    failed silently.

    Args:
        adata: AnnData with velocity_genes in var.

    Returns:
        ValidationResult with velocity gene count check.
    """
    if "velocity_genes" not in adata.var.columns:
        return ValidationResult(
            name="velocity_gene_coverage",
            passed=False,
            score=0.0,
            details="velocity_genes column not found in var",
        )

    n_velocity_genes = int(adata.var["velocity_genes"].sum())
    n_total_genes = adata.n_vars
    pct = n_velocity_genes / n_total_genes * 100

    # Expect at least 5% of genes to be velocity genes
    score = min(1.0, pct / 20.0)  # normalised to 20% as ideal
    passed = n_velocity_genes >= 100

    details = (
        f"{n_velocity_genes}/{n_total_genes} genes ({pct:.1f}%) "
        f"identified as velocity genes."
    )
    if not passed:
        details += " WARNING: Very few velocity genes -- consider checking preprocessing."

    logger.info(
        "validate_velocity_gene_coverage: n=%d (%.1f%%), passed=%s",
        n_velocity_genes, pct, passed,
    )
    return ValidationResult(
        name="velocity_gene_coverage",
        passed=passed,
        score=round(score, 4),
        details=details,
        evidence={
            "n_velocity_genes": n_velocity_genes,
            "n_total_genes": n_total_genes,
            "pct": round(pct, 2),
        },
    )


def validate_velocity_confidence(adata: ad.AnnData) -> ValidationResult:
    """Check that mean velocity confidence is above a meaningful threshold.

    Velocity confidence measures consistency of velocity direction across
    neighbouring cells. Mean confidence below 0.4 indicates very noisy
    or inconsistent velocity estimates, which would undermine trajectory
    inference.

    Args:
        adata: AnnData with velocity_confidence in obs.

    Returns:
        ValidationResult with confidence statistics.
    """
    if "velocity_confidence" not in adata.obs.columns:
        return ValidationResult(
            name="velocity_confidence",
            passed=False,
            score=0.0,
            details="velocity_confidence not found in obs",
        )

    conf = adata.obs["velocity_confidence"].values.astype(float)
    valid = conf[~np.isnan(conf)]

    if len(valid) == 0:
        return ValidationResult(
            name="velocity_confidence",
            passed=False,
            score=0.0,
            details="All velocity_confidence values are NaN",
        )

    mean_conf = float(valid.mean())
    pct_high = float((valid > 0.75).mean() * 100)
    score = float(np.clip(mean_conf / 0.75, 0, 1))
    passed = mean_conf >= 0.4

    details = (
        f"Mean velocity confidence: {mean_conf:.4f}. "
        f"Cells with confidence > 0.75: {pct_high:.1f}%."
    )

    logger.info(
        "validate_velocity_confidence: mean=%.4f, pct_high=%.1f%%, passed=%s",
        mean_conf, pct_high, passed,
    )
    return ValidationResult(
        name="velocity_confidence",
        passed=passed,
        score=round(score, 4),
        details=details,
        evidence={
            "mean_confidence": round(mean_conf, 4),
            "pct_high_confidence": round(pct_high, 2),
        },
    )


def run_all_validators(adata: ad.AnnData) -> list[ValidationResult]:
    """Run all biological constraint validators.

    Args:
        adata: AnnData with velocity computation complete.

    Returns:
        List of ValidationResult, one per validator.
    """
    results = [
        validate_root_cell_ordering(adata),
        validate_velocity_gene_coverage(adata),
        validate_velocity_confidence(adata),
    ]
    n_passed = sum(r.passed for r in results)
    logger.info("all_validators_complete: %d/%d passed", n_passed, len(results))
    return results
