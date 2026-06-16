"""Tests for biological constraint validators."""

from __future__ import annotations

import anndata as ad
import pytest

from src.benchmark.validators import (
    ValidationResult,
    run_all_validators,
    validate_root_cell_ordering,
    validate_velocity_confidence,
    validate_velocity_gene_coverage,
)


@pytest.fixture
def velocity_adata(synthetic_adata_with_velocity: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData with velocity columns and cluster labels."""
    import numpy as np
    adata = synthetic_adata_with_velocity.copy()
    adata.obs["clusters"] = adata.obs["cell_type_ground_truth"].copy()
    adata.var["velocity_genes"] = True
    # Add velocity_confidence (mean ~0.5 to pass the >0.4 threshold)
    rng = np.random.default_rng(42)
    adata.obs["velocity_confidence"] = rng.uniform(0.4, 0.8, adata.n_obs)
    return adata


class TestValidateRootCellOrdering:

    def test_returns_validation_result(self, velocity_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.validators.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.terminal_cell_types = ["Terminal"]
            result = validate_root_cell_ordering(velocity_adata)
        assert isinstance(result, ValidationResult)
        assert result.name == "root_cell_ordering"

    def test_score_in_unit_interval(self, velocity_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.validators.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.terminal_cell_types = ["Terminal"]
            result = validate_root_cell_ordering(velocity_adata)
        assert 0.0 <= result.score <= 1.0

    def test_correct_ordering_passes(self, velocity_adata: ad.AnnData) -> None:
        """Progenitor cells have lower pseudotime than Terminal -- should pass."""
        from unittest.mock import patch
        with patch("src.benchmark.validators.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.terminal_cell_types = ["Terminal"]
            result = validate_root_cell_ordering(velocity_adata)
        assert result.passed is True

    def test_missing_vpt_returns_failed(self, velocity_adata: ad.AnnData) -> None:
        adata = velocity_adata.copy()
        adata.obs = adata.obs.drop(columns=["velocity_pseudotime"])
        from unittest.mock import patch
        with patch("src.benchmark.validators.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.terminal_cell_types = ["Terminal"]
            result = validate_root_cell_ordering(adata)
        assert result.passed is False


class TestValidateVelocityGeneCoverage:

    def test_returns_validation_result(self, velocity_adata: ad.AnnData) -> None:
        result = validate_velocity_gene_coverage(velocity_adata)
        assert isinstance(result, ValidationResult)
        assert result.name == "velocity_gene_coverage"

    def test_passes_with_many_genes(self, velocity_adata: ad.AnnData) -> None:
        """All 100 genes flagged as velocity genes should pass."""
        result = validate_velocity_gene_coverage(velocity_adata)
        assert result.passed is True

    def test_fails_with_few_genes(self, velocity_adata: ad.AnnData) -> None:
        """Fewer than 100 velocity genes should fail."""
        adata = velocity_adata.copy()
        adata.var["velocity_genes"] = False
        adata.var.iloc[:10, adata.var.columns.get_loc("velocity_genes")] = True
        result = validate_velocity_gene_coverage(adata)
        assert result.passed is False

    def test_evidence_has_counts(self, velocity_adata: ad.AnnData) -> None:
        result = validate_velocity_gene_coverage(velocity_adata)
        assert "n_velocity_genes" in result.evidence
        assert "n_total_genes" in result.evidence


class TestValidateVelocityConfidence:

    def test_returns_validation_result(self, velocity_adata: ad.AnnData) -> None:
        result = validate_velocity_confidence(velocity_adata)
        assert isinstance(result, ValidationResult)
        assert result.name == "velocity_confidence"

    def test_passes_with_high_confidence(self, velocity_adata: ad.AnnData) -> None:
        """Mean confidence of ~0.5 (from noisy synthetic) should pass threshold."""
        result = validate_velocity_confidence(velocity_adata)
        assert result.passed is True

    def test_fails_with_low_confidence(self, velocity_adata: ad.AnnData) -> None:
        """Mean confidence below 0.4 should fail."""
        adata = velocity_adata.copy()
        adata.obs["velocity_confidence"] = 0.1
        result = validate_velocity_confidence(adata)
        assert result.passed is False

    def test_missing_column_returns_failed(self, velocity_adata: ad.AnnData) -> None:
        import pandas as pd
        adata = velocity_adata.copy()
        # Create new obs without velocity_confidence
        adata.obs = pd.DataFrame(
            {col: adata.obs[col] for col in adata.obs.columns
             if col != "velocity_confidence"},
            index=adata.obs.index,
        )
        result = validate_velocity_confidence(adata)
        assert result.passed is False


class TestRunAllValidators:

    def test_returns_three_results(self, velocity_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.validators.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.terminal_cell_types = ["Terminal"]
            results = run_all_validators(velocity_adata)
        assert len(results) == 3

    def test_all_are_validation_results(self, velocity_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.validators.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.terminal_cell_types = ["Terminal"]
            results = run_all_validators(velocity_adata)
        assert all(isinstance(r, ValidationResult) for r in results)

    def test_distinct_names(self, velocity_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.validators.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.terminal_cell_types = ["Terminal"]
            results = run_all_validators(velocity_adata)
        names = [r.name for r in results]
        assert len(names) == len(set(names))
