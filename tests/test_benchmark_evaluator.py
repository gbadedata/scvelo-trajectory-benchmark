"""Tests for benchmark evaluator module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from src.benchmark.evaluator import (
    BenchmarkResult,
    RankPreservationResult,
    TrajectoryRecoveryResult,
    _check_columns,
    evaluate_rank_preservation,
    evaluate_trajectory_recovery,
    run_benchmark,
)


@pytest.fixture
def full_benchmark_adata(synthetic_adata_with_velocity: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData with oracle and velocity pseudotime for benchmarking."""
    adata = synthetic_adata_with_velocity.copy()
    # Rename to match real pipeline column names
    adata.obs["clusters"] = adata.obs["cell_type_ground_truth"].copy()
    # velocity_pseudotime is already set in the fixture
    return adata


class TestEvaluateTrajectoryRecovery:
    """Test Task 1: global Spearman correlation."""

    def test_returns_trajectory_result(
        self, full_benchmark_adata: ad.AnnData
    ) -> None:
        result = evaluate_trajectory_recovery(full_benchmark_adata)
        assert isinstance(result, TrajectoryRecoveryResult)

    def test_rho_in_range(self, full_benchmark_adata: ad.AnnData) -> None:
        result = evaluate_trajectory_recovery(full_benchmark_adata)
        assert -1.0 <= result.spearman_rho <= 1.0

    def test_high_correlation_on_clean_signal(
        self, full_benchmark_adata: ad.AnnData
    ) -> None:
        """Synthetic fixture has velocity_pseudotime that closely tracks
        dpt_pseudotime (rho > 0.8 embedded in the fixture)."""
        result = evaluate_trajectory_recovery(full_benchmark_adata)
        assert result.spearman_rho > 0.5

    def test_n_cells_correct(self, full_benchmark_adata: ad.AnnData) -> None:
        result = evaluate_trajectory_recovery(full_benchmark_adata)
        assert result.n_cells <= full_benchmark_adata.n_obs

    def test_missing_column_raises(
        self, full_benchmark_adata: ad.AnnData
    ) -> None:
        adata = full_benchmark_adata.copy()
        adata.obs = adata.obs.drop(columns=["velocity_pseudotime"])
        with pytest.raises(ValueError, match="Missing required obs columns"):
            evaluate_trajectory_recovery(adata)

    def test_passes_threshold_with_high_rho(
        self, full_benchmark_adata: ad.AnnData
    ) -> None:
        result = evaluate_trajectory_recovery(full_benchmark_adata)
        # Fixture has rho > 0.8 which exceeds threshold of 0.5
        assert result.passes_threshold is True


class TestEvaluateRankPreservation:
    """Test Task 2: pairwise Mann-Whitney ordering."""

    def test_returns_rank_result(self, full_benchmark_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.evaluator.settings") as mock:
            mock.known_ordering = ["Progenitor", "Early", "Intermediate", "Terminal"]
            mock.terminal_ordering_pairs = [["Intermediate", "Terminal"]]
            mock.hidden_branch_type = "Terminal"
            mock.trajectory_rho_threshold = 0.5
            result = evaluate_rank_preservation(full_benchmark_adata)
        assert isinstance(result, RankPreservationResult)

    def test_pairs_tested_positive(self, full_benchmark_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.evaluator.settings") as mock:
            mock.known_ordering = ["Progenitor", "Early", "Intermediate", "Terminal"]
            mock.terminal_ordering_pairs = [["Intermediate", "Terminal"]]
            mock.hidden_branch_type = "Terminal"
            mock.trajectory_rho_threshold = 0.5
            result = evaluate_rank_preservation(full_benchmark_adata)
        assert result.n_pairs_tested > 0

    def test_pct_in_range(self, full_benchmark_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.evaluator.settings") as mock:
            mock.known_ordering = ["Progenitor", "Early", "Intermediate", "Terminal"]
            mock.terminal_ordering_pairs = [["Intermediate", "Terminal"]]
            mock.hidden_branch_type = "Terminal"
            mock.trajectory_rho_threshold = 0.5
            result = evaluate_rank_preservation(full_benchmark_adata)
        assert 0.0 <= result.pct_pairs_passing <= 100.0

    def test_correct_ordering_on_clean_data(self, full_benchmark_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.evaluator.settings") as mock:
            mock.known_ordering = ["Progenitor", "Early", "Intermediate", "Terminal"]
            mock.terminal_ordering_pairs = [["Intermediate", "Terminal"]]
            mock.hidden_branch_type = "Terminal"
            mock.trajectory_rho_threshold = 0.5
            result = evaluate_rank_preservation(full_benchmark_adata)
        assert result.n_pairs_tested >= 1


class TestCheckColumns:
    """Test _check_columns helper."""

    def test_passes_with_required_columns(
        self, full_benchmark_adata: ad.AnnData
    ) -> None:
        _check_columns(full_benchmark_adata, ["dpt_pseudotime", "velocity_pseudotime"])

    def test_raises_with_missing_column(
        self, full_benchmark_adata: ad.AnnData
    ) -> None:
        with pytest.raises(ValueError, match="Missing required obs columns"):
            _check_columns(full_benchmark_adata, ["nonexistent_column"])


class TestRunBenchmark:
    """Test the full benchmark orchestrator."""

    def _mock_velocity_fn(self, adata: ad.AnnData):
        """Mock velocity function for Task 3 testing."""
        from src.velocity import VelocityReport
        n = adata.n_obs
        adata.obs["velocity_pseudotime"] = np.random.default_rng(42).random(n)
        adata.obs["velocity_confidence"] = np.random.default_rng(43).random(n)
        return adata, VelocityReport(
            n_cells=n, n_velocity_genes=10, mode="deterministic",
            mean_velocity_confidence=0.5, pct_high_confidence=30.0,
            velocity_pseudotime_range=(0.0, 1.0),
        )

    def test_returns_benchmark_result(
        self, full_benchmark_adata: ad.AnnData
    ) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.evaluator.settings") as mock:
            mock.known_ordering = ["Progenitor", "Early", "Intermediate", "Terminal"]
            mock.terminal_ordering_pairs = [["Intermediate", "Terminal"]]
            mock.hidden_branch_type = "Terminal"
            mock.trajectory_rho_threshold = 0.5
            mock.root_cell_type = "Progenitor"
            mock.terminal_cell_types = ["Terminal"]
            result = run_benchmark(full_benchmark_adata, self._mock_velocity_fn)
        assert isinstance(result, BenchmarkResult)

    def test_result_has_summary(self, full_benchmark_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.benchmark.evaluator.settings") as mock:
            mock.known_ordering = ["Progenitor", "Early", "Intermediate", "Terminal"]
            mock.terminal_ordering_pairs = [["Intermediate", "Terminal"]]
            mock.hidden_branch_type = "Terminal"
            mock.trajectory_rho_threshold = 0.5
            mock.root_cell_type = "Progenitor"
            mock.terminal_cell_types = ["Terminal"]
            result = run_benchmark(full_benchmark_adata, self._mock_velocity_fn)
        assert "task1_trajectory_recovery" in result.summary
        assert "task2_rank_preservation" in result.summary
        assert "task3_hidden_branch" in result.summary
