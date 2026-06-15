"""Tests for velocity computation module.

Uses the synthetic_adata_with_velocity fixture which has a pre-computed
velocity layer, allowing tests to verify downstream computations
(velocity graph, pseudotime, confidence) without running the full
stochastic model fitting on synthetic data.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from src.preprocessing import compute_pca_and_neighbors, filter_and_normalise
from src.velocity import (
    VelocityReport,
    compute_moments,
    compute_velocity_confidence,
    compute_velocity_graph,
    compute_velocity_pseudotime,
    fit_velocity,
    run_velocity,
)


@pytest.fixture
def preprocessed_adata(synthetic_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData filtered, normalised, and with kNN graph."""
    adata = filter_and_normalise(synthetic_adata)
    return compute_pca_and_neighbors(adata)


@pytest.fixture
def adata_with_moments(preprocessed_adata: ad.AnnData) -> ad.AnnData:
    """AnnData with moment layers computed."""
    return compute_moments(preprocessed_adata)


@pytest.fixture
def adata_with_velocity(adata_with_moments: ad.AnnData) -> ad.AnnData:
    """AnnData with velocity layer fitted using deterministic mode.

    Deterministic mode is used for unit tests on synthetic data because
    the stochastic model's generalised least squares requires the covariance
    structure of real RNA data. Synthetic data with simulated velocity signal
    does not have the full second-order moment relationships, causing a
    numerical failure in scvelo's leastsq_generalized function. The
    deterministic model is the simpler steady-state baseline that runs
    correctly on synthetic data.
    """
    from unittest.mock import patch
    with patch("src.velocity.settings") as mock:
        mock.velocity_mode = "deterministic"
        mock.n_neighbors = 15
        mock.n_pcs = 10
        result = fit_velocity(adata_with_moments)
    return result


@pytest.fixture
def adata_with_graph(adata_with_velocity: ad.AnnData) -> ad.AnnData:
    """AnnData with velocity graph computed."""
    return compute_velocity_graph(adata_with_velocity)


class TestComputeMoments:
    """Test moment computation."""

    def test_ms_layer_added(self, preprocessed_adata: ad.AnnData) -> None:
        """Smoothed spliced moments layer is added."""
        result = compute_moments(preprocessed_adata)
        assert "Ms" in result.layers

    def test_mu_layer_added(self, preprocessed_adata: ad.AnnData) -> None:
        """Smoothed unspliced moments layer is added."""
        result = compute_moments(preprocessed_adata)
        assert "Mu" in result.layers

    def test_ms_shape(self, preprocessed_adata: ad.AnnData) -> None:
        """Ms layer has correct shape (n_cells x n_genes)."""
        result = compute_moments(preprocessed_adata)
        assert result.layers["Ms"].shape == (result.n_obs, result.n_vars)

    def test_ms_values_non_negative(self, preprocessed_adata: ad.AnnData) -> None:
        """Smoothed spliced moments are non-negative."""
        result = compute_moments(preprocessed_adata)
        assert np.all(result.layers["Ms"] >= 0)


class TestFitVelocity:
    """Test velocity model fitting."""

    @pytest.fixture
    def adata_det(self, adata_with_moments: ad.AnnData) -> ad.AnnData:
        """Use deterministic mode for synthetic data compatibility."""
        from unittest.mock import patch
        with patch("src.velocity.settings") as mock:
            mock.velocity_mode = "deterministic"
            mock.n_neighbors = 15
            mock.n_pcs = 10
            return fit_velocity(adata_with_moments)

    def test_velocity_layer_added(self, adata_det: ad.AnnData) -> None:
        """Velocity layer is added after fitting."""
        assert "velocity" in adata_det.layers

    def test_velocity_genes_flagged(self, adata_det: ad.AnnData) -> None:
        """velocity_genes column is added to var."""
        assert "velocity_genes" in adata_det.var.columns

    def test_at_least_one_velocity_gene(self, adata_det: ad.AnnData) -> None:
        """At least one gene passes velocity gene filter."""
        n_vel_genes = int(adata_det.var["velocity_genes"].sum())
        assert n_vel_genes > 0

    def test_velocity_shape(self, adata_det: ad.AnnData) -> None:
        """Velocity layer has correct shape."""
        assert adata_det.layers["velocity"].shape == (adata_det.n_obs, adata_det.n_vars)


class TestComputeVelocityGraph:
    """Test velocity graph computation."""

    def test_velocity_graph_in_uns(self, adata_with_graph: ad.AnnData) -> None:
        """Velocity graph is stored in uns."""
        assert "velocity_graph" in adata_with_graph.uns

    def test_velocity_graph_shape(self, adata_with_graph: ad.AnnData) -> None:
        """Velocity graph has shape (n_cells x n_cells)."""
        graph = adata_with_graph.uns["velocity_graph"]
        n = adata_with_graph.n_obs
        assert graph.shape == (n, n)


class TestComputeVelocityPseudotime:
    """Test velocity pseudotime computation."""

    def test_velocity_pseudotime_added(self, adata_with_graph: ad.AnnData) -> None:
        """velocity_pseudotime column is added to obs."""
        result = compute_velocity_pseudotime(adata_with_graph)
        assert "velocity_pseudotime" in result.obs.columns

    def test_velocity_pseudotime_range(self, adata_with_graph: ad.AnnData) -> None:
        """Velocity pseudotime values are in [0, 1]."""
        result = compute_velocity_pseudotime(adata_with_graph)
        vpt = result.obs["velocity_pseudotime"].values
        assert vpt.min() >= 0.0
        assert vpt.max() <= 1.0

    def test_velocity_pseudotime_no_all_nan(
        self, adata_with_graph: ad.AnnData
    ) -> None:
        """Velocity pseudotime should not be entirely NaN."""
        result = compute_velocity_pseudotime(adata_with_graph)
        vpt = result.obs["velocity_pseudotime"].values
        assert not np.all(np.isnan(vpt))


class TestComputeVelocityConfidence:
    """Test velocity confidence computation."""

    def test_confidence_column_added(self, adata_with_graph: ad.AnnData) -> None:
        """velocity_confidence column is added to obs."""
        result = compute_velocity_confidence(adata_with_graph)
        assert "velocity_confidence" in result.obs.columns

    def test_confidence_range(self, adata_with_graph: ad.AnnData) -> None:
        """Confidence values are in [0, 1]."""
        result = compute_velocity_confidence(adata_with_graph)
        conf = result.obs["velocity_confidence"].values
        valid = conf[~np.isnan(conf)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 1)


class TestRunVelocity:
    """Test the full velocity pipeline."""

    @pytest.fixture
    def patched_velocity_settings(self):
        """Patch velocity mode to deterministic for synthetic data."""
        from unittest.mock import patch
        with patch("src.velocity.settings") as mock:
            mock.velocity_mode = "deterministic"
            mock.n_neighbors = 15
            mock.n_pcs = 10
            yield mock

    def test_returns_adata_and_report(
        self, preprocessed_adata: ad.AnnData, patched_velocity_settings: object
    ) -> None:
        adata, report = run_velocity(preprocessed_adata)
        assert isinstance(adata, ad.AnnData)
        assert isinstance(report, VelocityReport)

    def test_report_mode_correct(
        self, preprocessed_adata: ad.AnnData, patched_velocity_settings: object
    ) -> None:
        _, report = run_velocity(preprocessed_adata)
        assert report.mode == "deterministic"

    def test_report_n_cells_consistent(
        self, preprocessed_adata: ad.AnnData, patched_velocity_settings: object
    ) -> None:
        adata, report = run_velocity(preprocessed_adata)
        assert report.n_cells == adata.n_obs

    def test_report_confidence_in_range(
        self, preprocessed_adata: ad.AnnData, patched_velocity_settings: object
    ) -> None:
        _, report = run_velocity(preprocessed_adata)
        assert 0.0 <= report.mean_velocity_confidence <= 1.0

    def test_report_vpt_range_valid(
        self, preprocessed_adata: ad.AnnData, patched_velocity_settings: object
    ) -> None:
        _, report = run_velocity(preprocessed_adata)
        lo, hi = report.velocity_pseudotime_range
        assert lo >= 0.0
        assert hi <= 1.0

    def test_all_velocity_layers_present(
        self, preprocessed_adata: ad.AnnData, patched_velocity_settings: object
    ) -> None:
        adata, _ = run_velocity(preprocessed_adata)
        for layer in ["velocity", "Ms", "Mu"]:
            assert layer in adata.layers

    def test_velocity_pseudotime_in_obs(
        self, preprocessed_adata: ad.AnnData, patched_velocity_settings: object
    ) -> None:
        adata, _ = run_velocity(preprocessed_adata)
        assert "velocity_pseudotime" in adata.obs.columns

    def test_velocity_confidence_in_obs(
        self, preprocessed_adata: ad.AnnData, patched_velocity_settings: object
    ) -> None:
        adata, _ = run_velocity(preprocessed_adata)
        assert "velocity_confidence" in adata.obs.columns
