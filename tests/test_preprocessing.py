"""Tests for preprocessing module."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from src.preprocessing import (
    PreprocessingReport,
    compute_diffusion_pseudotime,
    compute_pca_and_neighbors,
    filter_and_normalise,
    run_preprocessing,
)


class TestFilterAndNormalise:
    """Test scvelo filter and normalise."""

    def test_reduces_gene_count(self, synthetic_adata: ad.AnnData) -> None:
        """Filtering should remove low-expression genes."""
        n_genes_before = synthetic_adata.n_vars
        result = filter_and_normalise(synthetic_adata)
        # With min_shared_counts=20 and n_top_genes=2000, may reduce gene set
        assert result.n_vars <= n_genes_before

    def test_preserves_cell_count(self, synthetic_adata: ad.AnnData) -> None:
        """Cell count should not change during gene filtering."""
        n_cells_before = synthetic_adata.n_obs
        result = filter_and_normalise(synthetic_adata)
        assert result.n_obs == n_cells_before

    def test_normalised_layer_present(self, synthetic_adata: ad.AnnData) -> None:
        """Normalised spliced layer should be present after preprocessing."""
        result = filter_and_normalise(synthetic_adata)
        # scvelo adds Ms (smoothed spliced) layer during normalisation
        assert "spliced" in result.layers or "Ms" in result.layers

    def test_no_negative_values_after_normalise(
        self, synthetic_adata: ad.AnnData
    ) -> None:
        """Normalised counts should be non-negative."""
        import scipy.sparse as sp
        result = filter_and_normalise(synthetic_adata)
        spliced = result.layers["spliced"]
        if sp.issparse(spliced):
            spliced = spliced.toarray()
        assert np.all(spliced >= 0)


class TestComputePcaAndNeighbors:
    """Test PCA, neighbourhood graph, and UMAP computation."""

    @pytest.fixture
    def normalised_adata(self, synthetic_adata: ad.AnnData) -> ad.AnnData:
        return filter_and_normalise(synthetic_adata)

    def test_pca_embedding_present(self, normalised_adata: ad.AnnData) -> None:
        result = compute_pca_and_neighbors(normalised_adata)
        assert "X_pca" in result.obsm

    def test_umap_embedding_present(self, normalised_adata: ad.AnnData) -> None:
        result = compute_pca_and_neighbors(normalised_adata)
        assert "X_umap" in result.obsm

    def test_umap_shape(self, normalised_adata: ad.AnnData) -> None:
        result = compute_pca_and_neighbors(normalised_adata)
        assert result.obsm["X_umap"].shape == (normalised_adata.n_obs, 2)

    def test_neighbor_graph_present(self, normalised_adata: ad.AnnData) -> None:
        result = compute_pca_and_neighbors(normalised_adata)
        assert "connectivities" in result.obsp

    def test_umap_values_finite(self, normalised_adata: ad.AnnData) -> None:
        result = compute_pca_and_neighbors(normalised_adata)
        assert np.all(np.isfinite(result.obsm["X_umap"]))


class TestComputeDiffusionPseudotime:
    """Test oracle pseudotime computation."""

    @pytest.fixture
    def preprocessed_adata(self, synthetic_adata: ad.AnnData) -> ad.AnnData:
        adata = filter_and_normalise(synthetic_adata)
        return compute_pca_and_neighbors(adata)

    def test_dpt_pseudotime_added(self, preprocessed_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.preprocessing.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.random_seed = 42
            result = compute_diffusion_pseudotime(preprocessed_adata)
        assert "dpt_pseudotime" in result.obs.columns

    def test_dpt_range(self, preprocessed_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.preprocessing.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.random_seed = 42
            result = compute_diffusion_pseudotime(preprocessed_adata)
        pt = result.obs["dpt_pseudotime"].values
        assert pt.min() >= 0.0
        assert pt.max() <= 1.0

    def test_dpt_no_nan(self, preprocessed_adata: ad.AnnData) -> None:
        from unittest.mock import patch
        with patch("src.preprocessing.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.random_seed = 42
            result = compute_diffusion_pseudotime(preprocessed_adata)
        pt = result.obs["dpt_pseudotime"].values
        assert not np.any(np.isnan(pt))

    def test_root_type_has_low_pseudotime(
        self, preprocessed_adata: ad.AnnData
    ) -> None:
        """Root cell type should have lower mean pseudotime than terminal types."""
        from unittest.mock import patch
        with patch("src.preprocessing.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.random_seed = 42
            result = compute_diffusion_pseudotime(preprocessed_adata)
        pt = result.obs["dpt_pseudotime"].values
        clusters = result.obs["clusters"].astype(str)

        root_mean = pt[clusters == "Progenitor"].mean()
        terminal_mean = pt[clusters == "Terminal"].mean()

        assert root_mean < terminal_mean, (
            f"Root cells should have lower pseudotime ({root_mean:.3f}) "
            f"than terminal cells ({terminal_mean:.3f})"
        )

    def test_invalid_root_type_raises(
        self, preprocessed_adata: ad.AnnData
    ) -> None:
        """Unknown root cell type raises ValueError."""
        from unittest.mock import patch
        with patch("src.preprocessing.settings") as mock:
            mock.root_cell_type = "NonExistentType"
            mock.random_seed = 42
            with pytest.raises(ValueError, match="not found in dataset"):
                compute_diffusion_pseudotime(preprocessed_adata)


class TestRunPreprocessing:
    """Test the full preprocessing pipeline."""

    @pytest.fixture
    def patched_settings(self):
        from unittest.mock import patch
        with patch("src.preprocessing.settings") as mock:
            mock.root_cell_type = "Progenitor"
            mock.random_seed = 42
            mock.n_top_genes = 2000
            mock.n_neighbors = 15
            mock.n_pcs = 30
            yield mock

    def test_returns_adata_and_report(
        self, synthetic_adata: ad.AnnData, patched_settings: object
    ) -> None:
        adata, report = run_preprocessing(synthetic_adata)
        assert isinstance(adata, ad.AnnData)
        assert isinstance(report, PreprocessingReport)

    def test_report_cells_consistent(
        self, synthetic_adata: ad.AnnData, patched_settings: object
    ) -> None:
        adata, report = run_preprocessing(synthetic_adata)
        assert report.n_cells == adata.n_obs

    def test_report_has_pseudotime_range(
        self, synthetic_adata: ad.AnnData, patched_settings: object
    ) -> None:
        _, report = run_preprocessing(synthetic_adata)
        lo, hi = report.dpt_pseudotime_range
        assert 0.0 <= lo < hi <= 1.0

    def test_report_n_clusters_positive(
        self, synthetic_adata: ad.AnnData, patched_settings: object
    ) -> None:
        _, report = run_preprocessing(synthetic_adata)
        assert report.n_clusters > 0

    def test_all_embeddings_present(
        self, synthetic_adata: ad.AnnData, patched_settings: object
    ) -> None:
        adata, _ = run_preprocessing(synthetic_adata)
        assert "X_pca" in adata.obsm
        assert "X_umap" in adata.obsm

    def test_dpt_pseudotime_present(
        self, synthetic_adata: ad.AnnData, patched_settings: object
    ) -> None:
        adata, _ = run_preprocessing(synthetic_adata)
        assert "dpt_pseudotime" in adata.obs.columns
