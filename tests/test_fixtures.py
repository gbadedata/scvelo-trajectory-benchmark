"""Tests for synthetic trajectory fixtures.

Validates that fixtures produce biologically grounded AnnData structures
with correct velocity signal properties.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp


class TestSyntheticAdata:
    """Validate the synthetic_adata fixture."""

    def test_shape(self, synthetic_adata: ad.AnnData) -> None:
        assert synthetic_adata.n_obs == 200
        assert synthetic_adata.n_vars == 100

    def test_has_spliced_layer(self, synthetic_adata: ad.AnnData) -> None:
        assert "spliced" in synthetic_adata.layers

    def test_has_unspliced_layer(self, synthetic_adata: ad.AnnData) -> None:
        assert "unspliced" in synthetic_adata.layers

    def test_spliced_is_sparse(self, synthetic_adata: ad.AnnData) -> None:
        assert sp.issparse(synthetic_adata.layers["spliced"])

    def test_unspliced_is_sparse(self, synthetic_adata: ad.AnnData) -> None:
        assert sp.issparse(synthetic_adata.layers["unspliced"])

    def test_pseudotime_in_obs(self, synthetic_adata: ad.AnnData) -> None:
        assert "dpt_pseudotime" in synthetic_adata.obs.columns

    def test_pseudotime_range(self, synthetic_adata: ad.AnnData) -> None:
        pt = synthetic_adata.obs["dpt_pseudotime"].values
        assert pt.min() >= 0.0
        assert pt.max() <= 1.0

    def test_cell_types_in_obs(self, synthetic_adata: ad.AnnData) -> None:
        assert "clusters" in synthetic_adata.obs.columns
        assert "cell_type_ground_truth" in synthetic_adata.obs.columns

    def test_four_cell_types(self, synthetic_adata: ad.AnnData) -> None:
        types = synthetic_adata.obs["clusters"].unique()
        assert len(types) == 4
        expected = {"Progenitor", "Early", "Intermediate", "Terminal"}
        assert set(types) == expected

    def test_no_negative_counts(self, synthetic_adata: ad.AnnData) -> None:
        spliced = synthetic_adata.layers["spliced"].toarray()
        unspliced = synthetic_adata.layers["unspliced"].toarray()
        assert np.all(spliced >= 0)
        assert np.all(unspliced >= 0)

    def test_velocity_signal_in_early_cells(self, synthetic_adata: ad.AnnData) -> None:
        """Early cells should have higher unspliced:spliced ratio than late cells.

        This validates that the fixture embeds a detectable velocity signal:
        RNA accumulation (unspliced > spliced) marks transcriptional upregulation
        in early trajectory cells.
        """
        pt = synthetic_adata.obs["dpt_pseudotime"].values
        early_mask = pt < 0.25
        late_mask = pt > 0.75

        spliced = synthetic_adata.layers["spliced"].toarray()
        unspliced = synthetic_adata.layers["unspliced"].toarray()

        # Velocity genes (first 50) should show the signal
        vel_genes = slice(0, 50)
        early_ratio = (unspliced[early_mask, vel_genes] /
                       np.clip(spliced[early_mask, vel_genes], 0.1, None)).mean()
        late_ratio = (unspliced[late_mask, vel_genes] /
                      np.clip(spliced[late_mask, vel_genes], 0.1, None)).mean()

        assert early_ratio > late_ratio, (
            f"Early cells should have higher u/s ratio ({early_ratio:.3f}) "
            f"than late cells ({late_ratio:.3f})"
        )

    def test_progenitor_cells_earliest(self, synthetic_adata: ad.AnnData) -> None:
        """Progenitor cells should have the lowest mean pseudotime."""
        pt = synthetic_adata.obs["dpt_pseudotime"].values
        types = synthetic_adata.obs["clusters"].values

        progenitor_mean = pt[types == "Progenitor"].mean()
        terminal_mean = pt[types == "Terminal"].mean()

        assert progenitor_mean < terminal_mean


class TestSyntheticAdataWithVelocity:
    """Validate the synthetic_adata_with_velocity fixture."""

    def test_has_velocity_layer(self, synthetic_adata_with_velocity: ad.AnnData) -> None:
        assert "velocity" in synthetic_adata_with_velocity.layers

    def test_velocity_pseudotime_in_obs(
        self, synthetic_adata_with_velocity: ad.AnnData
    ) -> None:
        assert "velocity_pseudotime" in synthetic_adata_with_velocity.obs.columns

    def test_velocity_pseudotime_range(
        self, synthetic_adata_with_velocity: ad.AnnData
    ) -> None:
        vpt = synthetic_adata_with_velocity.obs["velocity_pseudotime"].values
        assert vpt.min() >= 0.0
        assert vpt.max() <= 1.0

    def test_velocity_pseudotime_correlates_with_ground_truth(
        self, synthetic_adata_with_velocity: ad.AnnData
    ) -> None:
        """Velocity pseudotime should correlate with ground-truth dpt_pseudotime."""
        from scipy.stats import spearmanr
        vpt = synthetic_adata_with_velocity.obs["velocity_pseudotime"].values
        dpt = synthetic_adata_with_velocity.obs["dpt_pseudotime"].values
        rho, _ = spearmanr(vpt, dpt)
        assert rho > 0.8, (
            f"Velocity pseudotime should correlate with ground truth "
            f"(rho={rho:.3f}, expected > 0.8)"
        )
