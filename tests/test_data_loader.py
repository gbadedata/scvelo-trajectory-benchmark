"""Tests for data_loader module."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pytest

from src.data_loader import get_dataset, load_h5ad, validate_dataset


class TestLoadH5ad:
    """Test h5ad loading."""

    def test_load_existing_h5ad(
        self, tmp_path: Path, synthetic_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_adata.write_h5ad(filepath)
        result = load_h5ad(filepath)
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == synthetic_adata.n_obs
        assert result.n_vars == synthetic_adata.n_vars

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_h5ad("/nonexistent/path/fake.h5ad")

    def test_preserves_layers(
        self, tmp_path: Path, synthetic_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_adata.write_h5ad(filepath)
        result = load_h5ad(filepath)
        assert "spliced" in result.layers
        assert "unspliced" in result.layers

    def test_preserves_obs_columns(
        self, tmp_path: Path, synthetic_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_adata.write_h5ad(filepath)
        result = load_h5ad(filepath)
        assert "dpt_pseudotime" in result.obs.columns
        assert "clusters" in result.obs.columns


class TestValidateDataset:
    """Test dataset structure validation."""

    def test_valid_dataset_passes(self, synthetic_adata: ad.AnnData) -> None:
        """The synthetic fixture satisfies all requirements."""
        validate_dataset(synthetic_adata)  # should not raise

    def test_missing_spliced_layer_raises(
        self, synthetic_adata: ad.AnnData
    ) -> None:
        adata = synthetic_adata.copy()
        del adata.layers["spliced"]
        with pytest.raises(ValueError, match="Missing required layers"):
            validate_dataset(adata)

    def test_missing_unspliced_layer_raises(
        self, synthetic_adata: ad.AnnData
    ) -> None:
        adata = synthetic_adata.copy()
        del adata.layers["unspliced"]
        with pytest.raises(ValueError, match="Missing required layers"):
            validate_dataset(adata)

    def test_missing_clusters_column_raises(
        self, synthetic_adata: ad.AnnData
    ) -> None:
        adata = synthetic_adata.copy()
        adata.obs = adata.obs.drop(columns=["clusters"])
        with pytest.raises(ValueError, match="Missing required obs columns"):
            validate_dataset(adata)

    def test_missing_pseudotime_column_does_not_raise(
        self, synthetic_adata: ad.AnnData
    ) -> None:
        """dpt_pseudotime is computed in preprocessing, not required at load time."""
        adata = synthetic_adata.copy()
        adata.obs = adata.obs.drop(columns=["dpt_pseudotime"])
        validate_dataset(adata)  # should NOT raise


class TestGetDataset:
    """Test the main get_dataset entry point."""

    def test_get_dataset_from_h5ad(
        self, tmp_path: Path, synthetic_adata: ad.AnnData
    ) -> None:
        filepath = tmp_path / "test.h5ad"
        synthetic_adata.write_h5ad(filepath)
        result = get_dataset(filepath=filepath)
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == 200

    def test_get_dataset_validates_structure(
        self, tmp_path: Path, synthetic_adata: ad.AnnData
    ) -> None:
        """get_dataset validates the loaded AnnData structure."""
        # Create an invalid dataset (no spliced layer)
        adata = synthetic_adata.copy()
        del adata.layers["spliced"]
        filepath = tmp_path / "invalid.h5ad"
        adata.write_h5ad(filepath)
        with pytest.raises(ValueError, match="Missing required layers"):
            get_dataset(filepath=filepath)
