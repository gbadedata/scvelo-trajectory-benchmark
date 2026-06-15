"""Data acquisition and loading.

Downloads the pancreatic endocrinogenesis dataset and loads it into
AnnData format. The dataset (Bastidas-Ponce et al. 2019) contains
spliced and unspliced count matrices required for RNA velocity
computation, along with diffusion pseudotime and cell-type labels
that serve as oracle ground truth for benchmark evaluation.

Supports loading any local h5ad file as an alternative to the
default download.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import scvelo as scv

from config.settings import settings

logger = logging.getLogger(__name__)


def download_pancreas(dest_dir: Path | str | None = None) -> ad.AnnData:
    """Download the pancreatic endocrinogenesis dataset via scvelo.

    Uses scvelo.datasets.pancreas() which fetches from the scvelo
    data repository. The file is cached locally so subsequent calls
    do not re-download.

    Dataset structure:
    - ~3,696 cells, ~27,998 genes (before filtering)
    - obs['clusters']: 8 cell-type labels along the endocrine trajectory
    - obs['dpt_pseudotime']: diffusion pseudotime (oracle ground truth)
    - layers['spliced']: spliced mRNA counts
    - layers['unspliced']: unspliced mRNA counts

    Args:
        dest_dir: Directory to cache the h5ad. Defaults to settings.data_dir.

    Returns:
        AnnData with raw counts and oracle pseudotime.
    """
    dest_dir = Path(dest_dir) if dest_dir else settings.data_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    cache_path = dest_dir / settings.dataset_filename
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info("dataset_cached: %s (%.1f MB)", cache_path, size_mb)
        return load_h5ad(cache_path)

    # scvelo saves to data/Pancreas/ by default; redirect to our data dir
    pancreas_dir = dest_dir / "Pancreas"
    pancreas_dir.mkdir(parents=True, exist_ok=True)
    scvelo_path = pancreas_dir / "endocrinogenesis_day15.h5ad"

    logger.info("downloading_pancreas via scvelo (figshare)")
    adata = scv.datasets.pancreas(file_path=str(scvelo_path))
    adata.var_names_make_unique()

    # Save to our canonical cache path
    adata.write_h5ad(cache_path)
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    logger.info(
        "downloaded_and_cached: %d cells x %d genes -> %s (%.1f MB)",
        adata.n_obs,
        adata.n_vars,
        cache_path,
        size_mb,
    )
    return adata


def load_h5ad(filepath: str | Path) -> ad.AnnData:
    """Load an h5ad file into AnnData.

    Args:
        filepath: Path to the .h5ad file.

    Returns:
        AnnData object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    logger.info("loading_h5ad: %s", filepath)
    adata = ad.read_h5ad(filepath)
    adata.var_names_make_unique()
    logger.info("loaded: %d cells x %d genes", adata.n_obs, adata.n_vars)
    return adata


def validate_dataset(adata: ad.AnnData) -> None:
    """Validate that the loaded dataset has the required structure.

    Checks for spliced/unspliced layers, cell-type labels, and
    oracle pseudotime. Raises ValueError if any required component
    is missing.

    Args:
        adata: Loaded AnnData to validate.

    Raises:
        ValueError: If required layers or obs columns are missing.
    """
    required_layers = ["spliced", "unspliced"]
    missing_layers = [lay for lay in required_layers if lay not in adata.layers]
    if missing_layers:
        raise ValueError(
            f"Missing required layers: {missing_layers}. "
            "The dataset must contain spliced and unspliced count matrices "
            "for RNA velocity computation."
        )

    required_obs = ["clusters", "dpt_pseudotime"]
    missing_obs = [col for col in required_obs if col not in adata.obs.columns]
    if missing_obs:
        raise ValueError(
            f"Missing required obs columns: {missing_obs}. "
            "The pancreas dataset must contain cell-type labels and "
            "diffusion pseudotime for benchmark evaluation."
        )

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    logger.info(
        "dataset_validated: %d cells x %d genes, layers=%s",
        n_cells,
        n_genes,
        list(adata.layers.keys()),
    )


def get_dataset(filepath: str | Path | None = None) -> ad.AnnData:
    """Download (if needed) and load the pancreas dataset.

    Main entry point. Downloads via scvelo if no explicit path given.
    Validates structure before returning.

    Args:
        filepath: Optional path to a local h5ad file. If None,
            downloads the pancreas dataset.

    Returns:
        Validated AnnData with spliced/unspliced layers and oracle pseudotime.
    """
    if filepath is not None:
        adata = load_h5ad(filepath)
    else:
        adata = download_pancreas()

    validate_dataset(adata)
    return adata
