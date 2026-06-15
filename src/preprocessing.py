"""Preprocessing for RNA velocity analysis.

Prepares the pancreas dataset for RNA velocity computation:
1. Filter and normalise spliced and unspliced count matrices
2. Select highly variable genes
3. PCA and neighbourhood graph
4. UMAP embedding
5. Compute diffusion pseudotime (oracle ground truth)

The diffusion pseudotime computed here serves as the oracle for the
benchmark. It is computed by an independent method (diffusion kernel
on the transcriptional similarity graph) from the RNA velocity-based
latent time that the pipeline infers later. High Spearman correlation
between the two is evidence that velocity correctly captures the
developmental trajectory direction.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import anndata as ad
import scanpy as sc
import scvelo as scv

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingReport:
    """Summary statistics from preprocessing.

    Attributes:
        n_cells: Cells after filtering.
        n_genes: Genes after filtering.
        n_velocity_genes: Genes retained for velocity computation.
        n_clusters: Number of cell-type clusters.
        root_cell_type: Cell type used as root for pseudotime.
        dpt_pseudotime_range: (min, max) of computed diffusion pseudotime.
        pct_root_cells: Percentage of cells identified as root type.
    """

    n_cells: int
    n_genes: int
    n_velocity_genes: int
    n_clusters: int
    root_cell_type: str
    dpt_pseudotime_range: tuple[float, float]
    pct_root_cells: float


def filter_and_normalise(adata: ad.AnnData) -> ad.AnnData:
    """Filter genes and normalise spliced and unspliced count matrices.

    Uses scvelo's preprocessing functions for consistent handling of
    both spliced and unspliced layers. The steps are:
    1. Filter genes with insufficient shared counts (scvelo)
    2. Normalise each cell to the same total count (scvelo)
    3. Select highly variable genes (scanpy, on spliced counts)
    4. Log1p transform (scanpy)

    Args:
        adata: Raw AnnData with spliced/unspliced layers.

    Returns:
        Filtered, normalised, and log1p-transformed AnnData.
    """
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Step 1: filter genes by minimum shared counts across both layers
        scv.pp.filter_genes(adata, min_shared_counts=20)
        # Step 2: normalise per cell (both layers consistently)
        scv.pp.normalize_per_cell(adata)

    # Step 3: HVG selection on the normalised (not yet log-transformed) counts
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=settings.n_top_genes,
        flavor="seurat",
        subset=True,
    )

    # Step 4: log1p transform
    sc.pp.log1p(adata)

    logger.info(
        "filter_and_normalise: %d->%d cells, %d->%d genes",
        n_cells_before,
        adata.n_obs,
        n_genes_before,
        adata.n_vars,
    )
    return adata


def compute_pca_and_neighbors(adata: ad.AnnData) -> ad.AnnData:
    """Compute PCA, neighbourhood graph, and UMAP.

    scvelo recommends n_neighbors=30 for velocity graph computation.
    PCA is computed on the normalised spliced counts.

    Args:
        adata: Filtered and normalised AnnData.

    Returns:
        AnnData with X_pca, X_umap, and neighbour graph.
    """
    sc.tl.pca(adata, svd_solver="arpack", random_state=settings.random_seed)

    sc.pp.neighbors(
        adata,
        n_neighbors=settings.n_neighbors,
        n_pcs=settings.n_pcs,
        random_state=settings.random_seed,
    )

    sc.tl.umap(adata, random_state=settings.random_seed)

    logger.info(
        "pca_neighbors_umap: n_pcs=%d, n_neighbors=%d",
        settings.n_pcs,
        settings.n_neighbors,
    )
    return adata


def compute_diffusion_pseudotime(adata: ad.AnnData) -> ad.AnnData:
    """Compute diffusion pseudotime rooted at the progenitor cell type.

    Diffusion pseudotime (Haghverdi et al. 2016) measures developmental
    distance from a root cell by simulating diffusion on the transcriptional
    similarity graph. Rooting at Ductal cells (the earliest progenitors)
    gives a continuous ordering that increases toward terminal fates.

    This pseudotime serves as the oracle ground truth for the benchmark.
    It is computed by a method entirely independent of RNA velocity,
    so correlation with velocity-inferred latent time is informative.

    Args:
        adata: AnnData with neighbour graph computed.

    Returns:
        AnnData with dpt_pseudotime added to obs.
    """
    root_type = settings.root_cell_type
    clusters = adata.obs["clusters"].astype(str)

    # Find root cells: all cells of the root type
    root_mask = clusters == root_type
    n_root = int(root_mask.sum())

    if n_root == 0:
        available = sorted(clusters.unique())
        raise ValueError(
            f"Root cell type '{root_type}' not found in dataset. "
            f"Available types: {available}"
        )

    # Use the cell with the highest total spliced count within the root
    # population as the single root cell -- this is the most mature progenitor
    # signal and gives the most stable DPT computation
    import numpy as np
    import scipy.sparse as sp

    spliced = adata.layers.get("Ms", adata.layers.get("spliced"))
    if sp.issparse(spliced):
        spliced_dense = spliced.toarray()
    else:
        spliced_dense = spliced

    root_indices = np.where(root_mask.values)[0]
    root_totals = spliced_dense[root_indices].sum(axis=1)
    root_cell_idx = int(root_indices[root_totals.argmax()])

    adata.uns["iroot"] = root_cell_idx
    sc.tl.dpt(adata, n_dcs=10)

    pt = adata.obs["dpt_pseudotime"].values
    logger.info(
        "diffusion_pseudotime_computed: root_type=%s, n_root_cells=%d, "
        "pt_range=[%.4f, %.4f]",
        root_type,
        n_root,
        float(pt.min()),
        float(pt.max()),
    )
    return adata


def run_preprocessing(adata: ad.AnnData) -> tuple[ad.AnnData, PreprocessingReport]:
    """Full preprocessing pipeline.

    Steps:
    1. Filter and normalise (scvelo)
    2. PCA, neighbours, UMAP (scanpy)
    3. Diffusion pseudotime oracle (scanpy)

    Args:
        adata: Raw AnnData from data loader.

    Returns:
        Tuple of (preprocessed AnnData, PreprocessingReport).
    """
    adata = filter_and_normalise(adata)
    adata = compute_pca_and_neighbors(adata)
    adata = compute_diffusion_pseudotime(adata)

    clusters = adata.obs["clusters"].astype(str)
    root_mask = clusters == settings.root_cell_type
    pt = adata.obs["dpt_pseudotime"].values

    report = PreprocessingReport(
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
        n_velocity_genes=int(adata.var["highly_variable"].sum())
        if "highly_variable" in adata.var.columns
        else adata.n_vars,
        n_clusters=clusters.nunique(),
        root_cell_type=settings.root_cell_type,
        dpt_pseudotime_range=(float(pt.min()), float(pt.max())),
        pct_root_cells=float(root_mask.mean() * 100),
    )

    logger.info(
        "preprocessing_complete: %d cells, %d genes, %d clusters, "
        "dpt_range=[%.4f, %.4f]",
        report.n_cells,
        report.n_genes,
        report.n_clusters,
        report.dpt_pseudotime_range[0],
        report.dpt_pseudotime_range[1],
    )
    return adata, report
