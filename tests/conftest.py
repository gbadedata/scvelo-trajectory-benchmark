"""Shared test fixtures.

Provides small synthetic AnnData objects with spliced and unspliced
count matrices simulating a developmental trajectory. Fixtures are
designed to be biologically grounded: cells are ordered along a
known pseudotime with velocity signal embedded in the spliced/unspliced
ratio.

No network access is required for any test.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _make_trajectory_counts(
    n_cells: int,
    n_genes: int,
    n_cell_types: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate spliced and unspliced count matrices with velocity signal.

    Simulates a simple linear trajectory where cells progress from
    progenitors (pseudotime=0) to terminal state (pseudotime=1).
    Genes increase in expression along the trajectory, creating a
    positive spliced/unspliced ratio gradient that RNA velocity
    should detect as forward-directed.

    Returns:
        spliced: (n_cells, n_genes) count matrix
        unspliced: (n_cells, n_genes) count matrix
        pseudotime: (n_cells,) ground-truth pseudotime values in [0, 1]
    """
    rng = np.random.default_rng(seed)

    # Ground-truth pseudotime: cells ordered from 0 to 1
    pseudotime = np.linspace(0, 1, n_cells)

    # Gene expression increases along pseudotime for "velocity genes"
    # and is flat for "housekeeping genes"
    n_velocity_genes = n_genes // 2
    n_housekeeping = n_genes - n_velocity_genes

    # Velocity genes: expression scales with pseudotime
    # Unspliced RNA leads spliced (precursor-product relationship)
    spliced_velocity = np.outer(pseudotime, np.ones(n_velocity_genes)) * 8
    unspliced_velocity = np.outer(
        np.clip(pseudotime + 0.15, 0, 1), np.ones(n_velocity_genes)
    ) * 8

    # Housekeeping genes: constant expression with noise
    spliced_house = rng.poisson(3, (n_cells, n_housekeeping)).astype(float)
    unspliced_house = rng.poisson(1, (n_cells, n_housekeeping)).astype(float)

    # Add Poisson noise
    spliced = np.hstack([
        rng.poisson(np.clip(spliced_velocity, 0.1, None)),
        spliced_house,
    ]).astype(np.float32)
    unspliced = np.hstack([
        rng.poisson(np.clip(unspliced_velocity, 0.1, None)),
        unspliced_house,
    ]).astype(np.float32)

    return spliced, unspliced, pseudotime


@pytest.fixture
def synthetic_adata() -> ad.AnnData:
    """Synthetic AnnData with spliced/unspliced matrices and known trajectory.

    Structure:
    - 200 cells ordered along a single linear trajectory
    - 100 genes: 50 with velocity signal, 50 housekeeping
    - 4 cell types corresponding to trajectory stages
    - Ground-truth pseudotime in obs['dpt_pseudotime']
    - Spliced counts in layers['spliced']
    - Unspliced counts in layers['unspliced']
    """
    n_cells = 200
    n_genes = 100
    n_types = 4

    spliced, unspliced, pseudotime = _make_trajectory_counts(
        n_cells, n_genes, n_types
    )

    # Cell type assignment based on pseudotime quartiles
    cell_types_list = ["Progenitor", "Early", "Intermediate", "Terminal"]
    quartile_boundaries = np.quantile(pseudotime, [0.25, 0.5, 0.75])
    cell_type_labels = np.array([
        cell_types_list[0] if pt < quartile_boundaries[0]
        else cell_types_list[1] if pt < quartile_boundaries[1]
        else cell_types_list[2] if pt < quartile_boundaries[2]
        else cell_types_list[3]
        for pt in pseudotime
    ])

    # Gene names: first 50 are velocity genes, rest are housekeeping
    gene_names = (
        [f"VEL_{i:03d}" for i in range(50)]
        + [f"HOUSE_{i:03d}" for i in range(50)]
    )

    obs = pd.DataFrame(
        {
            "dpt_pseudotime": pseudotime,
            "clusters": cell_type_labels,
            "cell_type_ground_truth": cell_type_labels,
        },
        index=[f"CELL_{i:04d}" for i in range(n_cells)],
    )

    var = pd.DataFrame(
        {"is_velocity_gene": [True] * 50 + [False] * 50},
        index=gene_names,
    )

    adata = ad.AnnData(
        X=sp.csr_matrix(spliced),
        obs=obs,
        var=var,
    )
    adata.layers["spliced"] = sp.csr_matrix(spliced)
    adata.layers["unspliced"] = sp.csr_matrix(unspliced)

    return adata


@pytest.fixture
def synthetic_adata_with_velocity(synthetic_adata: ad.AnnData) -> ad.AnnData:
    """Synthetic AnnData with a pre-computed velocity layer.

    Velocity is defined as unspliced - spliced (simplified), creating
    a positive signal for cells early in the trajectory and negative
    for cells near the terminal state. Used for testing downstream
    velocity-dependent modules without running scvelo's fitting.
    """
    adata = synthetic_adata.copy()
    rng = np.random.default_rng(42)

    spliced = adata.layers["spliced"].toarray()
    unspliced = adata.layers["unspliced"].toarray()

    # Simplified velocity: RNA accumulation rate
    # Positive where unspliced > spliced (early trajectory)
    # Near-zero or negative where system is at steady state (late trajectory)
    velocity = (unspliced - spliced) + rng.normal(0, 0.1, spliced.shape)

    adata.layers["velocity"] = velocity.astype(np.float32)

    # Simulated latent time: noisy version of ground truth pseudotime
    noise = rng.normal(0, 0.05, len(adata))
    adata.obs["velocity_pseudotime"] = np.clip(
        adata.obs["dpt_pseudotime"].values + noise, 0, 1
    )

    return adata
