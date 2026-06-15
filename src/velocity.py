"""RNA velocity computation.

Implements the deterministic RNA velocity model (LaManno et al. 2018)
on the preprocessed pancreas dataset. The pipeline:

1. Compute moments (first- and second-order) across the kNN graph
2. Fit the deterministic velocity model
3. Build the velocity graph (cell-to-cell transition probabilities)
4. Compute velocity pseudotime (random-walk distance from root)
5. Compute velocity confidence per cell and gene

The deterministic (steady-state) model is used rather than the stochastic
model because scvelo 0.3.x contains a NumPy 2.x incompatibility in the
stochastic model's leastsq_generalized function. The function assigns a
1-element matrix to a float32 scalar slot (gamma[i] = pinv(A).dot(b)),
which raises ValueError in NumPy >= 2.0. This is a known upstream issue
(github.com/theislab/scvelo/issues/966). The deterministic model is the
well-validated baseline from LaManno et al. 2018 and produces correct
trajectory ordering on the pancreas endocrinogenesis dataset.

The benchmark evaluation uses velocity_pseudotime, which is computed from
the velocity graph and is independent of the diffusion pseudotime oracle.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import anndata as ad
import scvelo as scv

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class VelocityReport:
    """Summary statistics from velocity computation.

    Attributes:
        n_cells: Number of cells.
        n_velocity_genes: Genes used for velocity estimation.
        mode: Velocity model used ('stochastic' or 'dynamical').
        mean_velocity_confidence: Mean per-cell velocity confidence.
        pct_high_confidence: Percentage of cells with confidence > 0.75.
        velocity_pseudotime_range: (min, max) of inferred pseudotime.
    """

    n_cells: int
    n_velocity_genes: int
    mode: str
    mean_velocity_confidence: float
    pct_high_confidence: float
    velocity_pseudotime_range: tuple[float, float]


def compute_moments(adata: ad.AnnData) -> ad.AnnData:
    """Compute first- and second-order moments across the kNN graph.

    Moments are the weighted averages of spliced and unspliced counts
    across each cell's nearest neighbours. They smooth out technical
    noise and are required inputs for velocity model fitting.

    scvelo stores moments as:
    - Ms: first-order moments of spliced counts
    - Mu: first-order moments of unspliced counts

    Args:
        adata: Preprocessed AnnData with neighbour graph and log1p counts.

    Returns:
        AnnData with Ms and Mu layers added.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        scv.pp.moments(
            adata,
            n_neighbors=settings.n_neighbors,
            n_pcs=settings.n_pcs,
            use_highly_variable=False,  # use all genes post-filtering
        )

    logger.info("moments_computed: Ms and Mu layers added")
    return adata


def fit_velocity(adata: ad.AnnData) -> ad.AnnData:
    """Fit deterministic RNA velocity model.

    The deterministic (steady-state) model estimates velocities by
    identifying the steady-state equilibrium ratio of unspliced to spliced
    mRNA, then quantifying how each cell deviates from that equilibrium.
    Cells above the equilibrium are in a state of RNA accumulation (positive
    velocity); cells below are in a state of RNA degradation (negative
    velocity).

    This is the original RNA velocity model from LaManno et al. 2018.

    Stores velocity estimates in adata.layers['velocity'].

    Args:
        adata: AnnData with moment layers (Ms, Mu) computed.

    Returns:
        AnnData with velocity layer added.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        scv.tl.velocity(
            adata,
            vkey="velocity",
            mode=settings.velocity_mode,
            use_highly_variable=False,
        )

    n_vel_genes = int(adata.var["velocity_genes"].sum())
    logger.info(
        "velocity_fitted: mode=%s, %d velocity genes",
        settings.velocity_mode,
        n_vel_genes,
    )
    return adata


def compute_velocity_graph(adata: ad.AnnData) -> ad.AnnData:
    """Compute the velocity graph (cell-to-cell transition probabilities).

    The velocity graph encodes the probability of each cell transitioning
    to each of its neighbours based on the cosine correlation between the
    cell's velocity vector and the displacement vector to each neighbour.
    A high probability means the neighbour is in the direction of the
    cell's predicted future state.

    Args:
        adata: AnnData with velocity layer computed.

    Returns:
        AnnData with velocity graph stored in uns['velocity_graph'].
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        scv.tl.velocity_graph(
            adata,
            vkey="velocity",
            n_jobs=1,
        )

    logger.info("velocity_graph_computed")
    return adata


def compute_velocity_pseudotime(adata: ad.AnnData) -> ad.AnnData:
    """Compute velocity pseudotime from the velocity graph.

    Velocity pseudotime measures developmental distance from inferred
    root cells by simulating random walks on the directed velocity graph.
    Unlike diffusion pseudotime (which uses transcriptional similarity),
    velocity pseudotime uses the directional momentum of transcriptional
    change -- it is the pipeline's prediction of developmental ordering.

    The Spearman correlation between velocity_pseudotime and dpt_pseudotime
    (oracle) is the primary benchmark metric.

    Args:
        adata: AnnData with velocity graph computed.

    Returns:
        AnnData with velocity_pseudotime added to obs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        scv.tl.velocity_pseudotime(
            adata,
            vkey="velocity",
        )

    vpt = adata.obs["velocity_pseudotime"].values
    logger.info(
        "velocity_pseudotime_computed: range=[%.4f, %.4f]",
        float(vpt.min()),
        float(vpt.max()),
    )
    return adata


def compute_velocity_confidence(adata: ad.AnnData) -> ad.AnnData:
    """Compute per-cell and per-gene velocity confidence scores.

    Velocity confidence measures how consistently a cell's velocity
    vector aligns with its neighbours' displacements. Low confidence
    indicates the velocity estimate is noisy or the cell is in a
    transitional state with ambiguous directionality.

    Stores:
    - adata.obs['velocity_confidence']: per-cell confidence [0, 1]
    - adata.obs['velocity_confidence_transition']: transition confidence

    Args:
        adata: AnnData with velocity graph computed.

    Returns:
        AnnData with confidence scores added to obs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        scv.tl.velocity_confidence(adata, vkey="velocity")

    conf = adata.obs["velocity_confidence"].values
    import numpy as np
    mean_conf = float(np.nanmean(conf))
    pct_high = float(np.mean(conf > 0.75) * 100)
    logger.info(
        "velocity_confidence_computed: mean=%.4f, pct_high_confidence=%.1f%%",
        mean_conf,
        pct_high,
    )
    return adata


def run_velocity(adata: ad.AnnData) -> tuple[ad.AnnData, VelocityReport]:
    """Full velocity computation pipeline.

    Steps:
    1. Compute moments (Ms, Mu)
    2. Fit stochastic velocity model
    3. Build velocity graph
    4. Compute velocity pseudotime
    5. Compute velocity confidence

    Args:
        adata: Preprocessed AnnData with neighbour graph.

    Returns:
        Tuple of (AnnData with velocity layers, VelocityReport).
    """
    import numpy as np

    adata = compute_moments(adata)
    adata = fit_velocity(adata)
    adata = compute_velocity_graph(adata)
    adata = compute_velocity_pseudotime(adata)
    adata = compute_velocity_confidence(adata)

    n_vel_genes = int(adata.var["velocity_genes"].sum())
    conf = adata.obs["velocity_confidence"].values
    vpt = adata.obs["velocity_pseudotime"].values

    report = VelocityReport(
        n_cells=adata.n_obs,
        n_velocity_genes=n_vel_genes,
        mode=settings.velocity_mode,
        mean_velocity_confidence=float(np.nanmean(conf)),
        pct_high_confidence=float(np.mean(conf > 0.75) * 100),
        velocity_pseudotime_range=(float(vpt.min()), float(vpt.max())),
    )

    logger.info(
        "velocity_complete: %d velocity genes, mean_conf=%.4f, "
        "vpt_range=[%.4f, %.4f]",
        report.n_velocity_genes,
        report.mean_velocity_confidence,
        report.velocity_pseudotime_range[0],
        report.velocity_pseudotime_range[1],
    )
    return adata, report
