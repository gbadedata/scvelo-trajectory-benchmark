"""Visualisation for RNA velocity trajectory analysis and benchmark results.

Generates publication-quality figures saved to evidence/figures/.
All functions write PNG files and do not display interactively.

Figures produced:
  01_umap_clusters.png         -- UMAP coloured by Leiden clusters
  02_umap_celltypes.png        -- UMAP coloured by cell-type labels
  03_umap_velocity_stream.png  -- UMAP with velocity streamlines
  04_umap_pseudotime_oracle.png -- UMAP coloured by DPT oracle
  05_umap_pseudotime_velocity.png -- UMAP coloured by velocity pseudotime
  06_pseudotime_correlation.png -- Scatter: DPT vs velocity pseudotime
  07_benchmark_task2.png       -- Bar chart: rank preservation per pair
  08_benchmark_summary.png     -- Summary panel: all three task results
"""

from __future__ import annotations

import logging

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from config.settings import settings

mpl.use("Agg")

logger = logging.getLogger(__name__)
FIGURE_DIR = settings.evidence_dir / "figures"
DPI = 150


def _save(fig: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("figure_saved: %s", path)


def _get_colours(n: int, palette: str = "tab10") -> list:
    cmap = mpl.colormaps[palette].resampled(max(n, 2))
    return [cmap(i) for i in range(n)]


def plot_umap_clusters(adata: ad.AnnData) -> None:
    """UMAP coloured by cell-type cluster labels."""
    if "X_umap" not in adata.obsm or "clusters" not in adata.obs.columns:
        logger.warning("Cannot plot cluster UMAP: missing embedding or column")
        return

    umap = adata.obsm["X_umap"]
    labels = adata.obs["clusters"].astype(str)
    unique = sorted(labels.unique())
    colours = _get_colours(len(unique))
    colour_map = dict(zip(unique, colours))

    fig, ax = plt.subplots(figsize=(9, 7))
    for ct in unique:
        mask = labels == ct
        ax.scatter(umap[mask, 0], umap[mask, 1],
                   c=[colour_map[ct]], s=6, alpha=0.7,
                   label=f"{ct} (n={mask.sum()})")

    ax.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.01, 1),
              loc="upper left", framealpha=0.9)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title("Pancreas endocrinogenesis -- cell-type clusters",
                 fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    _save(fig, "01_umap_clusters")


def plot_umap_velocity_stream(adata: ad.AnnData) -> None:
    """UMAP with RNA velocity streamlines overlaid."""
    import warnings

    import scvelo as scv

    if "X_umap" not in adata.obsm or "velocity_graph" not in adata.uns:
        logger.warning("Cannot plot velocity stream: missing embedding or graph")
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fig, ax = plt.subplots(figsize=(9, 7))
        scv.pl.velocity_embedding_stream(
            adata,
            basis="umap",
            color="clusters",
            ax=ax,
            show=False,
            legend_loc="right margin",
            title="RNA velocity streamlines -- pancreas endocrinogenesis",
            size=40,
            alpha=0.6,
            arrow_size=1.5,
            linewidth=1.2,
            figsize=None,
        )
    plt.tight_layout()
    _save(fig, "02_umap_velocity_stream")


def plot_umap_pseudotime(
    adata: ad.AnnData,
    col: str,
    title: str,
    filename: str,
) -> None:
    """UMAP coloured by a pseudotime column."""
    if "X_umap" not in adata.obsm or col not in adata.obs.columns:
        logger.warning("Cannot plot pseudotime UMAP: missing %s", col)
        return

    umap = adata.obsm["X_umap"]
    pt = adata.obs[col].values.astype(float)
    valid = ~np.isnan(pt)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc_plot = ax.scatter(
        umap[valid, 0], umap[valid, 1],
        c=pt[valid], cmap="viridis", s=6, alpha=0.8, vmin=0, vmax=1,
    )
    plt.colorbar(sc_plot, ax=ax, label="Pseudotime", shrink=0.7)
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    _save(fig, filename)


def plot_pseudotime_correlation(adata: ad.AnnData) -> None:
    """Scatter plot: velocity pseudotime vs oracle diffusion pseudotime."""
    required = ["dpt_pseudotime", "velocity_pseudotime", "clusters"]
    if not all(c in adata.obs.columns for c in required):
        logger.warning("Cannot plot pseudotime correlation: missing columns")
        return

    from scipy.stats import spearmanr

    dpt = adata.obs["dpt_pseudotime"].values.astype(float)
    vpt = adata.obs["velocity_pseudotime"].values.astype(float)
    clusters = adata.obs["clusters"].astype(str)
    valid = ~(np.isnan(dpt) | np.isnan(vpt))

    rho, pval = spearmanr(dpt[valid], vpt[valid])
    unique = sorted(clusters.unique())
    colours = _get_colours(len(unique))
    colour_map = dict(zip(unique, colours))

    fig, ax = plt.subplots(figsize=(8, 7))
    for ct in unique:
        mask = (clusters == ct) & valid
        ax.scatter(dpt[mask], vpt[mask],
                   c=[colour_map[ct]], s=8, alpha=0.5, label=ct)

    ax.set_xlabel("Oracle pseudotime (diffusion)", fontsize=11)
    ax.set_ylabel("Inferred pseudotime (velocity)", fontsize=11)
    ax.set_title(
        f"Trajectory recovery -- Spearman rho={rho:.4f} (p={pval:.2e})\n"
        f"n={valid.sum()} cells, threshold=0.50",
        fontsize=11, fontweight="bold",
    )
    ax.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.01, 1),
              loc="upper left", framealpha=0.9)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Perfect correlation")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, "05_pseudotime_correlation")


def plot_benchmark_task2(result: object) -> None:
    """Bar chart of rank preservation results per cell-type pair."""
    pairwise = result.rank_preservation.pairwise_results  # type: ignore[attr-defined]
    if not pairwise:
        return

    pairs = list(pairwise.keys())
    labels = [f"{a}\n→ {b}" for a, b in pairs]
    passes = [
        pairwise[p]["direction_correct"] and pairwise[p]["significant"]
        for p in pairs
    ]
    early_means = [pairwise[p]["early_mean_vpt"] for p in pairs]
    late_means = [pairwise[p]["late_mean_vpt"] for p in pairs]
    colours = ["#54B06D" if p else "#E25C4C" for p in passes]

    x = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(max(9, len(pairs) * 1.2), 5))
    ax.bar(x - 0.2, early_means, 0.35, label="Earlier type", color="#4C8BE2", alpha=0.8)
    ax.bar(x + 0.2, late_means, 0.35, label="Later type",
           color=[c for c in colours], alpha=0.8)

    for i, (p, label) in enumerate(zip(passes, labels)):
        ax.text(i, max(early_means[i], late_means[i]) + 0.02,
                "PASS" if p else "FAIL",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#54B06D" if p else "#E25C4C")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean velocity pseudotime", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Rank preservation -- {sum(passes)}/{len(passes)} pairs correctly ordered",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    _save(fig, "06_benchmark_task2_rank_preservation")


def plot_benchmark_summary(result: object) -> None:
    """Summary panel: all three benchmark task results."""
    t1 = result.trajectory_recovery  # type: ignore[attr-defined]
    t2 = result.rank_preservation  # type: ignore[attr-defined]
    t3 = result.hidden_branch  # type: ignore[attr-defined]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Task 1: Gauge chart for Spearman rho
    ax = axes[0]
    rho = t1.spearman_rho
    colour = "#54B06D" if t1.passes_threshold else "#E25C4C"
    ax.barh(["Spearman rho"], [rho], color=colour, height=0.4)
    ax.axvline(settings.trajectory_rho_threshold, color="grey",
               linestyle="--", linewidth=1.5, label=f"threshold={settings.trajectory_rho_threshold}")
    ax.set_xlim(0, 1.05)
    ax.set_title("Task 1: Trajectory Recovery\n(Global Spearman)", fontsize=10, fontweight="bold")
    ax.text(rho + 0.02, 0, f"rho={rho:.4f}", va="center", fontsize=10)
    ax.legend(fontsize=8)
    status = "PASS" if t1.passes_threshold else "FAIL"
    ax.text(0.5, -0.35, status, transform=ax.transAxes, ha="center",
            fontsize=13, fontweight="bold",
            color="#54B06D" if t1.passes_threshold else "#E25C4C")

    # Task 2: Fraction bar
    ax = axes[1]
    pct = t2.pct_pairs_passing / 100
    colour2 = "#54B06D" if t2.overall_passes else "#E25C4C"
    ax.barh(["% pairs correct"], [pct], color=colour2, height=0.4)
    ax.axvline(0.75, color="grey", linestyle="--", linewidth=1.5, label="threshold=75%")
    ax.set_xlim(0, 1.05)
    ax.set_title("Task 2: Rank Preservation\n(Mann-Whitney U)", fontsize=10, fontweight="bold")
    ax.text(pct + 0.02, 0, f"{t2.n_pairs_passing}/{t2.n_pairs_tested}", va="center", fontsize=10)
    ax.legend(fontsize=8)
    status2 = "PASS" if t2.overall_passes else "FAIL"
    ax.text(0.5, -0.35, status2, transform=ax.transAxes, ha="center",
            fontsize=13, fontweight="bold",
            color="#54B06D" if t2.overall_passes else "#E25C4C")

    # Task 3: rho comparison bar
    ax = axes[2]
    bars = ax.bar(["Full dataset", "Epsilon masked"],
                  [t3.spearman_rho_full, t3.spearman_rho_masked],
                  color=["#4C8BE2", "#54B06D" if abs(t3.rho_drop) < 0.05 else "#E2A43C"],
                  width=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Task 3: Hidden Branch Recovery\n(Epsilon masked, rho drop={t3.rho_drop:.4f})",
                 fontsize=10, fontweight="bold")
    for bar, val in zip(bars, [t3.spearman_rho_full, t3.spearman_rho_masked]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    status3 = "ROBUST" if abs(t3.rho_drop) < 0.05 else "DEGRADED"
    ax.text(0.5, -0.35, status3, transform=ax.transAxes, ha="center",
            fontsize=13, fontweight="bold",
            color="#54B06D" if status3 == "ROBUST" else "#E2A43C")

    fig.suptitle(
        "RNA Velocity Trajectory Benchmark -- Pancreas Endocrinogenesis",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, "07_benchmark_summary")


def generate_all_figures(
    adata: ad.AnnData,
    result: object,
) -> None:
    """Generate all evidence figures and save to evidence/figures/."""
    logger.info("generating_figures: %s", FIGURE_DIR)

    plot_umap_clusters(adata)
    plot_umap_velocity_stream(adata)
    plot_umap_pseudotime(
        adata, "dpt_pseudotime",
        "Oracle pseudotime (diffusion map, rooted at Ductal)",
        "03_umap_pseudotime_oracle",
    )
    plot_umap_pseudotime(
        adata, "velocity_pseudotime",
        "Inferred pseudotime (RNA velocity random walk)",
        "04_umap_pseudotime_velocity",
    )
    plot_pseudotime_correlation(adata)
    plot_benchmark_task2(result)
    plot_benchmark_summary(result)

    logger.info("all_figures_generated: 7 plots saved to %s", FIGURE_DIR)
