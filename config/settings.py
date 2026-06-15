"""Pipeline configuration.

All parameters are centralised here. Override via environment variables
prefixed VEL_ or a .env file (see .env.example).
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class PipelineSettings(BaseSettings):
    """RNA velocity trajectory pipeline settings."""

    # ── Paths ──────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    evidence_dir: Path = Path(__file__).resolve().parent.parent / "evidence"

    # ── Dataset ────────────────────────────────────────────────────────
    # Pancreatic endocrinogenesis, Bastidas-Ponce et al. 2019
    # Downloaded via scvelo.datasets.pancreas() from figshare
    dataset_name: str = "Pancreas endocrinogenesis (Bastidas-Ponce 2019)"
    dataset_filename: str = "endocrinogenesis_day15.h5ad"

    # ── Preprocessing ──────────────────────────────────────────────────
    n_top_genes: int = 2000
    n_pcs: int = 30
    n_neighbors: int = 30   # scvelo recommends 30 for velocity graph

    # ── Velocity ───────────────────────────────────────────────────────
    velocity_mode: str = "stochastic"  # 'stochastic' or 'dynamical'
    velocity_n_top_genes: int = 2000

    # ── Clustering ─────────────────────────────────────────────────────
    leiden_resolution: float = 0.8

    # ── Known biology (Pancreas endocrinogenesis) ──────────────────────
    # Root state for diffusion pseudotime computation.
    # Ductal cells are the earliest progenitors in the endocrine lineage.
    # dpt_pseudotime is computed via sc.tl.dpt() rooted here during Phase 2.
    root_cell_type: str = "Ductal"
    # Terminal fates for branch recovery benchmark
    terminal_cell_types: list[str] = ["Alpha", "Beta", "Delta", "Epsilon"]
    # Full known ordering from progenitor to terminal
    known_ordering: list[str] = [
        "Ductal",
        "Ngn3 low EP",
        "Ngn3 high EP",
        "Pre-endocrine",
        "Beta",
        "Alpha",
        "Delta",
        "Epsilon",
    ]

    # ── Benchmark ──────────────────────────────────────────────────────
    benchmark_seed: int = 42
    # Minimum Spearman rho to consider trajectory recovery successful
    trajectory_rho_threshold: float = 0.5
    # Cell type to mask in hidden-branch benchmark
    hidden_branch_type: str = "Epsilon"

    # ── Reproducibility ────────────────────────────────────────────────
    random_seed: int = 42

    model_config = {
        "env_prefix": "VEL_",
        "env_file": ".env",
        "extra": "ignore",
    }


settings = PipelineSettings()
