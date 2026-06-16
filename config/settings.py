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
    # 'deterministic' is used because scvelo 0.3.x stochastic mode contains
    # a NumPy 2.x incompatibility in leastsq_generalized:
    #   gamma[i] = np.linalg.pinv(A.T.dot(A)).dot(...)
    # returns a 1-element array in NumPy 2.x instead of a scalar, causing:
    #   ValueError: setting an array element with a sequence
    # The deterministic (steady-state) model is the well-validated baseline
    # used in the original LaManno et al. 2018 paper and produces correct
    # trajectory ordering on the pancreas dataset.
    # Reference: github.com/theislab/scvelo/issues/966
    velocity_mode: str = "deterministic"
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
    # Known ordering for rank preservation benchmark (Task 2).
    # Only the linear progenitor progression is tested -- the four terminal
    # fates (Alpha, Beta, Delta, Epsilon) branch simultaneously from
    # Pre-endocrine and have no defined relative ordering among themselves.
    # Testing Beta < Alpha < Delta would impose a false linear ordering on
    # what is a biological fan-out. The benchmark correctly stops at the
    # Pre-endocrine -> terminal transition.
    known_ordering: list[str] = [
        "Ductal",
        "Ngn3 low EP",
        "Ngn3 high EP",
        "Pre-endocrine",
        "Alpha",     # representative terminal fate: Pre-endocrine -> Alpha
    ]
    # Terminal comparison pairs to test separately (each vs Pre-endocrine)
    terminal_ordering_pairs: list[list[str]] = [
        ["Pre-endocrine", "Alpha"],
        ["Pre-endocrine", "Beta"],
        ["Pre-endocrine", "Delta"],
        ["Pre-endocrine", "Epsilon"],
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
