"""End-to-end pipeline runner.

Orchestrates all phases in sequence and writes a structured benchmark
report to evidence/reports/benchmark_report.json.

Usage:
    python3 -m src.pipeline
    python3 -m src.pipeline --data path/to/custom.h5ad
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import structlog

from config.settings import settings

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


def run_pipeline(data_path: str | None = None) -> dict:
    """Execute the full pipeline and return the benchmark report.

    Args:
        data_path: Optional path to a local h5ad file. If None,
            downloads the pancreas dataset via scvelo.

    Returns:
        Benchmark report dict (also written to evidence/reports/).
    """
    start = datetime.now(timezone.utc)
    log.info("pipeline_started", dataset=settings.dataset_name,
             timestamp=start.isoformat())

    # Phase 1: Load
    from src.data_loader import get_dataset
    adata = get_dataset(filepath=data_path)
    log.info("phase1_complete", n_cells=adata.n_obs, n_genes=adata.n_vars)

    # Phase 2: Preprocessing + oracle pseudotime
    from src.preprocessing import run_preprocessing
    adata, pp_report = run_preprocessing(adata)
    log.info(
        "phase2_complete",
        n_cells=pp_report.n_cells,
        n_genes=pp_report.n_genes,
        dpt_range=pp_report.dpt_pseudotime_range,
    )

    # Phase 3: RNA velocity
    from src.velocity import run_velocity
    adata, vel_report = run_velocity(adata)
    log.info(
        "phase3_complete",
        mode=vel_report.mode,
        n_velocity_genes=vel_report.n_velocity_genes,
        mean_confidence=round(vel_report.mean_velocity_confidence, 4),
    )

    # Phase 4: Benchmark
    from src.benchmark.evaluator import run_benchmark
    from src.benchmark.validators import run_all_validators

    result = run_benchmark(adata, run_velocity)
    validator_results = run_all_validators(adata)

    log.info(
        "phase4_complete",
        spearman_rho=result.trajectory_recovery.spearman_rho,
        rank_pairs=f"{result.rank_preservation.n_pairs_passing}/{result.rank_preservation.n_pairs_tested}",
        hidden_rho_drop=result.hidden_branch.rho_drop,
        validators_passed=sum(v.passed for v in validator_results),
    )

    # Phase 5: Visualisation
    from src.visualization import generate_all_figures
    generate_all_figures(adata, result)
    log.info("phase5_complete", figures_dir=str(settings.evidence_dir / "figures"))

    # Build and save report
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    report = {
        "metadata": {
            "timestamp": start.isoformat(),
            "dataset": settings.dataset_name,
            "runtime_seconds": round(elapsed, 1),
        },
        "pipeline": {
            "n_cells": pp_report.n_cells,
            "n_genes": pp_report.n_genes,
            "n_velocity_genes": vel_report.n_velocity_genes,
            "velocity_mode": vel_report.mode,
            "mean_velocity_confidence": round(vel_report.mean_velocity_confidence, 4),
            "dpt_pseudotime_range": list(pp_report.dpt_pseudotime_range),
        },
        "benchmark": result.summary,
        "validators": [
            {"name": v.name, "passed": v.passed, "score": v.score, "details": v.details}
            for v in validator_results
        ],
    }

    report_dir = settings.evidence_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("pipeline_complete", runtime_seconds=round(elapsed, 1),
             report=str(report_path))

    # Save annotated AnnData
    output_path = settings.data_dir / "pancreas_annotated.h5ad"
    adata.write_h5ad(output_path)
    log.info("adata_saved", path=str(output_path))

    return report


def _print_summary(report: dict) -> None:
    bm = report["benchmark"]
    t1 = bm["task1_trajectory_recovery"]
    t2 = bm["task2_rank_preservation"]
    t3 = bm["task3_hidden_branch"]
    pl = report["pipeline"]

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY -- RNA Velocity Trajectory Inference")
    print("=" * 65)
    print(f"  Dataset:          {report['metadata']['dataset']}")
    print(f"  Cells:            {pl['n_cells']}")
    print(f"  Velocity genes:   {pl['n_velocity_genes']}")
    print(f"  Mean confidence:  {pl['mean_velocity_confidence']:.4f}")
    print()
    print("Task 1 -- Trajectory Recovery (Spearman rho)")
    print(f"  rho = {t1['spearman_rho']:.4f}  |  {'PASS' if t1['passes'] else 'FAIL'}")
    print()
    print("Task 2 -- Rank Preservation (Mann-Whitney U)")
    print(f"  {t2['pairs_passing']} pairs ({t2['pct_passing']:.1f}%)  |  {'PASS' if t2['passes'] else 'FAIL'}")
    print()
    print("Task 3 -- Hidden Branch Recovery")
    print(f"  rho_full={t3['rho_full']:.4f}  rho_masked={t3['rho_masked']:.4f}  "
          f"drop={t3['rho_drop']:.4f}")
    print(f"  Ordering preserved: {t3['ordering_preserved']}")
    print()
    print("Biological validators:")
    for v in report["validators"]:
        status = "PASS" if v["passed"] else "FAIL"
        print(f"  [{status}] {v['name']}: {v['score']:.4f}")
    print(f"\nRuntime: {report['metadata']['runtime_seconds']}s")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="scvelo-trajectory-benchmark: RNA velocity pipeline"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to local h5ad file. Defaults to pancreas download.",
    )
    args = parser.parse_args()

    try:
        report = run_pipeline(data_path=args.data)
        _print_summary(report)
        sys.exit(0)
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        sys.exit(1)
