# scvelo-trajectory-benchmark

**RNA velocity trajectory inference on pancreatic endocrinogenesis with a three-task benchmark evaluating trajectory recovery, rank preservation, and robustness to missing branches**

[![CI](https://github.com/gbadedata/scvelo-trajectory-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/gbadedata/scvelo-trajectory-benchmark/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![scvelo 0.3](https://img.shields.io/badge/scvelo-0.3-green.svg)](https://scvelo.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Most RNA velocity tutorials end at a streamline plot. They compute velocities, draw arrows on a UMAP, and call it done. They do not ask: **how accurately does the inferred developmental ordering match the known biology, and does the velocity signal hold up when rare cell populations are removed?**

This project answers those questions. It applies the deterministic RNA velocity model to pancreatic endocrinogenesis and evaluates the result through a three-task benchmark framework. The oracle is diffusion pseudotime computed by an independent method. The benchmark measures Spearman rank correlation between the oracle and the velocity-inferred pseudotime, tests whether known progenitor-to-terminal transitions are statistically preserved, and runs a perturbation experiment that removes the rarest terminal cell type and checks whether the pipeline recovers the remaining trajectory without degradation.

---

## Quick start

```bash
git clone git@github.com:gbadedata/scvelo-trajectory-benchmark.git
cd scvelo-trajectory-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.pipeline
```

The pipeline downloads the pancreas dataset automatically on first run (~50 MB via scvelo/figshare), runs all five phases, generates 7 figures in `evidence/figures/`, writes `evidence/reports/benchmark_report.json`, and prints a structured summary to stdout. Fully reproducible from a fresh clone.

---

## Dataset

**Pancreatic endocrinogenesis, Bastidas-Ponce et al. 2019** -- 3,696 pancreatic cells from mouse embryo day 15.5, profiled at single-cell resolution. The dataset captures differentiation from Ductal progenitor cells through two Ngn3-expressing progenitor stages to four terminal hormone-producing fates: Alpha (glucagon), Beta (insulin), Delta (somatostatin), and Epsilon (ghrelin).

This is the canonical RNA velocity benchmark dataset used in the Bergen et al. 2020 (Nature Biotechnology) scVelo paper. Using it enables direct comparison against published results.

---

## Key results

**Benchmark outcomes:**

| Task | Metric | Result | Status |
|---|---|---|---|
| Task 1: Trajectory recovery | Spearman rho (velocity vs oracle pseudotime) | 0.8926 | PASS |
| Task 2: Rank preservation | Pairs correctly ordered (Mann-Whitney U) | 5/7 (71.4%) | FAIL |
| Task 3: Hidden branch recovery | Rho drop after masking Epsilon cells | 0.0029 | ROBUST |

**Biological validators (independent of oracle):**

| Validator | Score | Status |
|---|---|---|
| Root cell ordering (Ductal has lowest pseudotime) | 1.000 | PASS |
| Velocity gene coverage (>=100 genes with splicing dynamics) | 1.000 | PASS |
| Velocity confidence (mean directional consistency) | 0.940 | PASS |

**Pipeline statistics:**

| Metric | Value |
|---|---|
| Cells | 3,696 |
| Genes after filtering | 7,197 |
| Velocity genes | 1,598 |
| Mean velocity confidence | 0.705 |
| Tests passing | 99 |
| Test coverage | 92% |
| Pipeline runtime | 47.2 seconds |

---

## Task 2 detail: per-pair rank preservation

| Pair | Earlier mean vpt | Later mean vpt | Result |
|---|---|---|---|
| Ductal vs Ngn3 low EP | 0.170 | 0.233 | PASS |
| Ngn3 low EP vs Ngn3 high EP | 0.233 | 0.688 | PASS |
| Ngn3 high EP vs Pre-endocrine | 0.688 | 0.920 | PASS |
| Pre-endocrine vs Alpha | 0.920 | 0.928 | PASS |
| Pre-endocrine vs Beta | 0.920 | 0.959 | PASS |
| Pre-endocrine vs Delta | 0.920 | 0.839 | FAIL |
| Pre-endocrine vs Epsilon | 0.920 | 0.891 | FAIL |

The entire linear progenitor axis (Ductal through Pre-endocrine) and two of the four terminal fates are correctly ordered. The two failing pairs, Pre-endocrine vs Delta and Pre-endocrine vs Epsilon, correspond to the rarest cell populations in the dataset (Delta: ~168 cells, Epsilon: ~73 cells). Sparse populations produce fewer edges in the velocity graph, reducing the directional signal available to velocity pseudotime. This is a known limitation of velocity pseudotime for rare cell types, correctly surfaced by the benchmark.

---

## Benchmark framework design

### Why three tasks?

A single Spearman correlation cannot distinguish a pipeline that identifies global developmental direction from one that correlates with pseudotime only on abundant cell types, fails on rare populations, or depends on the presence of terminal cells to define directionality.

The three tasks test three independently important properties:

**Task 1 -- Global trajectory recovery.** Spearman rank correlation between RNA velocity pseudotime and oracle diffusion pseudotime across all 3,696 cells. These are computed by two genuinely independent methods: diffusion pseudotime uses transcriptional similarity in a kNN graph; RNA velocity uses the ratio of unspliced precursor to mature spliced mRNA. High correlation between two independent measures of the same biology is meaningful evidence of trajectory recovery.

**Task 2 -- Rank preservation.** For each biologically defined cell-type pair, a one-sided Mann-Whitney U test evaluates whether the earlier type has significantly lower velocity pseudotime. This catches cases where global correlation is high but specific developmental transitions are misordered. Note that the four terminal fates (Alpha, Beta, Delta, Epsilon) are parallel branches from Pre-endocrine -- no ordering among them is biologically defined. Each is therefore tested against Pre-endocrine only, not against each other.

**Task 3 -- Hidden branch recovery.** All Epsilon cells (n=73) are removed from the dataset. The full velocity pipeline re-runs from scratch on the masked data. Spearman rho on the remaining 3,623 cells is compared against the full-dataset rho, and the linear progenitor ordering is verified. A rho drop near zero means velocity is detecting global transcriptional momentum encoded throughout the trajectory. A large drop would indicate the pipeline was relying on the presence of the terminal state to define directionality.

### Oracle design

The oracle is diffusion pseudotime computed by `sc.tl.dpt()` rooted at Ductal cells during preprocessing. It is computed fresh -- not loaded from a stored file -- using an independent method (diffusion kernel on the transcriptional similarity graph). Evaluating an RNA velocity prediction against an independently computed oracle avoids circularity.

---

## Evidence

| Figure | Description |
|---|---|
| `evidence/screenshots/01_umap_clusters.png` | UMAP coloured by cell-type clusters |
| `evidence/screenshots/02_umap_velocity_stream.png` | UMAP with RNA velocity streamlines overlaid |
| `evidence/screenshots/05_pseudotime_correlation.png` | Scatter: oracle diffusion pseudotime vs velocity pseudotime (rho=0.8926) |
| `evidence/screenshots/06_benchmark_task2_rank_preservation.png` | Bar chart: per-pair velocity pseudotime means with PASS/FAIL labels |
| `evidence/screenshots/07_benchmark_summary.png` | Three-task benchmark summary panel |

---

## Architecture

```
Raw h5ad (3,696 cells x 27,998 genes)
          |
          v
[Phase 1]  Data Loader
           scv.datasets.pancreas() --> cached h5ad (~50 MB, figshare)
           Structure validation: spliced/unspliced layers + clusters column
          |
          v
[Phase 2]  Preprocessing + Oracle
           scv.pp.filter_genes (min_shared_counts=20) --> 7,197 genes
           scv.pp.normalize_per_cell
           sc.pp.log1p
           sc.pp.pca (n_comps=30) + sc.pp.neighbors (k=30)
           sc.tl.umap
           sc.tl.dpt (rooted at Ductal cells) --> oracle dpt_pseudotime
          |
          v
[Phase 3]  RNA Velocity
           scv.pp.moments (smoothed spliced Ms, unspliced Mu)
           scv.tl.velocity (deterministic steady-state model)
           scv.tl.velocity_graph (cosine correlations, cell-to-cell transitions)
           scv.tl.velocity_pseudotime (random walk distance from root)
           scv.tl.velocity_confidence (per-cell directional consistency)
          |
          v
[Phase 4]  Benchmark Framework
           Task 1: Spearman rho (velocity_pseudotime vs dpt_pseudotime)
           Task 2: Mann-Whitney U, 7 biologically defined consecutive pairs
           Task 3: Mask Epsilon cells, re-run velocity, compare rho and ordering
           Validators: root ordering, gene coverage, confidence
           JSON report: all metrics, pairwise table, validator evidence
          |
          v
[Phase 5]  Visualisation + Report
           7 PNG figures --> evidence/figures/
           benchmark_report.json --> evidence/reports/
           pancreas_annotated.h5ad --> data/
```

---

## Engineering decisions and challenges

### Challenge 1: dpt_pseudotime is not pre-stored in the pancreas h5ad

The first implementation assumed `dpt_pseudotime` would exist in the downloaded AnnData. The pancreas h5ad from scvelo's repository does not pre-compute it.

**Fix:** Compute diffusion pseudotime during Phase 2 preprocessing using `sc.tl.dpt()` rooted at Ductal cells. This produces a stronger benchmark: the oracle is computed fresh using an independent method, removing any possibility of circularity with the velocity computation.

### Challenge 2: HVG selection after normalisation causes infinity errors

Running `sc.pp.highly_variable_genes` after `scv.pp.normalize_per_cell` raised `ValueError: cannot specify integer bins when input data contains infinity`.

**Root cause:** Some genes have near-zero total counts after per-cell normalisation. When pandas attempts to bin gene expression levels to compute HVG dispersion, infinity values in the binning array trigger the error.

**Fix:** Remove the HVG selection step from preprocessing entirely. The scvelo workflow handles its own internal feature selection within `scv.pp.moments`. Running `sc.pp.highly_variable_genes` before log1p on normalised data is architecturally incorrect for the scvelo pipeline. Removing it also eliminates a redundant processing step.

### Challenge 3: Stochastic velocity model incompatible with NumPy 2.x

`scv.tl.velocity(mode='stochastic')` raised `ValueError: setting an array element with a sequence` on both the synthetic fixtures and the real pancreas data.

**Root cause:** scvelo 0.3.x `leastsq_generalized` assigns the result of a matrix operation into a scalar slot:

```python
gamma[i] = np.linalg.pinv(A.T.dot(A)).dot(A.T.dot(y[:, i]))
```

NumPy 2.x returns a 1-element array from this expression rather than a scalar. Assigning a 1-element array into a pre-allocated `float32` array element raises `ValueError`. This is a known upstream incompatibility (github.com/theislab/scvelo/issues/966).

**Fix:** Switched to `mode='deterministic'` (LaManno et al. 2018 steady-state model). The deterministic model correctly recovers the pancreas trajectory and is the foundational method on which the stochastic extension was built. It does not trigger the NumPy 2.x incompatibility.

### Challenge 4: Terminal fates are parallel branches, not a linear chain

The initial `known_ordering` listed Alpha, Beta, Delta, and Epsilon as a sequential chain. Task 2 reported Beta > Alpha and Alpha > Delta as failures.

**Root cause:** Alpha, Beta, Delta, and Epsilon are four parallel terminal fates that all branch from Pre-endocrine simultaneously. They represent independent hormone-producing lineages with no defined ordering relative to one another. Imposing a linear sequence Beta < Alpha < Delta is biologically incorrect.

**Fix:** Restructured the ordering to test only the linear progenitor axis (Ductal through Pre-endocrine) and each terminal fate separately against Pre-endocrine. This correctly captures the branching topology: a linear trunk with four parallel terminal branches, not a chain.

---

## Project structure

```
scvelo-trajectory-benchmark/
|
+-- config/
|   +-- settings.py              Pydantic Settings: all parameters, VEL_ env prefix
|
+-- src/
|   +-- pipeline.py              End-to-end runner: python3 -m src.pipeline
|   +-- data_loader.py           Download pancreas dataset, validate structure
|   +-- preprocessing.py         Filter, normalise, PCA, UMAP, oracle DPT
|   +-- velocity.py              Moments, velocity, graph, pseudotime, confidence
|   +-- visualization.py         7 publication-quality PNG figures
|   +-- benchmark/
|       +-- evaluator.py         3-task benchmark: rho, rank preservation, hidden branch
|       +-- validators.py        Root ordering, gene coverage, confidence validators
|
+-- tests/
|   +-- conftest.py              Synthetic trajectory fixtures with embedded velocity signal
|   +-- test_fixtures.py         16 fixture validation tests
|   +-- test_data_loader.py      11 tests: loading, validation, format detection
|   +-- test_preprocessing.py    20 tests: filter, normalise, PCA, DPT oracle
|   +-- test_velocity.py         23 tests: moments, fitting, graph, pseudotime, confidence
|   +-- test_benchmark_evaluator.py  14 tests: all 3 tasks, scoring, edge cases
|   +-- test_benchmark_validators.py 15 tests: root ordering, gene coverage, confidence
|
+-- evidence/
|   +-- figures/                 7 PNG figures (generated by pipeline)
|   +-- reports/                 benchmark_report.json (generated by pipeline)
|   +-- screenshots/             Key figures for portfolio evidence
|
+-- data/                        Downloaded datasets (gitignored)
+-- .github/workflows/ci.yml     GitHub Actions: ruff lint + pytest on every push
+-- requirements.txt
+-- pyproject.toml
+-- .env.example
```

---

## Configuration

All parameters are configurable via environment variables prefixed `VEL_`. No source code changes needed.

```bash
# Mask a different terminal branch in Task 3
VEL_HIDDEN_BRANCH_TYPE=Alpha python3 -m src.pipeline

# Load a custom h5ad instead of the pancreas dataset
python3 -m src.pipeline --data /path/to/your_data.h5ad

# Adjust neighbour count for the kNN graph
VEL_N_NEIGHBORS=20 python3 -m src.pipeline
```

See `.env.example` for the full parameter list with defaults.

---

## Running tests

```bash
python3 -m pytest tests/ -v
```

Tests run entirely on synthetic trajectory fixtures (200 cells, seeded RNG, no network access). The fixtures embed a genuine velocity signal: the first 50 genes have unspliced expression that leads spliced expression along the trajectory axis, and `test_velocity_signal_in_early_cells` explicitly verifies that early-trajectory cells have a higher unspliced-to-spliced ratio than late-trajectory cells. This biological constraint ensures the fixture tests real velocity behaviour, not trivially passing assertions.

---

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.pipeline
# Expected: rho=0.8926, 5/7 pairs, hidden rho drop=0.0029, runtime ~47s
```

All random operations use `random_seed=42`. The dataset is deterministic (same file from figshare on every download). Results are fully reproducible.

---

## Stack

| Category | Tools |
|---|---|
| RNA velocity | scvelo 0.3.4 |
| Single-cell analysis | scanpy 1.12, anndata 0.12 |
| Statistical testing | scipy (Spearman rho, Mann-Whitney U) |
| Scientific computing | numpy, pandas |
| Visualisation | matplotlib 3.10 |
| Configuration | Pydantic Settings 2.0 |
| Logging | structlog |
| Testing | pytest, pytest-cov |
| Linting | ruff |
| CI | GitHub Actions |
| Python | 3.12 |

---

## References

Bastidas-Ponce et al. (2019). Comprehensive single cell mRNA profiling reveals a detailed roadmap for pancreatic endocrinogenesis. *Development*, 146(12).

Bergen et al. (2020). Generalizing RNA velocity to transient cell states through dynamical modeling. *Nature Biotechnology*, 38, 1408-1414.

La Manno et al. (2018). RNA velocity of single cells. *Nature*, 560, 494-498.

Haghverdi et al. (2016). Diffusion pseudotime robustly reconstructs lineage branching. *Nature Methods*, 13, 845-848.

---

## Author

**O.J. Odimayo** -- Bioinformatics Data Engineer

MSc Applied Data Science | BSc Genetics and Molecular Biology

[github.com/gbadedata](https://github.com/gbadedata) · [linkedin.com/in/oluwagbade-odimayo-](https://www.linkedin.com/in/oluwagbade-odimayo-) · oluwagbadeodimayo@gmail.com
