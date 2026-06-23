# AttackPath-PGAS Supplementary Implementation

This archive contains the reference implementation for AttackPath-PGAS, a
hierarchical rare-event detector that combines event-level discriminative risk
estimation with Particle Gibbs with Ancestor Sampling (PGAS) over temporal
attack states.

The implementation supports:

- memory-bounded CSV ingestion and temporal-window construction;
- chronological train, validation, and test partitioning;
- leakage-safe event-risk and window-level baseline models;
- binary-state PGAS posterior inference with hybrid emissions;
- risk-anchored posterior fusion and segment-aware decision policies;
- classification, calibration, uncertainty, transition, and segment metrics;
- convergence diagnostics, five-seed evaluation, and runtime/memory profiling;
- manuscript-aligned tables and figures written to an output directory.

## Archive layout

```text
config/default.yaml                 Experiment configuration
notebooks/AttackPath_PGAS_Implementation.ipynb
                                    End-to-end executable notebook (no outputs)
scripts/run_experiment.py           Command-line experiment entry point
src/attackpath_pgas/                Importable implementation modules
tests/                              Deterministic unit and smoke tests
requirements.txt                    Runtime and notebook dependencies
pyproject.toml                      Package and test configuration
```

## Data

The CICAPT-IIoT network files are not redistributed. Place the Phase 1 and
Phase 2 CSV files in a local directory and identify that directory with
`--dataset-root` or `ATTACKPATH_DATASET_ROOT`. Column names are detected from
common source, destination, label, event-type, and timestamp aliases. Explicit
column overrides are available in `config/default.yaml`.

Expected high-level protocol:

1. Phase 1 contributes benign background windows to training.
2. The first 50% of Phase 2 windows are assigned to training.
3. The next 25% and final 25% are assigned to validation and test.
4. Model selection, calibration, and decision thresholds use training and
   validation data only.
5. Test labels are used exclusively for final evaluation.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Command-line execution

```bash
python scripts/run_experiment.py \
  --config config/default.yaml \
  --dataset-root /path/to/CICAPT-IIoT \
  --output-dir results/main_run
```

For a fast installation check that does not require the research dataset:

```bash
pytest
python scripts/run_experiment.py --smoke-test --output-dir results/smoke
```

## Notebook execution

Start Jupyter from the archive root so the editable package is importable:

```bash
jupyter lab notebooks/AttackPath_PGAS_Implementation.ipynb
```

The distributed notebook contains no stored outputs or execution counts.

## Reproducibility controls

- The default seed is 42.
- Five-seed evaluation uses 42, 123, 202, 777, and 999.
- All splits are chronological.
- Thresholds and temporal policies are selected on validation data.
- PGAS chains use non-overlapping deterministic random-number streams.
- Configuration and environment metadata are written with every run.

## Computational reporting

Runtime and memory measurements depend on hardware, thread count, data type,
and whether event-risk or posterior values are cached. The benchmarking module
reports wall-clock seconds, throughput, and incremental peak resident memory.
Posterior sampling time is reported separately from predictive inference time.

## Outputs

The command-line and notebook workflows create:

```text
results/<run>/
  figures/
  logs/
  models/
  tables/
  run_config.json
  environment.json
```

No manuscript result is embedded as a hard-coded model output. Tables and
figures are produced from the supplied dataset and saved experiment artifacts.

