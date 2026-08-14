# AttackPath-PGAS

**AttackPath-PGAS** is a hierarchical Bayesian framework for uncertainty-aware attack-state trajectory inference in Industrial Internet of Things (IIoT) traffic. The implementation connects full-row event streaming, rare-event event-level learning, temporal-window representation, calibrated window risk, a binary state-space model, Particle Gibbs with Ancestor Sampling (PGAS), posterior smoothing/uncertainty quantification, and validation-governed alert generation.

This package is prepared for a clean GitHub upload and contains only:

```text
AttackPath_PGAS_Implementation.ipynb
README.md
requirements.txt
```

The notebook is distributed **without saved outputs**. Runtime artifacts are generated under `artifacts/` and are not included in this ZIP.

---

## 1. Primary dataset — CIC APT IIoT Dataset 2024 / CICAPT-IIoT

**Official source:** https://www.unb.ca/cic/datasets/iiot-dataset-2024.html

The Canadian Institute for Cybersecurity dataset provides an APT-oriented IIoT testbed with network/provenance evidence. AttackPath-PGAS uses the **network-event files** from both phases while preserving native file/row order as pseudo-time.

### Documented primary-corpus profile

| Phase | Events | Windows | Attack-event rows | Attack-active windows |
|---|---:|---:|---:|---:|
| Phase 1 | 12,062,396 | 2,413 | 0 | 0 |
| Phase 2 | 9,536,823 | 1,908 | 1,004 | 62 |
| **Total** | **21,599,219** | **4,321** | **1,004** | **62** |

The primary window is **5,000 rows, non-overlapping**. The final partial window in a phase is retained. A window is attack-active when:

```text
attack_event_count >= 3 OR attack_event_fraction >= 0.001
```

The documented chronological ownership is:

| Split | Windows | Phase coverage | Attack windows |
|---|---:|---|---:|
| Training | 3,367 | all 2,413 Phase-1 + first 954 Phase-2 | 35 |
| Validation | 477 | next 25% of Phase 2 | 12 |
| Test | 477 | final 25% of Phase 2 | 15 |

### Suggested local layout

```text
data/
└── CICAPT-IIoT/
    ├── Phase1/
    │   └── ...network CSV/Parquet files...
    └── Phase2/
        └── ...network CSV/Parquet files...
```

Set either:

```bash
export ATTACKPATH_DATA_ROOT=/absolute/path/to/CICAPT-IIoT
```

or edit `CFG["data"]["root"]` in the notebook. If filenames/directories do not contain Phase-1/Phase-2 markers, set the explicit `phase1_files` and `phase2_files` lists in `CFG`.

> Do not place unrelated provenance/derived tables in the network-event folders unless they are explicitly intended as inputs.

---

## 2. External-validation datasets

External-validation adapters are included but disabled by default because downloaded layouts and ordering fields differ.

### Edge-IIoTset

- Dataset DOI / record: https://doi.org/10.21227/mbc1-1h68
- Related project/publication record: https://ro.ecu.edu.au/ecuworks2022-2026/552/

Edge-IIoTset is a realistic IoT/IIoT cybersecurity dataset generated using a purpose-built testbed. For the AttackPath-PGAS external protocol, benign traffic is mapped to state 0 and all attacks to state 1. The manuscript reports a **chronological device/capture split**. The notebook deliberately requires the local file/capture ordering to be supplied rather than inventing chronology from labels.

### ToN-IoT

- Official UNSW source: https://research.unsw.edu.au/projects/toniot-datasets

ToN-IoT contains heterogeneous Industry-4.0/IoT/IIoT sources including network traffic, telemetry and host data. The AttackPath-PGAS external protocol uses the network-oriented material with a **chronological source-file split** and a binary benign/attack mapping.

---

## 3. Environment

Recommended:

- Python 3.11
- 64-bit OS
- 16 GB RAM minimum for development
- 32–64 GB RAM recommended for the complete primary pipeline
- multi-core CPU
- GPU not required by the core PGAS implementation

The manuscript benchmark configuration recorded a 12-core CPU and 64 GB RAM. Runtime varies with CPU, storage, BLAS backend, package versions and optional XGBoost availability.

### Install

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\\Scripts\\activate       # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open `AttackPath_PGAS_Implementation.ipynb` and run cells sequentially after configuring the dataset path.

---

## 4. Primary implementation controls

The notebook centralizes the documented controls in a single configuration dictionary.

### Streaming/windowing

```text
chunk size                 200,000 rows
primary window              5,000 rows, non-overlapping
attack-fraction threshold   0.001
minimum attack events       3
max numeric window columns  65
max event-type levels       30
minimum feature variance    1e-10
max core features           90
```

### Feature selection and causal temporal context

```text
MI-selected features        35 (training labels only)
shift-screen candidates     45 (Phase-1 vs Phase-2-training covariates only)
lags                        1, 2, 3, 5
rolling windows             3, 5, 9
EWMA spans                  3, 7
temporal base-feature cap   35
```

The source protocol defines the ownership and number of shift candidates but does not identify a unique library primitive for its shift score. This repository therefore makes the rule explicit and auditable: **absolute standardized mean shift**, calculated only from Phase-1 and Phase-2-training covariates. It can be replaced through one function without changing the no-peeking boundary.

### Event learner

```text
maximum event features      45
training attack cap         350,000
training benign cap         650,000
validation attack cap       120,000
validation benign cap       240,000
sampling seed               2026
top-risk events/window      20
high-risk thresholds        .50, .70, .90, .95
```

Event models: balanced logistic regression, Extra Trees, histogram gradient boosting, and optional XGBoost.

For calibrated event model `m`, the documented validation score is:

```text
s_m = AUPR_m + 0.25 * best_F2_m
```

Positive scores are normalized into event-ensemble weights. The **complete event stream** is then rescored and aggregated by window using mean, standard deviation, maximum, top-20 mean, noisy-or and threshold-exceedance ratios.

### Window learner

The notebook includes:

- L2 and L1 logistic regression;
- random forest;
- Extra Trees;
- histogram gradient boosting;
- optional XGBoost;
- Isolation Forest background deviation;
- PCA reconstruction error;
- second-level calibrated score stackers.

For each calibrated window candidate:

```text
q_m = 0.45*AUPR + 0.30*best_F2 + 0.15*best_MCC + 0.10*AUROC
```

The hierarchical validation objective is:

```text
J = F1 + 0.25*F2 + 0.20*MCC + 0.25*Recall - 0.20*FPR + 0.10*AUPR
```

with primary constraints `Recall >= 0.65` and `FPR <= 0.05`.

---

## 5. PGAS model

The primary model contains two hidden states:

```text
0 = benign
1 = attack-active
```

### Priors

```text
pi     ~ Dirichlet([300, 2])
A[0]   ~ Dirichlet([140, 4])
A[1]   ~ Dirichlet([6, 42])

sigma^2 ~ InverseGamma(a0=3, b0=3)
mu | sigma^2 ~ Normal(m0=0, sigma^2/kappa0)
kappa0 = 0.05
```

This package follows the method/Appendix-B attack-row prior `[6, 42]` consistently.

### Hybrid emission

For state `k` at window `t`:

```text
log Psi(t,k)
 = lambda_G * log Normal(x_t | mu_k, diag(sigma_k^2))
 + lambda_R * log B_tk
```

Primary weights:

```text
lambda_R = 9.0
lambda_G = 0.005
```

### PGAS controls

```text
independent chains          2
particles                   200
iterations/chain            550
burn-in                     180
thinning                    5
retained draws/chain        74
pooled retained draws       148
uncertainty draws used      up to 80
```

The notebook implements conditional SMC, ancestor sampling, trajectory back-tracing and conjugate updates for the initial state, transition rows and diagonal-Gaussian emission parameters.

---

## 6. Posterior outputs

The notebook supports:

- forward filtering;
- full forward-backward smoothing;
- posterior mean attack probability;
- draw-wise 95% posterior probability intervals;
- posterior standard deviation;
- predictive entropy and normalized entropy;
- aleatoric variance;
- epistemic variance;
- mutual information;
- transition summaries;
- R-hat;
- bulk/tail ESS;
- MCSE;
- lag-1 autocorrelation.

Probability intervals are treated as posterior probability dispersion, **not** as binary-outcome coverage intervals. Posterior-predictive checking and simulation-based parameter recovery are kept separate.

---

## 7. Final PGAS-guided policy

Main search controls:

```text
risk-anchor weights          .80, .85, .90, .95, .97, .99
transition boost             .35
guarded PGAS noisy-or        .15
hysteresis low ratios        .50, .55
minimum segment lengths      1, 2, 3
merge gaps                   0, 1, 2
minimum validation recall    .65
maximum validation FPR       .05
minimum validation precision .20
```

The final validation score is:

```text
J_val = F1 + 0.35*F2 + 0.35*MCC + 0.25*Recall
        + 0.15*AUPR + 0.05*AUROC - 0.35*FPR
```

A `0.015` tie preference is added to PGAS-containing candidates. The selected calibration and policy are frozen before held-out test evaluation.

---

## 8. Leakage-control boundary

| Operation | Training | Validation | Test |
|---|---|---|---|
| feature ranking | fit | no | no |
| shift screen | Phase-1 + Phase-2-train covariates | no | no |
| scaler/imputer | fit | frozen apply | frozen apply |
| event/window learners | fit | calibration/selection only | no refit |
| PGAS parameter estimation | fit | no refit | no refit |
| probability calibration | no | fit | frozen apply |
| fusion/policy selection | no | fit | frozen apply |
| final metrics | development evidence | development evidence | **labels first consumed after prediction freeze** |

---

## 9. Notebook workflow

1. Imports/environment
2. Central configuration
3. Reproducibility manifest and SHA-256 utilities
4. Phase file discovery
5. Schema detection
6. Streaming windowization
7. Chronological split ownership
8. Training-only feature screening
9. Event sampling
10. Event models and validation calibration
11. Full-stream event rescoring
12. Event-to-window enrichment
13. Frozen final preprocessing
14. Window candidates/stackers
15. Hierarchical risk selection
16. PGAS inference
17. R-hat/ESS/MCSE/autocorrelation
18. posterior smoothing and uncertainty
19. PGAS-guided fusion and policy selection
20. frozen held-out evaluation
21. top-k and temporal evaluation
22. posterior-predictive / recovery helpers
23. sensitivity and robustness helpers
24. external-validation adapters
25. artifact integrity manifest

---

## 10. Runtime artifacts

A complete run produces, for example:

```text
artifacts/
├── config/
│   ├── config.json
│   └── config.sha256
├── manifests/
│   ├── dataset_manifest.csv
│   ├── environment.json
│   ├── run_manifest.json
│   └── artifact_integrity.csv
├── data/
│   ├── windows_base.parquet
│   ├── windows_enriched.parquet
│   └── split_indices.csv
├── features/
│   ├── mi_features.json
│   ├── shift_candidates.json
│   └── final_features.json
├── models/
├── posterior/
│   ├── retained_parameters.npz
│   ├── chain_monitor.csv
│   └── diagnostics.csv
├── predictions/
│   ├── validation_predictions.csv
│   ├── test_predictions.csv
│   ├── test_calibration_bins.csv
│   └── test_topk.csv
└── metrics/
    └── test_metrics.json
```

Raw datasets and large generated artifacts should normally be excluded from Git.

---

## 11. Reproducibility checklist

Preserve with any archival run:

- exact Phase-1 and Phase-2 filenames;
- SHA-256 for every input file;
- phase/file order;
- split indices;
- MI-selected 35 features;
- shift-screen 45 candidates;
- final feature list;
- global, robustness and chain-specific seeds;
- complete configuration JSON + SHA-256;
- Python/package versions;
- OS, CPU/RAM and available BLAS details;
- optional XGBoost status/version;
- retained posterior parameter arrays;
- chain diagnostics;
- frozen validation/test probabilities;
- alerts, segments and calibration bins.

---

## 12. Reference study values

The latest manuscript reports the following primary held-out CICAPT-IIoT values:

```text
Accuracy             0.99371
Balanced accuracy    0.93225
Precision            0.92857
Recall               0.86667
F1                    0.89655
MCC                   0.89388
AUROC                 0.99290
AUPR                  0.86540
ECE                   0.00650
```

These values are included as **reference metadata only**. The notebook never hard-codes them into model fitting or prediction.

Reported external-validation reference rows are also stored separately in the notebook for comparison, not fitting.

---

## 13. Optional dependency policy

XGBoost is used when import and fitting succeed. When unavailable, the notebook records it explicitly rather than silently substituting another learner. Comparisons across runs should therefore retain the environment manifest.

---

## 14. Related work / citation

When the AttackPath-PGAS article has final bibliographic details, cite the published article alongside this implementation.

Related uncertainty-aware graph-Bayesian IDS work from the same research programme:

- Osei-Wusu, F., Appiah, O., Mensah, P. K., Nimbe, P., & Donkoh, E. K. (2026). *A Graph-Based Bayesian Intrusion Detection Framework With No-U-Turn Sampling for Uncertainty-Aware IoT Security*. **Concurrency and Computation: Practice and Experience, 38**(14), e70817. https://doi.org/10.1002/cpe.70817
- Osei-Wusu, F., Appiah, O., Mensah, P. K., Nimbe, P., & Donkoh, E. K. (2026). *A graph-based Bayesian intrusion detection framework with Gibbs sampling for uncertainty-aware IoT security*. **Journal of Reliable Intelligent Environments, 12**, 9. https://doi.org/10.1007/s40860-026-00272-8

---

## 15. Responsible use

This implementation is for defensive cybersecurity research, reproducibility, benchmarking and intrusion-detection development. Dataset licenses/terms remain those of the original providers.
