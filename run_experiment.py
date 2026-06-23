#!/usr/bin/env python3
"""Command-line entry point for the AttackPath-PGAS reference workflow."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
SRC = ARCHIVE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from attackpath_pgas.pgas import HybridPGAS, PGASConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ARCHIVE_ROOT / "config" / "default.yaml")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ARCHIVE_ROOT / "results" / "main_run")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small deterministic synthetic test instead of reading the research dataset.",
    )
    return parser.parse_args()


def synthetic_sequence(seed: int = 42, length: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.zeros(length, dtype=np.int8)
    states[35:48] = 1
    states[82:96] = 1
    observations = rng.normal(0.0, 0.55, size=(length, 4))
    observations[states == 1] += 2.1
    logits = -4.0 + 6.0 * states + rng.normal(0.0, 0.65, length)
    risk = 1.0 / (1.0 + np.exp(-logits))
    return observations, risk, states


def run_smoke_test(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    observations, risk, truth = synthetic_sequence()
    sampler = HybridPGAS(
        PGASConfig(
            particles=32,
            iterations=28,
            burn_in=8,
            thin=2,
            risk_emission_weight=3.0,
            gaussian_emission_weight=0.2,
            seed=42,
        )
    )
    result = sampler.fit(observations, risk)
    probability = result.posterior_attack_probability
    prediction = (probability >= 0.5).astype(np.int8)
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = {
        "accuracy": float((tp + tn) / len(truth)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    pd.DataFrame({"truth": truth, "risk": risk, "posterior": probability}).to_csv(
        output_dir / "smoke_predictions.csv", index=False
    )
    (output_dir / "smoke_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8"
    )
    print(f"Smoke test completed: {output_dir}")


def run_full_experiment(args: argparse.Namespace) -> None:
    if not args.config.exists():
        raise FileNotFoundError(args.config)
    if args.dataset_root is None:
        raise SystemExit("--dataset-root is required unless --smoke-test is used")
    os.environ["ATTACKPATH_CONFIG"] = str(args.config.resolve())
    os.environ["ATTACKPATH_DATASET_ROOT"] = str(args.dataset_root.resolve())
    os.environ["ATTACKPATH_OUTPUT_DIR"] = str(args.output_dir.resolve())
    os.chdir(ARCHIVE_ROOT)
    runpy.run_path(str(ARCHIVE_ROOT / "scripts" / "AttackPath_PGAS_Implementation.py"), run_name="__main__")


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test(args.output_dir)
    else:
        run_full_experiment(args)


if __name__ == "__main__":
    main()
