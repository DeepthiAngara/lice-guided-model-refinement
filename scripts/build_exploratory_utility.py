#!/usr/bin/env python3
"""Exploratory decision-curve calculations on the existing test predictions.

The thresholds are hypothetical decision thresholds used to calculate net
benefit. They are not clinically specified cutoffs, do not replace the fixed
0.50 classification threshold in the original experiment, and are not used for
model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "Baseline": "baseline_probability",
    "M1-Interactions": "m1_interactions_probability",
    "M2-Mild": "m2_mild_probability",
    "M2-Balanced": "m2_balanced_probability",
    "M2-High": "m2_high_probability",
    "M3-Mild": "m3_mild_probability",
    "M3-Balanced": "m3_balanced_probability",
    "M3-High": "m3_high_probability",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=Path("predictions/test/eight_configuration_test_predictions_wide.csv.gz"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/calibration_prevalence"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.predictions)
    y = data.y_true.to_numpy(dtype=int)
    n = len(y)
    prevalence = y.mean()
    thresholds = np.round(np.arange(0.05, 0.501, 0.01), 2)
    rows = []
    for threshold in thresholds:
        odds = threshold / (1 - threshold)
        rows.extend([
            {"Strategy": "Treat none", "Hypothetical_Decision_Threshold": threshold, "Net_Benefit": 0.0},
            {"Strategy": "Treat all", "Hypothetical_Decision_Threshold": threshold, "Net_Benefit": prevalence - (1 - prevalence) * odds},
        ])
        for model, probability_column in MODELS.items():
            pred = data[probability_column].to_numpy(float) >= threshold
            tp = int(np.sum((y == 1) & pred))
            fp = int(np.sum((y == 0) & pred))
            net_benefit = tp / n - fp / n * odds
            rows.append({
                "Strategy": model,
                "Hypothetical_Decision_Threshold": threshold,
                "Net_Benefit": net_benefit,
                "TP": tp,
                "FP": fp,
            })
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "exploratory_utility_assessment.csv", index=False)
    manifest = {
        "artifact_id": "exploratory_utility_assessment",
        "method": "standard net-benefit decision-curve calculation",
        "data": "existing nearly balanced 13,812-row test partition",
        "thresholds": {"minimum": 0.05, "maximum": 0.50, "increment": 0.01},
        "classification_threshold_in_original_experiment": 0.50,
        "used_for_model_selection": False,
        "clinical_threshold_specification": False,
        "interpretation_boundary": "Exploratory model-based utility only; no clinical recommendation or validated cost ratio.",
    }
    (args.output_dir / "exploratory_utility_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Decision-curve rows: {len(results)}")
    print(f"Observed test outcome frequency: {prevalence:.6f}")


if __name__ == "__main__":
    main()
