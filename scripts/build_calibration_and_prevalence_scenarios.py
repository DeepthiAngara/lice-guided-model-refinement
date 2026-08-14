#!/usr/bin/env python3
"""Calibration and hypothetical prevalence-scenario analyses.

The prevalence calculations are mathematical transport scenarios based on the
observed test sensitivity and specificity. They are not cohort validation and
do not introduce a different dataset into the experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss


MODELS = {
    "Baseline": ("baseline_prediction", "baseline_probability"),
    "M1-Interactions": ("m1_interactions_prediction", "m1_interactions_probability"),
    "M2-Mild": ("m2_mild_prediction", "m2_mild_probability"),
    "M2-Balanced": ("m2_balanced_prediction", "m2_balanced_probability"),
    "M2-High": ("m2_high_prediction", "m2_high_probability"),
    "M3-Mild": ("m3_mild_prediction", "m3_mild_probability"),
    "M3-Balanced": ("m3_balanced_prediction", "m3_balanced_probability"),
    "M3-High": ("m3_high_prediction", "m3_high_probability"),
}

PREVALENCE_SCENARIOS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def calibration_statistics(y: np.ndarray, probability: np.ndarray) -> dict:
    eps = np.finfo(float).eps
    clipped = np.clip(probability, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2_000)
    calibration_model.fit(logits, y)

    bins = pd.qcut(probability, q=10, labels=False, duplicates="drop")
    frame = pd.DataFrame({"y": y, "probability": probability, "bin": bins})
    grouped = frame.groupby("bin", observed=True).agg(
        Count=("y", "size"), Mean_Predicted_Probability=("probability", "mean"), Observed_Frequency=("y", "mean")
    )
    ece = np.sum(grouped.Count / len(frame) * np.abs(grouped.Mean_Predicted_Probability - grouped.Observed_Frequency))
    return {
        "Brier_Score": brier_score_loss(y, probability),
        "Log_Loss": log_loss(y, probability),
        "Mean_Predicted_Probability": probability.mean(),
        "Observed_Outcome_Frequency": y.mean(),
        "Mean_Probability_Minus_Observed_Frequency": probability.mean() - y.mean(),
        "Calibration_Intercept": float(calibration_model.intercept_[0]),
        "Calibration_Slope": float(calibration_model.coef_[0, 0]),
        "Expected_Calibration_Error_10_Quantile_Bins": float(ece),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=Path("predictions/test/eight_configuration_test_predictions_wide.csv.gz"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/calibration_prevalence"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.predictions)
    assert len(data) == 13_812 and data.Source_Row_Index.is_unique
    y = data.y_true.to_numpy(dtype=int)
    calibration_rows, bin_rows, scenario_rows = [], [], []

    for model, (pred_col, probability_col) in MODELS.items():
        pred = data[pred_col].to_numpy(dtype=int)
        probability = data[probability_col].to_numpy(dtype=float)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        calibration_rows.append({"Model": model, **calibration_statistics(y, probability)})

        bins = pd.qcut(probability, q=10, labels=False, duplicates="drop")
        frame = pd.DataFrame({"Bin": bins, "y": y, "probability": probability})
        grouped = frame.groupby("Bin", observed=True).agg(
            Count=("y", "size"),
            Minimum_Probability=("probability", "min"),
            Maximum_Probability=("probability", "max"),
            Mean_Predicted_Probability=("probability", "mean"),
            Observed_Frequency=("y", "mean"),
        ).reset_index()
        grouped.insert(0, "Model", model)
        bin_rows.append(grouped)

        for prevalence in PREVALENCE_SCENARIOS:
            true_positive_rate_per_1000 = sensitivity * prevalence * 1_000
            false_negative_rate_per_1000 = (1 - sensitivity) * prevalence * 1_000
            false_positive_rate_per_1000 = (1 - specificity) * (1 - prevalence) * 1_000
            true_negative_rate_per_1000 = specificity * (1 - prevalence) * 1_000
            predicted_positive_per_1000 = true_positive_rate_per_1000 + false_positive_rate_per_1000
            ppv = true_positive_rate_per_1000 / predicted_positive_per_1000
            predicted_negative_per_1000 = true_negative_rate_per_1000 + false_negative_rate_per_1000
            npv = true_negative_rate_per_1000 / predicted_negative_per_1000
            scenario_rows.append({
                "Model": model,
                "Hypothetical_Outcome_Prevalence": prevalence,
                "Sensitivity_From_Existing_Test_Set": sensitivity,
                "Specificity_From_Existing_Test_Set": specificity,
                "Expected_TP_per_1000": true_positive_rate_per_1000,
                "Expected_FP_per_1000": false_positive_rate_per_1000,
                "Expected_FN_per_1000": false_negative_rate_per_1000,
                "Expected_TN_per_1000": true_negative_rate_per_1000,
                "Expected_Predicted_Positive_per_1000": predicted_positive_per_1000,
                "Scenario_PPV": ppv,
                "Scenario_NPV": npv,
            })

    calibration = pd.DataFrame(calibration_rows)
    calibration_bins = pd.concat(bin_rows, ignore_index=True)
    scenarios = pd.DataFrame(scenario_rows)
    calibration.to_csv(args.output_dir / "calibration_assessment.csv", index=False)
    calibration_bins.to_csv(args.output_dir / "calibration_quantile_bins.csv", index=False)
    scenarios.to_csv(args.output_dir / "prevalence_scenario_assessment.csv", index=False)

    manifest = {
        "artifact_ids": ["calibration_assessment", "prevalence_scenario_assessment"],
        "data": "existing 13,812-row untouched test partition from the 69,057-row duplicate-removed modelling dataset",
        "calibration": {
            "measures": ["Brier score", "log loss", "calibration intercept", "calibration slope", "10-quantile-bin ECE"],
            "limitation": "Calibration is assessed on a nearly balanced test partition.",
        },
        "prevalence_scenarios": PREVALENCE_SCENARIOS,
        "scenario_method": "PPV, NPV and expected counts per 1,000 derived mathematically from observed test sensitivity and specificity",
        "interpretation_boundary": "Hypothetical transport scenarios, not validation in prevalence-preserving or clinical cohorts.",
        "new_dataset_used": False,
        "clinical_recommendation": False,
    }
    (args.output_dir / "calibration_prevalence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Calibration assessment")
    print(calibration.to_string(index=False))
    print("\nPrevalence scenarios: 48 model-scenario rows")


if __name__ == "__main__":
    main()
