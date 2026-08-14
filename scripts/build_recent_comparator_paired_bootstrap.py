#!/usr/bin/env python3
"""Build paired bootstrap intervals for the recent same-split comparators."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/recent_comparators/independent_verification"
OUT.mkdir(parents=True, exist_ok=True)

primary = pd.read_csv(
    ROOT / "predictions/test/eight_configuration_test_predictions_wide.csv.gz"
)
y = primary.y_true.to_numpy(int)
ids = primary.Source_Row_Index.to_numpy(int)
values = {
    "Baseline": (
        primary.baseline_prediction.to_numpy(int),
        primary.baseline_probability.to_numpy(float),
    ),
    "M3-Balanced": (
        primary.m3_balanced_prediction.to_numpy(int),
        primary.m3_balanced_probability.to_numpy(float),
    ),
    "M3-High": (
        primary.m3_high_prediction.to_numpy(int),
        primary.m3_high_probability.to_numpy(float),
    ),
}
for model, filename in [
    ("Jose-LightGBM", "jose_lightgbm_test_predictions.csv.gz"),
    ("Pang-MARS", "pang_mars_test_predictions.csv.gz"),
]:
    frame = (
        pd.read_csv(ROOT / "predictions/recent_comparators" / filename)
        .set_index("Source_Row_Index")
        .loc[ids]
    )
    values[model] = (
        frame.Predicted_Class.to_numpy(int),
        frame.Predicted_Probability.to_numpy(float),
    )


def metrics(y_true, prediction, probability):
    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "AUC": roc_auc_score(y_true, probability),
        "Recall": recall_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction),
        "F1": f1_score(y_true, prediction),
        "Kappa": cohen_kappa_score(y_true, prediction),
        "MCC": matthews_corrcoef(y_true, prediction),
    }


comparisons = [
    (comparator, reference)
    for comparator in ["Jose-LightGBM", "Pang-MARS"]
    for reference in ["Baseline", "M3-Balanced", "M3-High"]
]
metric_names = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]
n_resamples = 5000
seed = 20260812
rng = np.random.default_rng(seed)
differences = {
    (comparator, reference, metric): np.empty(n_resamples)
    for comparator, reference in comparisons
    for metric in metric_names
}

for index in range(n_resamples):
    sampled = rng.integers(0, len(y), len(y))
    sampled_y = y[sampled]
    required_models = set(sum(([c, r] for c, r in comparisons), []))
    sampled_metrics = {
        model: metrics(
            sampled_y,
            values[model][0][sampled],
            values[model][1][sampled],
        )
        for model in required_models
    }
    for comparator, reference in comparisons:
        for metric in metric_names:
            differences[comparator, reference, metric][index] = (
                sampled_metrics[comparator][metric]
                - sampled_metrics[reference][metric]
            )
    if (index + 1) % 500 == 0:
        print(f"Completed bootstrap {index + 1} of {n_resamples}", flush=True)

point_estimates = {model: metrics(y, *model_values) for model, model_values in values.items()}
rows = []
for comparator, reference in comparisons:
    for metric in metric_names:
        distribution = differences[comparator, reference, metric]
        rows.append(
            {
                "Comparison": f"{comparator} vs {reference}",
                "Comparator": comparator,
                "Reference": reference,
                "Metric": metric,
                "Reference_Estimate": point_estimates[reference][metric],
                "Comparator_Estimate": point_estimates[comparator][metric],
                "Difference_Comparator_Minus_Reference": (
                    point_estimates[comparator][metric]
                    - point_estimates[reference][metric]
                ),
                "CI_95_Lower": np.quantile(distribution, 0.025),
                "CI_95_Upper": np.quantile(distribution, 0.975),
                "Bootstrap_Resamples": n_resamples,
                "Bootstrap_Seed": seed,
                "CI_Method": "paired instance-level percentile bootstrap",
            }
        )

output = OUT / "head_to_head_paired_bootstrap_intervals.csv"
pd.DataFrame(rows).to_csv(output, index=False)
print(f"Saved {len(rows)} interval rows to {output.relative_to(ROOT)}")
