#!/usr/bin/env python3
"""Paired test-set uncertainty and multiplicity analysis.

All model differences use the same resampled test-row indices. The analysis
does not tune models, thresholds, or configurations on the test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


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

METRICS = ("Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC")


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Kappa": cohen_kappa_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=Path("predictions/test/eight_configuration_test_predictions_wide.csv.gz"),
        type=Path,
    )
    parser.add_argument("--output-dir", default=Path("results/statistical_analysis"), type=Path)
    parser.add_argument("--resamples", default=5_000, type=int)
    parser.add_argument("--seed", default=20260812, type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.predictions)
    assert len(data) == 13_812
    assert data.Source_Row_Index.is_unique
    y = data.y_true.to_numpy(dtype=int)
    values = {
        model: (
            data[pred_col].to_numpy(dtype=int),
            data[prob_col].to_numpy(dtype=float),
        )
        for model, (pred_col, prob_col) in MODELS.items()
    }
    base_pred, base_prob = values["Baseline"]
    base_point = metrics(y, base_pred, base_prob)

    rng = np.random.default_rng(args.seed)
    candidates = [model for model in MODELS if model != "Baseline"]
    differences = {
        (model, metric): np.empty(args.resamples, dtype=float)
        for model in candidates
        for metric in METRICS
    }

    for replicate in range(args.resamples):
        idx = rng.integers(0, len(y), size=len(y))
        y_b = y[idx]
        base_b = metrics(y_b, base_pred[idx], base_prob[idx])
        for model in candidates:
            pred, prob = values[model]
            candidate_b = metrics(y_b, pred[idx], prob[idx])
            for metric in METRICS:
                differences[(model, metric)][replicate] = candidate_b[metric] - base_b[metric]

    interval_rows = []
    for model in candidates:
        pred, prob = values[model]
        point = metrics(y, pred, prob)
        for metric in METRICS:
            bootstrap = differences[(model, metric)]
            interval_rows.append(
                {
                    "Comparison": f"{model} vs Baseline",
                    "Model": model,
                    "Metric": metric,
                    "Baseline_Estimate": base_point[metric],
                    "Candidate_Estimate": point[metric],
                    "Difference_Candidate_Minus_Baseline": point[metric] - base_point[metric],
                    "CI_95_Lower": np.quantile(bootstrap, 0.025),
                    "CI_95_Upper": np.quantile(bootstrap, 0.975),
                    "Bootstrap_Resamples": args.resamples,
                    "Bootstrap_Seed": args.seed,
                    "CI_Method": "paired instance-level percentile bootstrap",
                }
            )
    intervals = pd.DataFrame(interval_rows)

    mcnemar_rows = []
    for scope, mask in (
        ("Full test set", np.ones(len(y), dtype=bool)),
        ("Actual positive cases only", y == 1),
    ):
        yt = y[mask]
        baseline_correct = base_pred[mask] == yt
        for model in candidates:
            pred = values[model][0][mask]
            candidate_correct = pred == yt
            b = int(np.sum(baseline_correct & ~candidate_correct))
            c = int(np.sum(~baseline_correct & candidate_correct))
            discordant = b + c
            p_value = binomtest(min(b, c), discordant, 0.5).pvalue if discordant else 1.0
            mcnemar_rows.append(
                {
                    "Scope": scope,
                    "Comparison": f"{model} vs Baseline",
                    "Both_Correct": int(np.sum(baseline_correct & candidate_correct)),
                    "Baseline_Correct_Candidate_Wrong": b,
                    "Baseline_Wrong_Candidate_Correct": c,
                    "Both_Wrong": int(np.sum(~baseline_correct & ~candidate_correct)),
                    "Discordant_Total": discordant,
                    "Exact_Two_Sided_P_Value": p_value,
                }
            )
    mcnemar = pd.DataFrame(mcnemar_rows)
    mcnemar["Holm_Adjusted_P_Value_Within_Scope"] = np.nan
    for scope, indices in mcnemar.groupby("Scope").groups.items():
        adjusted = holm_adjust(mcnemar.loc[indices, "Exact_Two_Sided_P_Value"].tolist())
        mcnemar.loc[indices, "Holm_Adjusted_P_Value_Within_Scope"] = adjusted
    mcnemar["Significant_After_Holm_0.05"] = (
        mcnemar.Holm_Adjusted_P_Value_Within_Scope < 0.05
    )

    intervals.to_csv(args.output_dir / "paired_bootstrap_intervals.csv", index=False)
    mcnemar.to_csv(args.output_dir / "mcnemar_paired_comparisons.csv", index=False)

    manifest = {
        "artifact_ids": ["paired_bootstrap_intervals", "mcnemar_paired_comparisons", "multiplicity_adjusted_inference"],
        "test_rows": len(data),
        "comparisons": len(candidates),
        "threshold": 0.5,
        "bootstrap": {
            "unit": "test instance",
            "pairing": "identical sampled row indices for baseline and candidate",
            "resamples": args.resamples,
            "seed": args.seed,
            "interval": "95% percentile",
        },
        "mcnemar": {
            "test": "exact two-sided binomial McNemar",
            "scopes": ["Full test set", "Actual positive cases only"],
            "multiplicity": "Holm adjustment across seven comparisons separately within each scope",
        },
        "interpretation_boundary": "Uncertainty analysis does not correct the non-nested configuration-comparison design.",
    }
    (args.output_dir / "paired_test_inference_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Paired bootstrap rows: {len(intervals)}")
    print(f"McNemar rows: {len(mcnemar)}")
    print("\nFull-test McNemar comparisons:")
    print(mcnemar[mcnemar.Scope == "Full test set"].to_string(index=False))


if __name__ == "__main__":
    main()
