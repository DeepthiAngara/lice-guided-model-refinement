#!/usr/bin/env python3
"""Independent structural and numerical checks for the extended analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OOF_RESULTS = ROOT / "results" / "oof"
STAT_RESULTS = ROOT / "results" / "statistical_analysis"
CONTROL_RESULTS = ROOT / "results" / "conventional_comparators"
CAL_RESULTS = ROOT / "results" / "calibration_prevalence"
OOF_PREDICTIONS = ROOT / "predictions" / "oof"
CONTROL_PREDICTIONS = ROOT / "predictions" / "conventional_comparators"
checks: list[dict] = []


def check(name: str, condition: bool, observed, expected) -> None:
    checks.append({"Check": name, "Observed": observed, "Expected": expected, "Status": "PASS" if condition else "FAIL"})


oof = pd.read_csv(OOF_PREDICTIONS / "eight_configuration_oof_predictions_long.csv.gz")
baseline_oof = pd.read_csv(OOF_PREDICTIONS / "baseline_oof_predictions_full_precision.csv.gz").sort_values("Train_Position")
baseline_long = oof.loc[oof.Model == "Baseline"].sort_values("Train_Position")
check("Canonical baseline OOF source-row agreement", np.array_equal(baseline_oof.Source_Row_Index, baseline_long.Source_Row_Index), int(np.sum(baseline_oof.Source_Row_Index.to_numpy() != baseline_long.Source_Row_Index.to_numpy())), 0)
baseline_probability_max_difference = float(np.max(np.abs(baseline_oof.oof_probability.to_numpy(float) - baseline_long.Predicted_Probability.to_numpy(float))))
check("Canonical baseline OOF probability agreement", baseline_probability_max_difference <= 1e-15, baseline_probability_max_difference, "<=1e-15")
check("Canonical baseline OOF class agreement", np.array_equal(baseline_oof.oof_prediction.to_numpy(int), baseline_long.Predicted_Class.to_numpy(int)), int(np.sum(baseline_oof.oof_prediction.to_numpy(int) != baseline_long.Predicted_Class.to_numpy(int))), 0)
check("OOF long row count", len(oof) == 441_960, len(oof), 441_960)
check("OOF unique instance-model keys", not oof.duplicated(["Source_Row_Index", "Model"]).any(), int(oof.duplicated(["Source_Row_Index", "Model"]).sum()), 0)
check("OOF configuration count", oof.Model.nunique() == 8, oof.Model.nunique(), 8)
check("OOF fixed threshold", set(oof.Decision_Threshold) == {0.5}, sorted(oof.Decision_Threshold.unique()), [0.5])
check("OOF threshold-class agreement", np.array_equal(oof.Predicted_Class, (oof.Predicted_Probability >= 0.5).astype(int)), int(np.sum(oof.Predicted_Class != (oof.Predicted_Probability >= 0.5))), 0)

oof_metrics = pd.read_csv(OOF_RESULTS / "oof_ablation_performance_full_precision.csv")
for model, frame in oof.groupby("Model"):
    reported = oof_metrics[oof_metrics.Model == model].iloc[0]
    y, pred, prob = frame.y_true.to_numpy(int), frame.Predicted_Class.to_numpy(int), frame.Predicted_Probability.to_numpy(float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    check(f"{model} OOF confusion matrix", (tp, tn, fp, fn) == (reported.TP, reported.TN, reported.FP, reported.FN), (tp, tn, fp, fn), (reported.TP, reported.TN, reported.FP, reported.FN))
    max_diff = max(abs(roc_auc_score(y, prob) - reported.AUC), abs(f1_score(y, pred) - reported.F1))
    check(f"{model} OOF AUC/F1", max_diff <= 1e-12, max_diff, "<=1e-12")

intervals = pd.read_csv(STAT_RESULTS / "paired_bootstrap_intervals.csv")
check("Paired bootstrap result rows", len(intervals) == 49, len(intervals), 49)
check("Paired bootstrap resamples", set(intervals.Bootstrap_Resamples) == {5000}, sorted(intervals.Bootstrap_Resamples.unique()), [5000])
check("Paired bootstrap seed", set(intervals.Bootstrap_Seed) == {20260812}, sorted(intervals.Bootstrap_Seed.unique()), [20260812])
check("Paired bootstrap ordered intervals", (intervals.CI_95_Lower <= intervals.CI_95_Upper).all(), bool((intervals.CI_95_Lower <= intervals.CI_95_Upper).all()), True)

mcnemar = pd.read_csv(STAT_RESULTS / "mcnemar_paired_comparisons.csv")
check("McNemar result rows", len(mcnemar) == 14, len(mcnemar), 14)
check("McNemar adjusted p-value range", mcnemar.Holm_Adjusted_P_Value_Within_Scope.between(0, 1).all(), [mcnemar.Holm_Adjusted_P_Value_Within_Scope.min(), mcnemar.Holm_Adjusted_P_Value_Within_Scope.max()], "[0,1]")

comparator_metrics = pd.read_csv(CONTROL_RESULTS / "conventional_comparator_performance.csv")
comparator_predictions = pd.read_csv(CONTROL_PREDICTIONS / "conventional_comparator_predictions_long.csv.gz")
check("Comparator metric rows", len(comparator_metrics) == 40, len(comparator_metrics), 40)
check("Comparator prediction rows", len(comparator_predictions) == 1_381_140, len(comparator_predictions), 1_381_140)
fixed = comparator_predictions[~comparator_predictions.Model.str.startswith("BaselineThreshold")]
check("Weight-control fixed threshold", set(fixed.Decision_Threshold) == {0.5}, sorted(fixed.Decision_Threshold.unique()), [0.5])
check("Comparator probability-class agreement", np.array_equal(comparator_predictions.Predicted_Class, (comparator_predictions.Predicted_Probability >= comparator_predictions.Decision_Threshold).astype(int)), int(np.sum(comparator_predictions.Predicted_Class != (comparator_predictions.Predicted_Probability >= comparator_predictions.Decision_Threshold))), 0)

thresholds = pd.read_csv(CONTROL_RESULTS / "matched_operating_point_selection.csv")
check("OOF-selected threshold rows", len(thresholds) == 4, len(thresholds), 4)
check("Threshold matching error", (thresholds.Absolute_OOF_Matching_Error <= 0.0001).all(), thresholds.Absolute_OOF_Matching_Error.max(), "<=0.0001")

calibration = pd.read_csv(CAL_RESULTS / "calibration_assessment.csv")
check("Calibration models", len(calibration) == 8, len(calibration), 8)
check("Brier-score range", calibration.Brier_Score.between(0, 1).all(), [calibration.Brier_Score.min(), calibration.Brier_Score.max()], "[0,1]")

scenarios = pd.read_csv(CAL_RESULTS / "prevalence_scenario_assessment.csv")
check("Prevalence-scenario rows", len(scenarios) == 48, len(scenarios), 48)
total_per_1000 = scenarios[["Expected_TP_per_1000", "Expected_FP_per_1000", "Expected_FN_per_1000", "Expected_TN_per_1000"]].sum(axis=1)
check("Scenario counts sum to 1000", np.allclose(total_per_1000, 1000, atol=1e-10), float(np.max(np.abs(total_per_1000 - 1000))), "<=1e-10")

utility = pd.read_csv(CAL_RESULTS / "exploratory_utility_assessment.csv")
check("Exploratory utility rows", len(utility) == 460, len(utility), 460)
check("Exploratory utility threshold range", utility.Hypothetical_Decision_Threshold.between(0.05, 0.50).all(), [utility.Hypothetical_Decision_Threshold.min(), utility.Hypothetical_Decision_Threshold.max()], "[0.05,0.50]")

results = pd.DataFrame(checks)
failed = results[results.Status != "PASS"]
summary = {"verification_checks": len(results), "passed": int((results.Status == "PASS").sum()), "failed": len(failed)}
print(results.to_string(index=False))
print(f"\nVerification checks: {summary['passed']} passed, {summary['failed']} failed")
print("VERIFICATION_SUMMARY " + json.dumps({"checks": len(results), "passed": summary["passed"], "failed": summary["failed"]}))
if len(failed):
    raise SystemExit(1)
