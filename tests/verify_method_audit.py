#!/usr/bin/env python3
"""Verify released LICE explanation-quality and counterfactual audit artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions" / "method_audit"
RESULT = ROOT / "results" / "method_audit"
OOF = ROOT / "predictions" / "oof"
checks: list[dict] = []


def check(name: str, condition: bool, observed, expected) -> None:
    checks.append({"Check": name, "Observed": observed, "Expected": expected,
                   "Status": "PASS" if condition else "FAIL"})


quality = pd.read_csv(PRED / "explanation_quality_case_level.csv.gz")
status = pd.read_csv(PRED / "counterfactual_generation_status.csv.gz")
counterfactuals = pd.read_csv(PRED / "counterfactuals_wide.csv.gz")
changes = pd.read_csv(PRED / "counterfactual_feature_changes_long.csv.gz")
lime = pd.read_csv(PRED / "fn_lime_explanations_long.csv.gz")
baseline = pd.read_csv(OOF / "baseline_oof_predictions_full_precision.csv.gz")
baseline_fn = baseline.loc[(baseline.y_true == 1) & (baseline.oof_prediction == 0)]

check("Generation-status coverage of baseline OOF false negatives",
      set(status.Original_Index) == set(baseline_fn.Source_Row_Index),
      len(set(status.Original_Index).symmetric_difference(set(baseline_fn.Source_Row_Index))), 0)
check("One generation-status row per case", status.Original_Index.is_unique,
      int(status.Original_Index.duplicated().sum()), 0)
successful = status.loc[status.Generation_Status == "Success"]
check("Quality rows equal successful cases", set(quality.Original_Index) == set(successful.Original_Index),
      len(set(quality.Original_Index).symmetric_difference(set(successful.Original_Index))), 0)
check("Generated counterfactual row count", len(counterfactuals) == int(successful.Generated_CFs.sum()),
      len(counterfactuals), int(successful.Generated_CFs.sum()))
check("Feature-change matrix width", len(changes) == 4 * len(counterfactuals),
      len(changes), 4 * len(counterfactuals))
check("Counterfactual desired-class agreement", (counterfactuals.CF_Outcome == 1).all(),
      int((counterfactuals.CF_Outcome != 1).sum()), 0)

configuration = json.loads((ROOT / "config" / "method_audit" / "counterfactual_configuration.json").read_text())
guided_features = configuration["lime_guided_dice_features"]
check("Prespecified LIME-guided feature set", guided_features == ["GenHlth", "HighBP", "HighChol", "BMI"],
      guided_features, ["GenHlth", "HighBP", "HighChol", "BMI"])
check("Feature-change rows use only guided features", set(changes.Feature) == set(guided_features),
      sorted(changes.Feature.unique()), sorted(guided_features))
check("FN LIME case coverage", set(lime.Original_Index) == set(baseline_fn.Source_Row_Index),
      len(set(lime.Original_Index).symmetric_difference(set(baseline_fn.Source_Row_Index))), 0)
check("FN LIME fold coverage", set(lime.Fold_Number) == {1, 2, 3},
      sorted(lime.Fold_Number.unique()), [1, 2, 3])

summary = pd.read_csv(RESULT / "explanation_quality_summary.csv").set_index("Metric")
for metric, column in [("LIME Fidelity", "LIME_Fidelity_R2"),
                       ("DiCE Stability (Jaccard)", "DiCE_Stability_Jaccard")]:
    values = quality[column].dropna().to_numpy(float)
    observed = [values.mean(), values.std(ddof=1), np.median(values), values.min(), values.max()]
    expected = summary.loc[metric, ["Mean", "SD", "Median", "Minimum", "Maximum"]].to_numpy(float)
    difference = float(np.max(np.abs(np.asarray(observed) - expected)))
    check(f"{metric} summary reproduction", difference <= 5e-7, difference, "<=5e-7")

fold_summary = pd.read_csv(RESULT / "explanation_quality_by_fold.csv").set_index("Fold_Number")
for fold, frame in quality.groupby("Fold_Number"):
    reported = fold_summary.loc[fold]
    difference = max(
        abs(frame.LIME_Fidelity_R2.mean() - reported.LIME_Fidelity_Mean),
        abs(frame.DiCE_Stability_Jaccard.mean() - reported.DiCE_Stability_Mean),
    )
    check(f"Fold {fold} quality-summary reproduction", difference <= 1e-12,
          float(difference), "<=1e-12")

constraints = pd.read_csv(RESULT / "feature_constraint_audit.csv").set_index("Feature")
for feature in guided_features:
    frame = changes.loc[changes.Feature == feature]
    reported = constraints.loc[feature]
    values = frame.Counterfactual_Value.to_numpy(float)
    below = int((values < reported.Training_Minimum).sum())
    above = int((values > reported.Training_Maximum).sum())
    cases = int(frame.loc[(values < reported.Training_Minimum) |
                          (values > reported.Training_Maximum), "Original_Index"].nunique())
    check(f"{feature} constraint counts", (below, above, cases) ==
          (reported.Rows_Below_Training_Minimum, reported.Rows_Above_Training_Maximum,
           reported.Unique_Cases_Outside_Training_Range),
          (below, above, cases),
          (reported.Rows_Below_Training_Minimum, reported.Rows_Above_Training_Maximum,
           reported.Unique_Cases_Outside_Training_Range))

weights = pd.read_csv(RESULT / "sample_weight_identity_check.csv")
check("Sample-weight identity confirmation", weights.Identical_To_Previous_LICE_Weight.all(),
      weights.Identical_To_Previous_LICE_Weight.tolist(), [True] * len(weights))
features = pd.read_csv(RESULT / "feature_confirmation.csv")
check("LIME-selected features confirmed by DiCE",
      features.Selected_By_LIME.all() and features.Confirmed_By_DiCE.all(),
      int((features.Selected_By_LIME & features.Confirmed_By_DiCE).sum()), len(features))
stability = pd.read_csv(RESULT / "lime_feature_stability_by_fold.csv")
recalculated = (lime.groupby(["Feature", "Fold_Number"], as_index=False)
                .agg(Unique_Cases=("Original_Index", "nunique")))
reported = stability[["Feature", "Fold_Number", "Unique_Cases"]]
check("Fold-wise LIME feature-frequency reproduction",
      recalculated.equals(reported),
      int((recalculated.Unique_Cases != reported.Unique_Cases).sum()), 0)

results = pd.DataFrame(checks)
failed = results.loc[results.Status != "PASS"]
print(results.to_string(index=False))
print("VERIFICATION_SUMMARY " + json.dumps({"checks": len(results),
      "passed": int((results.Status == "PASS").sum()), "failed": len(failed)}))
if len(failed):
    raise SystemExit(1)
