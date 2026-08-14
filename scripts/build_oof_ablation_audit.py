#!/usr/bin/env python3
"""Build the eight-configuration training-side OOF audit.

This script does not retrain or select a model. It reads the released canonical
long-format OOF predictions, recalculates metrics, and checks their alignment
with the fixed training-row and fold assignments. No external array folder is
required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODELS = {
    "Baseline": (
        "Step5_oof_predictions_Baseline_TunedGBDT_NoWeights.npy",
        "Step5_oof_probabilities_Baseline_TunedGBDT_NoWeights.npy",
        "none",
        False,
    ),
    "M1-Interactions": (
        "Step4_stage2_interactions_only_oof_predictions.npy",
        "Step4_stage2_interactions_only_oof_probabilities.npy",
        "none",
        True,
    ),
    "M2-Mild": (
        "Step5_oof_predictions_LICE_Mild.npy",
        "Step5_oof_probabilities_LICE_Mild.npy",
        "mild",
        False,
    ),
    "M2-Balanced": (
        "Step5_oof_predictions_LICE_Balanced.npy",
        "Step5_oof_probabilities_LICE_Balanced.npy",
        "balanced",
        False,
    ),
    "M2-High": (
        "Step5_oof_predictions_LICE_High_Sensitivity.npy",
        "Step5_oof_probabilities_LICE_High_Sensitivity.npy",
        "high_sensitivity",
        False,
    ),
    "M3-Mild": (
        "Step5_stage3_oof_predictions_Interactions_LICE_Mild.npy",
        "Step5_stage3_oof_probabilities_Interactions_LICE_Mild.npy",
        "mild",
        True,
    ),
    "M3-Balanced": (
        "Step5_stage3_oof_predictions_Interactions_LICE_Balanced.npy",
        "Step5_stage3_oof_probabilities_Interactions_LICE_Balanced.npy",
        "balanced",
        True,
    ),
    "M3-High": (
        "Step5_stage3_oof_predictions_Interactions_LICE_High_Sensitivity.npy",
        "Step5_stage3_oof_probabilities_Interactions_LICE_High_Sensitivity.npy",
        "high_sensitivity",
        True,
    ),
}


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def error_type(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.select(
        [
            (y_true == 1) & (y_pred == 1),
            (y_true == 0) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="invalid",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof-predictions",
        default=Path("predictions/oof/eight_configuration_oof_predictions_long.csv.gz"),
        type=Path,
    )
    parser.add_argument(
        "--training-audit",
        default=Path("predictions/oof/training_lice_assignment_audit.csv.gz"),
        type=Path,
    )
    parser.add_argument(
        "--existing-overall-results",
        type=Path,
        help="Optional historical eight-model OOF summary for agreement checks.",
    )
    parser.add_argument("--output-dir", default=Path("results/oof"), type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.training_audit).sort_values("Train_Position")
    assert len(audit) == 55_245
    assert audit.Train_Position.tolist() == list(range(55_245))
    assert audit.Source_Row_Index.is_unique
    assert set(audit.OOF_Validation_Fold.unique()) == {1, 2, 3}

    y_true = audit.y_true.to_numpy(dtype=int)
    released = pd.read_csv(args.oof_predictions)
    long_frames: list[pd.DataFrame] = []
    overall_rows: list[dict] = []
    fold_rows: list[dict] = []
    checks: list[dict] = []

    for model, (_, _, weight_scheme, interactions) in MODELS.items():
        model_released = released.loc[released.Model == model].sort_values("Train_Position")
        if len(model_released) != len(audit):
            raise AssertionError(f"{model}: expected {len(audit)} released OOF rows, found {len(model_released)}")
        if not np.array_equal(model_released.Source_Row_Index.to_numpy(), audit.Source_Row_Index.to_numpy()):
            raise AssertionError(f"{model}: source-row alignment failure")
        if not np.array_equal(model_released.OOF_Validation_Fold.to_numpy(), audit.OOF_Validation_Fold.to_numpy()):
            raise AssertionError(f"{model}: fold alignment failure")
        pred = model_released.Predicted_Class.to_numpy(dtype=int)
        prob = model_released.Predicted_Probability.to_numpy(dtype=float)
        checks.extend(
            [
                {"Check": f"{model}: row count", "Observed": len(pred), "Expected": 55_245, "Status": "PASS" if len(pred) == 55_245 else "FAIL"},
                {"Check": f"{model}: probability row count", "Observed": len(prob), "Expected": 55_245, "Status": "PASS" if len(prob) == 55_245 else "FAIL"},
                {"Check": f"{model}: probability range", "Observed": f"[{prob.min():.17g}, {prob.max():.17g}]", "Expected": "[0, 1]", "Status": "PASS" if np.all((prob >= 0) & (prob <= 1)) else "FAIL"},
                {"Check": f"{model}: fixed 0.50 threshold", "Observed": int(np.sum(pred != (prob >= 0.5))), "Expected": 0, "Status": "PASS" if np.array_equal(pred, (prob >= 0.5).astype(int)) else "FAIL"},
            ]
        )

        model_frame = model_released.copy()
        long_frames.append(model_frame)

        overall_rows.append({"Model": model, **metric_row(y_true, pred, prob)})
        for fold in (1, 2, 3):
            mask = audit.OOF_Validation_Fold.to_numpy() == fold
            fold_rows.append(
                {
                    "Model": model,
                    "OOF_Validation_Fold": fold,
                    "Validation_Rows": int(mask.sum()),
                    **metric_row(y_true[mask], pred[mask], prob[mask]),
                }
            )

    long_df = pd.concat(long_frames, ignore_index=True)
    overall_df = pd.DataFrame(overall_rows)
    fold_df = pd.DataFrame(fold_rows)

    checks.extend(
        [
            {"Check": "long audit row count", "Observed": len(long_df), "Expected": 441_960, "Status": "PASS" if len(long_df) == 441_960 else "FAIL"},
            {"Check": "unique source-row/model pairs", "Observed": int(long_df.duplicated(["Source_Row_Index", "Model"]).sum()), "Expected": 0, "Status": "PASS" if not long_df.duplicated(["Source_Row_Index", "Model"]).any() else "FAIL"},
            {"Check": "one OOF fold per source-row/model", "Observed": int(long_df.groupby(["Source_Row_Index", "Model"]).OOF_Validation_Fold.nunique().max()), "Expected": 1, "Status": "PASS" if long_df.groupby(["Source_Row_Index", "Model"]).OOF_Validation_Fold.nunique().max() == 1 else "FAIL"},
        ]
    )

    if args.existing_overall_results:
        historical = pd.read_csv(args.existing_overall_results)
        aliases = {
            "Baseline_TunedGBDT_NoWeights": "Baseline",
            "TunedGBDT_LICE_Interactions_Only": "M1-Interactions",
            "LICE_Mild": "M2-Mild",
            "LICE_Balanced": "M2-Balanced",
            "LICE_High_Sensitivity": "M2-High",
            "Interactions_LICE_Mild": "M3-Mild",
            "Interactions_LICE_Balanced": "M3-Balanced",
            "Interactions_LICE_High_Sensitivity": "M3-High",
        }
        historical["Model"] = historical.Model.map(aliases)
        merged = overall_df.merge(historical, on="Model", suffixes=("_new", "_historical"))
        count_cols = ["TP", "TN", "FP", "FN"]
        metric_cols = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]
        count_match = all((merged[f"{c}_new"] == merged[f"{c}_historical"]).all() for c in count_cols)
        max_diff = max(float(np.max(np.abs(merged[f"{c}_new"] - merged[f"{c}_historical"]))) for c in metric_cols)
        checks.extend(
            [
                {"Check": "historical OOF confusion-count agreement", "Observed": count_match, "Expected": True, "Status": "PASS" if count_match else "FAIL"},
                {"Check": "historical OOF metric agreement", "Observed": max_diff, "Expected": "<=1e-12", "Status": "PASS" if max_diff <= 1e-12 else "FAIL"},
            ]
        )

    checks_df = pd.DataFrame(checks)
    if (checks_df.Status != "PASS").any():
        raise AssertionError(checks_df.loc[checks_df.Status != "PASS"].to_string(index=False))

    long_df.to_csv(args.output_dir / "oof_ablation_predictions_long.csv.gz", index=False)
    overall_df.to_csv(args.output_dir / "oof_ablation_metrics_full_precision.csv", index=False)
    fold_df.to_csv(args.output_dir / "oof_ablation_fold_metrics_full_precision.csv", index=False)
    checks_df.to_csv(args.output_dir / "oof_ablation_verification_checks.csv", index=False)

    manifest = {
        "artifact_id": "oof_ablation_performance",
        "interpretation": "training-side OOF consistency evidence; not nested model selection",
        "row_counts": {
            "training_instances": 55_245,
            "configurations": 8,
            "instance_model_records": 441_960,
        },
        "decision_threshold": 0.5,
        "folds": 3,
        "fold_construction": "StratifiedKFold(n_splits=3, shuffle=False)",
        "nested_assessment": False,
        "limitation": "Globally derived LICE patterns and weights were reused across configuration OOF folds.",
        "models": list(MODELS),
        "all_checks_passed": True,
    }
    (args.output_dir / "oof_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(overall_df.to_string(index=False))
    print(f"\nOOF audit rows: {len(long_df):,}")
    print(f"Verification checks: {len(checks_df)} passed, 0 failed")


if __name__ == "__main__":
    main()
