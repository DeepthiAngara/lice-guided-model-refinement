#!/usr/bin/env python3
"""Run prespecified non-LICE controls on the existing fixed data workflow.

Controls:
1. Global positive-class weighting at the original three weight strengths.
2. Direct weighting of baseline OOF false-negative rows at the same strengths.
3. Random targeting of 7,174 positive rows at balanced strength across 10 seeds.
4. Baseline threshold controls selected from baseline OOF predictions to match
   the OOF recall or FPR of M3-Balanced and M3-High, then transferred unchanged
   to baseline test probabilities.

No test outcome or test prediction guides training, weighting, or threshold
selection. These are new controls and do not alter the original ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


STRENGTHS = {"Mild": 1.15, "Balanced": 1.25, "High": 1.35}
RANDOM_SEEDS = [101, 211, 307, 401, 503, 601, 701, 809, 907, 1009]
TARGET_COUNT = 7_174


def model_template() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        ccp_alpha=0.0,
        criterion="friedman_mse",
        learning_rate=0.1,
        loss="log_loss",
        max_depth=5,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=100,
        n_iter_no_change=None,
        random_state=42,
        subsample=1.0,
        tol=0.0001,
        validation_fraction=0.1,
        verbose=0,
        warm_start=False,
    )


def metric_row(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, prob),
        "Recall": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "FPR": fp / (tn + fp),
        "Precision": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "Kappa": cohen_kappa_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


def evaluate_weight_control(
    name: str,
    weights: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    folds: np.ndarray,
) -> tuple[dict, dict, pd.DataFrame]:
    oof_prob = np.zeros(len(y_train), dtype=float)
    for fold in (1, 2, 3):
        validation = folds == fold
        fitted = model_template().fit(
            X_train.loc[~validation], y_train[~validation],
            sample_weight=weights[~validation],
        )
        oof_prob[validation] = fitted.predict_proba(X_train.loc[validation])[:, 1]
    oof_pred = (oof_prob >= 0.5).astype(int)

    fitted = model_template().fit(X_train, y_train, sample_weight=weights)
    test_prob = fitted.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    records = pd.concat(
        [
            pd.DataFrame({"Split": "OOF", "Position": np.arange(len(y_train)), "Model": name, "y_true": y_train, "Predicted_Probability": oof_prob, "Predicted_Class": oof_pred, "Decision_Threshold": 0.5}),
            pd.DataFrame({"Split": "test", "Position": np.arange(len(y_test)), "Model": name, "y_true": y_test, "Predicted_Probability": test_prob, "Predicted_Class": test_pred, "Decision_Threshold": 0.5}),
        ], ignore_index=True,
    )
    return (
        {"Model": name, "Split": "OOF", **metric_row(y_train, oof_pred, oof_prob)},
        {"Model": name, "Split": "test", **metric_row(y_test, test_pred, test_prob)},
        records,
    )


def select_threshold(y: np.ndarray, prob: np.ndarray, target: float, target_metric: str) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y, prob)
    observed = tpr if target_metric == "Recall" else fpr
    distance = np.abs(observed - target)
    candidates = np.flatnonzero(distance == distance.min())
    chosen = candidates[np.argmin(np.abs(thresholds[candidates] - 0.5))]
    return float(thresholds[chosen]), float(observed[chosen])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=Path("data/diabetes_brfss2015_prepared.csv"), type=Path)
    parser.add_argument("--splits", default=Path("splits/split_assignments.csv.gz"), type=Path)
    parser.add_argument("--training-audit", default=Path("predictions/oof/training_lice_assignment_audit.csv.gz"), type=Path)
    parser.add_argument("--baseline-oof", default=Path("predictions/oof/baseline_oof_predictions_full_precision.csv.gz"), type=Path)
    parser.add_argument("--released-test", default=Path("predictions/test/eight_configuration_test_predictions_wide.csv.gz"), type=Path)
    parser.add_argument("--released-oof", default=Path("predictions/oof/eight_configuration_oof_predictions_long.csv.gz"), type=Path)
    parser.add_argument("--output-dir", default=Path("results/conventional_comparators"), type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.dataset)
    splits = pd.read_csv(args.splits)
    train_map = splits[splits.Outer_Split == "train"].sort_values("Train_Position")
    test_map = splits[splits.Outer_Split == "test"].sort_values("Test_Position")
    features = [column for column in data.columns if column != "Outcome"]
    X_train = data.loc[train_map.Source_Row_Index.astype(int), features].reset_index(drop=True)
    X_test = data.loc[test_map.Source_Row_Index.astype(int), features].reset_index(drop=True)
    y_train = data.loc[train_map.Source_Row_Index.astype(int), "Outcome"].astype(int).to_numpy()
    y_test = data.loc[test_map.Source_Row_Index.astype(int), "Outcome"].astype(int).to_numpy()
    folds = train_map.OOF_Validation_Fold.astype(int).to_numpy()

    audit = pd.read_csv(args.training_audit).sort_values("Train_Position")
    assert np.array_equal(audit.Source_Row_Index, train_map.Source_Row_Index)
    assert np.array_equal(audit.y_true.astype(int), y_train)
    baseline_oof = pd.read_csv(args.baseline_oof).sort_values("Train_Position")
    assert np.array_equal(baseline_oof.Source_Row_Index, train_map.Source_Row_Index)
    raw_fn = (baseline_oof.y_true.to_numpy(dtype=int) == 1) & (baseline_oof.oof_prediction.to_numpy(dtype=int) == 0)
    assert int(raw_fn.sum()) == 5_698

    definitions: list[tuple[str, np.ndarray, str, int | None]] = []
    for strength, weight in STRENGTHS.items():
        global_weights = np.where(y_train == 1, weight, 1.0)
        definitions.append((f"GlobalPositive-{strength}", global_weights, "all positive training rows", None))
        raw_weights = np.where(raw_fn, weight, 1.0)
        definitions.append((f"RawOOFFN-{strength}", raw_weights, "baseline OOF false-negative rows", None))

    positive_positions = np.flatnonzero(y_train == 1)
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(positive_positions, size=TARGET_COUNT, replace=False)
        weights = np.ones(len(y_train), dtype=float)
        weights[chosen] = STRENGTHS["Balanced"]
        definitions.append((f"RandomTarget-Balanced-seed{seed}", weights, "random positive training rows", seed))

    oof_rows, test_rows, prediction_frames, definition_rows = [], [], [], []
    for index, (name, weights, target_rule, seed) in enumerate(definitions, start=1):
        print(f"[{index}/{len(definitions)}] {name}", flush=True)
        oof, test, records = evaluate_weight_control(name, weights, X_train, y_train, X_test, y_test, folds)
        oof_rows.append(oof); test_rows.append(test); prediction_frames.append(records)
        definition_rows.append({
            "Model": name, "Target_Rule": target_rule, "Random_Seed": seed,
            "Weighted_Rows": int(np.sum(weights > 1.0)),
            "Weighted_Positive_Rows": int(np.sum((weights > 1.0) & (y_train == 1))),
            "Weighted_Negative_Rows": int(np.sum((weights > 1.0) & (y_train == 0))),
            "Minimum_Weight": float(weights.min()), "Maximum_Weight": float(weights.max()),
            "Mean_Weight": float(weights.mean()),
        })

    released_test = pd.read_csv(args.released_test)
    base_test_prob = released_test.baseline_probability.to_numpy(float)
    assert np.array_equal(released_test.y_true.to_numpy(int), y_test)
    base_oof_prob = baseline_oof.oof_probability.to_numpy(float)
    threshold_selection_rows, threshold_result_rows, threshold_predictions = [], [], []
    released_oof = pd.read_csv(args.released_oof)
    targets = {
        candidate: released_oof.loc[released_oof.Model == candidate]
        .sort_values("Train_Position").Predicted_Probability.to_numpy(float)
        for candidate in ("M3-Balanced", "M3-High")
    }
    if any(len(probabilities) != len(y_train) for probabilities in targets.values()):
        raise AssertionError("Canonical M3 OOF predictions are incomplete")
    for candidate, candidate_prob in targets.items():
        candidate_pred = (candidate_prob >= 0.5).astype(int)
        candidate_metrics = metric_row(y_train, candidate_pred, candidate_prob)
        for target_metric in ("Recall", "FPR"):
            target_value = candidate_metrics[target_metric]
            threshold, achieved_oof = select_threshold(y_train, base_oof_prob, target_value, target_metric)
            name = f"BaselineThreshold-Match{target_metric}-{candidate}"
            oof_pred = (base_oof_prob >= threshold).astype(int)
            test_pred = (base_test_prob >= threshold).astype(int)
            threshold_selection_rows.append({
                "Comparator": name, "Target_Configuration": candidate,
                "Target_Metric": target_metric, "Target_OOF_Value": target_value,
                "Selected_Baseline_Threshold": threshold, "Achieved_Baseline_OOF_Value": achieved_oof,
                "Absolute_OOF_Matching_Error": abs(achieved_oof - target_value),
                "Selection_Data": "baseline and candidate training-side OOF predictions",
            })
            threshold_result_rows.extend([
                {"Model": name, "Split": "OOF", **metric_row(y_train, oof_pred, base_oof_prob)},
                {"Model": name, "Split": "test", **metric_row(y_test, test_pred, base_test_prob)},
            ])
            threshold_predictions.extend([
                pd.DataFrame({"Split": "OOF", "Position": np.arange(len(y_train)), "Model": name, "y_true": y_train, "Predicted_Probability": base_oof_prob, "Predicted_Class": oof_pred, "Decision_Threshold": threshold}),
                pd.DataFrame({"Split": "test", "Position": np.arange(len(y_test)), "Model": name, "y_true": y_test, "Predicted_Probability": base_test_prob, "Predicted_Class": test_pred, "Decision_Threshold": threshold}),
            ])

    metrics = pd.DataFrame(oof_rows + test_rows + threshold_result_rows)
    definitions_df = pd.DataFrame(definition_rows)
    threshold_selection = pd.DataFrame(threshold_selection_rows)
    predictions = pd.concat(prediction_frames + threshold_predictions, ignore_index=True)
    random_summary = (
        metrics[metrics.Model.str.startswith("RandomTarget")]
        .groupby("Split")[["TP", "TN", "FP", "FN", "Accuracy", "AUC", "Recall", "Specificity", "FPR", "Precision", "F1", "Kappa", "MCC"]]
        .agg(["mean", "std", "min", "max"])
    )
    random_summary.columns = [f"{metric}_{stat}" for metric, stat in random_summary.columns]
    random_summary = random_summary.reset_index()

    metrics.to_csv(args.output_dir / "conventional_comparator_performance.csv", index=False)
    definitions_df.to_csv(args.output_dir / "conventional_comparator_definitions.csv", index=False)
    threshold_selection.to_csv(args.output_dir / "matched_operating_point_selection.csv", index=False)
    predictions.to_csv(args.output_dir / "conventional_comparator_predictions_long.csv.gz", index=False)
    random_summary.to_csv(args.output_dir / "random_targeting_summary.csv", index=False)

    manifest = {
        "artifact_ids": ["conventional_comparator_performance", "matched_operating_point_comparison"],
        "dataset_rows": 69_057, "training_rows": 55_245, "test_rows": 13_812,
        "original_threshold": 0.5,
        "model": "GradientBoostingClassifier with the released baseline parameters",
        "controls": {
            "global_positive_weighting": STRENGTHS,
            "raw_oof_false_negative_weighting": STRENGTHS,
            "random_positive_targeting": {"weight": 1.25, "target_count": TARGET_COUNT, "seeds": RANDOM_SEEDS},
            "threshold": "selected using training-side OOF predictions only and transferred unchanged to test probabilities",
        },
        "excluded_controls": {
            "clinically_specified_interactions": "not claimed because no clinical specification exercise was conducted",
            "focal_loss": "not included because it is not a native loss of the fixed scikit-learn GradientBoostingClassifier workflow",
        },
        "test_guided_training_or_threshold_selection": False,
    }
    (args.output_dir / "conventional_comparator_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nTest results")
    print(metrics[metrics.Split == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
