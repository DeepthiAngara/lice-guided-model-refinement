"""Build prediction-level audit artifacts and verify Tables 3-10.

Run from the repository root with:

    python tests/verify_released_predictions.py

The program exits with a non-zero status if any released prediction, metric,
transition, McNemar result, or presentation table cannot be reproduced.
"""

from __future__ import annotations

import hashlib
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
from scipy.stats import chi2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_brfss2015_prepared.csv"
SPLIT_PATH = PROJECT_ROOT / "splits" / "split_assignments.csv.gz"
WEIGHT_PATH = PROJECT_ROOT / "input_artifacts" / "lice_sample_weights.csv.gz"
PATTERN_PATH = PROJECT_ROOT / "results" / "pattern_match_trace.csv.gz"
PREDICTION_PATH = PROJECT_ROOT / "results" / "predictions_all_models.csv.gz"
METRIC_PATH = PROJECT_ROOT / "results" / "metrics_full_precision.csv"
MCNEMAR_PATH = PROJECT_ROOT / "results" / "mcnemar_results_full_precision.csv"
REFERENCE_PATH = (
    PROJECT_ROOT / "reference_inputs" / "published_gbc_reference.csv"
)
TABLE_DIR = PROJECT_ROOT / "generated_outputs" / "tables"
RESULTS_DIR = PROJECT_ROOT / "results"

DECISION_THRESHOLD = 0.50
FULL_PRECISION_ATOL = 1e-12
FOUR_DECIMAL_TOLERANCE = 5e-5
TWO_DECIMAL_PP_TOLERANCE = 0.005

MODEL_SPECIFICATIONS = [
    {
        "model": "Baseline",
        "prefix": "baseline",
        "training_weight_scheme": "None",
        "uses_lice_interactions": False,
        "uses_lice_sample_weighting": False,
    },
    {
        "model": "M1-Interactions",
        "prefix": "m1_interactions",
        "training_weight_scheme": "None",
        "uses_lice_interactions": True,
        "uses_lice_sample_weighting": False,
    },
    {
        "model": "M2-Mild",
        "prefix": "m2_mild",
        "training_weight_scheme": "Mild",
        "uses_lice_interactions": False,
        "uses_lice_sample_weighting": True,
    },
    {
        "model": "M2-Balanced",
        "prefix": "m2_balanced",
        "training_weight_scheme": "Balanced",
        "uses_lice_interactions": False,
        "uses_lice_sample_weighting": True,
    },
    {
        "model": "M2-High",
        "prefix": "m2_high",
        "training_weight_scheme": "High",
        "uses_lice_interactions": False,
        "uses_lice_sample_weighting": True,
    },
    {
        "model": "M3-Mild",
        "prefix": "m3_mild",
        "training_weight_scheme": "Mild",
        "uses_lice_interactions": True,
        "uses_lice_sample_weighting": True,
    },
    {
        "model": "M3-Balanced",
        "prefix": "m3_balanced",
        "training_weight_scheme": "Balanced",
        "uses_lice_interactions": True,
        "uses_lice_sample_weighting": True,
    },
    {
        "model": "M3-High",
        "prefix": "m3_high",
        "training_weight_scheme": "High",
        "uses_lice_interactions": True,
        "uses_lice_sample_weighting": True,
    },
]

MODEL_ORDER = [specification["model"] for specification in MODEL_SPECIFICATIONS]
DISPLAY_NAMES = {
    "Baseline": "M0: baseline TunedGBDT",
    "M1-Interactions": "M1: LICE-derived interaction-only model",
    "M2-Mild": "M2-Mild: LICE-weighted-only",
    "M2-Balanced": "M2-Balanced: LICE-weighted-only",
    "M2-High": "M2-High: LICE-weighted-only",
    "M3-Mild": "M3-Mild: LICE-Interaction + LICE-weighting",
    "M3-Balanced": "M3-Balanced: LICE-Interaction + LICE-weighting",
    "M3-High": "M3-High: LICE-Interaction + LICE-weighting",
}
SELECTED_NAMES = {
    "Baseline": "Baseline",
    "M3-Balanced": "LICE-BalancedGBDT",
    "M3-High": "LICE-HighSensitivityGBDT",
}


checks: list[dict[str, str]] = []


def record_check(
    check_id: str,
    scope: str,
    passed: bool,
    observed: object,
    expected: object,
    tolerance: str,
) -> None:
    checks.append(
        {
            "Check_ID": check_id,
            "Scope": scope,
            "Observed": str(observed),
            "Expected": str(expected),
            "Tolerance": tolerance,
            "Status": "PASS" if passed else "FAIL",
        }
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f4(value: float) -> str:
    return f"{float(value):.4f}"


def f2(value: float) -> str:
    return f"{float(value):.2f}"


def error_type(y_true: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.select(
        [
            (y_true == 0) & (prediction == 0),
            (y_true == 0) & (prediction == 1),
            (y_true == 1) & (prediction == 0),
            (y_true == 1) & (prediction == 1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="Unexpected",
    )


def mcnemar_categories(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> np.ndarray:
    baseline_correct = baseline_prediction == y_true
    candidate_correct = candidate_prediction == y_true
    return np.select(
        [
            baseline_correct & candidate_correct,
            baseline_correct & ~candidate_correct,
            ~baseline_correct & candidate_correct,
            ~baseline_correct & ~candidate_correct,
        ],
        [
            "Both_Correct",
            "Baseline_Correct_Model_Wrong",
            "Baseline_Wrong_Model_Correct",
            "Both_Wrong",
        ],
        default="Unexpected",
    )


def fn_transition_categories(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> np.ndarray:
    return np.select(
        [
            y_true == 0,
            (y_true == 1)
            & (baseline_prediction == 0)
            & (candidate_prediction == 1),
            (y_true == 1)
            & (baseline_prediction == 1)
            & (candidate_prediction == 0),
            (y_true == 1)
            & (baseline_prediction == 0)
            & (candidate_prediction == 0),
            (y_true == 1)
            & (baseline_prediction == 1)
            & (candidate_prediction == 1),
        ],
        [
            "Not_Applicable_Actual_Negative",
            "FN_to_TP",
            "TP_to_FN",
            "FN_to_FN",
            "TP_to_TP",
        ],
        default="Unexpected",
    )


def compare_presentation_table(number: int, observed: pd.DataFrame) -> None:
    table_paths = {
        3: "table_3_test_set_performance.csv",
        4: "table_4_false_negative_tradeoff.csv",
        5: "table_5_incremental_interaction_effect.csv",
        6: "table_6_criterion_summary.csv",
        7: "table_7_mcnemar_positive_cases.csv",
        8: "table_8_operating_point_summary.csv",
        9: "table_9_metric_change_summary.csv",
        10: "table_10_cross_study_comparison.csv",
    }
    expected = pd.read_csv(
        TABLE_DIR / table_paths[number], dtype=str, keep_default_na=False
    )
    observed_text = observed.astype(str)
    same_columns = observed_text.columns.tolist() == expected.columns.tolist()
    same_shape = observed_text.shape == expected.shape
    same_values = same_columns and same_shape and observed_text.equals(expected)
    detail = "exact formatted match"
    if not same_values:
        detail = "schema, shape, or formatted values differ"
    record_check(
        f"table_{number}_reproduction",
        f"Table {number}",
        same_values,
        detail,
        "exact formatted match",
        "four-decimal tolerance 0.00005; two-decimal pp tolerance 0.005",
    )


required_paths = [
    DATA_PATH,
    SPLIT_PATH,
    WEIGHT_PATH,
    PATTERN_PATH,
    PREDICTION_PATH,
    METRIC_PATH,
    MCNEMAR_PATH,
    REFERENCE_PATH,
]
for required_path in required_paths:
    record_check(
        f"input_exists_{required_path.name}",
        "Input availability",
        required_path.exists(),
        required_path.exists(),
        True,
        "exact",
    )

if not all(path.exists() for path in required_paths):
    missing = [str(path) for path in required_paths if not path.exists()]
    raise FileNotFoundError(f"Required release files are missing: {missing}")


dataset = pd.read_csv(DATA_PATH)
dataset_context = dataset.reset_index(names="Source_Row_Index")
splits = pd.read_csv(SPLIT_PATH)
wide_predictions = pd.read_csv(PREDICTION_PATH, float_precision="round_trip")
released_metrics = pd.read_csv(
    METRIC_PATH, float_precision="round_trip"
).set_index("Model")
released_mcnemar = pd.read_csv(
    MCNEMAR_PATH, float_precision="round_trip"
)
published_reference = pd.read_csv(
    REFERENCE_PATH, float_precision="round_trip"
).iloc[0]

required_prediction_columns = {"Source_Row_Index", "y_true"}
for specification in MODEL_SPECIFICATIONS:
    prefix = specification["prefix"]
    required_prediction_columns.update(
        {f"{prefix}_prediction", f"{prefix}_probability"}
    )
record_check(
    "prediction_schema",
    "Released wide prediction file",
    required_prediction_columns.issubset(wide_predictions.columns),
    len(required_prediction_columns.intersection(wide_predictions.columns)),
    len(required_prediction_columns),
    "exact required-column inclusion",
)
record_check(
    "prediction_row_count",
    "Released wide prediction file",
    len(wide_predictions) == 13812,
    len(wide_predictions),
    13812,
    "exact",
)
record_check(
    "prediction_unique_source_rows",
    "Released wide prediction file",
    wide_predictions["Source_Row_Index"].is_unique,
    wide_predictions["Source_Row_Index"].nunique(),
    len(wide_predictions),
    "exact",
)


test_split = splits.loc[
    splits["Outer_Split"].eq("test"),
    [
        "Source_Row_Index",
        "Outer_Split",
        "Test_Position",
        "OOF_Validation_Fold",
    ],
].copy()
record_check(
    "test_split_row_count",
    "Split assignment",
    len(test_split) == len(wide_predictions),
    len(test_split),
    len(wide_predictions),
    "exact",
)
record_check(
    "test_fold_not_applicable",
    "Untouched test-set semantics",
    test_split["OOF_Validation_Fold"].isna().all(),
    int(test_split["OOF_Validation_Fold"].notna().sum()),
    0,
    "exact",
)

test_context = wide_predictions.merge(
    test_split, on="Source_Row_Index", how="left", validate="one_to_one"
).merge(
    dataset_context[
        ["Source_Row_Index", "Outcome", "GenHlth", "HighBP", "HighChol", "BMI"]
    ],
    on="Source_Row_Index",
    how="left",
    validate="one_to_one",
)
record_check(
    "prediction_split_join",
    "Prediction-to-split linkage",
    test_context["Outer_Split"].eq("test").all()
    and test_context["Test_Position"].notna().all(),
    int(test_context["Test_Position"].notna().sum()),
    len(test_context),
    "exact",
)
record_check(
    "prediction_true_label_linkage",
    "Prediction-to-dataset linkage",
    np.array_equal(
        test_context["y_true"].to_numpy(dtype=int),
        test_context["Outcome"].to_numpy(dtype=int),
    ),
    "linked labels",
    "dataset Outcome",
    "exact",
)

test_context["Match_FN_P1_GenHlth_le_2"] = (
    test_context["GenHlth"] <= 2
).astype(int)
test_context["Match_FN_P2_HighBP_le_0"] = (
    test_context["HighBP"] <= 0
).astype(int)
test_context["Match_FN_P3_HighChol_le_0"] = (
    test_context["HighChol"] <= 0
).astype(int)
test_context["Match_FN_P4_BMI_le_25"] = (
    test_context["BMI"] <= 25
).astype(int)
pattern_columns = [
    "Match_FN_P1_GenHlth_le_2",
    "Match_FN_P2_HighBP_le_0",
    "Match_FN_P3_HighChol_le_0",
    "Match_FN_P4_BMI_le_25",
]
test_context["LICE_Pattern_Match_Count"] = test_context[pattern_columns].sum(
    axis=1
)
test_context["Matches_At_Least_Two_LICE_Patterns"] = (
    test_context["LICE_Pattern_Match_Count"] >= 2
).astype(int)
test_context["LICE_Targeted_Positive_Region"] = (
    test_context["y_true"].eq(1)
    & test_context["Matches_At_Least_Two_LICE_Patterns"].eq(1)
).astype(int)

baseline_prediction = test_context["baseline_prediction"].to_numpy(dtype=int)
y_true = test_context["y_true"].to_numpy(dtype=int)
long_frames = []
for specification in MODEL_SPECIFICATIONS:
    prefix = specification["prefix"]
    prediction = test_context[f"{prefix}_prediction"].to_numpy(dtype=int)
    probability = test_context[f"{prefix}_probability"].to_numpy(dtype=float)

    threshold_prediction = (probability >= DECISION_THRESHOLD).astype(int)
    record_check(
        f"threshold_{prefix}",
        specification["model"],
        np.array_equal(prediction, threshold_prediction),
        int(np.sum(prediction != threshold_prediction)),
        0,
        "exact at threshold 0.50",
    )
    record_check(
        f"probability_range_{prefix}",
        specification["model"],
        bool(np.all((probability >= 0.0) & (probability <= 1.0))),
        f"[{probability.min():.17g}, {probability.max():.17g}]",
        "[0, 1]",
        "inclusive",
    )

    frame = test_context[
        [
            "Source_Row_Index",
            "Outer_Split",
            "Test_Position",
            "OOF_Validation_Fold",
            "y_true",
            "GenHlth",
            "HighBP",
            "HighChol",
            "BMI",
            *pattern_columns,
            "LICE_Pattern_Match_Count",
            "Matches_At_Least_Two_LICE_Patterns",
            "LICE_Targeted_Positive_Region",
        ]
    ].copy()
    frame.insert(4, "Fold_Assignment_Status", "Not applicable: untouched test set")
    frame.insert(5, "Model", specification["model"])
    frame.insert(7, "Predicted_Probability", probability)
    frame.insert(8, "Predicted_Class", prediction)
    frame.insert(9, "Decision_Threshold", DECISION_THRESHOLD)
    frame.insert(10, "Evaluation_Sample_Weight", 1.0)
    frame.insert(11, "Training_Weight_Scheme", specification["training_weight_scheme"])
    frame.insert(12, "Uses_LICE_Interactions", specification["uses_lice_interactions"])
    frame.insert(
        13,
        "Uses_LICE_Sample_Weighting",
        specification["uses_lice_sample_weighting"],
    )
    frame["Error_Type"] = error_type(y_true, prediction)
    frame["Baseline_Predicted_Class"] = baseline_prediction
    if specification["model"] == "Baseline":
        frame["FN_Transition_vs_Baseline"] = "Baseline_Reference"
        frame["McNemar_Category_vs_Baseline"] = "Baseline_Reference"
    else:
        frame["FN_Transition_vs_Baseline"] = fn_transition_categories(
            y_true, baseline_prediction, prediction
        )
        frame["McNemar_Category_vs_Baseline"] = mcnemar_categories(
            y_true, baseline_prediction, prediction
        )
    long_frames.append(frame)

test_predictions_long = pd.concat(long_frames, ignore_index=True)
test_predictions_long["Model"] = pd.Categorical(
    test_predictions_long["Model"], categories=MODEL_ORDER, ordered=True
)
test_predictions_long = test_predictions_long.sort_values(
    ["Test_Position", "Model"], kind="stable"
).reset_index(drop=True)
test_predictions_long["Model"] = test_predictions_long["Model"].astype(str)

record_check(
    "long_prediction_row_count",
    "Long-format prediction audit",
    len(test_predictions_long) == 13812 * 8,
    len(test_predictions_long),
    13812 * 8,
    "exact",
)
record_check(
    "long_prediction_unique_key",
    "Long-format prediction audit",
    not test_predictions_long.duplicated(["Source_Row_Index", "Model"]).any(),
    int(
        test_predictions_long.duplicated(["Source_Row_Index", "Model"]).sum()
    ),
    0,
    "exact",
)
record_check(
    "test_evaluation_weights",
    "Untouched test-set semantics",
    test_predictions_long["Evaluation_Sample_Weight"].eq(1.0).all(),
    test_predictions_long["Evaluation_Sample_Weight"].unique().tolist(),
    [1.0],
    "exact",
)

test_audit_path = RESULTS_DIR / "test_predictions_long.csv.gz"
test_predictions_long.to_csv(
    test_audit_path,
    index=False,
    float_format="%.17g",
    compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
)


training_split = splits.loc[
    splits["Outer_Split"].eq("train"),
    ["Source_Row_Index", "Outer_Split", "Train_Position", "OOF_Validation_Fold"],
].copy()
weights = pd.read_csv(WEIGHT_PATH)
patterns = pd.read_csv(PATTERN_PATH)
training_audit = training_split.merge(
    weights,
    on=["Source_Row_Index", "Train_Position"],
    how="left",
    validate="one_to_one",
).merge(
    patterns,
    on=["Source_Row_Index", "Train_Position"],
    how="left",
    validate="one_to_one",
).merge(
    dataset_context[
        ["Source_Row_Index", "Outcome", "GenHlth", "HighBP", "HighChol", "BMI"]
    ],
    on="Source_Row_Index",
    how="left",
    validate="one_to_one",
)
training_audit = training_audit.sort_values("Train_Position").reset_index(drop=True)
record_check(
    "training_audit_row_count",
    "Training LICE assignment audit",
    len(training_audit) == 55245,
    len(training_audit),
    55245,
    "exact",
)
record_check(
    "training_audit_label_linkage",
    "Training LICE assignment audit",
    np.array_equal(
        training_audit["Actual"].to_numpy(dtype=int),
        training_audit["Outcome"].to_numpy(dtype=int),
    ),
    "linked labels",
    "dataset Outcome",
    "exact",
)

recalculated_training_patterns = pd.DataFrame(
    {
        "Match_FN_P1": (training_audit["GenHlth"] <= 2).astype(int),
        "Match_FN_P2": (training_audit["HighBP"] <= 0).astype(int),
        "Match_FN_P3": (training_audit["HighChol"] <= 0).astype(int),
        "Match_FN_P4": (training_audit["BMI"] <= 25).astype(int),
    }
)
for pattern_column in recalculated_training_patterns.columns:
    record_check(
        f"training_{pattern_column}",
        "Training LICE pattern linkage",
        np.array_equal(
            training_audit[pattern_column].to_numpy(dtype=int),
            recalculated_training_patterns[pattern_column].to_numpy(dtype=int),
        ),
        int(
            np.sum(
                training_audit[pattern_column].to_numpy(dtype=int)
                != recalculated_training_patterns[pattern_column].to_numpy(dtype=int)
            )
        ),
        0,
        "exact",
    )

training_audit = training_audit.rename(
    columns={
        "Actual": "y_true",
        "Mild": "Mild_Sample_Weight",
        "Balanced": "Balanced_Sample_Weight",
        "High": "High_Sensitivity_Sample_Weight",
        "Targeted_Positive": "LICE_Targeted_Positive",
    }
)
training_audit = training_audit[
    [
        "Source_Row_Index",
        "Outer_Split",
        "Train_Position",
        "OOF_Validation_Fold",
        "y_true",
        "GenHlth",
        "HighBP",
        "HighChol",
        "BMI",
        "Match_FN_P1",
        "Match_FN_P2",
        "Match_FN_P3",
        "Match_FN_P4",
        "Pattern_Match_Count",
        "LICE_Targeted_Positive",
        "Mild_Sample_Weight",
        "Balanced_Sample_Weight",
        "High_Sensitivity_Sample_Weight",
    ]
]
training_audit_path = RESULTS_DIR / "training_lice_assignment_audit.csv.gz"
training_audit.to_csv(
    training_audit_path,
    index=False,
    float_format="%.17g",
    compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
)


metric_rows = []
fn_transition_rows = []
for specification in MODEL_SPECIFICATIONS:
    model = specification["model"]
    prefix = specification["prefix"]
    prediction = wide_predictions[f"{prefix}_prediction"].to_numpy(dtype=int)
    probability = wide_predictions[f"{prefix}_probability"].to_numpy(dtype=float)
    true_values = wide_predictions["y_true"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(true_values, prediction).ravel()
    metric_rows.append(
        {
            "Model": model,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "Accuracy": accuracy_score(true_values, prediction),
            "AUC": roc_auc_score(true_values, probability),
            "Recall": recall_score(true_values, prediction, zero_division=0),
            "Precision": precision_score(true_values, prediction, zero_division=0),
            "F1": f1_score(true_values, prediction, zero_division=0),
            "Kappa": cohen_kappa_score(true_values, prediction),
            "MCC": matthews_corrcoef(true_values, prediction),
        }
    )
    if model != "Baseline":
        transitions = fn_transition_categories(
            true_values, baseline_prediction, prediction
        )
        fn_transition_rows.append(
            {
                "Model": model,
                "FN_to_TP": int(np.sum(transitions == "FN_to_TP")),
                "TP_to_FN": int(np.sum(transitions == "TP_to_FN")),
                "FN_to_FN": int(np.sum(transitions == "FN_to_FN")),
                "TP_to_TP": int(np.sum(transitions == "TP_to_TP")),
                "Net_FN_Reduction": int(
                    np.sum(transitions == "FN_to_TP")
                    - np.sum(transitions == "TP_to_FN")
                ),
            }
        )

computed_metrics = pd.DataFrame(metric_rows).set_index("Model")
fn_transition_summary = pd.DataFrame(fn_transition_rows)
fn_transition_path = RESULTS_DIR / "fn_transition_summary_from_predictions.csv"
fn_transition_summary.to_csv(fn_transition_path, index=False)

count_columns = ["TN", "FP", "FN", "TP"]
metric_columns = ["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa", "MCC"]
for model in MODEL_ORDER:
    counts_match = all(
        int(computed_metrics.loc[model, column])
        == int(released_metrics.loc[model, column])
        for column in count_columns
    )
    record_check(
        f"confusion_matrix_{model}",
        model,
        counts_match,
        computed_metrics.loc[model, count_columns].astype(int).to_dict(),
        released_metrics.loc[model, count_columns].astype(int).to_dict(),
        "exact",
    )
    maximum_metric_difference = max(
        abs(
            float(computed_metrics.loc[model, column])
            - float(released_metrics.loc[model, column])
        )
        for column in metric_columns
    )
    record_check(
        f"full_precision_metrics_{model}",
        model,
        maximum_metric_difference <= FULL_PRECISION_ATOL,
        f"max_abs_difference={maximum_metric_difference:.17g}",
        "max_abs_difference<=1e-12",
        "absolute tolerance 1e-12",
    )


baseline_metrics = computed_metrics.loc["Baseline"]
comparison = computed_metrics.copy()
for metric in metric_columns:
    comparison[f"Delta_{metric}"] = comparison[metric] - baseline_metrics[metric]
comparison["FN_Reduction"] = baseline_metrics["FN"] - comparison["FN"]
comparison["FP_Increase"] = comparison["FP"] - baseline_metrics["FP"]
comparison["Relative_FN_Reduction_Percentage"] = (
    comparison["FN_Reduction"] / baseline_metrics["FN"] * 100.0
)


def calculate_mcnemar(
    true_values: np.ndarray,
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    comparison_name: str,
    scope: str,
) -> dict[str, object]:
    baseline_correct = baseline_values == true_values
    candidate_correct = candidate_values == true_values
    both_correct = int(np.sum(baseline_correct & candidate_correct))
    baseline_correct_candidate_wrong = int(
        np.sum(baseline_correct & ~candidate_correct)
    )
    baseline_wrong_candidate_correct = int(
        np.sum(~baseline_correct & candidate_correct)
    )
    both_wrong = int(np.sum(~baseline_correct & ~candidate_correct))
    discordant_total = (
        baseline_correct_candidate_wrong + baseline_wrong_candidate_correct
    )
    statistic = (
        (abs(
            baseline_correct_candidate_wrong
            - baseline_wrong_candidate_correct
        ) - 1.0) ** 2
        / discordant_total
    )
    p_value = float(chi2.sf(statistic, df=1))
    return {
        "Scope": scope,
        "Comparison": comparison_name,
        "Both_Correct": both_correct,
        "Baseline_Correct_Candidate_Wrong": baseline_correct_candidate_wrong,
        "Baseline_Wrong_Candidate_Correct": baseline_wrong_candidate_correct,
        "Both_Wrong": both_wrong,
        "McNemar_Statistic": float(statistic),
        "p_value": p_value,
    }


mcnemar_rows = []
positive_mask = y_true == 1
for candidate_model, candidate_prefix in [
    ("M3-Balanced", "m3_balanced"),
    ("M3-High", "m3_high"),
]:
    candidate_prediction = wide_predictions[
        f"{candidate_prefix}_prediction"
    ].to_numpy(dtype=int)
    mcnemar_rows.append(
        calculate_mcnemar(
            y_true,
            baseline_prediction,
            candidate_prediction,
            f"Baseline vs {candidate_model}",
            "Full test set",
        )
    )
    mcnemar_rows.append(
        calculate_mcnemar(
            y_true[positive_mask],
            baseline_prediction[positive_mask],
            candidate_prediction[positive_mask],
            f"Baseline vs {candidate_model}",
            "Actual positive cases only",
        )
    )
computed_mcnemar = pd.DataFrame(mcnemar_rows)
for _, computed_row in computed_mcnemar.iterrows():
    expected_row = released_mcnemar.loc[
        released_mcnemar["Scope"].eq(computed_row["Scope"])
        & released_mcnemar["Comparison"].eq(computed_row["Comparison"])
    ].iloc[0]
    count_match = all(
        int(computed_row[column]) == int(expected_row[column])
        for column in [
            "Both_Correct",
            "Baseline_Correct_Candidate_Wrong",
            "Baseline_Wrong_Candidate_Correct",
            "Both_Wrong",
        ]
    )
    numeric_difference = max(
        abs(float(computed_row[column]) - float(expected_row[column]))
        for column in ["McNemar_Statistic", "p_value"]
    )
    record_check(
        "mcnemar_"
        + computed_row["Scope"].lower().replace(" ", "_")
        + "_"
        + computed_row["Comparison"].lower().replace(" ", "_"),
        f"{computed_row['Scope']}: {computed_row['Comparison']}",
        count_match and numeric_difference <= FULL_PRECISION_ATOL,
        f"counts_match={count_match}; max_abs_difference={numeric_difference:.17g}",
        "released McNemar result",
        "counts exact; numeric absolute tolerance 1e-12",
    )


# Table 3
table_3 = pd.DataFrame(
    [
        {
            "Variant": DISPLAY_NAMES[model],
            "TP": str(int(computed_metrics.loc[model, "TP"])),
            "TN": str(int(computed_metrics.loc[model, "TN"])),
            "FP": str(int(computed_metrics.loc[model, "FP"])),
            "FN": str(int(computed_metrics.loc[model, "FN"])),
            "Accuracy": f4(computed_metrics.loc[model, "Accuracy"]),
            "AUC": f4(computed_metrics.loc[model, "AUC"]),
            "Recall": f4(computed_metrics.loc[model, "Recall"]),
            "Precision": f4(computed_metrics.loc[model, "Precision"]),
            "F1": f4(computed_metrics.loc[model, "F1"]),
            "MCC": f4(computed_metrics.loc[model, "MCC"]),
        }
        for model in MODEL_ORDER
    ]
)

# Table 4
table_4_rows = []
for model in MODEL_ORDER:
    row = comparison.loc[model]
    if model == "Baseline":
        relative_fn = delta_recall = delta_precision = delta_f1 = delta_mcc = "0"
    else:
        relative_fn = f2(row["Relative_FN_Reduction_Percentage"])
        delta_recall = f4(row["Delta_Recall"])
        delta_precision = f4(row["Delta_Precision"])
        delta_f1 = f4(row["Delta_F1"])
        delta_mcc = f4(row["Delta_MCC"])
    table_4_rows.append(
        {
            "Variant": DISPLAY_NAMES[model],
            "FN Reduction": str(int(row["FN_Reduction"])),
            "Relative FN Reduction (%)": relative_fn,
            "FP Increase": str(int(row["FP_Increase"])),
            "Delta Recall": delta_recall,
            "Delta Precision": delta_precision,
            "Delta F1": delta_f1,
            "Delta MCC": delta_mcc,
        }
    )
table_4 = pd.DataFrame(table_4_rows)

# Table 5
table_5_rows = []
for label, weighting_model, combined_model in [
    ("Mild", "M2-Mild", "M3-Mild"),
    ("Balanced", "M2-Balanced", "M3-Balanced"),
    ("High-Sensitivity", "M2-High", "M3-High"),
]:
    weighting = comparison.loc[weighting_model]
    combined = comparison.loc[combined_model]
    table_5_rows.append(
        {
            "Weighting Configuration": label,
            "Weighting-only FN Reduction": str(int(weighting["FN_Reduction"])),
            "Interaction + Weighting FN Reduction": str(
                int(combined["FN_Reduction"])
            ),
            "Additional FN Reduction": str(
                int(combined["FN_Reduction"] - weighting["FN_Reduction"])
            ),
            "Additional FP Change": str(
                int(combined["FP_Increase"] - weighting["FP_Increase"])
            ),
            "Additional Delta Recall": f4(
                combined["Delta_Recall"] - weighting["Delta_Recall"]
            ),
            "Additional Delta F1": f4(
                combined["Delta_F1"] - weighting["Delta_F1"]
            ),
            "Additional Delta MCC": f4(
                combined["Delta_MCC"] - weighting["Delta_MCC"]
            ),
        }
    )
table_5 = pd.DataFrame(table_5_rows)

# Table 6
table_6_rows = []
for criterion, column, direction, formatter in [
    ("Lowest FN / Highest FN Reduction", "FN", "min", "{:.0f} FN"),
    ("Highest Recall", "Recall", "max", "{:.4f}"),
    ("Highest F1-Score", "F1", "max", "{:.4f}"),
    ("Highest MCC", "MCC", "max", "{:.4f}"),
    ("Highest AUC", "AUC", "max", "{:.4f}"),
    ("Highest Precision", "Precision", "max", "{:.4f}"),
]:
    model = (
        computed_metrics.loc[MODEL_ORDER, column].idxmin()
        if direction == "min"
        else computed_metrics.loc[MODEL_ORDER, column].idxmax()
    )
    table_6_rows.append(
        {
            "Criterion": criterion,
            "Best Variant": DISPLAY_NAMES[model],
            "Value": formatter.format(float(computed_metrics.loc[model, column])),
        }
    )
table_6 = pd.DataFrame(table_6_rows)

# Table 7
positive_mcnemar = computed_mcnemar.loc[
    computed_mcnemar["Scope"].eq("Actual positive cases only")
].copy()
table_7 = pd.DataFrame(
    {
        "Comparison": positive_mcnemar["Comparison"]
        .str.replace("M3-Balanced", "LICE-BalancedGBDT", regex=False)
        .str.replace("M3-High", "LICE-HighSensitivityGBDT", regex=False),
        "Baseline Correct and LICE Wrong": positive_mcnemar[
            "Baseline_Correct_Candidate_Wrong"
        ].map(lambda value: str(int(value))),
        "Baseline Wrong and LICE Correct": positive_mcnemar[
            "Baseline_Wrong_Candidate_Correct"
        ].map(lambda value: str(int(value))),
        "Chi-square": positive_mcnemar["McNemar_Statistic"].map(
            lambda value: f"{float(value):.2f}"
        ),
        "p": positive_mcnemar["p_value"].map(
            lambda value: "<0.001" if float(value) < 0.001 else f4(value)
        ),
    }
).reset_index(drop=True)

# Tables 8 and 9
selected_models = ["Baseline", "M3-Balanced", "M3-High"]
table_8 = pd.DataFrame(
    [
        {
            "Model": SELECTED_NAMES[model],
            "FN": str(int(computed_metrics.loc[model, "FN"])),
            "FP": str(int(computed_metrics.loc[model, "FP"])),
            "Recall": f4(computed_metrics.loc[model, "Recall"]),
            "Precision": f4(computed_metrics.loc[model, "Precision"]),
            "F1": f4(computed_metrics.loc[model, "F1"]),
            "AUC": f4(computed_metrics.loc[model, "AUC"]),
            "MCC": f4(computed_metrics.loc[model, "MCC"]),
        }
        for model in selected_models
    ]
)

table_9_rows = []
for model in selected_models:
    row = comparison.loc[model]
    if model == "Baseline":
        relative_fn = delta_recall = delta_precision = delta_f1 = "0"
        delta_auc = delta_mcc = "0"
    else:
        relative_fn = f2(row["Relative_FN_Reduction_Percentage"])
        delta_recall = f4(row["Delta_Recall"])
        delta_precision = f4(row["Delta_Precision"])
        delta_f1 = f4(row["Delta_F1"])
        delta_auc = f4(row["Delta_AUC"])
        delta_mcc = f4(row["Delta_MCC"])
    table_9_rows.append(
        {
            "Model": SELECTED_NAMES[model],
            "FN Reduction": str(int(row["FN_Reduction"])),
            "Relative FN Reduction": relative_fn,
            "Recall": delta_recall,
            "Precision": delta_precision,
            "F1": delta_f1,
            "AUC": delta_auc,
            "MCC": delta_mcc,
        }
    )
table_9 = pd.DataFrame(table_9_rows)

# Table 10
table_10_rows = [
    {
        "Model": str(published_reference["Model"]),
        "Recall Difference (pp)": "0.00",
        "Precision Difference (pp)": "0.00",
        "F1 Difference (pp)": "0.00",
        "AUC Difference (pp)": "0.00",
        "MCC Difference (pp)": "0.00",
    }
]
for model in selected_models:
    row = computed_metrics.loc[model]
    table_10_rows.append(
        {
            "Model": SELECTED_NAMES[model],
            "Recall Difference (pp)": f2(
                100.0 * (row["Recall"] - published_reference["Recall"])
            ),
            "Precision Difference (pp)": f2(
                100.0 * (row["Precision"] - published_reference["Precision"])
            ),
            "F1 Difference (pp)": f2(
                100.0 * (row["F1"] - published_reference["F1"])
            ),
            "AUC Difference (pp)": f2(
                100.0 * (row["AUC"] - published_reference["AUC"])
            ),
            "MCC Difference (pp)": f2(
                100.0 * (row["MCC"] - published_reference["MCC"])
            ),
        }
    )
table_10 = pd.DataFrame(table_10_rows)

for table_number, table_frame in [
    (3, table_3),
    (4, table_4),
    (5, table_5),
    (6, table_6),
    (7, table_7),
    (8, table_8),
    (9, table_9),
    (10, table_10),
]:
    compare_presentation_table(table_number, table_frame)


check_report = pd.DataFrame(checks)
check_report_path = RESULTS_DIR / "prediction_reproduction_checks.csv"
check_report.to_csv(check_report_path, index=False)
all_checks_passed = check_report["Status"].eq("PASS").all()

manifest = {
    "release_tag": "v1.2.0",
    "purpose": (
        "Prediction-level audit and independent reproduction of manuscript "
        "Tables 3-10 from released case-level predictions."
    ),
    "test_set_semantics": {
        "outer_split": "test",
        "oof_validation_fold": "not applicable to the untouched test set",
        "evaluation_sample_weight": 1.0,
        "decision_threshold": DECISION_THRESHOLD,
    },
    "row_counts": {
        "test_instances": int(len(wide_predictions)),
        "model_variants": int(len(MODEL_ORDER)),
        "long_prediction_rows": int(len(test_predictions_long)),
        "training_audit_rows": int(len(training_audit)),
    },
    "tolerances": {
        "integer_counts": "exact",
        "full_precision_metrics_absolute": FULL_PRECISION_ATOL,
        "four_decimal_presentation_half_unit": FOUR_DECIMAL_TOLERANCE,
        "two_decimal_percentage_point_half_unit": TWO_DECIMAL_PP_TOLERANCE,
        "formatted_table_comparison": "exact after declared rounding",
    },
    "lice_patterns": {
        "FN_P1": "GenHlth <= 2",
        "FN_P2": "HighBP <= 0",
        "FN_P3": "HighChol <= 0",
        "FN_P4": "BMI <= 25",
        "target_region": "y_true = 1 and at least two patterns match",
    },
    "artifacts": {
        "test_predictions_long": {
            "file": str(test_audit_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(test_audit_path),
            "rows": int(len(test_predictions_long)),
            "columns": test_predictions_long.columns.tolist(),
        },
        "training_lice_assignment_audit": {
            "file": str(training_audit_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(training_audit_path),
            "rows": int(len(training_audit)),
            "columns": training_audit.columns.tolist(),
        },
        "fn_transition_summary": {
            "file": str(fn_transition_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(fn_transition_path),
            "rows": int(len(fn_transition_summary)),
        },
        "reproduction_checks": {
            "file": str(check_report_path.relative_to(PROJECT_ROOT)),
            "sha256": file_sha256(check_report_path),
            "checks": int(len(check_report)),
            "passed": int(check_report["Status"].eq("PASS").sum()),
            "failed": int(check_report["Status"].eq("FAIL").sum()),
        },
    },
    "all_checks_passed": bool(all_checks_passed),
}
manifest_path = RESULTS_DIR / "prediction_audit_manifest.json"
with manifest_path.open("w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)

print(check_report.to_string(index=False))
print()
print(f"Test prediction rows: {len(wide_predictions):,}")
print(f"Long prediction-model rows: {len(test_predictions_long):,}")
print(f"Training audit rows: {len(training_audit):,}")
print(
    "Verification checks: "
    f"{int(check_report['Status'].eq('PASS').sum())} passed, "
    f"{int(check_report['Status'].eq('FAIL').sum())} failed"
)

if not all_checks_passed:
    failed_checks = check_report.loc[check_report["Status"].eq("FAIL")]
    raise AssertionError(
        "Released predictions did not reproduce every required artifact:\n"
        + failed_checks.to_string(index=False)
    )

print("Prediction-level audit and Tables 3-10 verification completed successfully.")
