# Machine-readable result artifacts

The files in this directory contain full-precision aggregate results,
case-level predictions, explanation-guided assignment records, and automated
reproduction checks.

## Prediction-level audit files

### `predictions_all_models.csv.gz`

Wide-format reference predictions for all 13,812 untouched test instances.
Each row contains the true label and predicted class and probability for all
eight model variants.

### `test_predictions_long.csv.gz`

Long-format audit output with one row per test instance and model variant
(13,812 x 8 = 110,496 rows). Its composite key is
`Source_Row_Index` plus `Model`.

Important fields include:

- `Outer_Split` and `Test_Position`: the released test assignment;
- `OOF_Validation_Fold`: blank because OOF folds apply only to training rows;
- `Fold_Assignment_Status`: explicit explanation of the blank fold field;
- `Predicted_Probability`, `Predicted_Class`, and `Decision_Threshold`;
- `Evaluation_Sample_Weight`: always 1.0 because test evaluation is unweighted;
- `Training_Weight_Scheme`: the training emphasis used by the model variant;
- `Uses_LICE_Interactions` and `Uses_LICE_Sample_Weighting`;
- four case-level LICE-pattern match indicators and their total;
- `Error_Type`: TN, FP, FN, or TP for that model;
- `FN_Transition_vs_Baseline`; and
- `McNemar_Category_vs_Baseline`.

The test-set pattern fields describe whether the observed feature values fall
within the LICE-identified region. They do not imply that test observations
were given training sample weights.

### `training_lice_assignment_audit.csv.gz`

One row for each of the 55,245 training observations. It connects the source
row, training position, OOF validation fold, true label, LICE-pattern matches,
targeted-positive status, and Mild, Balanced, and High-Sensitivity sample
weights. This is the applicable artifact for auditing training weights and OOF
fold assignments.

### `fn_transition_summary_from_predictions.csv`

Counts of baseline false negatives corrected to true positives, baseline true
positives changed to false negatives, unchanged false negatives, unchanged
true positives, and the resulting net false-negative reduction for each
non-baseline model.

### `prediction_reproduction_checks.csv`

Machine-readable PASS/FAIL output from the independent verification program.
It records schema checks, split and label linkage, probability thresholding,
confusion matrices, full-precision metrics, McNemar results, and exact
post-rounding reproduction of Tables 3-10.

### `prediction_audit_manifest.json`

Declares row counts, test-set semantics, LICE patterns, rounding tolerances,
artifact checksums, and the overall verification status.

## Reproduction command

From the repository root, run:

```bash
python tests/verify_released_predictions.py
```
