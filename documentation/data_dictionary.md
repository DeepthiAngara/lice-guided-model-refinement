# Data and prediction dictionary

## Shared identifiers

| Field | Meaning |
|---|---|
| `Source_Row_Index` | Deterministic row identifier in the prepared dataset |
| `Train_Position` | Position within the fixed training partition |
| `Test_Position` | Position within the fixed test partition |
| `OOF_Validation_Fold` | Training-only validation fold; not applicable to test rows |
| `Outcome` / `y_true` | Binary observed outcome: 0 non-diabetes, 1 diabetes |

## Prediction fields

| Field | Meaning |
|---|---|
| `Model` | Stable model/configuration name |
| `Predicted_Probability` | Full-precision positive-class probability |
| `Predicted_Class` | Class obtained at the recorded threshold |
| `Decision_Threshold` | Classification threshold; 0.50 for original and recent comparators |
| `Evaluation_Sample_Weight` | Always 1.0 on the untouched test set |
| `Training_Weight_Scheme` | Weighting applied during model training; not a test weight |
| `Error_Type` | TP, TN, FP or FN for the relevant model prediction |

## LICE audit fields

Pattern indicators identify whether a row meets each selected false-negative
condition. `LICE_Pattern_Match_Count` counts satisfied conditions, and
`LICE_Targeted_Positive_Region` identifies positive training rows receiving
LICE-guided sample weighting. These fields describe training-side construction;
they do not assign evaluation weights to test rows.
