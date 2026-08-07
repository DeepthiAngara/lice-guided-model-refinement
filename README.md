# LICE-Guided Model Refinement and Ablation Evaluation

This repository contains the module-level implementation and full-precision
results for LICE-guided diabetes model refinement using the BRFSS 2015 health
indicator dataset. It is organized as a Google Colab workflow so that each
stage can be inspected and run from top to bottom.

This release does not contain the manuscript or unrelated parts of the broader
research project. It includes only the result-generation code and the single
published benchmark input needed to regenerate the experimental tables and
figures associated with this module.

## Google Drive location

Place the extracted repository folder at exactly:

```text
/content/drive/MyDrive/Research/LICE_Guided_Model_Refinement_Release_V3
```

All five notebooks already use this path.

### First Colab setup run

The notebooks install pinned dependency versions. When the installed binary
stack differs from the required one, the setup cell automatically restarts the
Colab runtime after installation. After Colab reconnects, run the same setup
cell once more and then continue to the next cell. This prevents a mixture of
old in-memory NumPy components and newly installed packages.

## Repository structure

```text
LICE_Guided_Model_Refinement_Release_V3/
├── README.md
├── requirements.txt
├── requirements_oof.txt
├── environment/
│   ├── requirements_final_lock.txt
│   └── requirements_oof_lock.txt
├── input_artifact_manifest.json
├── data/
│   ├── README.md
│   ├── LICENSE.md
│   └── diabetes_brfss2015_prepared.csv
├── input_artifacts/
│   └── lice_sample_weights.csv.gz
├── notebook/
│   ├── 01_Data_Preparation_Splits_and_Tuning.ipynb
│   ├── 02_OOF_LIME_DiCE_and_Weight_Identification.ipynb
│   ├── 03_Model_Refinement_and_Ablation.ipynb
│   ├── 04_Result_Tables_and_Figures.ipynb
│   └── 05_Prediction_Level_Audit_and_Table_Verification.ipynb
├── tests/
│   ├── README.md
│   └── verify_released_predictions.py
├── reference_inputs/
│   └── published_gbc_reference.csv
├── generated_outputs/
│   ├── tables/
│   └── figures/
├── splits/
│   └── split_assignments.csv.gz
├── models/
│   ├── M3_Balanced.joblib
│   └── M3_High.joblib
└── results/
    ├── README.md
    ├── metrics_full_precision.csv
    ├── predictions_all_models.csv.gz
    ├── test_predictions_long.csv.gz
    ├── training_lice_assignment_audit.csv.gz
    ├── fn_transition_summary_from_predictions.csv
    ├── prediction_reproduction_checks.csv
    ├── prediction_audit_manifest.json
    ├── mcnemar_results_full_precision.csv
    ├── OOF and tuning results
    └── LIME, DiCE, and sample-weight summaries
```

## Recommended execution

### Quick independent verification of the released predictions

Open the following notebook in Google Colab:

```text
notebook/05_Prediction_Level_Audit_and_Table_Verification.ipynb
```

This notebook does not retrain the models. It reloads the released predictions,
creates the consolidated prediction-level audit files, and independently
reconstructs confusion matrices, metrics, false-negative transitions, McNemar
results, and Tables 3-10.

### Re-execution of the reported ablation

Open the following notebook in Google Colab:

```text
notebook/03_Model_Refinement_and_Ablation.ipynb
```

Run every cell from top to bottom. The notebook mounts Google Drive, installs
`requirements.txt`, verifies the dataset and weight checksums, trains all eight
model configurations, recalculates the metrics and McNemar tests, and saves the
full-precision results.

### Complete module workflow

Run the notebooks in this order:

1. `01_Data_Preparation_Splits_and_Tuning.ipynb`
2. `02_OOF_LIME_DiCE_and_Weight_Identification.ipynb`
3. `03_Model_Refinement_and_Ablation.ipynb`
4. `04_Result_Tables_and_Figures.ipynb`
5. `05_Prediction_Level_Audit_and_Table_Verification.ipynb`

Notebook 1 contains the Kaggle acquisition command, deterministic preprocessing,
the 80:20 split, three-fold OOF assignments, and the disclosed GBDT tuning
space and selection rule.

Notebook 2 generates OOF predictions and contains the complete LIME, DiCE,
false-negative aggregation, pattern-matching, and sample-weighting implementation
with separate weights for exactly two matches and for three or four matches.
Full LIME and DiCE execution is disabled by default because it
is computationally expensive. Set `RUN_FULL_LIME = True` and
`RUN_FULL_DICE = True` in that notebook for the complete case-level run. The
released patterns, confirmation summary, weights, and final results remain
available for the quick verification path.

Notebook 3 performs the final eight-configuration ablation on the untouched
test set.

Notebook 4 reads the machine-readable outputs and regenerates Tables 3–10 and
Figures 2–5. Tables 3–9 are calculated only from this repository's experiment
outputs. Table 10 is a descriptive cross-study calculation and reads the
published GBC values recorded in `reference_inputs/published_gbc_reference.csv`.
Figure 1 is a conceptual framework diagram rather than a data-generated result;
its editable source is maintained separately from the executable workflow.

Notebook 5 reads the released case-level predictions from disk, creates the
long-format test prediction audit and consolidated training assignment audit,
and verifies Tables 3-10 independently of the in-memory training workflow.

## Prediction-level reproducibility artifacts

`results/test_predictions_long.csv.gz` contains one row per untouched test
instance and model variant: 13,812 test cases multiplied by eight variants,
giving 110,496 rows. It includes the true label, probability, predicted class,
decision threshold, split position, evaluation weight, training-weight scheme,
LICE-pattern indicators, error type, false-negative transition, and McNemar
discordance category.

The untouched test cases have no OOF validation-fold assignment because OOF
folds apply only to training rows. They are evaluated without weighting, so
their evaluation sample weight is 1.0. Model-specific sample weights must not
be assigned to test observations.

`results/training_lice_assignment_audit.csv.gz` provides the applicable
training-side linkage among source row, OOF validation fold, true label,
LICE-pattern matches, targeted-positive status, and Mild, Balanced, and
High-Sensitivity sample weights.

The automated command is:

```bash
python tests/verify_released_predictions.py
```

It exits with a non-zero status if a check fails and writes the complete
machine-readable result to `results/prediction_reproduction_checks.csv`.

## Dataset acquisition and preparation

The original input is
`diabetes_binary_5050split_health_indicators_BRFSS2015.csv` from the
[Diabetes Health Indicators Dataset on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
The source dataset is published under the **CC0: Public Domain** license. The
prepared derivative used for exact reproduction is therefore included in this
repository. See `data/LICENSE.md` for the source and redistribution statement.

The preparation procedure is implemented in Notebook 1:

1. rename `Diabetes_binary` to `Outcome`;
2. remove exact duplicate rows, retaining the first occurrence;
3. identify values outside `Q1 - 1.5×IQR` and `Q3 + 1.5×IQR` for `BMI`,
   `MentHlth`, and `PhysHlth`;
4. replace those values with the corresponding feature median; and
5. retain the original row order of the remaining 69,057 rows.

The prepared file is included so the exact released split can be checked even
when Kaggle credentials are not configured.

## Fixed experimental settings

| Setting | Value |
|---|---|
| Train/test split | 80:20, `random_state=42`, no stratification |
| Training/test rows | 55,245 / 13,812 |
| OOF validation | Three-fold `StratifiedKFold`, `shuffle=False` |
| Classifier | `GradientBoostingClassifier` |
| Selected parameters | 100 estimators, learning rate 0.1, maximum depth 5 |
| Model and tuning seed | 42 |
| Decision threshold | 0.50 |
| LIME deterministic release seed | `42 + Train_Position` |
| DiCE seed | `42 + Train_Position` |

The original LIME notebook did not set an explicit `random_state`. This is
recorded as a historical limitation. The clean Notebook 2 uses a per-case seed
so checkpointed and resumed runs are deterministic.

## Environment records

The short requirements files identify the intended direct dependencies for the
two execution stages. The files under `environment/` record complete installed
package sets, including transitive dependencies, for the verified executions:

- `environment/requirements_oof_lock.txt` for Notebooks 1 and 2; and
- `environment/requirements_final_lock.txt` for Notebooks 3, 4, and 5.

The notebooks install the short requirements files by default for readability.
For an exact environment reconstruction, install the corresponding lock file.

## Exact LICE-guided weighting rule

The selected false-negative conditions are:

- `GenHlth <= 2`;
- `HighBP <= 0`;
- `HighChol <= 0`; and
- `BMI <= 25`.

Only positive-class training cases matching at least two conditions receive a
weight above 1.0.

| Matching conditions | Mild | Balanced | High |
|---:|---:|---:|---:|
| Exactly 2 | 1.10 | 1.15 | 1.20 |
| 3 or 4 | 1.15 | 1.25 | 1.35 |

All other cases receive a weight of 1.0. Notebook 2 recalculates every weight
and compares all regenerated rows and values exactly with the released CSV
artifact. It separately verifies the SHA-256 checksum of the released compressed
file. (Valid gzip byte streams can differ across compression-library builds even
when their decompressed CSV contents are identical.)

## Full precision and rounding

No intermediate rounding is applied. Machine-readable floating-point CSV files
use 17 significant digits. Values may be rounded to four decimal places only
when presented in the manuscript. Small AUC or MCC changes should be interpreted
from the full-precision files rather than four-decimal summaries alone.

Automated prediction checks require exact integer counts and use an absolute
tolerance of `1e-12` for full-precision metrics. Four-decimal manuscript values
use a half-unit tolerance of `0.00005`; two-decimal percentage-point values use
a half-unit tolerance of `0.005`. Generated table strings must match exactly
after applying these declared presentation rules.

The released files represent the recorded reference execution. The final few
digits of probability-based metrics can vary across operating systems or
compiled numerical-library builds without changing predicted classes or the
confusion matrices.

## Primary result checks

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Baseline | 4,767 | 1,956 | 1,417 | 5,672 |
| M3-Balanced | 4,613 | 2,110 | 1,245 | 5,844 |
| M3-High | 4,561 | 2,162 | 1,221 | 5,868 |

Notebook 3 recalculates these counts and metrics from predictions produced in
the ablation run. Notebook 5 separately reloads the released prediction CSV,
reconstructs the results, and verifies all generated Tables 3-10.
