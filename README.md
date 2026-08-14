# LICE-Guided Model Refinement — v1.3.0

This repository provides the data, code, prediction-level outputs, and
verification resources for the LICE-guided diabetes-model refinement workflow.
It includes the fixed train–test split, out-of-fold analyses, untouched-test
evaluation, conventional controls, and same-split implementations of the Pang
MARS and Jose LightGBM approaches.

The repository is frozen for the public `v1.3.0` release after all numerical
and inventory checks pass.

## Study boundary

- Prepared dataset: duplicate-removed BRFSS 2015 balanced-source records.
- Fixed training and untouched-test assignments are provided in `splits/`.
- Model-refinement assessment: predefined GBDT configurations evaluated at the
  prespecified decision threshold.
- OOF assessment: fixed training folds used as training-side consistency
  evidence, not as a nested model-selection analysis.
- The untouched test set is used only for final evaluation; it is not used to
  rank configurations or select a final model.
- Recent same-split comparators: Pang MARS and Jose LightGBM.
- No new dataset is introduced.
- Prevalence outputs are hypothetical scenarios, not population-cohort
  validation.
- Utility outputs are exploratory and do not define clinical thresholds.

## Repository structure

| Directory | Contents |
|---|---|
| `data/` | Prepared modelling dataset and licensing information |
| `splits/` | Fixed train/test and OOF-fold assignments |
| `notebooks/` | Ordered modelling, explanation, evaluation, and comparator workflows |
| `scripts/` | OOF, comparator, statistical, calibration, and prevalence-analysis programs |
| `predictions/oof/` | Baseline and eight-configuration OOF predictions and audits |
| `predictions/test/` | Eight-configuration untouched-test predictions |
| `predictions/conventional_comparators/` | Conventional-control predictions |
| `predictions/recent_comparators/` | Pang MARS and Jose LightGBM OOF/test predictions |
| `predictions/method_audit/` | Case-level LIME, explanation-quality, and counterfactual audit records |
| `results/oof/` | Pooled and fold-wise OOF performance |
| `results/test_ablation/` | Eight-configuration untouched-test results |
| `results/conventional_comparators/` | Global weighting, random targeting, and threshold controls |
| `results/recent_comparators/` | Same-split Pang/Jose search, performance, and paired inference |
| `results/method_audit/` | Fold-level quality, feature-constraint, feature-confirmation, and weight-identity audits |
| `results/statistical_analysis/` | Bootstrap, McNemar, and multiplicity-adjusted outputs |
| `results/calibration_prevalence/` | Calibration, scenario, and exploratory-utility outputs |
| `results/verification/` | Master and component verification results |
| `generated_outputs/` | Workflow-named result summaries and visualisations |
| `documentation/` | Analysis boundaries, release verification, and data dictionary |

## Verification

The `pang_mars_earth.rds` file is tracked with Git LFS. Install and
initialize Git LFS before adding the extracted release files to a Git working
copy. Do not upload the release ZIP as repository content.

For an isolated verification environment, run from the repository root:

```bash
python -m pip install virtualenv
python -m virtualenv .venv
.venv/bin/python -m pip install -r environment/requirements_verification.txt
.venv/bin/python tests/verify_all_results.py
.venv/bin/python tests/verify_release_inventory.py
```

On Windows, use `.venv\Scripts\python` in place of `.venv/bin/python`.

The master verifier runs independent components for core prediction results,
extended OOF/comparator/statistical/calibration analyses, and recent same-split
comparators. The program derives all check totals from the component outputs at
runtime. Release requires every reported check to pass and zero failed
component programs.

## Recommended execution order

- Run notebooks `01` through `05` in filename order for data preparation,
  explanation-guided refinement, evaluation, result generation, and the
  prediction-level audit.
- Run `06_Recent_Same_Split_Comparators_Pang_MARS_Jose_LightGBM.ipynb` for the
  recent same-split comparators.
- Run the applicable OOF, comparator, statistical, calibration, and
  prevalence-analysis programs under `scripts/` when regenerating those
  analyses.
- Run `tests/verify_all_results.py`, followed by
  `tests/verify_release_inventory.py`, for final verification.
