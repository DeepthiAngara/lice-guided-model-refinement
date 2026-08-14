# Analysis scope and interpretation boundaries

## Evaluation design

- The prepared BRFSS 2015 dataset contains duplicate-removed records.
- Fixed training and untouched-test assignments are supplied with the release.
- Fixed out-of-fold assignments are supplied for the training partition.
- OOF results provide training-side consistency evidence. They do not represent
  a fully nested model-selection analysis.
- The untouched test set is evaluated without sample weighting and is not used
  to select model parameters or decision thresholds.

## LICE-guided refinement

The release includes the explanation-guided feature patterns, targeted-positive
trace, configuration-specific training weights, interaction definitions, and
prediction-level audits required to reproduce the predefined model-refinement
assessment. The mild, balanced, and high-sensitivity weighting strengths are
controlled ablation levels applied to the LICE-targeted training region.

## Comparator analyses

- Conventional controls include global weighting, deterministic and random
  targeting controls, and OOF-selected threshold controls.
- The Pang MARS and Jose LightGBM approaches are implemented on the same
  prepared dataset, fixed split, preprocessing boundary, search protocol, and
  evaluation code used for the primary workflow.
- These implementations support a same-split comparison. They are not claimed
  to be exact replications of experiments reported on different datasets or
  under incompletely specified tuning procedures.

## Statistical and calibration analyses

- Paired bootstrap intervals use the prespecified resampling configuration
  recorded in the statistical-analysis manifest.
- McNemar analyses include the full test set and a restricted positive-case
  scope, with Holm adjustment within the declared comparison families.
- Positive-case-only inference is not interpreted as evidence of overall
  superiority.
- Calibration results apply to the prepared evaluation cohort.
- Prevalence scenarios are hypothetical mathematical transports rather than
  external-cohort validation.
- Exploratory utility calculations do not establish clinical decision
  thresholds.

## Reproducibility boundary

The repository provides deterministic source-row identifiers, fixed split and
OOF assignments, full-precision predictions, configuration definitions,
training-weight audits, model artifacts, statistical settings, environment
requirements, checksums, and automated verification programs. Historical
stochastic explanation-generation limitations are retained where applicable
and should not be interpreted as prospective clinical validation.
