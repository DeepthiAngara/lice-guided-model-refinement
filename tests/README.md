# Prediction-level reproducibility verification

Run the automated verification from the repository root:

```bash
python tests/verify_released_predictions.py
```

The program reads the released case-level predictions rather than using
in-memory model outputs. It reconstructs the confusion matrices, all reported
metrics, false-negative transitions, McNemar discordant counts and statistics,
and Tables 3-10.

The verification also creates or refreshes:

- `results/test_predictions_long.csv.gz`;
- `results/training_lice_assignment_audit.csv.gz`;
- `results/fn_transition_summary_from_predictions.csv`;
- `results/prediction_reproduction_checks.csv`; and
- `results/prediction_audit_manifest.json`.

The process exits with a non-zero status if any check fails. Integer counts
must match exactly. Full-precision metrics use an absolute tolerance of
`1e-12`. Four-decimal values use a half-unit tolerance of `0.00005`, and
two-decimal percentage-point values use a half-unit tolerance of `0.005`.

Table 10 combines prediction-recalculated metrics with the explicitly declared
external reference row in `reference_inputs/published_gbc_reference.csv`.
