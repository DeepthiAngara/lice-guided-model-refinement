# Release verification

The release is accepted only when all automated numerical and inventory checks
complete without failure.

| Verification component | Scope | Required result |
|---|---|---|
| Core results | Prediction schema, confusion matrices, full-precision metrics, false-negative transitions, McNemar results, and workflow-named result summaries | Every executed check passes |
| Extended analyses | OOF consistency, conventional comparators, paired inference, calibration, prevalence scenarios, and exploratory utility | Every executed check passes |
| Method audit | Case coverage, explanation quality, fold summaries, feature constraints, and weight identity | Every executed check passes |
| Recent comparators | Pang MARS and Jose LightGBM protocol, predictions, metrics, and paired comparisons | Every executed check passes |
| Release inventory | Presence, byte size, and SHA-256 checksum of every frozen release file | Zero failures |

The verification environment and commands are specified in the repository
README. The public `v1.3.0` tag should be created only from a repository state
for which every dynamically reported numerical check and the complete
file-inventory check pass.
