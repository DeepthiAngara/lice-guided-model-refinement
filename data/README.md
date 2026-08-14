# Prepared BRFSS 2015 dataset

`diabetes_brfss2015_prepared.csv` is the de-identified prepared dataset used by
this module. It contains the prepared predictors and binary target
`Outcome`, no missing values, and the row order used by the released split.

The original Kaggle file and complete deterministic preparation code are
specified in `notebooks/01_Data_Preparation_Splits_and_Tuning.ipynb`.

Expected prepared-file SHA-256:

```text
cea4e25cd6304a5f35f5f71cd1b374bef9613ecb9e1ff2a9b2056c7a3d8b7cc8
```

The original source CSV is not duplicated in this folder. Users can download it
from the Kaggle source or place an existing copy under `data/raw/` before
running the preparation stage. The Kaggle source dataset is released under
CC0: Public Domain, which permits redistribution and derivative preparation.
The included prepared CSV is the deterministic derivative required to reproduce
the published split and evaluation. See `LICENSE.md` in this directory.
