# IE-CQES-21 Course Quality Assessment


## Architecture

```
raw questionnaires (n = 1,620)
        │  inclusion / exclusion, logic checks, missing > 20% dropped
        ▼
valid sample (n = 1,500)     Y = 1{IE-CQES-21 total ≥ 29}
        │  stratified 7:3 split (Y rates preserved)
        ▼
train-only preprocess
  MICE (continuous) · mode (categorical)
  |Z| > 3 → Winsorize at the 95th percentile
  standardize continuous (mean 0, sd 1) · one-hot categorical
        │
        ├─► stepwise AIC + L2 logistic  →  p_LR(x)
        ├─► Gini top-20 + CART + CCP pruning
        └─► extended CART on [x, p_LR(x)]   Ŷ = 1{p_ext ≥ τ}
                 τ from Youden on training folds
```

Baselines on the same training fold: linear SVM (squared-hinge, class-balanced, Platt scaling), random forest (500 trees, max depth 12), GBDT (learning rate 0.05, 300 trees).

## Data

`scripts/generate_data.py` writes:

| file | content |
| --- | --- |
| `data/raw/ie_cqes21_survey_1620.csv` | 1,620 collected questionnaires |
| `data/processed/ie_cqes21_valid_1500.csv` | 1,500 valid undergraduate records after screening |


## Metrics

Reported on the test fold for the below-standard class (`Y = 1`):

- accuracy, precision, recall, F1, 95% Wilson CI
- AUC-ROC
- sensitivity / specificity at τ
- majority-class baseline 67.60%
- accuracy by university type
- McNemar tests of extended CART vs the other learners
- logistic AIC, Hosmer–Lemeshow, OR = exp(β)
- CART / extended-CART Gini importance


## Setup

```bash
cd logistics-main
pip install -r requirements.txt
```

## Commands

```bash


# 1) stratified split, preprocess, train all models
python scripts/train.py

# 2) test-set tables, ROC, McNemar, university strata
python scripts/test.py

# 3) inference on CSV or JSON
python scripts/infer.py --json examples/sample.json
python scripts/infer.py --input data/processed/test.csv --out outputs/pred_test.csv

# full pipeline
python scripts/run_all.py
```

Optional flags:

```bash
python scripts/train.py --device cuda --epochs 400
python scripts/test.py --checkpoint checkpoints/framework.joblib
```

Outputs: `checkpoints/framework.joblib`, `outputs/test_metrics.json`, `outputs/table_model_comparison.csv`, `outputs/logistic_odds_ratios.csv`, `outputs/figures/roc_test.png`.

## Package layout

```
logistics-main/
  iecq/data/       schema, simulation, MICE / Winsor / split, Dataset
  iecq/models/     LogisticRegressionNet, CART, ExtendedCART, baselines, framework
  iecq/metrics/    Results metrics
  iecq/analysis/   OR, Gini, grouping rates
  scripts/         train, test, infer, run_all
  examples/        sample.json
```

## Label rule

IE-CQES-21 total = TC + TM + PS (21 items × 0–3). `Y = 1` (below-standard) if total ≥ 29, else `Y = 0`.
