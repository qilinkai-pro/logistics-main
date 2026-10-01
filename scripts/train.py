"""Train logistic regression, CART, extended CART and baselines (Methods §2.4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iecq.analysis.factors import grouping_analysis, logistic_or_table, tree_importance_table
from iecq.config import CHECKPOINT_DIR, DATA_PROCESSED, OUTPUT_DIR, SEED, TRAIN_RATIO, TrainConfig
from iecq.models.framework import AssessmentFramework, prepare_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PROCESSED / "ie_cqes21_valid_1500.csv")
    parser.add_argument("--out", type=Path, default=CHECKPOINT_DIR / "framework.joblib")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.data)
    train_df, test_df = prepare_splits(df, seed=args.seed)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_PROCESSED / "train.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(DATA_PROCESSED / "test.csv", index=False, encoding="utf-8-sig")

    cfg = TrainConfig(seed=args.seed, device=args.device, lr_epochs=args.epochs)
    fw = AssessmentFramework(cfg=cfg)
    fw.fit(train_df)
    fw.save(args.out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    or_tbl = logistic_or_table(fw.lr_fit)
    or_tbl.to_csv(OUTPUT_DIR / "logistic_odds_ratios.csv", index=False, encoding="utf-8-sig")
    tree_importance_table(fw.cart_fit).to_csv(
        OUTPUT_DIR / "cart_gini_importance.csv", index=False, encoding="utf-8-sig"
    )
    tree_importance_table(fw.ext_fit).to_csv(
        OUTPUT_DIR / "extended_cart_gini_importance.csv", index=False, encoding="utf-8-sig"
    )
    groups, tests = grouping_analysis(train_df)
    groups.to_csv(OUTPUT_DIR / "factor_groups_train.csv", index=False, encoding="utf-8-sig")

    meta = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_ratio": TRAIN_RATIO,
        "tau": fw.tau,
        "lambda": fw.lr_fit.lam,
        "lr_selected": fw.lr_fit.selected_names,
        "lr_aic": fw.lr_fit.aic,
        "hosmer_lemeshow_chi2": fw.lr_fit.hl_stat,
        "hosmer_lemeshow_p": fw.lr_fit.hl_p,
        "cart_best_params": fw.cart_fit.best_params,
        "cart_cv_accuracy": fw.cart_fit.cv_accuracy,
        "extended_cart_best_params": fw.ext_fit.best_params,
        "extended_cart_cv_accuracy": fw.ext_fit.cv_accuracy,
        "factor_chi2": tests,
        "checkpoint": str(args.out),
    }
    (OUTPUT_DIR / "train_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(json.dumps({k: meta[k] for k in ("n_train", "n_test", "tau", "lambda", "lr_aic")}, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
