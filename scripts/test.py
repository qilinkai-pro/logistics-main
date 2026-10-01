"""Evaluate trained models on the held-out test set (Results §3.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iecq.analysis.factors import grouping_analysis, sample_summary
from iecq.config import CHECKPOINT_DIR, DATA_PROCESSED, FIGURE_DIR, OUTPUT_DIR
from iecq.metrics.evaluation import evaluate_predictions, roc_points
from iecq.models.framework import AssessmentFramework


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _write_tables(report: dict, path: Path) -> None:
    models = report["models"]
    lines = [
        "model,accuracy,precision,recall,f1,auc",
        "Majority class,{:.3f},---,{:.3f},---,{:.3f}".format(
            models["majority_class"]["accuracy"],
            models["majority_class"]["recall"],
            models["majority_class"]["auc"],
        ),
    ]
    order = ["logistic", "svm", "cart", "random_forest", "gbdt", "extended_cart"]
    names = {
        "logistic": "Logistic regression",
        "svm": "Linear SVM",
        "cart": "CART",
        "random_forest": "Random forest",
        "gbdt": "GBDT",
        "extended_cart": "Extended CART (fused)",
    }
    for k in order:
        m = models[k]
        lines.append(
            f"{names[k]},{m['accuracy']:.3f},{m['precision']:.3f},{m['recall']:.3f},{m['f1']:.3f},{m['auc']:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_roc(pack: dict, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skip ROC figure")
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = {
        "logistic": "Logistic regression",
        "svm": "Linear SVM",
        "cart": "CART",
        "random_forest": "Random forest",
        "gbdt": "GBDT",
        "extended_cart": "Extended CART",
    }
    plt.figure(figsize=(6.4, 5.2))
    for key, lab in labels.items():
        pts = roc_points(pack["y"], pack[key])
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(pack["y"], pack[key])
        plt.plot(pts["fpr"], pts["tpr"], label=f"{lab} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def _plot_bars(report: dict, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skip bar figure")
        return
    keys = ["logistic", "svm", "cart", "random_forest", "gbdt", "extended_cart"]
    labels = ["LR", "SVM", "CART", "RF", "GBDT", "Ext.CART"]
    acc = [report["models"][k]["accuracy"] for k in keys]
    auc = [report["models"][k]["auc"] for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w / 2, acc, w, label="Accuracy")
    ax.bar(x + w / 2, auc, w, label="AUC")
    ax.axhline(0.676, color="gray", linestyle="--", linewidth=1, label="Majority 0.676")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "framework.joblib")
    parser.add_argument("--test", type=Path, default=DATA_PROCESSED / "test.csv")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import pandas as pd

    fw = AssessmentFramework.load(args.checkpoint, device=args.device)
    test_df = pd.read_csv(args.test)
    pack = fw.predict_all(test_df)
    report = evaluate_predictions(pack, tau=fw.tau)
    report["sample_summary_test"] = sample_summary(test_df)
    groups, tests = grouping_analysis(test_df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    groups.to_csv(OUTPUT_DIR / "factor_groups_test.csv", index=False, encoding="utf-8-sig")
    report["factor_chi2_test"] = tests

    serial = _jsonify(report)
    (OUTPUT_DIR / "test_metrics.json").write_text(
        json.dumps(serial, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_tables(report, OUTPUT_DIR / "table_model_comparison.csv")
    _plot_roc(pack, FIGURE_DIR / "roc_test.png")
    _plot_bars(report, FIGURE_DIR / "bars_test.png")

    print("tau=", fw.tau, "n=", report["n"])
    for k in ["logistic", "cart", "extended_cart", "svm", "random_forest", "gbdt"]:
        m = report["models"][k]
        print(
            f"{k:16s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
            f"rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}"
        )


if __name__ == "__main__":
    main()
