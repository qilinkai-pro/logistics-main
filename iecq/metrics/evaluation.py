"""Results-section metrics: accuracy, precision, recall, F1, AUC, CI, McNemar, stratum accuracy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


POSITIVE = 1


def _wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def metric_bundle(y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    yhat = (p >= tau).astype(int)
    n = len(y)
    acc = float(accuracy_score(y, yhat))
    prec = float(precision_score(y, yhat, pos_label=POSITIVE, zero_division=0))
    rec = float(recall_score(y, yhat, pos_label=POSITIVE, zero_division=0))
    f1 = float(f1_score(y, yhat, pos_label=POSITIVE, zero_division=0))
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
    acc_lo, acc_hi = _wilson_ci(int((yhat == y).sum()), n)
    prec_lo, prec_hi = _wilson_ci(int(tp), int(tp + fp)) if (tp + fp) else (float("nan"), float("nan"))
    rec_lo, rec_hi = _wilson_ci(int(tp), int(tp + fn)) if (tp + fn) else (float("nan"), float("nan"))
    return {
        "accuracy": acc,
        "accuracy_ci": [acc_lo, acc_hi],
        "precision": prec,
        "precision_ci": [prec_lo, prec_hi],
        "recall": rec,
        "recall_ci": [rec_lo, rec_hi],
        "f1": f1,
        "auc": auc,
        "sensitivity": sens,
        "specificity": spec,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(n),
        "tau": float(tau),
        "yhat": yhat,
    }


def majority_baseline(y: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    n = len(y)
    yhat = np.zeros(n, dtype=int)
    p = np.full(n, 0.0)
    bundle = metric_bundle(y, p, tau=0.5)
    bundle["accuracy"] = float((y == 0).mean())
    bundle["recall"] = 0.0
    bundle["precision"] = float("nan")
    bundle["f1"] = float("nan")
    bundle["auc"] = 0.5
    bundle["name"] = "majority_class"
    return bundle


def mcnemar_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar χ² of model A vs B (Results Table 5). A is typically extended CART."""
    y = np.asarray(y).astype(int)
    a = np.asarray(pred_a).astype(int)
    b = np.asarray(pred_b).astype(int)
    a_ok = a == y
    b_ok = b == y
    b01 = int((~a_ok & b_ok).sum())
    b10 = int((a_ok & ~b_ok).sum())
    discordant = b01 + b10
    if discordant == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / discordant
        from scipy.stats import chi2 as chi2dist

        p = float(chi2dist.sf(chi2, 1))
    return {
        "discordant_pairs": int(discordant),
        "chi2": float(chi2),
        "p": p,
        "interpretation": "Significant" if p < 0.05 else "Not significant",
    }


def accuracy_by_university(y: np.ndarray, yhat: np.ndarray, univ: np.ndarray) -> dict[str, dict]:
    out = {}
    for u in ["double_first_class", "univ_211", "general_undergraduate", "vocational"]:
        m = univ == u
        if m.sum() == 0:
            continue
        acc = float((yhat[m] == y[m]).mean())
        out[u] = {"n": int(m.sum()), "accuracy": acc}
    out["overall"] = {"n": int(len(y)), "accuracy": float((yhat == y).mean())}
    return out


def roc_points(y: np.ndarray, p: np.ndarray) -> dict:
    fpr, tpr, thr = roc_curve(y, p)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "threshold": thr.tolist()}


def evaluate_predictions(pack: dict, tau: float) -> dict:
    y = pack["y"]
    names = [
        "logistic",
        "svm",
        "cart",
        "random_forest",
        "gbdt",
        "extended_cart",
    ]
    display = {
        "logistic": "Logistic regression",
        "svm": "Linear SVM",
        "cart": "CART",
        "random_forest": "Random forest",
        "gbdt": "GBDT",
        "extended_cart": "Extended CART (fused)",
    }
    models = {}
    models["majority_class"] = majority_baseline(y)
    yhats = {}
    for name in names:
        models[name] = metric_bundle(y, pack[name], tau)
        models[name]["name"] = display[name]
        yhats[name] = models[name]["yhat"]
        models[name] = {k: v for k, v in models[name].items() if k != "yhat"}

    mcnemar = {}
    ext = yhats["extended_cart"]
    for other in ["logistic", "svm", "cart", "random_forest", "gbdt"]:
        mcnemar[other] = mcnemar_test(y, ext, yhats[other])

    univ = {}
    for name in ["logistic", "cart", "random_forest", "extended_cart"]:
        univ[name] = accuracy_by_university(y, yhats[name], pack["university_type"])

    return {
        "models": models,
        "mcnemar_extended_vs": mcnemar,
        "accuracy_by_university": univ,
        "tau": tau,
        "n": int(len(y)),
        "below_rate": float(y.mean()),
    }
