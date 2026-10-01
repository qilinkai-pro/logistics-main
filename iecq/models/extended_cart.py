"""Two-stage complementary scheme: extended CART on [x, p_LR(x)] (Methods)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from iecq.config import CART_PARAM_GRID, N_FOLDS, SEED, TOP_K_GINI


@dataclass
class ExtendedCARTFit:
    model: DecisionTreeClassifier
    selected_idx: list[int]
    selected_names: list[str]
    importances: np.ndarray
    best_params: dict
    cv_accuracy: float
    meta_name: str = "p_lr"


class ExtendedCART:
    """Ŷ = 1{p_ext(x, p_LR(x)) ≥ τ}, p_ext = leaf frequency."""

    def __init__(self, param_grid: dict | None = None, top_k: int = TOP_K_GINI):
        self.param_grid = param_grid or CART_PARAM_GRID
        self.top_k = top_k

    def augment(self, x: np.ndarray, p_lr: np.ndarray) -> np.ndarray:
        return np.column_stack([x, np.asarray(p_lr).reshape(-1, 1)]).astype(np.float32)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        p_lr: np.ndarray,
        names: list[str],
    ) -> ExtendedCARTFit:
        xa = self.augment(x, p_lr)
        names_a = list(names) + ["p_lr"]
        probe = DecisionTreeClassifier(criterion="gini", random_state=SEED)
        probe.fit(xa, y)
        imp = probe.feature_importances_
        order = np.argsort(-imp)
        k_use = min(self.top_k, xa.shape[1])
        idx = order[:k_use].tolist()
        xs = xa[:, idx]
        sel_names = [names_a[i] for i in idx]
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        grid = GridSearchCV(
            DecisionTreeClassifier(criterion="gini", random_state=SEED),
            param_grid=self.param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(xs, y)
        model: DecisionTreeClassifier = grid.best_estimator_
        return ExtendedCARTFit(
            model=model,
            selected_idx=idx,
            selected_names=sel_names,
            importances=model.feature_importances_,
            best_params=dict(grid.best_params_),
            cv_accuracy=float(grid.best_score_),
        )

    def predict_proba(self, fit: ExtendedCARTFit, x: np.ndarray, p_lr: np.ndarray) -> np.ndarray:
        xa = self.augment(x, p_lr)
        xs = xa[:, fit.selected_idx]
        proba = fit.model.predict_proba(xs)
        if proba.shape[1] == 1:
            return np.zeros(len(xa))
        return proba[:, 1]
