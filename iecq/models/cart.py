"""CART with Gini ranking, grid-search 5-fold CV, and cost-complexity pruning (Methods)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from iecq.config import CART_PARAM_GRID, N_FOLDS, SEED, TOP_K_GINI


@dataclass
class CARTFit:
    model: DecisionTreeClassifier
    selected_idx: list[int]
    selected_names: list[str]
    importances: np.ndarray
    best_params: dict
    cv_accuracy: float


def gini_rank(x: np.ndarray, y: np.ndarray, names: list[str], k: int = TOP_K_GINI) -> tuple[list[int], np.ndarray]:
    probe = DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=SEED,
    )
    probe.fit(x, y)
    imp = probe.feature_importances_
    order = np.argsort(-imp)
    k_use = min(k, x.shape[1])
    idx = order[:k_use].tolist()
    return idx, imp


class CARTClassifier:
    def __init__(self, param_grid: dict | None = None, top_k: int = TOP_K_GINI):
        self.param_grid = param_grid or CART_PARAM_GRID
        self.top_k = top_k

    def fit(self, x: np.ndarray, y: np.ndarray, names: list[str]) -> CARTFit:
        idx, imp = gini_rank(x, y, names, self.top_k)
        xs = x[:, idx]
        sel_names = [names[i] for i in idx]
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
        return CARTFit(
            model=model,
            selected_idx=idx,
            selected_names=sel_names,
            importances=model.feature_importances_,
            best_params=dict(grid.best_params_),
            cv_accuracy=float(grid.best_score_),
        )

    @staticmethod
    def predict_proba(fit: CARTFit, x: np.ndarray) -> np.ndarray:
        xs = x[:, fit.selected_idx]
        proba = fit.model.predict_proba(xs)
        if proba.shape[1] == 1:
            return proba[:, 0] * 0.0
        return proba[:, 1]
