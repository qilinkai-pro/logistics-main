"""Comparison learners from Methods: linear SVM, random forest, GBDT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC

from iecq.config import GBDT_LR, GBDT_N_ESTIMATORS, RF_MAX_DEPTH, RF_N_ESTIMATORS, SEED


@dataclass
class BaselineFits:
    svm: CalibratedClassifierCV
    rf: RandomForestClassifier
    gbdt: GradientBoostingClassifier


def fit_baselines(x: np.ndarray, y: np.ndarray) -> BaselineFits:
    svm_base = LinearSVC(
        loss="squared_hinge",
        class_weight="balanced",
        dual="auto",
        max_iter=4000,
        random_state=SEED,
    )
    svm = CalibratedClassifierCV(svm_base, method="sigmoid", cv=5)
    svm.fit(x, y)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=SEED,
        n_jobs=-1,
    )
    rf.fit(x, y)

    gbdt = GradientBoostingClassifier(
        learning_rate=GBDT_LR,
        n_estimators=GBDT_N_ESTIMATORS,
        random_state=SEED,
    )
    gbdt.fit(x, y)
    return BaselineFits(svm=svm, rf=rf, gbdt=gbdt)


def predict_baseline_proba(model, x: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.ndim == 1:
        return proba
    return proba[:, 1]
