"""End-to-end Methods architecture: preprocess → LR → CART → extended CART → baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch

from iecq.config import TrainConfig, YOUDEN_TAU_PAPER, resolve_device
from iecq.data.preprocess import Preprocessor, stratified_split
from iecq.models.baselines import BaselineFits, fit_baselines, predict_baseline_proba
from iecq.models.cart import CARTClassifier, CARTFit
from iecq.models.extended_cart import ExtendedCART, ExtendedCARTFit
from iecq.models.logistic import LogisticFit, StepwiseLogisticTrainer


@dataclass
class AssessmentFramework:
    """Fig. 2 architecture: logistic probability as CART meta-feature."""

    cfg: TrainConfig = field(default_factory=TrainConfig)
    preprocessor: Optional[Preprocessor] = None
    lr_fit: Optional[LogisticFit] = None
    cart_fit: Optional[CARTFit] = None
    ext_fit: Optional[ExtendedCARTFit] = None
    baselines: Optional[BaselineFits] = None
    feature_names: list[str] = field(default_factory=list)
    tau: float = YOUDEN_TAU_PAPER
    device: str = "cpu"

    def fit(self, train_df: pd.DataFrame) -> "AssessmentFramework":
        self.device = resolve_device(self.cfg.device)
        self.preprocessor = Preprocessor()
        x, y, _ = self.preprocessor.fit_transform(train_df)
        y = y.astype(int)
        self.feature_names = list(self.preprocessor.feature_names_)

        lr_trainer = StepwiseLogisticTrainer(self.cfg)
        self.lr_fit = lr_trainer.fit(x, y, self.feature_names)
        self.tau = float(self.lr_fit.tau) if self.cfg.tau is None else float(self.cfg.tau)

        cart = CARTClassifier(param_grid=self.cfg.cart_param_grid)
        self.cart_fit = cart.fit(x, y, self.feature_names)

        p_lr_train = self._p_lr(x)
        ext = ExtendedCART(param_grid=self.cfg.cart_param_grid)
        self.ext_fit = ext.fit(x, y, p_lr_train, self.feature_names)

        self.baselines = fit_baselines(x, y)
        return self

    def _p_lr(self, x: np.ndarray) -> np.ndarray:
        return StepwiseLogisticTrainer.predict_proba(self.lr_fit, x, self.device)

    def predict_all(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        x, y, meta = self.preprocessor.transform(df)
        p_lr = self._p_lr(x)
        p_cart = CARTClassifier.predict_proba(self.cart_fit, x)
        ext = ExtendedCART()
        p_ext = ext.predict_proba(self.ext_fit, x, p_lr)
        p_svm = predict_baseline_proba(self.baselines.svm, x)
        p_rf = predict_baseline_proba(self.baselines.rf, x)
        p_gbdt = predict_baseline_proba(self.baselines.gbdt, x)
        out = {
            "y": y,
            "logistic": p_lr,
            "cart": p_cart,
            "extended_cart": p_ext,
            "svm": p_svm,
            "random_forest": p_rf,
            "gbdt": p_gbdt,
            "university_type": meta["university_type"].to_numpy(),
        }
        return out

    def predict_labels(self, proba: np.ndarray) -> np.ndarray:
        return (proba >= self.tau).astype(int)

    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        pack = self.predict_all(df)
        res = df.copy()
        res["p_lr"] = pack["logistic"]
        res["p_cart"] = pack["cart"]
        res["p_extended_cart"] = pack["extended_cart"]
        res["yhat_extended_cart"] = self.predict_labels(pack["extended_cart"])
        res["yhat_logistic"] = self.predict_labels(pack["logistic"])
        res["tau"] = self.tau
        return res

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_model = self.lr_fit.model.cpu()
        payload = {
            "cfg": self.cfg,
            "preprocessor": self.preprocessor,
            "lr_fit": self.lr_fit,
            "cart_fit": self.cart_fit,
            "ext_fit": self.ext_fit,
            "baselines": self.baselines,
            "feature_names": self.feature_names,
            "tau": self.tau,
            "lr_state": cpu_model.state_dict(),
            "n_features_lr": int(cpu_model.linear.in_features),
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> "AssessmentFramework":
        payload = joblib.load(path)
        obj = cls(cfg=payload["cfg"])
        obj.preprocessor = payload["preprocessor"]
        obj.lr_fit = payload["lr_fit"]
        obj.cart_fit = payload["cart_fit"]
        obj.ext_fit = payload["ext_fit"]
        obj.baselines = payload["baselines"]
        obj.feature_names = payload["feature_names"]
        obj.tau = payload["tau"]
        obj.device = device or resolve_device(obj.cfg.device)
        from iecq.models.logistic import LogisticRegressionNet

        net = LogisticRegressionNet(payload["n_features_lr"])
        net.load_state_dict(payload["lr_state"])
        net.to(obj.device)
        net.eval()
        obj.lr_fit.model = net
        return obj


def prepare_splits(valid_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return stratified_split(valid_df, seed=seed)
