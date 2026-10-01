"""Methods preprocessing: MICE, mode fill, Z-score Winsorization, standardize, one-hot, 7:3 split."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from iecq.config import SEED, TEST_RATIO, TRAIN_RATIO, WINSOR_Q, ZSCORE_OUTLIER
from iecq.data.schema import CATEGORICAL_PREDICTORS, CONTINUOUS_PREDICTORS, PREDICTOR_COLS


@dataclass
class Preprocessor:
    continuous: list[str] = field(default_factory=lambda: list(CONTINUOUS_PREDICTORS))
    categorical: list[str] = field(default_factory=lambda: list(CATEGORICAL_PREDICTORS))
    mice: Optional[IterativeImputer] = None
    cat_imputer: Optional[SimpleImputer] = None
    scaler: Optional[StandardScaler] = None
    encoder: Optional[OneHotEncoder] = None
    winsor_lo: dict[str, float] = field(default_factory=dict)
    winsor_hi: dict[str, float] = field(default_factory=dict)
    z_mean: dict[str, float] = field(default_factory=dict)
    z_std: dict[str, float] = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    fitted: bool = False

    def _continuous_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.continuous].apply(pd.to_numeric, errors="coerce")

    def _categorical_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.categorical].copy()
        for c in self.categorical:
            out[c] = out[c].astype("object")
            out[c] = out[c].where(pd.notna(out[c]), np.nan)
        return out

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        x_num = self._continuous_frame(df)
        x_cat = self._categorical_frame(df)

        self.mice = IterativeImputer(
            random_state=SEED,
            max_iter=20,
            sample_posterior=False,
            skip_complete=True,
        )
        x_num_imp = pd.DataFrame(self.mice.fit_transform(x_num), columns=self.continuous, index=df.index)

        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        x_cat_imp = pd.DataFrame(
            self.cat_imputer.fit_transform(x_cat),
            columns=self.categorical,
            index=df.index,
        )

        self.z_mean = {c: float(x_num_imp[c].mean()) for c in self.continuous}
        self.z_std = {c: float(x_num_imp[c].std(ddof=0) or 1.0) for c in self.continuous}
        self.winsor_lo = {c: float(x_num_imp[c].quantile(1.0 - WINSOR_Q)) for c in self.continuous}
        self.winsor_hi = {c: float(x_num_imp[c].quantile(WINSOR_Q)) for c in self.continuous}
        x_num_w = self._winsorize(x_num_imp)

        self.scaler = StandardScaler()
        x_num_s = self.scaler.fit_transform(x_num_w)

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
        x_cat_o = self.encoder.fit_transform(x_cat_imp)
        cat_names = list(self.encoder.get_feature_names_out(self.categorical))
        self.feature_names_ = list(self.continuous) + cat_names
        _ = np.hstack([x_num_s, x_cat_o])
        self.fitted = True
        return self

    def _winsorize(self, x_num: pd.DataFrame) -> pd.DataFrame:
        out = x_num.copy()
        for c in self.continuous:
            z = (out[c] - self.z_mean[c]) / (self.z_std[c] if self.z_std[c] else 1.0)
            hi_mask = z.abs() > ZSCORE_OUTLIER
            out.loc[hi_mask & (out[c] > self.winsor_hi[c]), c] = self.winsor_hi[c]
            out.loc[hi_mask & (out[c] < self.winsor_lo[c]), c] = self.winsor_lo[c]
        return out

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if not self.fitted:
            raise RuntimeError("Preprocessor is not fitted")
        x_num = self._continuous_frame(df)
        x_cat = self._categorical_frame(df)
        x_num_imp = pd.DataFrame(self.mice.transform(x_num), columns=self.continuous, index=df.index)
        x_cat_imp = pd.DataFrame(
            self.cat_imputer.transform(x_cat),
            columns=self.categorical,
            index=df.index,
        )
        x_num_w = self._winsorize(x_num_imp)
        x_num_s = self.scaler.transform(x_num_w)
        x_cat_o = self.encoder.transform(x_cat_imp)
        x = np.hstack([x_num_s, x_cat_o]).astype(np.float32)
        y = df["y"].to_numpy(dtype=np.int64) if "y" in df.columns else np.full(len(df), -1, dtype=np.int64)
        meta = df.copy()
        return x, y, meta

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        self.fit(df)
        return self.transform(df)

    def to_state(self) -> dict:
        from io import BytesIO
        import pickle

        buf = BytesIO()
        pickle.dump(
            {
                "continuous": self.continuous,
                "categorical": self.categorical,
                "mice": self.mice,
                "cat_imputer": self.cat_imputer,
                "scaler": self.scaler,
                "encoder": self.encoder,
                "winsor_lo": self.winsor_lo,
                "winsor_hi": self.winsor_hi,
                "z_mean": self.z_mean,
                "z_std": self.z_std,
                "feature_names_": self.feature_names_,
                "fitted": self.fitted,
            },
            buf,
        )
        return {"pickle_b64": __import__("base64").b64encode(buf.getvalue()).decode("ascii")}

    @classmethod
    def from_state(cls, state: dict) -> "Preprocessor":
        import pickle
        import base64

        obj = pickle.loads(base64.b64decode(state["pickle_b64"]))
        inst = cls()
        inst.__dict__.update(obj)
        return inst


def stratified_split(
    df: pd.DataFrame,
    seed: int = SEED,
    test_size: float = TEST_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 7:3 split keeping below/above-standard rates (Methods)."""
    assert abs(TRAIN_RATIO + TEST_RATIO - 1.0) < 1e-8
    strat = df["y"].astype(str) + "_" + df["university_type"].astype(str)
    counts = strat.value_counts()
    rare = counts[counts < 2].index
    if len(rare) > 0:
        strat = strat.where(~strat.isin(rare), df["y"].astype(str))
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def dump_feature_names(path, names: list[str]) -> None:
    path.write_text(json.dumps(names, indent=2), encoding="utf-8")
