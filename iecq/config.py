"""Hyperparameters and protocol constants from Methods / Results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

SEED = 20250518
N_COLLECTED = 1620
N_VALID = 1500
N_BELOW = 486
N_ABOVE = 1014
BELOW_RATE = 0.324
MAJORITY_BASELINE = 0.676
Y_THRESHOLD = 29
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3
MISSING_DROP_FRAC = 0.20
ZSCORE_OUTLIER = 3.0
WINSOR_Q = 0.95
TOP_K_GINI = 20
LAMBDA_MIN = 0.01
LAMBDA_MAX = 10.0
N_LAMBDA = 12
N_FOLDS = 5
YOUDEN_TAU_PAPER = 0.48
MAX_MISSING_IMPLIED = 0.20

RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 12
GBDT_LR = 0.05
GBDT_N_ESTIMATORS = 300

CART_PARAM_GRID = {
    "max_depth": [6, 8, 10, 12],
    "min_samples_leaf": [3, 5, 8],
    "min_samples_split": [8, 10, 16],
    "ccp_alpha": [0.0, 0.001, 0.002, 0.005],
}

UNIV_COUNTS = {
    "double_first_class": 420,
    "univ_211": 570,
    "general_undergraduate": 315,
    "vocational": 195,
}

LEVEL_COUNTS = {
    "undergraduate_core": 690,
    "undergraduate_elective": 810,
}

DEPT_COUNTS = {
    "business_economics": 225,
    "engineering_technology": 675,
    "education_humanities": 210,
    "interdisciplinary": 390,
}

TITLE_COUNTS = {
    "lecturer": 330,
    "associate_professor": 660,
    "professor": 225,
    "industry_mentor": 285,
}

BELOW_TOTAL_MEAN = 41.27
BELOW_TOTAL_SD = 9.35
ABOVE_TOTAL_MEAN = 15.62
ABOVE_TOTAL_SD = 6.81
BELOW_TC_MEAN = 14.32
BELOW_TC_SD = 4.21
BELOW_TM_MEAN = 12.56
BELOW_TM_SD = 3.89
BELOW_PS_MEAN = 14.39
BELOW_PS_SD = 4.02

BELOW_BY_UNIV = {
    "double_first_class": 103,
    "univ_211": 172,
    "general_undergraduate": 122,
    "vocational": 89,
}


@dataclass
class TrainConfig:
    seed: int = SEED
    device: str = "auto"
    lr_epochs: int = 400
    lr_lr: float = 0.08
    batch_size: int = 256
    n_folds: int = N_FOLDS
    lambda_grid: list[float] = field(
        default_factory=lambda: [
            round(x, 6)
            for x in __import__("numpy").geomspace(LAMBDA_MIN, LAMBDA_MAX, N_LAMBDA)
        ]
    )
    cart_param_grid: dict = field(default_factory=lambda: dict(CART_PARAM_GRID))
    tau: float | None = None
    n_jobs: int = -1


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
