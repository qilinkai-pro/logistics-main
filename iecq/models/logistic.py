"""PyTorch logistic regression with AIC stepwise screening and L2 (Methods §2.2 / §2.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from iecq.config import LAMBDA_MAX, LAMBDA_MIN, SEED, TrainConfig, resolve_device


class LogisticRegressionNet(nn.Module):
    """p_LR(x) = σ(β0 + βᵀ x)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def coefficients(self) -> tuple[float, np.ndarray]:
        b0 = float(self.linear.bias.detach().cpu().numpy()[0])
        beta = self.linear.weight.detach().cpu().numpy().ravel()
        return b0, beta


def _nll_with_l2(model: LogisticRegressionNet, x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    p = model(x).clamp(1e-7, 1 - 1e-7)
    nll = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    l2 = (model.linear.weight ** 2).sum()
    return nll + lam * l2


def train_logistic_torch(
    x: np.ndarray,
    y: np.ndarray,
    lam: float,
    cfg: TrainConfig,
    device: str | None = None,
) -> LogisticRegressionNet:
    device = device or resolve_device(cfg.device)
    model = LogisticRegressionNet(x.shape[1]).to(device)
    nn.init.zeros_(model.linear.weight)
    nn.init.zeros_(model.linear.bias)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_lr)
    model.train()
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    for _ in range(cfg.lr_epochs):
        opt.zero_grad(set_to_none=True)
        loss = _nll_with_l2(model, x_t, y_t, lam)
        loss.backward()
        opt.step()
    model.eval()
    return model


def logistic_aic(x: np.ndarray, y: np.ndarray, lam: float = 0.0) -> tuple[float, np.ndarray, float]:
    """Unpenalized (or lightly penalized) MLE AIC via Newton steps on CPU numpy."""
    n, p = x.shape
    xb = np.column_stack([np.ones(n), x])
    k = xb.shape[1]
    beta = np.zeros(k)
    for _ in range(40):
        z = np.clip(xb @ beta, -30, 30)
        p_hat = 1.0 / (1.0 + np.exp(-z))
        w = p_hat * (1 - p_hat) + 1e-6
        grad = xb.T @ (p_hat - y) + lam * np.concatenate([[0.0], beta[1:]])
        hess = xb.T @ (w[:, None] * xb)
        hess[1:, 1:] += lam * np.eye(p)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta = beta - step
        if np.max(np.abs(step)) < 1e-6:
            break
    z = np.clip(xb @ beta, -30, 30)
    p_hat = 1.0 / (1.0 + np.exp(-z))
    p_hat = np.clip(p_hat, 1e-12, 1 - 1e-12)
    ll = float(np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)))
    aic = 2 * k - 2 * ll
    return aic, beta, ll


def stepwise_aic(x: np.ndarray, y: np.ndarray, names: list[str], max_features: int | None = None) -> list[int]:
    """Forward introduction + backward elimination with AIC (Methods)."""
    n, p = x.shape
    selected: list[int] = []
    remaining = list(range(p))
    best_aic, _, _ = logistic_aic(np.zeros((n, 0)), y)
    improved = True
    limit = max_features or p
    while improved:
        improved = False
        # forward
        if remaining and len(selected) < limit:
            scores = []
            for j in remaining:
                cols = selected + [j]
                aic, _, _ = logistic_aic(x[:, cols], y)
                scores.append((aic, j))
            aic_f, j_f = min(scores, key=lambda t: t[0])
            if aic_f + 1e-6 < best_aic:
                selected.append(j_f)
                remaining.remove(j_f)
                best_aic = aic_f
                improved = True
        # backward
        if len(selected) >= 2:
            scores_b = []
            for j in selected:
                cols = [c for c in selected if c != j]
                aic, _, _ = logistic_aic(x[:, cols], y)
                scores_b.append((aic, j))
            aic_b, j_b = min(scores_b, key=lambda t: t[0])
            if aic_b + 1e-6 < best_aic:
                selected.remove(j_b)
                remaining.append(j_b)
                best_aic = aic_b
                improved = True
    if not selected:
        # fallback: keep all
        selected = list(range(p))
    return selected


@dataclass
class LogisticFit:
    model: LogisticRegressionNet
    lam: float
    selected_idx: list[int]
    selected_names: list[str]
    intercept: float
    beta: np.ndarray
    odds_ratio: np.ndarray
    or_ci_low: np.ndarray
    or_ci_high: np.ndarray
    aic: float
    loglik: float
    hl_stat: float
    hl_p: float
    oof_proba: np.ndarray
    tau: float
    feature_names: list[str]


def _hosmer_lemeshow(y: np.ndarray, p: np.ndarray, g: int = 10) -> tuple[float, float]:
    from scipy.stats import chi2

    order = np.argsort(p)
    y = y[order]
    p = p[order]
    n = len(y)
    stat = 0.0
    edges = np.linspace(0, n, g + 1, dtype=int)
    df = 0
    for i in range(g):
        sl = slice(edges[i], edges[i + 1])
        if edges[i + 1] <= edges[i]:
            continue
        obs = y[sl].sum()
        exp = p[sl].sum()
        ng = edges[i + 1] - edges[i]
        var = exp * (1 - exp / ng) if ng > 0 else 0.0
        if var > 1e-8:
            stat += (obs - exp) ** 2 / var
            df += 1
    df = max(df - 2, 1)
    pval = float(chi2.sf(stat, df))
    return float(stat), pval


def odds_ratio_with_ci(x: np.ndarray, y: np.ndarray, intercept: float, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    xb = np.column_stack([np.ones(n), x])
    b = np.concatenate([[intercept], beta])
    z = np.clip(xb @ b, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    w = p * (1 - p) + 1e-8
    hess = xb.T @ (w[:, None] * xb)
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hess)
    se = np.sqrt(np.clip(np.diag(cov)[1:], 1e-12, None))
    zcrit = 1.959964
    lo = np.exp(beta - zcrit * se)
    hi = np.exp(beta + zcrit * se)
    return np.exp(beta), lo, hi


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.nanargmax(j))
    tau = float(thr[k])
    if not np.isfinite(tau):
        tau = 0.5
    return float(np.clip(tau, 0.05, 0.95))


class StepwiseLogisticTrainer:
    def __init__(self, cfg: TrainConfig | None = None):
        self.cfg = cfg or TrainConfig()

    def fit(self, x: np.ndarray, y: np.ndarray, names: list[str]) -> LogisticFit:
        y = y.astype(np.float64)
        selected = stepwise_aic(x, y, names)
        xs = x[:, selected]
        sel_names = [names[i] for i in selected]
        device = resolve_device(self.cfg.device)

        skf = StratifiedKFold(n_splits=self.cfg.n_folds, shuffle=True, random_state=SEED)
        lam_grid = list(self.cfg.lambda_grid)
        lam_scores = []
        for lam in lam_grid:
            accs = []
            for tr, va in skf.split(xs, y.astype(int)):
                m = train_logistic_torch(xs[tr], y[tr], lam, self.cfg, device)
                with torch.no_grad():
                    p = m(torch.as_tensor(xs[va], dtype=torch.float32, device=device)).cpu().numpy().ravel()
                pred = (p >= 0.5).astype(int)
                accs.append((pred == y[va].astype(int)).mean())
            lam_scores.append((float(np.mean(accs)), lam))
        _, best_lam = max(lam_scores, key=lambda t: t[0])
        best_lam = float(np.clip(best_lam, LAMBDA_MIN, LAMBDA_MAX))

        oof = np.zeros(len(y))
        for tr, va in skf.split(xs, y.astype(int)):
            m = train_logistic_torch(xs[tr], y[tr], best_lam, self.cfg, device)
            with torch.no_grad():
                oof[va] = m(torch.as_tensor(xs[va], dtype=torch.float32, device=device)).cpu().numpy().ravel()
        tau = youden_threshold(y.astype(int), oof)

        model = train_logistic_torch(xs, y, best_lam, self.cfg, device)
        intercept, beta = model.coefficients()
        aic, beta_mle, ll = logistic_aic(xs, y, lam=0.0)
        or_hat, or_lo, or_hi = odds_ratio_with_ci(xs, y, intercept, beta)
        with torch.no_grad():
            p_full = model(torch.as_tensor(xs, dtype=torch.float32, device=device)).cpu().numpy().ravel()
        hl_stat, hl_p = _hosmer_lemeshow(y, p_full)
        return LogisticFit(
            model=model,
            lam=best_lam,
            selected_idx=selected,
            selected_names=sel_names,
            intercept=intercept,
            beta=beta,
            odds_ratio=or_hat,
            or_ci_low=or_lo,
            or_ci_high=or_hi,
            aic=aic,
            loglik=ll,
            hl_stat=hl_stat,
            hl_p=hl_p,
            oof_proba=oof,
            tau=tau,
            feature_names=sel_names,
        )

    @staticmethod
    def predict_proba(fit: LogisticFit, x: np.ndarray, device: str | None = None) -> np.ndarray:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        xs = x[:, fit.selected_idx]
        fit.model.to(device)
        fit.model.eval()
        with torch.no_grad():
            p = fit.model(torch.as_tensor(xs, dtype=torch.float32, device=device)).cpu().numpy().ravel()
        return p
