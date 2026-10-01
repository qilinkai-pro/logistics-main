"""Simulate IE-CQES-21 survey records matching Methods sampling and Results Table 1.

All stored fields are coded values. Instructional remarks are not written to data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from iecq.config import (
    ABOVE_TOTAL_MEAN,
    ABOVE_TOTAL_SD,
    BELOW_BY_UNIV,
    BELOW_PS_MEAN,
    BELOW_PS_SD,
    BELOW_TC_MEAN,
    BELOW_TC_SD,
    BELOW_TM_MEAN,
    BELOW_TM_SD,
    BELOW_TOTAL_MEAN,
    BELOW_TOTAL_SD,
    DEPT_COUNTS,
    LEVEL_COUNTS,
    MISSING_DROP_FRAC,
    N_ABOVE,
    N_BELOW,
    N_COLLECTED,
    N_VALID,
    SEED,
    TITLE_COUNTS,
    UNIV_COUNTS,
    Y_THRESHOLD,
)
from iecq.data.schema import (
    CATEGORICAL_PREDICTORS,
    CONTINUOUS_PREDICTORS,
    DEPT_LEVELS,
    EXCLUDE_REASON_COL,
    FUNDING_BAND_LEVELS,
    IE_ITEMS,
    IE_PS,
    IE_TC,
    IE_TM,
    LEVEL_LEVELS,
    PREDICTOR_COLS,
    RAW_COLUMNS,
    TITLE_LEVELS,
    UNIV_LEVELS,
)


def _expand(counts: dict[str, int], rng: np.random.Generator) -> np.ndarray:
    labels: list[str] = []
    for k, n in counts.items():
        labels.extend([k] * int(n))
    arr = np.array(labels)
    rng.shuffle(arr)
    return arr


def _clip_int(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.clip(np.rint(x), lo, hi).astype(int)


def _split_items(total: int, n: int, rng: np.random.Generator, max_item: int = 3) -> np.ndarray:
    total = int(np.clip(total, 0, n * max_item))
    scores = np.zeros(n, dtype=int)
    for _ in range(total):
        eligible = np.where(scores < max_item)[0]
        if eligible.size == 0:
            break
        w = (max_item - scores[eligible]).astype(float) + 0.3
        w = w / w.sum()
        idx = rng.choice(eligible, p=w)
        scores[idx] += 1
    return scores


def _truncated_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    lo: float,
    hi: float,
    size: int,
) -> np.ndarray:
    out = np.empty(size)
    filled = 0
    while filled < size:
        draw = rng.normal(mean, sd, size=size)
        ok = draw[(draw >= lo) & (draw <= hi)]
        take = min(ok.size, size - filled)
        out[filled : filled + take] = ok[:take]
        filled += take
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def generate_valid_records(rng: np.random.Generator, n: int = N_VALID) -> pd.DataFrame:
    univ = np.empty(n, dtype=object)
    y = np.zeros(n, dtype=int)
    start = 0
    for u, n_u in UNIV_COUNTS.items():
        n_b = BELOW_BY_UNIV[u]
        n_a = n_u - n_b
        univ[start : start + n_u] = u
        y[start : start + n_b] = 1
        y[start + n_b : start + n_u] = 0
        start += n_u
    perm = rng.permutation(n)
    univ = univ[perm]
    y = y[perm]

    course_level = _expand(LEVEL_COUNTS, rng)
    department = _expand(DEPT_COUNTS, rng)
    instructor_title = _expand(TITLE_COUNTS, rng)

    semester = rng.choice(["spring", "autumn"], size=n, p=[0.48, 0.52])
    semester_year = rng.choice([2023, 2024, 2025], size=n, p=[0.22, 0.48, 0.30])
    respondent_role = rng.choice(
        ["student", "instructor", "administrator", "industry_mentor"],
        size=n,
        p=[0.82, 0.10, 0.04, 0.04],
    )

    class_band = rng.choice([0, 1, 2, 3], size=n, p=[0.18, 0.42, 0.26, 0.14])
    class_size = np.empty(n, dtype=int)
    class_size[class_band == 0] = rng.integers(18, 31, size=(class_band == 0).sum())
    class_size[class_band == 1] = rng.integers(31, 61, size=(class_band == 1).sum())
    class_size[class_band == 2] = rng.integers(61, 101, size=(class_band == 2).sum())
    class_size[class_band == 3] = rng.integers(101, 181, size=(class_band == 3).sum())

    univ_code = pd.Series(univ).map(
        {u: i for i, u in enumerate(UNIV_LEVELS)}
    ).to_numpy()

    weekly_class_hours = np.clip(
        rng.normal(4.2 - 0.15 * univ_code + 0.35 * (course_level == "undergraduate_core"), 1.15, size=n),
        1.0,
        10.0,
    )
    semester_total_hours = np.clip(weekly_class_hours * rng.uniform(15.5, 18.5, size=n), 16, 180)
    perceived_weekly_hours = np.clip(weekly_class_hours + rng.normal(0, 0.35, size=n), 0.5, 12.0)

    funding_logit = (
        1.2
        - 0.55 * univ_code
        + 0.25 * (instructor_title == "professor")
        + 0.15 * (instructor_title == "industry_mentor")
        + rng.normal(0, 0.7, size=n)
    )
    funding_score = _sigmoid(funding_logit)
    funding_band_idx = np.digitize(funding_score, [0.22, 0.42, 0.62, 0.82])
    funding_band = np.array(FUNDING_BAND_LEVELS)[funding_band_idx]
    band_lo = np.array([800, 3000, 6000, 10000, 20000])
    band_hi = np.array([2999, 5999, 9999, 19999, 42000])
    funding_amount = rng.integers(band_lo[funding_band_idx], band_hi[funding_band_idx] + 1)
    perceived_funding_level = _clip_int(
        1.5 + 3.2 * funding_score + rng.normal(0, 0.55, size=n), 1, 5
    )

    learning_pressure = _clip_int(
        3.15 + 0.35 * (y == 1) + 0.18 * univ_code + rng.normal(0, 0.95, size=n), 1, 5
    )
    interaction_satisfaction = _clip_int(
        3.45 - 0.55 * (y == 1) - 0.12 * univ_code + rng.normal(0, 0.90, size=n), 1, 5
    )
    mentor_guidance = _clip_int(
        3.35 - 0.50 * (y == 1) + rng.normal(0, 0.95, size=n), 1, 5
    )
    teaching_resources = _clip_int(
        3.40 - 0.45 * (y == 1) - 0.18 * univ_code + rng.normal(0, 0.90, size=n), 1, 5
    )
    industry_project_integration = _clip_int(
        3.20 - 0.55 * (y == 1) - 0.15 * univ_code + rng.normal(0, 0.95, size=n), 1, 5
    )
    coursework_heaviness = _clip_int(
        3.05 + 0.40 * (y == 1) + rng.normal(0, 0.90, size=n), 1, 5
    )

    weekly_practice_freq = _clip_int(
        2.15 - 0.70 * (y == 1) + 0.20 * (1 - univ_code / 3) + rng.normal(0, 1.05, size=n),
        0,
        6,
    )
    monthly_project_practice = _clip_int(
        2.40 - 0.85 * (y == 1) + rng.normal(0, 1.15, size=n), 0, 8
    )
    weekly_team_project = _clip_int(
        1.85 - 0.65 * (y == 1) + rng.normal(0, 1.10, size=n), 0, 5
    )

    se_p = np.clip(0.62 - 0.28 * (y == 1) - 0.06 * univ_code, 0.12, 0.88)
    school_enterprise_support = np.where(rng.random(n) < se_p, "yes", "no")
    enterprise_mentor_hours = np.clip(
        rng.normal(
            2.4 + 2.2 * (school_enterprise_support == "yes") - 1.1 * (y == 1),
            1.4,
            size=n,
        ),
        0,
        10,
    )

    intensity_p = np.vstack(
        [
            np.clip(0.28 + 0.22 * (y == 1), 0.08, 0.7),
            np.clip(0.46 - 0.10 * (y == 1), 0.2, 0.6),
            np.clip(0.26 - 0.12 * (y == 1), 0.08, 0.5),
        ]
    ).T
    intensity_p = intensity_p / intensity_p.sum(axis=1, keepdims=True)
    practice_intensity = np.array(["low", "moderate", "high"])[
        [rng.choice(3, p=intensity_p[i]) for i in range(n)]
    ]

    fmt_p = np.vstack(
        [
            np.clip(0.30 + 0.18 * (y == 1), 0.1, 0.6),
            np.clip(0.38 - 0.14 * (y == 1), 0.15, 0.55),
            np.clip(0.26 - 0.04 * (y == 1), 0.1, 0.4),
            np.full(n, 0.06),
        ]
    ).T
    fmt_p = fmt_p / fmt_p.sum(axis=1, keepdims=True)
    practice_format = np.array(["solo_theory", "team_competition", "both", "other"])[
        [rng.choice(4, p=fmt_p[i]) for i in range(n)]
    ]

    feedback_regular = np.where(
        rng.random(n) < np.clip(0.68 - 0.32 * (y == 1), 0.2, 0.9), "regular", "irregular"
    )
    feedback_interval_days = np.where(
        feedback_regular == "regular",
        rng.integers(7, 31, size=n),
        rng.integers(31, 95, size=n),
    ).astype(float)
    extra_irreg = (feedback_regular == "irregular") & (rng.random(n) < 0.35)
    feedback_interval_days[extra_irreg] = rng.integers(61, 120, size=extra_irreg.sum())

    qshare_p = np.vstack(
        [
            np.clip(0.28 + 0.22 * (y == 1), 0.1, 0.6),
            np.clip(0.40 - 0.05 * (y == 1), 0.2, 0.5),
            np.clip(0.24 - 0.16 * (y == 1), 0.06, 0.4),
            np.full(n, 0.08),
        ]
    ).T
    qshare_p = qshare_p / qshare_p.sum(axis=1, keepdims=True)
    quality_investment_share = np.array(["lt_10", "10_20", "gt_20", "unknown"])[
        [rng.choice(4, p=qshare_p[i]) for i in range(n)]
    ]

    fund_stab_p = np.clip(0.70 - 0.28 * (y == 1) - 0.08 * univ_code, 0.25, 0.88)
    funding_stability = np.array(
        [
            rng.choice(["stable", "unstable", "unknown"], p=[p, 0.85 * (1 - p), 0.15 * (1 - p)])
            if False
            else None
            for p in fund_stab_p
        ]
    )
    funding_stability = np.empty(n, dtype=object)
    for i, p in enumerate(fund_stab_p):
        rest = 1 - p
        funding_stability[i] = rng.choice(
            ["stable", "unstable", "unknown"], p=[p, 0.82 * rest, 0.18 * rest]
        )

    res_p = np.clip(0.72 - 0.30 * (y == 1) - 0.06 * univ_code, 0.22, 0.9)
    resource_allocation_stability = np.where(rng.random(n) < res_p, "stable", "unstable")

    z = (
        0.50 * (learning_pressure - 3.0)
        - 0.38 * (interaction_satisfaction - 3.0)
        - 0.30 * (weekly_practice_freq - 2.0)
        - 0.26 * ((weekly_class_hours - 4.0) / 1.5)
        - 0.19 * (perceived_funding_level - 3.0)
        - 0.42 * (school_enterprise_support == "yes")
        - 0.20 * (practice_intensity == "moderate")
        - 0.16 * (resource_allocation_stability == "stable")
        - 0.18 * (feedback_regular == "regular")
        - 0.14 * ((enterprise_mentor_hours >= 5).astype(float))
        - 0.13 * (industry_project_integration - 3.0)
        + 0.20 * (coursework_heaviness - 3.0)
        - 0.11 * (mentor_guidance - 3.0)
        + 0.70 * ((learning_pressure >= 4) & (interaction_satisfaction <= 2))
        + 0.50 * ((learning_pressure > 3) & (monthly_project_practice < 2))
        + 0.32 * ((feedback_regular == "irregular") & (weekly_class_hours < 3.0))
        + 0.16 * univ_code
        + rng.normal(0, 0.62 + 0.10 * univ_code, size=n)
    )

    def _rate(b: float) -> float:
        return float(_sigmoid(b + z).mean())

    lo, hi = -6.0, 6.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _rate(mid) > 0.324:
            hi = mid
        else:
            lo = mid
    intercept = 0.5 * (lo + hi)
    p = _sigmoid(intercept + z)

    order = np.argsort(-p)
    y_model = np.zeros(n, dtype=int)
    y_model[order[:N_BELOW]] = 1
    flip_n = int(0.10 * n)
    flip_idx = order[N_BELOW - flip_n // 2 : N_BELOW + flip_n // 2]
    y_model[flip_idx] = 1 - y_model[flip_idx]
    if y_model.sum() > N_BELOW:
        extra = np.where(y_model == 1)[0]
        extra = extra[np.argsort(p[extra])[: y_model.sum() - N_BELOW]]
        y_model[extra] = 0
    elif y_model.sum() < N_BELOW:
        need = np.where(y_model == 0)[0]
        need = need[np.argsort(-p[need])[: N_BELOW - y_model.sum()]]
        y_model[need] = 1
    y = y_model

    below = y == 1
    totals = np.empty(n, dtype=float)
    totals[below] = _truncated_normal(rng, BELOW_TOTAL_MEAN, BELOW_TOTAL_SD, Y_THRESHOLD, 63, int(below.sum()))
    totals[~below] = _truncated_normal(rng, ABOVE_TOTAL_MEAN, ABOVE_TOTAL_SD, 0, Y_THRESHOLD - 1, int((~below).sum()))
    totals = _clip_int(totals, 0, 63)
    totals[below] = np.maximum(totals[below], Y_THRESHOLD)
    totals[~below] = np.minimum(totals[~below], Y_THRESHOLD - 1)

    tc = np.empty(n, dtype=int)
    tm = np.empty(n, dtype=int)
    ps = np.empty(n, dtype=int)
    tc[below] = _clip_int(
        rng.normal(BELOW_TC_MEAN, BELOW_TC_SD, size=int(below.sum())), 0, 21
    )
    tm[below] = _clip_int(
        rng.normal(BELOW_TM_MEAN, BELOW_TM_SD, size=int(below.sum())), 0, 21
    )
    ps[below] = np.clip(totals[below] - tc[below] - tm[below], 0, 21)
    overflow = below & ((tc + tm + ps) != totals)
    for i in np.where(overflow)[0]:
        w = np.array(
            [
                max(BELOW_TC_MEAN, 1.0),
                max(BELOW_TM_MEAN, 1.0),
                max(BELOW_PS_MEAN, 1.0),
            ]
        )
        w = w / w.sum()
        parts = np.array([int(round(totals[i] * w[0])), int(round(totals[i] * w[1])), 0])
        parts[2] = totals[i] - parts[0] - parts[1]
        parts = np.clip(parts, 0, 21)
        parts[2] = np.clip(totals[i] - parts[0] - parts[1], 0, 21)
        tc[i], tm[i], ps[i] = parts

    n_above = int((~below).sum())
    share = np.array([0.34, 0.32, 0.34])
    tc[~below] = _clip_int(totals[~below] * share[0] + rng.normal(0, 1.4, n_above), 0, 21)
    tm[~below] = _clip_int(totals[~below] * share[1] + rng.normal(0, 1.3, n_above), 0, 21)
    ps[~below] = np.clip(totals[~below] - tc[~below] - tm[~below], 0, 21)
    for i in np.where(~below)[0]:
        s = int(tc[i] + tm[i] + ps[i])
        if s != totals[i]:
            ps[i] = int(np.clip(totals[i] - tc[i] - tm[i], 0, 21))
            tc[i] = int(np.clip(tc[i], 0, 21))
            tm[i] = int(np.clip(totals[i] - tc[i] - ps[i], 0, 21))

    item_mat = np.zeros((n, 21), dtype=int)
    for i in range(n):
        item_mat[i, 0:7] = _split_items(int(tc[i]), 7, rng)
        item_mat[i, 7:14] = _split_items(int(tm[i]), 7, rng)
        item_mat[i, 14:21] = _split_items(int(ps[i]), 7, rng)

    l2 = _clip_int(5 - item_mat[:, 6] + rng.normal(0, 0.4, size=n), 1, 5)

    df = pd.DataFrame(
        {
            "survey_id": [f"C{i:04d}" for i in range(1, n + 1)],
            "completion_time_min": np.round(rng.uniform(15.0, 20.0, size=n), 2),
            "s1_complete_semester": 1,
            "s2_complete_records": 1,
            "s3_major_restructure": 0,
            "s4_course_level": course_level,
            "university_type": univ,
            "department": department,
            "semester": semester,
            "semester_year": semester_year.astype(int),
            "class_size": class_size,
            "instructor_title": instructor_title,
            "weekly_class_hours": np.round(weekly_class_hours, 2),
            "semester_total_hours": np.round(semester_total_hours, 1),
            "funding_band": funding_band,
            "funding_amount": funding_amount,
            "respondent_role": respondent_role,
        }
    )
    for j, col in enumerate(IE_ITEMS):
        df[col] = item_mat[:, j]
    df["learning_pressure"] = learning_pressure
    df["interaction_satisfaction"] = interaction_satisfaction
    df["mentor_guidance"] = mentor_guidance
    df["teaching_resources"] = teaching_resources
    df["industry_project_integration"] = industry_project_integration
    df["coursework_heaviness"] = coursework_heaviness
    df["weekly_practice_freq"] = weekly_practice_freq
    df["monthly_project_practice"] = monthly_project_practice
    df["practice_intensity"] = practice_intensity
    df["practice_format"] = practice_format
    df["weekly_team_project"] = weekly_team_project
    df["school_enterprise_support"] = school_enterprise_support
    df["enterprise_mentor_hours"] = np.round(enterprise_mentor_hours, 2)
    df["perceived_weekly_hours"] = np.round(perceived_weekly_hours, 2)
    df["feedback_interval_days"] = np.round(feedback_interval_days, 1)
    df["feedback_regular"] = feedback_regular
    df["quality_investment_share"] = quality_investment_share
    df["funding_stability"] = funding_stability
    df["resource_allocation_stability"] = resource_allocation_stability
    df["perceived_funding_level"] = perceived_funding_level
    df["l1_select_2"] = 2
    df["l2_content_supports_practice"] = l2
    df["l3_restructure_confirm"] = 0
    df["l4_proxy_fill"] = 0
    df["tc_subtotal"] = df[IE_TC].sum(axis=1).astype(int)
    df["tm_subtotal"] = df[IE_TM].sum(axis=1).astype(int)
    df["ps_subtotal"] = df[IE_PS].sum(axis=1).astype(int)
    df["ie_cqes21_total"] = df["tc_subtotal"] + df["tm_subtotal"] + df["ps_subtotal"]
    df["y"] = (df["ie_cqes21_total"] >= Y_THRESHOLD).astype(int)
    df[EXCLUDE_REASON_COL] = ""

    df = _inject_item_noise_to_match_y(df, y, rng)
    df = _inject_missing_and_outliers(df, rng, valid=True)
    return df.reset_index(drop=True)


def _inject_item_noise_to_match_y(
    df: pd.DataFrame, y_target: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Keep IE-CQES-21 totals consistent with the designed binary label."""
    df = df.copy()
    cur = (df["ie_cqes21_total"] >= Y_THRESHOLD).to_numpy()
    mismatch = np.where(cur != y_target)[0]
    for i in mismatch:
        if y_target[i] == 1 and df.at[i, "ie_cqes21_total"] < Y_THRESHOLD:
            need = Y_THRESHOLD - int(df.at[i, "ie_cqes21_total"])
            items = IE_ITEMS.copy()
            rng.shuffle(items)
            for col in items:
                if need <= 0:
                    break
                room = 3 - int(df.at[i, col])
                add = min(room, need)
                df.at[i, col] = int(df.at[i, col]) + add
                need -= add
        elif y_target[i] == 0 and df.at[i, "ie_cqes21_total"] >= Y_THRESHOLD:
            need = int(df.at[i, "ie_cqes21_total"]) - (Y_THRESHOLD - 1)
            items = IE_ITEMS.copy()
            rng.shuffle(items)
            for col in items:
                if need <= 0:
                    break
                room = int(df.at[i, col])
                sub = min(room, need)
                df.at[i, col] = int(df.at[i, col]) - sub
                need -= sub
    df["tc_subtotal"] = df[IE_TC].sum(axis=1).astype(int)
    df["tm_subtotal"] = df[IE_TM].sum(axis=1).astype(int)
    df["ps_subtotal"] = df[IE_PS].sum(axis=1).astype(int)
    df["ie_cqes21_total"] = df["tc_subtotal"] + df["tm_subtotal"] + df["ps_subtotal"]
    df["y"] = (df["ie_cqes21_total"] >= Y_THRESHOLD).astype(int)

    n_below = int(df["y"].sum())
    if n_below != N_BELOW:
        if n_below > N_BELOW:
            idx = df.index[df["y"] == 1].to_numpy()
            idx = idx[np.argsort(df.loc[idx, "ie_cqes21_total"].to_numpy())[: n_below - N_BELOW]]
            for i in idx:
                need = int(df.at[i, "ie_cqes21_total"]) - (Y_THRESHOLD - 1)
                for col in IE_ITEMS:
                    if need <= 0:
                        break
                    room = int(df.at[i, col])
                    sub = min(room, need)
                    df.at[i, col] = int(df.at[i, col]) - sub
                    need -= sub
        else:
            idx = df.index[df["y"] == 0].to_numpy()
            idx = idx[np.argsort(-df.loc[idx, "ie_cqes21_total"].to_numpy())[: N_BELOW - n_below]]
            for i in idx:
                need = Y_THRESHOLD - int(df.at[i, "ie_cqes21_total"])
                for col in IE_ITEMS:
                    if need <= 0:
                        break
                    room = 3 - int(df.at[i, col])
                    add = min(room, need)
                    df.at[i, col] = int(df.at[i, col]) + add
                    need -= add
        df["tc_subtotal"] = df[IE_TC].sum(axis=1).astype(int)
        df["tm_subtotal"] = df[IE_TM].sum(axis=1).astype(int)
        df["ps_subtotal"] = df[IE_PS].sum(axis=1).astype(int)
        df["ie_cqes21_total"] = df["tc_subtotal"] + df["tm_subtotal"] + df["ps_subtotal"]
        df["y"] = (df["ie_cqes21_total"] >= Y_THRESHOLD).astype(int)
    return df


def _inject_missing_and_outliers(
    df: pd.DataFrame, rng: np.random.Generator, valid: bool
) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    miss_cols = [
        "class_size",
        "weekly_class_hours",
        "funding_amount",
        "mentor_guidance",
        "teaching_resources",
        "enterprise_mentor_hours",
        "feedback_interval_days",
        "perceived_weekly_hours",
        "practice_intensity",
        "funding_stability",
        "quality_investment_share",
    ]
    rate = 0.035 if valid else 0.08
    for col in miss_cols:
        mask = rng.random(n) < rate
        if valid:
            # keep row-wise missing well below 20%
            df.loc[mask, col] = np.nan
        else:
            df.loc[mask, col] = np.nan
    if valid:
        out_n = max(4, int(0.012 * n))
        out_idx = rng.choice(df.index, size=out_n, replace=False)
        df.loc[out_idx[: out_n // 2], "class_size"] = rng.integers(260, 420, size=out_n // 2)
        df.loc[out_idx[out_n // 2 :], "weekly_class_hours"] = np.round(
            rng.uniform(14.5, 22.0, size=out_n - out_n // 2), 2
        )
    return df


def _random_fill_records(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Unconstrained records used only for excluded questionnaires."""
    df = pd.DataFrame(
        {
            "survey_id": [f"X{i:04d}" for i in range(1, n + 1)],
            "completion_time_min": np.round(rng.uniform(12.0, 22.0, size=n), 2),
            "s1_complete_semester": 1,
            "s2_complete_records": 1,
            "s3_major_restructure": 0,
            "s4_course_level": rng.choice(
                ["undergraduate_core", "undergraduate_elective"], size=n
            ),
            "university_type": rng.choice(UNIV_LEVELS, size=n),
            "department": rng.choice(DEPT_LEVELS, size=n),
            "semester": rng.choice(["spring", "autumn"], size=n),
            "semester_year": rng.choice([2023, 2024, 2025], size=n),
            "class_size": rng.integers(20, 120, size=n),
            "instructor_title": rng.choice(TITLE_LEVELS, size=n),
            "weekly_class_hours": np.round(rng.uniform(2.0, 8.0, size=n), 2),
            "semester_total_hours": np.round(rng.uniform(32.0, 128.0, size=n), 1),
            "funding_band": rng.choice(FUNDING_BAND_LEVELS, size=n),
            "funding_amount": rng.integers(1200, 25000, size=n),
            "respondent_role": rng.choice(
                ["student", "instructor", "administrator", "industry_mentor"], size=n
            ),
        }
    )
    for col in IE_ITEMS:
        df[col] = rng.integers(0, 4, size=n)
    df["learning_pressure"] = rng.integers(1, 6, size=n)
    df["interaction_satisfaction"] = rng.integers(1, 6, size=n)
    df["mentor_guidance"] = rng.integers(1, 6, size=n)
    df["teaching_resources"] = rng.integers(1, 6, size=n)
    df["industry_project_integration"] = rng.integers(1, 6, size=n)
    df["coursework_heaviness"] = rng.integers(1, 6, size=n)
    df["weekly_practice_freq"] = rng.integers(0, 6, size=n)
    df["monthly_project_practice"] = rng.integers(0, 7, size=n)
    df["practice_intensity"] = rng.choice(["low", "moderate", "high"], size=n)
    df["practice_format"] = rng.choice(
        ["solo_theory", "team_competition", "both", "other"], size=n
    )
    df["weekly_team_project"] = rng.integers(0, 5, size=n)
    df["school_enterprise_support"] = rng.choice(["yes", "no"], size=n)
    df["enterprise_mentor_hours"] = np.round(rng.uniform(0, 8, size=n), 2)
    df["perceived_weekly_hours"] = np.round(rng.uniform(1.5, 9.0, size=n), 2)
    df["feedback_interval_days"] = rng.integers(7, 90, size=n).astype(float)
    df["feedback_regular"] = rng.choice(["regular", "irregular"], size=n)
    df["quality_investment_share"] = rng.choice(["lt_10", "10_20", "gt_20", "unknown"], size=n)
    df["funding_stability"] = rng.choice(["stable", "unstable", "unknown"], size=n)
    df["resource_allocation_stability"] = rng.choice(["stable", "unstable"], size=n)
    df["perceived_funding_level"] = rng.integers(1, 6, size=n)
    df["l1_select_2"] = 2
    df["l2_content_supports_practice"] = rng.integers(1, 6, size=n)
    df["l3_restructure_confirm"] = 0
    df["l4_proxy_fill"] = 0
    df["tc_subtotal"] = df[IE_TC].sum(axis=1).astype(int)
    df["tm_subtotal"] = df[IE_TM].sum(axis=1).astype(int)
    df["ps_subtotal"] = df[IE_PS].sum(axis=1).astype(int)
    df["ie_cqes21_total"] = df["tc_subtotal"] + df["tm_subtotal"] + df["ps_subtotal"]
    df["y"] = (df["ie_cqes21_total"] >= Y_THRESHOLD).astype(int)
    df[EXCLUDE_REASON_COL] = ""
    return df


def generate_invalid_records(rng: np.random.Generator, n: int = N_COLLECTED - N_VALID) -> pd.DataFrame:
    reasons = (
        ["s1_incomplete"] * 22
        + ["s2_incomplete"] * 22
        + ["s3_restructure"] * 18
        + ["postgraduate"] * 16
        + ["l1_fail"] * 14
        + ["l3_mismatch"] * 10
        + ["l4_proxy"] * 8
        + ["missing_gt_20pct"] * 10
    )
    reasons = reasons[:n]
    rng.shuffle(reasons)
    base = _random_fill_records(rng, n)
    base["survey_id"] = [f"X{i:04d}" for i in range(1, n + 1)]
    base[EXCLUDE_REASON_COL] = reasons
    for i, reason in enumerate(reasons):
        if reason == "s1_incomplete":
            base.at[i, "s1_complete_semester"] = 0
        elif reason == "s2_incomplete":
            base.at[i, "s2_complete_records"] = 0
        elif reason == "s3_restructure":
            base.at[i, "s3_major_restructure"] = 1
            base.at[i, "l3_restructure_confirm"] = 1
        elif reason == "postgraduate":
            base.at[i, "s4_course_level"] = "postgraduate"
        elif reason == "l1_fail":
            base.at[i, "l1_select_2"] = int(rng.choice([0, 1, 3]))
        elif reason == "l3_mismatch":
            base.at[i, "l3_restructure_confirm"] = 1
            base.at[i, "s3_major_restructure"] = 0
        elif reason == "l4_proxy":
            base.at[i, "l4_proxy_fill"] = 1
        elif reason == "missing_gt_20pct":
            cols = CONTINUOUS_PREDICTORS + CATEGORICAL_PREDICTORS + IE_ITEMS
            k = max(int(MISSING_DROP_FRAC * len(cols)) + 3, 12)
            drop = rng.choice(cols, size=k, replace=False)
            for c in drop:
                base.at[i, c] = np.nan
        if reason in {"s1_incomplete", "s2_incomplete", "l1_fail", "l4_proxy"}:
            base.at[i, "completion_time_min"] = round(float(rng.choice([4.2, 6.5, 42.0, 55.0])), 2)
    return base


def _row_missing_frac(row: pd.Series) -> float:
    cols = PREDICTOR_COLS + IE_ITEMS
    vals = row[cols]
    return float(vals.isna().mean())


def _as_int(v, default: int = 0) -> int:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return int(v)


def screen_valid_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply Methods inclusion / exclusion and logic checks."""
    work = df.copy()
    reasons = []
    for _, row in work.iterrows():
        reason = ""
        if _as_int(row["s1_complete_semester"]) != 1:
            reason = "s1_incomplete"
        elif _as_int(row["s2_complete_records"]) != 1:
            reason = "s2_incomplete"
        elif _as_int(row["s3_major_restructure"]) != 0:
            reason = "s3_restructure"
        elif str(row["s4_course_level"]) == "postgraduate":
            reason = "postgraduate"
        elif _as_int(row["l1_select_2"]) != 2:
            reason = "l1_fail"
        elif _as_int(row["l3_restructure_confirm"]) != _as_int(row["s3_major_restructure"]):
            reason = "l3_mismatch"
        elif _as_int(row["l4_proxy_fill"]) != 0:
            reason = "l4_proxy"
        elif _row_missing_frac(row) > MISSING_DROP_FRAC:
            reason = "missing_gt_20pct"
        reasons.append(reason)
    work["_screen_reason"] = reasons
    valid = work[work["_screen_reason"] == ""].drop(columns=["_screen_reason"]).copy()
    invalid = work[work["_screen_reason"] != ""].copy()
    if EXCLUDE_REASON_COL in invalid.columns:
        invalid[EXCLUDE_REASON_COL] = invalid["_screen_reason"]
        invalid = invalid.drop(columns=["_screen_reason"])
    valid[EXCLUDE_REASON_COL] = ""
    if "ie_cqes21_total" in valid.columns:
        valid["y"] = (valid["ie_cqes21_total"] >= Y_THRESHOLD).astype(int)
    return valid.reset_index(drop=True), invalid.reset_index(drop=True)


def generate_full_survey(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    valid = generate_valid_records(rng, N_VALID)
    invalid = generate_invalid_records(np.random.default_rng(seed + 7), N_COLLECTED - N_VALID)
    raw = pd.concat([valid, invalid], ignore_index=True)
    raw = raw.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    raw["survey_id"] = [f"R{i:04d}" for i in range(1, len(raw) + 1)]
    ordered = [c for c in RAW_COLUMNS if c in raw.columns]
    extra = [c for c in raw.columns if c not in ordered]
    raw = raw[ordered + extra]
    screened, _ = screen_valid_samples(raw)
    if EXCLUDE_REASON_COL in screened.columns:
        screened = screened.drop(columns=[EXCLUDE_REASON_COL])
    return raw, screened


def save_survey(
    raw: pd.DataFrame,
    valid: pd.DataFrame,
    raw_path,
    valid_path,
) -> None:
    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    valid.to_csv(valid_path, index=False, encoding="utf-8-sig")
