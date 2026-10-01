"""Influencing-factor analysis from logistic OR and CART paths (Methods §2.3, Results §3.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def logistic_or_table(fit) -> pd.DataFrame:
    rows = []
    for name, b, orh, lo, hi in zip(
        fit.selected_names, fit.beta, fit.odds_ratio, fit.or_ci_low, fit.or_ci_high
    ):
        rows.append(
            {
                "feature": name,
                "beta": float(b),
                "or": float(orh),
                "or_ci_low": float(lo),
                "or_ci_high": float(hi),
                "role": "risk" if orh > 1 else "protective",
            }
        )
    return pd.DataFrame(rows).sort_values("or", ascending=False).reset_index(drop=True)


def tree_importance_table(fit, top: int = 10) -> pd.DataFrame:
    imp = np.asarray(fit.importances)
    names = list(fit.selected_names)
    order = np.argsort(-imp)
    rows = []
    for i in order[:top]:
        rows.append({"feature": names[i], "gini_importance": float(imp[i])})
    return pd.DataFrame(rows)


def _rate(mask: np.ndarray, y: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "below_rate": float("nan"), "above_rate": float("nan")}
    a = float(y[mask].mean())
    return {"n": n, "below_rate": a, "above_rate": 1.0 - a}


def grouping_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y = df["y"].to_numpy()
    pressure = pd.to_numeric(df["learning_pressure"], errors="coerce").to_numpy()
    practice_m = pd.to_numeric(df["monthly_project_practice"], errors="coerce").to_numpy()
    coursework = pd.to_numeric(df["coursework_heaviness"], errors="coerce").to_numpy()
    mentor = pd.to_numeric(df["mentor_guidance"], errors="coerce").to_numpy()
    se = df["school_enterprise_support"].astype(str).to_numpy()
    hours = pd.to_numeric(df["enterprise_mentor_hours"], errors="coerce").to_numpy()
    industry = pd.to_numeric(df["industry_project_integration"], errors="coerce").to_numpy()
    resources = pd.to_numeric(df["teaching_resources"], errors="coerce").to_numpy()
    weekly_h = pd.to_numeric(df["weekly_class_hours"], errors="coerce").to_numpy()
    fb_int = pd.to_numeric(df["feedback_interval_days"], errors="coerce").to_numpy()
    fb_reg = df["feedback_regular"].astype(str).to_numpy()
    team = pd.to_numeric(df["weekly_team_project"], errors="coerce").to_numpy()
    fmt = df["practice_format"].astype(str).to_numpy()
    intensity = df["practice_intensity"].astype(str).to_numpy()
    qshare = df["quality_investment_share"].astype(str).to_numpy()
    fund_st = df["funding_stability"].astype(str).to_numpy()
    perc_fund = pd.to_numeric(df["perceived_funding_level"], errors="coerce").to_numpy()
    wprac = pd.to_numeric(df["weekly_practice_freq"], errors="coerce").to_numpy()

    specs = []

    combined = (pressure > 3) & (practice_m < 2) & (coursework >= 4) & (mentor <= 2)
    high_p = pressure > 3
    practice_only = practice_m < 2
    specs.append(("learning_pressure", "combined_negative", _rate(combined, y), None))
    specs.append(("learning_pressure", "high_pressure", _rate(high_p, y), None))
    specs.append(("learning_pressure", "insufficient_practice", _rate(practice_only, y), None))

    weak_mentor = mentor <= 2
    se_yes = weak_mentor & (hours >= 5)
    se_no = weak_mentor & (hours < 5) & (se == "no")
    specs.append(("teacher_student_interaction", "enterprise_mentor_ge5h", _rate(se_yes, y), None))
    specs.append(("teacher_student_interaction", "no_school_enterprise", _rate(se_no, y), None))

    high_ind = industry >= 4
    good_se = se == "yes"
    good_res = resources >= 4
    specs.append(("teacher_student_interaction", "high_industry_integration", _rate(high_ind, y), None))
    specs.append(("teacher_student_interaction", "school_enterprise_support", _rate(good_se, y), None))
    specs.append(("teacher_student_interaction", "adequate_resources", _rate(good_res, y), None))

    insuff_h = weekly_h < 3.0
    regular = fb_reg == "regular"
    irregular = fb_reg == "irregular"
    fb_gt60 = fb_int > 60
    specs.append(("class_hours_feedback", "adequate_hours_poor_feedback", _rate((~insuff_h) & irregular, y), None))
    specs.append(("class_hours_feedback", "insufficient_hours_regular", _rate(insuff_h & regular, y), None))
    specs.append(("class_hours_feedback", "regular_feedback", _rate(regular, y), None))
    specs.append(("class_hours_feedback", "feedback_interval_gt60", _rate(fb_gt60, y), None))
    specs.append(("class_hours_feedback", "irregular_feedback", _rate(irregular, y), None))
    specs.append(
        (
            "class_hours_feedback",
            "high_pressure_practice_ge3",
            _rate((pressure > 3) & (wprac >= 3), y),
            None,
        )
    )
    specs.append(
        (
            "class_hours_feedback",
            "high_pressure_low_practice",
            _rate((pressure > 3) & (wprac < 3), y),
            None,
        )
    )

    team3 = (fmt == "team_competition") & (team >= 3)
    solo = fmt == "solo_theory"
    specs.append(("practice_frequency", "team_projects_3_per_week", _rate(team3, y), None))
    specs.append(("practice_frequency", "solo_theory", _rate(solo, y), None))
    for lab in ["low", "moderate", "high"]:
        specs.append(("practice_frequency", f"{lab}_intensity", _rate(intensity == lab, y), None))

    specs.append(("funding", "adequate_funding_resources", _rate((perc_fund >= 4) & (resources >= 4), y), None))
    specs.append(("funding", "quality_investment_gt20", _rate(qshare == "gt_20", y), None))
    specs.append(("funding", "quality_investment_lt10", _rate(qshare == "lt_10", y), None))
    specs.append(("funding", "stable_funding", _rate(fund_st == "stable", y), None))
    specs.append(("funding", "unstable_funding", _rate(fund_st == "unstable", y), None))

    rows = []
    for feat, group, rates, _extra in specs:
        rows.append(
            {
                "feature": feat,
                "grouping": group,
                "n": rates["n"],
                "below_rate": rates["below_rate"],
                "above_rate": rates["above_rate"],
            }
        )
    out = pd.DataFrame(rows)

    # attach chi-square for a few key 2x2 splits
    tests = {}
    tests["school_enterprise"] = _chi2(se == "yes", y)
    tests["regular_feedback"] = _chi2(regular, y)
    tests["moderate_intensity"] = _chi2(intensity == "moderate", y)
    tests["stable_funding"] = _chi2(fund_st == "stable", y)
    return out, tests


def _chi2(group: np.ndarray, y: np.ndarray) -> dict:
    tab = np.array(
        [
            [((~group) & (y == 0)).sum(), ((~group) & (y == 1)).sum()],
            [(group & (y == 0)).sum(), (group & (y == 1)).sum()],
        ],
        dtype=float,
    )
    chi2, p, _, _ = chi2_contingency(tab)
    return {"chi2": float(chi2), "p": float(p)}


def sample_summary(df: pd.DataFrame) -> dict:
    y = df["y"]
    below = df[df["y"] == 1]
    above = df[df["y"] == 0]
    from scipy.stats import ttest_ind

    t, p = ttest_ind(below["ie_cqes21_total"], above["ie_cqes21_total"], equal_var=False)
    return {
        "n": int(len(df)),
        "below_n": int((y == 1).sum()),
        "above_n": int((y == 0).sum()),
        "below_rate": float(y.mean()),
        "university_type": df["university_type"].value_counts().to_dict(),
        "course_level": df["s4_course_level"].value_counts().to_dict(),
        "department": df["department"].value_counts().to_dict(),
        "instructor_title": df["instructor_title"].value_counts().to_dict(),
        "below_total_mean": float(below["ie_cqes21_total"].mean()),
        "below_total_sd": float(below["ie_cqes21_total"].std(ddof=1)),
        "above_total_mean": float(above["ie_cqes21_total"].mean()),
        "above_total_sd": float(above["ie_cqes21_total"].std(ddof=1)),
        "t_total": float(t),
        "p_total": float(p),
        "below_tc_mean": float(below["tc_subtotal"].mean()),
        "below_tc_sd": float(below["tc_subtotal"].std(ddof=1)),
        "below_tm_mean": float(below["tm_subtotal"].mean()),
        "below_tm_sd": float(below["tm_subtotal"].std(ddof=1)),
        "below_ps_mean": float(below["ps_subtotal"].mean()),
        "below_ps_sd": float(below["ps_subtotal"].std(ddof=1)),
    }
