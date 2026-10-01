"""Questionnaire field schema (IE-CQES-21 + process module). Labels only, no remarks."""

from __future__ import annotations

IE_TC = [f"tc{i}" for i in range(1, 8)]
IE_TM = [f"tm{i}" for i in range(1, 8)]
IE_PS = [f"ps{i}" for i in range(1, 8)]
IE_ITEMS = IE_TC + IE_TM + IE_PS

UNIV_LEVELS = [
    "double_first_class",
    "univ_211",
    "general_undergraduate",
    "vocational",
]
LEVEL_LEVELS = ["undergraduate_core", "undergraduate_elective", "postgraduate"]
DEPT_LEVELS = [
    "business_economics",
    "engineering_technology",
    "education_humanities",
    "interdisciplinary",
]
TITLE_LEVELS = ["lecturer", "associate_professor", "professor", "industry_mentor"]
SEMESTER_LEVELS = ["spring", "autumn"]
INTENSITY_LEVELS = ["low", "moderate", "high"]
FORMAT_LEVELS = ["solo_theory", "team_competition", "both", "other"]
ROLE_LEVELS = ["student", "instructor", "administrator", "industry_mentor"]
FUNDING_BAND_LEVELS = ["lt_3000", "3000_5999", "6000_9999", "10000_19999", "ge_20000"]
QUALITY_SHARE_LEVELS = ["lt_10", "10_20", "gt_20", "unknown"]
STABILITY_LEVELS = ["unstable", "stable", "unknown"]
BINARY_YN = ["no", "yes"]

SCREEN_COLS = ["s1_complete_semester", "s2_complete_records", "s3_major_restructure", "s4_course_level"]
ADMIN_COLS = [
    "university_type",
    "department",
    "semester",
    "semester_year",
    "class_size",
    "instructor_title",
    "weekly_class_hours",
    "semester_total_hours",
    "funding_band",
    "funding_amount",
    "respondent_role",
]
PROCESS_COLS = [
    "learning_pressure",
    "interaction_satisfaction",
    "mentor_guidance",
    "teaching_resources",
    "industry_project_integration",
    "coursework_heaviness",
    "weekly_practice_freq",
    "monthly_project_practice",
    "practice_intensity",
    "practice_format",
    "weekly_team_project",
    "school_enterprise_support",
    "enterprise_mentor_hours",
    "perceived_weekly_hours",
    "feedback_interval_days",
    "feedback_regular",
    "quality_investment_share",
    "funding_stability",
    "resource_allocation_stability",
    "perceived_funding_level",
]
LOGIC_COLS = ["l1_select_2", "l2_content_supports_practice", "l3_restructure_confirm", "l4_proxy_fill"]
META_COLS = ["survey_id", "completion_time_min"]
OUTCOME_COLS = [
    "tc_subtotal",
    "tm_subtotal",
    "ps_subtotal",
    "ie_cqes21_total",
    "y",
]
EXCLUDE_REASON_COL = "exclude_reason"

CONTINUOUS_PREDICTORS = [
    "class_size",
    "weekly_class_hours",
    "semester_total_hours",
    "funding_amount",
    "learning_pressure",
    "interaction_satisfaction",
    "mentor_guidance",
    "teaching_resources",
    "industry_project_integration",
    "coursework_heaviness",
    "weekly_practice_freq",
    "monthly_project_practice",
    "weekly_team_project",
    "enterprise_mentor_hours",
    "perceived_weekly_hours",
    "feedback_interval_days",
    "perceived_funding_level",
]

CATEGORICAL_PREDICTORS = [
    "university_type",
    "s4_course_level",
    "department",
    "instructor_title",
    "funding_band",
    "practice_intensity",
    "practice_format",
    "school_enterprise_support",
    "feedback_regular",
    "quality_investment_share",
    "funding_stability",
    "resource_allocation_stability",
]

PREDICTOR_COLS = CONTINUOUS_PREDICTORS + CATEGORICAL_PREDICTORS

STRATIFY_COLS = ["university_type", "s4_course_level", "y"]

RAW_COLUMNS = (
    META_COLS
    + SCREEN_COLS
    + ADMIN_COLS
    + IE_ITEMS
    + PROCESS_COLS
    + LOGIC_COLS
    + OUTCOME_COLS
    + [EXCLUDE_REASON_COL]
)

ITEM_LABELS = {
    "tc1": "objectives_alignment",
    "tc2": "content_currency",
    "tc3": "innovation_knowledge_depth",
    "tc4": "case_authenticity",
    "tc5": "interdisciplinary_industry",
    "tc6": "assessment_alignment",
    "tc7": "content_practice_support",
    "tm1": "project_inquiry_teaching",
    "tm2": "inclass_interaction",
    "tm3": "digital_blended",
    "tm4": "differentiated_guidance",
    "tm5": "formative_feedback",
    "tm6": "team_competition_learning",
    "tm7": "intention_activation",
    "ps1": "practice_hours",
    "ps2": "platform_equipment",
    "ps3": "school_enterprise",
    "ps4": "industry_mentor_guidance",
    "ps5": "real_project_access",
    "ps6": "competition_incubation",
    "ps7": "outcome_conversion",
}
