from typing import Optional, Dict, Any
from src.mathematical_ops import clamp, hr_dev, hrv_dev, sleep_debt_24h, steps_ratio

# Core Stress Components ----------------------------------------------------------------
def physiological_stress(hr_deviation: float, hrv_deviation: Optional[float], model_prob: Optional[float], steps_ratio_v: float,) -> float:
    """
    This function computes the physiological stress score based on heart rate deviation, heart rate variability deviation,
    optional model probability, and steps ratio.

    Args:
        hr_deviation (float): Heart rate deviation from baseline.
        hrv_deviation (Optional[float]): Heart rate variability deviation from baseline.
        model_prob (Optional[float]): Model probability of stress.
        steps_ratio_v (float): Ratio of steps in session to steps in baseline.

    Returns:
        float: Physiological stress score.
    """
    # Activity buffer
    activity_buffer = max(0.0, steps_ratio_v - 1.0) * 10.0

    if model_prob is not None:
        ps = 100.0 * clamp(model_prob, 0.0, 1.0)
    elif hrv_deviation is not None:
        ps = 100.0 * clamp(
            0.6 * hrv_deviation + 0.4 * hr_deviation + 0.1, 0.0, 1.0
        )
    else:
        ps = 100.0 * clamp(
            0.5 * hr_deviation + 0.15, 0.0, 1.0
        )

    return clamp(ps - activity_buffer, 0.0, 100.0)


def self_reported_stress(pre: Optional[int], post: Optional[int], weekly: Optional[int]) -> float:
    """
    This function computes the self-reported stress score based on pre, post, and weekly questionnaires.

    Args:
        pre (Optional[int]): Pre-session questionnaire score.
        post (Optional[int]): Post-session questionnaire score.
        weekly (Optional[int]): Weekly questionnaire score.

    Returns:
        float: Self-reported stress score.
    """
    values = []
    weights = []

    if post is not None:
        values.append(post * 50.0)
        weights.append(0.5)
    if pre is not None:
        values.append(pre * 50.0)
        weights.append(0.35)
    if weekly is not None:
        values.append(weekly * 50.0)
        weights.append(0.15)

    if not values:
        return 20.0 

    return clamp(sum(v * w for v, w in zip(values, weights)), 0.0, 100.0)


def academic_context(deadlines_72h: int, exam_hours: Optional[float], back2back_sessions: int, credit_overload: int, 
                     work_hours_week: int, commute_minutes_day: int, sleep_debt_24h_v: float,) -> float:
    """
    This function computes the academic context stress score based on deadlines, exam hours, back-to-back sessions, 
    credit overload, work hours, commute time, and sleep debt.

    Args:
        deadlines_72h (int): Number of deadlines in the next 72 hours.
        exam_hours (Optional[float]): Hours until the next exam.
        back2back_sessions (int): Number of back-to-back sessions.
        credit_overload (int): Credit overload.
        work_hours_week (int): Work hours per week.
        commute_minutes_day (int): Commute minutes per day.
        sleep_debt_24h_v (float): Sleep debt in the last 24 hours.

    Returns:
        float: Academic context stress score.
    """
    points = 0.0

    points += min(deadlines_72h * 6.0, 18.0)

    if exam_hours is not None:
        if exam_hours <= 48:
            points += 20.0
        elif exam_hours <= 96:
            points += 12.0

    points += min(max(0, back2back_sessions - 1) * 4.0, 12.0)
    points += sleep_debt_24h_v * 3.0
    points += min(credit_overload, 6.0)
    points += min(max(0, work_hours_week - 10), 8.0)
    points += min(commute_minutes_day // 20, 6.0)

    return clamp(points, 0.0, 50.0)
# --------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------------
# Overrides
def apply_overrides(ps: float, srs: float, alc: float, *, exam_hours_until: Optional[float], deadlines_72h: int,
    back_to_back_sessions: int, hrv_session: Optional[float], hrv_base: float, sleep_hours: Optional[float], ) -> Dict[str, Any]:
    """
    This function applies override rules to adjust the physiological stress (ps) and academic context (alc) scores
    based on specific conditions.

    Args:
        ps (float): Physiological stress score.
        srs (float): Self-reported stress score.
        alc (float): Academic context stress score.
        exam_hours_until (Optional[float]): Hours until the next exam.
        deadlines_72h (int): Number of deadlines in the next 72 hours.
        back_to_back_sessions (int): Number of back-to-back sessions.
        hrv_session (Optional[float]): Heart rate variability session value.
        hrv_base (float): Heart rate variability baseline value.
        sleep_hours (Optional[float]): Hours of sleep in the last 24 hours.

    Returns:
        Dict[str, Any]: A dictionary containing the adjusted physiological stress score, academic context stress score, and a list of applied overrides.
    """
    overrides = []

    if exam_hours_until is not None and exam_hours_until < 48 and srs >= 67:
        ps = max(ps, 80.0)
        overrides.append("exam<48h_and_high_SRS")

    if hrv_session is not None and hrv_session <= 0.6 * hrv_base:
        ps += 8.0
        overrides.append("hrv_collapse")

    if deadlines_72h >= 3 and back_to_back_sessions >= 2:
        alc += 8.0
        overrides.append("deadline_cluster")

    if sleep_hours is not None and sleep_hours < 5.0:
        ps += 5.0
        alc += 5.0
        overrides.append("severe_sleep_loss")

    return {
        "ps": clamp(ps, 0.0, 100.0),
        "alc": clamp(alc, 0.0, 50.0),
        "overrides": overrides,
    }


# Final Aggregation
def overall_stress(ps: float, srs: float, alc: float) -> float:
    """
    This function computes the overall stress score based on physiological stress, self-reported stress, and academic context stress.
    
    Args:
        ps (float): Physiological stress score.
        srs (float): Self-reported stress score.
        alc (float): Academic context stress score.

    Returns:
        float: Overall stress score.
    """
    return clamp(0.5 * ps + 0.25 * srs + 0.25 * alc, 0.0, 100.0)

# --------------------------------------------------------------------------------------------------------------------------


# Main Entry Point ---------------------------------------------------------------------------------------------------------
def education_compute_stress(
    *,
    # Baseline
    hr_base: float,
    hrv_base: float,
    steps_base: int,

    # Session
    hr_session: float,
    hrv_session: Optional[float],
    steps_session: int,
    sleep_last_24h: Optional[float],

    # ML
    model_stress_prob: Optional[float],

    # Questionnaires
    pre_sr: Optional[int],
    post_sr: Optional[int],
    weekly_sr: Optional[int],

    # Academic context
    deadlines_72h: int,
    exam_hours_until: Optional[float],
    back_to_back_sessions: int,
    credit_overload: int,
    work_hours_week: int,
    commute_minutes_day: int,
) -> Dict[str, Any]:

    # Normalization
    hr_d = hr_dev(hr_session, hr_base)
    hrv_d = hrv_dev(hrv_session, hrv_base)
    steps_r = steps_ratio(steps_session, steps_base)
    sleep_debt = sleep_debt_24h(sleep_last_24h)

    # Core scores
    ps = physiological_stress(hr_d, hrv_d, model_stress_prob, steps_r)
    srs = self_reported_stress(pre_sr, post_sr, weekly_sr)
    alc = academic_context(deadlines_72h, exam_hours_until, back_to_back_sessions, credit_overload, work_hours_week, commute_minutes_day, sleep_debt,)

    # Overrides
    override_out = apply_overrides(ps, srs, alc,
        exam_hours_until=exam_hours_until,
        deadlines_72h=deadlines_72h,
        back_to_back_sessions=back_to_back_sessions,
        hrv_session=hrv_session,
        hrv_base=hrv_base,
        sleep_hours=sleep_last_24h,
    )

    ps = override_out["ps"]
    alc = override_out["alc"]

    # Final
    total = overall_stress(ps, srs, alc)

    if total < 34:
        label = "low"
    elif total < 67:
        label = "moderate"
    else:
        label = "high"

    return {
        "physiological_stress": round(ps, 2),
        "self_reported_stress": round(srs, 2),
        "academic_load": round(alc, 2),
        "overall_stress": round(total, 2),
        "classification": label,
        "overrides": override_out["overrides"],
    }
    
def education_compute_confidence(*, hrv_session: Optional[float], hr_missing_ratio: float, model_used: bool, post_sr_present: bool, sleep_present: bool,) -> float:
    """
    This function computes a confidence score for the stress computation based on the availability of certain data points.

    Args:
        hrv_session (Optional[float]): HRV session value.
        hr_missing_ratio (float): Ratio of missing heart rate data.
        model_used (bool): Whether the model was used.
        post_sr_present (bool): Whether post-stress questionnaire is present.
        sleep_present (bool): Whether sleep data is present.

    Returns:
        float:  confidence score between 0.4 and 1.0
    """
    confidence = 1.0

    if hrv_session is None:
        confidence -= 0.25

    if hr_missing_ratio > 0.10:
        confidence -= 0.15

    if not model_used:
        confidence -= 0.10

    if not post_sr_present:
        confidence -= 0.10

    if not sleep_present:
        confidence -= 0.10

    return max(confidence, 0.4)

# --------------------------------------------------------------------------------------------------------------------------