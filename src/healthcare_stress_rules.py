from typing import Optional, Dict, Any

from src.mathematical_ops import clamp, hr_dev, hrv_dev, steps_ratio

# Core Stress Components ----------------------------------------------------------------

def physiological_stress(hr_d: float, hrv_d: Optional[float], model_prob: Optional[float], steps_r: float,) -> float:
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
    activity_buffer = max(0.0, steps_r - 1.0) * 10.0

    if model_prob is not None:
        ps = 100.0 * clamp(model_prob, 0.0, 1.0)
    elif hrv_d is not None:
        ps = 100.0 * clamp(
            0.6 * hrv_d + 0.4 * hr_d + 0.1, 0.0, 1.0
        )
    else:
        ps = 100.0 * clamp(
            0.5 * hr_d + 0.15, 0.0, 1.0
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
    pre_shift = {0: 10, 1: 20, 2: 35}
    post_shift = {0: 15, 1: 30, 2: 55}
    weekly_map = {0: 5, 1: 10, 2: 20}
    
    if pre is None and post is None and weekly is None:
        return 20.0  # documented fallback

    pre_v = pre_shift.get(pre if pre is not None else -1, 0)
    post_v = post_shift.get(post if post is not None else -1, 0)
    weekly_v = weekly_map.get(weekly if weekly is not None else -1, 0)

    return clamp(
        0.5 * post_v + 0.35 * pre_v + 0.15 * weekly_v,
        0.0,
        100.0,
    )

def contextual_load(shift_type: str, pref_match: bool, consecutive_shifts: int, hours_since_last_shift: float, overtime_hours: float,) -> float:
    """
    This function computes the contextual load score based on shift type, preference match, consecutive shifts,
    hours since last shift, and overtime hours.

    Args:
        shift_type (str): Type of shift ("day", "evening", "night").
        pref_match (bool): Preference match for shift type.
        consecutive_shifts (int): Number of consecutive shifts.
        hours_since_last_shift (float): Hours since last shift.
        overtime_hours (float): Overtime hours.

    Returns:
        float: Contextual load score.
    """
    points = 0.0

    if not pref_match:
        points += 10.0

    if shift_type == "evening":
        points += 4.0
    elif shift_type == "night":
        points += 12.0

    if consecutive_shifts > 2:
        points += min((consecutive_shifts - 2) * 3.0, 15.0)

    if hours_since_last_shift < 12:
        points += 10.0
    elif hours_since_last_shift < 16:
        points += 6.0

    points += min(overtime_hours * 2.0, 10.0)

    return clamp(points, 0.0, 40.0)

# ------------------------------------------------------------------------------------------------


# Overrides ---------------------------------------------------------------------------------------
def apply_overrides(ps: float, srs: float, cls: float, *, post_sr: Optional[int], hrv_shift: Optional[float], 
                    hrv_base: float, shift_type: str, pref_match: bool, consecutive_shifts: int,) -> Dict[str, Any]:
    
    overrides = []

    if ps >= 70 and post_sr == 2:
        overrides.append("high_ps_and_high_post_sr")

    if hrv_shift is not None and hrv_shift <= 0.6 * hrv_base:
        ps += 8.0
        overrides.append("hrv_collapse")

    if shift_type == "night" and not pref_match and consecutive_shifts >= 4:
        cls += 10.0
        overrides.append("extended_night_mismatch")

    return {
        "ps": clamp(ps, 0.0, 100.0),
        "cls": clamp(cls, 0.0, 40.0),
        "overrides": overrides,
    }

def overall_stress(ps: float, srs: float, cls: float) -> float:
    return clamp(0.5 * ps + 0.3 * srs + 0.2 * cls, 0.0, 100.0)

# -----------------------------------------------------------------------------------------------------------------



# Confidence -------------------------------------------------------------------------------------------------------
def compute_confidence(*, hrv_present: bool, hr_missing_ratio: float, model_used: bool, post_sr_present: bool,) -> float:
    conf = 1.0

    if not hrv_present:
        conf -= 0.25
    if hr_missing_ratio > 0.10:
        conf -= 0.15
    if not model_used:
        conf -= 0.10
    if not post_sr_present:
        conf -= 0.10

    return max(conf, 0.4)

# -------------------------------------------------------------------------------------------------------


# Main Entry Point ---------------------------------------------------------------------------------------
def compute_stress_healthcare(
    *,
    # Baseline
    hr_base: float,
    hrv_base: float,
    steps_base: int,

    # Shift
    hr_shift: float,
    hrv_shift: Optional[float],
    steps_shift: int,

    # ML
    model_stress_prob: Optional[float],

    # Questionnaires
    pre_sr: Optional[int],
    post_sr: Optional[int],
    weekly_sr: Optional[int],

    # Context
    shift_type: str,
    pref_match: bool,
    consecutive_shifts: int,
    hours_since_last_shift: float,
    overtime_hours: float,
) -> Dict[str, Any]:

    hr_d = hr_dev(hr_shift, hr_base)
    hrv_d = hrv_dev(hrv_shift, hrv_base)
    steps_r = steps_ratio(steps_shift, steps_base)

    ps = physiological_stress(hr_d, hrv_d, model_stress_prob, steps_r)
    srs = self_reported_stress(pre_sr, post_sr, weekly_sr)
    cls = contextual_load(
        shift_type,
        pref_match,
        consecutive_shifts,
        hours_since_last_shift,
        overtime_hours,
    )

    override_out = apply_overrides(
        ps,
        srs,
        cls,
        post_sr=post_sr,
        hrv_shift=hrv_shift,
        hrv_base=hrv_base,
        shift_type=shift_type,
        pref_match=pref_match,
        consecutive_shifts=consecutive_shifts,
    )

    ps = override_out["ps"]
    cls = override_out["cls"]

    total = overall_stress(ps, srs, cls)

    label = "low" if total < 34 else "moderate" if total < 67 else "high"

    return {
        "physiological_stress": round(ps, 2),
        "self_reported_stress": round(srs, 2),
        "contextual_load": round(cls, 2),
        "overall_stress": round(total, 2),
        "classification": label,
        "overrides": override_out["overrides"],
    }
