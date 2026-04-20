import math
from typing import Optional


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def hr_dev(hr_session: float, hr_base: float) -> float:
    return clamp((hr_session - hr_base) / max(10.0, hr_base), -0.5, 0.5)


def hrv_dev(hrv_session: Optional[float], hrv_base: float) -> float:
    if hrv_session is None:
        return 0.0  # neutral if missing
    return clamp((hrv_base - hrv_session) / max(10.0, hrv_base), -0.5, 0.5)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def compute_stress_academic_physio(
    hr_base: float,
    hrv_base: float,
    hr_session: float,
    hrv_session: Optional[float],
) -> float:

    # --- deviations ---
    hr_d = hr_dev(hr_session, hr_base)
    hrv_d = hrv_dev(hrv_session, hrv_base)

    # --- weighted combination (HRV dominates) ---
    raw_score = 0.3 * hr_d + 0.7 * hrv_d

    # --- scale via sigmoid ---
    stress = sigmoid(raw_score * 5.0) * 100.0

    return round(stress, 2)