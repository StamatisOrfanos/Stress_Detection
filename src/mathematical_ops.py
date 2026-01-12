from typing import Optional


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def hr_dev(hr_session: float, hr_base: float) -> float:
    return clamp((hr_session - hr_base) / max(10.0, hr_base), -0.5, 0.5)


def hrv_dev(hrv_session: Optional[float], hrv_base: float) -> Optional[float]:
    if hrv_session is None:
        return None
    return clamp((hrv_base - hrv_session) / max(10.0, hrv_base), -0.5, 0.5)


def steps_ratio(steps_session: int, steps_base: int) -> float:
    return clamp(steps_session / max(1000, steps_base), 0.5, 2.0)


def sleep_debt_24h(sleep_hours: Optional[float]) -> float:
    if sleep_hours is None:
        return 0.0
    return max(0.0, 7.5 - sleep_hours)
