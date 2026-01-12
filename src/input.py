from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel


class EducationComputeInput(BaseModel):
    dataframe_split: Dict[str, Any]

    # Baseline
    hr_base: float
    hrv_base: float
    steps_base: int

    # Session
    hr_session: float
    hrv_session: Optional[float]
    steps_session: int
    sleep_last_24h: Optional[float]

    # Questionnaires
    pre_sr: Optional[int]
    post_sr: Optional[int]
    weekly_sr: Optional[int]

    # Academic context
    deadlines_72h: int = 0
    exam_hours_until: Optional[float]
    back_to_back_sessions: int = 0
    credit_overload: int = 0
    work_hours_week: int = 0
    commute_minutes_day: int = 0
    
class HealthcareStressInput(BaseModel):
    # --- ML input ---
    dataframe_split: Dict[str, Any]

    # --- Baseline (per nurse) ---
    hr_base: float
    hrv_base: float
    steps_base: int

    # --- Shift observations ---
    hr_shift: float
    hrv_shift: Optional[float] = None
    steps_shift: int

    # --- Questionnaires ---
    pre_sr: Optional[int] = None    # 0,1,2
    post_sr: Optional[int] = None   # 0,1,2
    weekly_sr: Optional[int] = None # 0,1,2

    # --- Scheduling / context ---
    shift_type: Literal["day", "evening", "night"]
    pref_match: bool
    consecutive_shifts: int
    hours_since_last_shift: float
    overtime_hours: float = 0.0


class DataFrameInput(BaseModel):
    dataframe_split: dict
    questionnaire: Optional[int] = None