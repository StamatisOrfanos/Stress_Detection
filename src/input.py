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
    
class AcademicPhysioStressInput(BaseModel):
    # Baseline
    hr_base: float
    hrv_base: float

    # Current session
    hr_session: float
    hrv_session: Optional[float]
    
class HealthcareStressInput(BaseModel):
    dataframe_split: Dict[str, Any]


class DataFrameInput(BaseModel):
    dataframe_split: dict
    questionnaire: Optional[int] = None