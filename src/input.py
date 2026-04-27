from typing import Any, Dict, Optional
from pydantic import BaseModel


class EducationComputeInput(BaseModel):
    dataframe_split: Dict[str, Any]
    
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