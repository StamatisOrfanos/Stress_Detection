import numpy as np
from typing import List, Union, Optional, Dict

# Re-import necessary packages due to environment reset
import numpy as np
from typing import List, Optional, Union, Dict

def compute_hrv_from_heart_rate(heart_rates: List[float], windowed: bool = False, timestamps: Optional[List[str]] = None, 
                                use_jitter: bool = False, jitter_std: float = 20.0, upsample: bool = False, upsample_factor: int = 10
                                ) -> Union[Dict, List[Dict]]:

    def rr_intervals(hr_segment: List[float]) -> np.ndarray:
        """Convert HR (bpm) to RR intervals (ms)."""
        return np.array([60000 / hr for hr in hr_segment if hr > 0])

    def rr_intervals_with_jitter(hr_segment: List[float], jitter_std: float = 5.0) -> np.ndarray:
        """Estimate RR intervals and add random jitter (simulating beat-to-beat variation)."""
        base_rr = rr_intervals(hr_segment)
        jitter = np.random.normal(0, jitter_std, size=len(base_rr))
        return base_rr + jitter

    def upsample_hr_to_rr(hr_segment: List[float], factor: int = 4) -> np.ndarray:
        """Upsample HR data and compute RR intervals."""
        hr_array = np.array(hr_segment)
        x = np.arange(len(hr_array))
        x_upsampled = np.linspace(0, len(hr_array) - 1, len(hr_array) * factor)
        interpolated_hr = np.interp(x_upsampled, x, hr_array)
        return 60000 / interpolated_hr

    def hrv_metrics(rr: np.ndarray) -> Dict[str, float]:
        """Calculate RMSSD and SDNN from RR intervals."""
        rr_diffs = np.diff(rr)
        rmssd = np.sqrt(np.mean(rr_diffs ** 2)) if len(rr_diffs) > 0 else np.nan
        sdnn = np.std(rr, ddof=1) if len(rr) > 1 else np.nan
        return {"rmssd": round(rmssd, 2), "sdnn": round(sdnn, 2)}

    if not windowed:
        if upsample:
            rr = upsample_hr_to_rr(heart_rates, upsample_factor)
        elif use_jitter:
            rr = rr_intervals_with_jitter(heart_rates, jitter_std)
        else:
            rr = rr_intervals(heart_rates)
        result = hrv_metrics(rr)
        if timestamps:
            result["start_timestamp"] = timestamps[0]
        return result

    else:
        results = []
        for i in range(0, len(heart_rates), 15):
            segment = heart_rates[i:i + 15]
            if len(segment) == 15:
                if upsample:
                    rr = upsample_hr_to_rr(segment, upsample_factor)
                elif use_jitter:
                    rr = rr_intervals_with_jitter(segment, jitter_std)
                else:
                    rr = rr_intervals(segment)
                metrics = hrv_metrics(rr)
                metrics["window"] = f"{i}-{i+14}"
                if timestamps:
                    metrics["start_timestamp"] = timestamps[i]
                results.append(metrics)
        return results



# data = [75, 74, 73, 75, 76, 77, 76, 75, 74, 73, 72, 71, 72, 73, 74]
# timestamps = ["2023-10-01T00:00:00Z"] * len(data)
# result = compute_hrv_from_heart_rate(heart_rates=data, windowed=True, timestamps=timestamps, use_jitter=True, jitter_std=5.0, upsample=True, upsample_factor=4)
# print(result)
