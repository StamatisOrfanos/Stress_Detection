import numpy as np
from typing import List, Union, Optional, Dict

def compute_hrv_from_heart_rate(heart_rates: List[int], windowed: bool = False, timestamps: Optional[List[str]] = None) -> Union[Dict, List[Dict]]:
    
    def rr_intervals(hr_segment):
        return [60000 / hr for hr in hr_segment if hr > 0]

    def hrv_metrics(rr):
        rr_diffs = np.diff(rr)
        rmssd = np.sqrt(np.mean(rr_diffs ** 2))
        sdnn = np.std(rr, ddof=1)
        return {"rmssd": round(rmssd, 2), "sdnn": round(sdnn, 2)}

    if not windowed:
        rr = rr_intervals(heart_rates)
        result = hrv_metrics(rr)
        if timestamps:
            result["start_timestamp"] = timestamps[0]
        return result
    else:
        results = []
        for i in range(0, len(heart_rates), 15):
            segment = heart_rates[i:i+15]
            if len(segment) == 15:
                rr = rr_intervals(segment)
                metrics = hrv_metrics(rr)
                metrics["window"] = f"{i}-{i+14}"
                if timestamps:
                    metrics["start_timestamp"] = timestamps[i]
                results.append(metrics)
        return results


data = [75, 74, 73, 75, 76, 77, 76, 75, 74, 73, 72, 71, 72, 73, 74]
timestamps = ["2023-10-01T00:00:00Z"] * len(data)
result = compute_hrv_from_heart_rate(heart_rates=data, windowed=True, timestamps=timestamps)
print(result)
