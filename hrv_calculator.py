import numpy as np
from typing import List, Optional, Union, Dict

def compute_hrv_per_second(heart_rates: List[float], window_size: int = 15, jitter_std: float = 5.0, 
                           use_jitter: bool = False, timestamps: Optional[List[str]] = None) -> List[Dict]:

    def rr_intervals(hr_segment: List[float]) -> np.ndarray:
        return np.array([60000 / hr for hr in hr_segment if hr > 0])

    def rr_intervals_with_jitter(hr_segment: List[float], jitter_std: float = 5.0) -> np.ndarray:
        base_rr = rr_intervals(hr_segment)
        jitter = np.random.normal(0, jitter_std, size=len(base_rr))
        return base_rr + jitter

    def hrv_metrics(rr: np.ndarray) -> Dict[str, float]:
        rr_diffs = np.diff(rr)
        rmssd = np.sqrt(np.mean(rr_diffs ** 2)) if len(rr_diffs) > 0 else np.nan
        sdnn = np.std(rr, ddof=1) if len(rr) > 1 else np.nan
        return {
            "rmssd": round(rmssd, 2),
            "sdnn": round(sdnn, 2)
        }

    results = []
    for i in range(0, len(heart_rates) - window_size + 1):
        segment = heart_rates[i:i + window_size]
        if use_jitter:
            rr = rr_intervals_with_jitter(segment, jitter_std)
        else:
            rr = rr_intervals(segment)
        metrics = hrv_metrics(rr)
        metrics["second"] = i
        if timestamps:
            metrics["start_timestamp"] = timestamps[i]
        results.append(metrics)

    return results




heart_rates = [75 + np.random.randn() for _ in range(120)]
results = compute_hrv_per_second(heart_rates, window_size=15, use_jitter=True)

# Print first few results
for r in results[:50]:
    print(r)

