import numpy as np
import pandas as pd
from datetime import timedelta

def compute_hrv_column_fast(data_pd: pd.DataFrame, step: int = 15, jitter_std: float = 5.0, use_jitter: bool = False) -> pd.Series:
    def rr_intervals(hr_segment):
        return np.array([60000 / hr for hr in hr_segment if hr > 0])

    def rr_intervals_with_jitter(hr_segment):
        base_rr = rr_intervals(hr_segment)
        jitter = np.random.normal(0, jitter_std, size=len(base_rr))
        return base_rr + jitter

    def hrv_rmssd(rr):
        rr_diffs = np.diff(rr)
        return round(np.sqrt(np.mean(rr_diffs ** 2)), 2) if len(rr_diffs) > 0 else np.nan

    # Ensure datetime is parsed
    data_pd = data_pd.copy()
    data_pd['datetime'] = pd.to_datetime(data_pd['datetime'])

    # Compute time gaps between consecutive rows
    time_deltas = data_pd['datetime'].diff().dt.total_seconds()
    time_deltas.iloc[0] = 1  # prevent NaN at start
    gap_mask = (time_deltas != 1).astype(int).cumsum()

    # Preallocate HRV column
    hrv_values = [np.nan] * len(data_pd)

    for _, group in data_pd.groupby(gap_mask):
        for i in range(0, len(group) - step + 1, step):
            chunk = group.iloc[i:i+step]
            if len(chunk) == step:
                hr_segment = chunk['HR'].tolist()
                rr = rr_intervals_with_jitter(hr_segment) if use_jitter else rr_intervals(hr_segment)
                rmssd = hrv_rmssd(rr)
                hrv_values[chunk.index[0]:chunk.index[0]+step] = [rmssd] * step

    return pd.Series(hrv_values, index=data_pd.index)
