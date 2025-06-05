from pymongo import MongoClient
from typing import Optional
from datetime import datetime, timedelta
import numpy as np


def compute_hrv_from_heart_rate(heart_rates, window_size: int = 30, step_size: int = 1, jitter_std: float = 5.0, use_jitter: bool = False,timestamps: Optional[list] = None):
    
    def rr_intervals(hr_segment):
        return np.array([60000 / hr for hr in hr_segment if hr > 0])

    def rr_intervals_with_jitter(hr_segment):
        base_rr = rr_intervals(hr_segment)
        jitter = np.random.normal(0, jitter_std, size=len(base_rr))
        return base_rr + jitter

    def hrv_metrics(rr):
        rr_diffs = np.diff(rr)
        rmssd = np.sqrt(np.mean(rr_diffs ** 2)) if len(rr_diffs) > 0 else np.nan
        sdnn = np.std(rr, ddof=1) if len(rr) > 1 else np.nan
        return {
            'rmssd': round(rmssd, 2),
            'sdnn': round(sdnn, 2)
        }

    results = []
    for i in range(0, len(heart_rates) - window_size + 1, step_size):
        segment = heart_rates[i:i + window_size]
        if len(segment) < window_size:
            continue
        rr = rr_intervals_with_jitter(segment) if use_jitter else rr_intervals(segment)
        metrics = hrv_metrics(rr)
        metrics['start_index'] = i
        if timestamps and i < len(timestamps):
            metrics['start_timestamp'] = timestamps[i]
        results.append(metrics)

    return results


def process_hrv_documents(mongo_uri='mongodb://localhost:27017', db_name='phr-db', collection_name='PHR', window_size=30):
    '''
    Process PHR documents in MongoDB to compute HRV from heart rate entries.
    Args:
        mongo_uri (str): MongoDB connection URI.
        db_name (str): Name of the database.
        collection_name (str): Name of the collection containing PHR documents.
        window_size (int): Size of the window for HRV computation in seconds.
    '''
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    phr_collection = db[collection_name]

    # Find all PHR documents with heart rate entries
    phr_docs = phr_collection.find({'entries.type': 'heartrate'})
    
    # Iterate through each document and compute HRV
    for doc in phr_docs:
        
        # Get user ID and effective data time to calculate timestamps
        user_id = doc.get('userId')
        effective_time = doc.get('effectiveDatetime')
        if not effective_time:
            continue

        updated = False
        entries = doc.get('entries', [])

        for entry in entries:
            # Process only heart rate entries
            if entry.get('type') != 'heartrate':
                continue

            signal = entry.get('signal', [])
            sampling_freq = entry.get('samplingFrequency', 1)

            if isinstance(signal, str):
                signal = eval(signal)

            if not signal or len(signal) < window_size:
                continue

            try:
                start_dt = effective_time if isinstance(effective_time, datetime) \
                    else datetime.fromisoformat(effective_time.replace('Z', '+00:00'))
            except Exception as e:
                print(f'[User {user_id}] Error parsing datetime: {e}', flush=True)
                continue

            timestamps = [(start_dt + timedelta(seconds=i / sampling_freq)).isoformat() for i in range(len(signal))]

            hrv_results = compute_hrv_from_heart_rate(signal, window_size=window_size, timestamps=timestamps)
            entry['hrv'] = hrv_results
            updated = True

            for result in hrv_results:
                ts = result.get('start_timestamp', 'N/A')
                print(f"  Timestamp: {ts}: RMSSD={result['rmssd']}, SDNN={result['sdnn']}", flush=True)

        if updated:
            phr_collection.update_one({'_id': doc['_id']}, {'$set': {'entries': entries}})

if __name__ == "__main__":
    process_hrv_documents()