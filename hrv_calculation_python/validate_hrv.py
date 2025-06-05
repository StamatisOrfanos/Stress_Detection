from pymongo import MongoClient


def validate_hrv_fields(mongo_uri="mongodb://localhost:27017", db_name="phr-db", collection_name="PHR"):
    '''
    Validate the HRV fields in PHR documents in MongoDB.
    Args:
        mongo_uri (str): MongoDB connection URI.
        db_name (str): Name of the database.
        collection_name (str): Name of the collection containing PHR documents.
    '''
    # Connect to MongoDB and validate HRV fields
    required_keys = {"rmssd", "sdnn", "start_index", "start_timestamp"}
    client = MongoClient(mongo_uri)
    db = client[db_name]
    phr_collection = db[collection_name]

    docs_checked = 0
    valid_entries = 0
    invalid_entries = 0

    phr_docs = phr_collection.find({"entries.type": "heartrate"})

    for doc in phr_docs:
        entries = doc.get("entries", [])
        for entry in entries:
            if entry.get("type") != "heartrate":
                continue

            hrv = entry.get("hrv")
            docs_checked += 1

            if not isinstance(hrv, list):
                print(f"[DOC {doc['_id']}]'hrv' field is missing or not a list.", flush=True)
                invalid_entries += 1
                continue

            for i, hrv_item in enumerate(hrv):
                if not isinstance(hrv_item, dict):
                    print(f"[DOC {doc['_id']}] hrv[{i}] is not a dictionary.", flush=True)
                    invalid_entries += 1
                    break
                missing = required_keys - hrv_item.keys()
                if missing:
                    print(f"[DOC {doc['_id']}] hrv[{i}] missing fields: {missing}", flush=True)
                    invalid_entries += 1
                    break
            else:
                valid_entries += 1

    print("\n HRV Field Validation Summary:", flush=True)
    print(f"Documents checked: {docs_checked}", flush=True)
    print(f"Valid entries    : {valid_entries}", flush=True)
    print(f"Invalid entries  : {invalid_entries}", flush=True)

if __name__ == "__main__":
    validate_hrv_fields()