#!/bin/bash

set -e  
npx tsc --noEmit

TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOG_FILE="pipeline_${TIMESTAMP}.log"

echo "Logging output to $LOG_FILE"
echo "--------------------- HRV Pipeline Run: $(date) ---------------------" > "$LOG_FILE"

exec > >(tee -a "$LOG_FILE") 2>&1  # Log both stdout and stderr to file and terminal

echo "Starting MongoDB with Docker Compose..."
docker-compose up -d

echo "Waiting for MongoDB to become available..."
sleep 5

echo "Importing mock phr_data.json into MongoDB..."
docker cp phr_data.json phr-mongo:/phr_data.json
docker exec phr-mongo mongoimport \
  --db phr-db \
  --collection PHR \
  --file /phr_data.json \
  --jsonArray

echo "Running HRV computation..."
npx ts-node src/compute_hrv.ts

echo "Validating stored HRV entries..."
npx ts-node src/validate_hrv.ts

echo "Pipeline complete!"