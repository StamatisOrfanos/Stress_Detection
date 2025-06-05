#!/bin/bash

docker cp phr_data.json phr-mongo:/phr_data.json

docker exec -it phr-mongo mongoimport \
  --db phr-db \
  --collection PHR \
  --file /phr_data.json \
  --jsonArray
