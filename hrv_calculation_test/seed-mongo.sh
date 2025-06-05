#!/bin/bash

mongoimport --host localhost --port 27017 --db phr-db --collection PHR --file phr_data.json --jsonArray
