import { MongoClient } from "mongodb";

async function validateHRV() {
  const client = await MongoClient.connect("mongodb://localhost:27017");
  const db = client.db("phr-db");
  const collection = db.collection("PHR");

  const requiredKeys = ["rmssd", "sdnn", "start_index", "start_timestamp"];
  const docs = await collection.find({ "entries.type": "heartrate" }).toArray();

  let checked = 0;
  let valid = 0;
  let invalid = 0;

  for (const doc of docs) {
    for (const entry of doc.entries) {
      if (entry.type !== "heartrate") continue;
      checked++;

      const hrv = entry.hrv;
      if (!Array.isArray(hrv)) {
        console.error(`[DOC ${doc._id}] No HRV array.`);
        invalid++;
        continue;
      }

      const first = hrv[0];
      if (!first || !requiredKeys.every((k) => k in first)) {
        console.error(`[DOC ${doc._id}] HRV entry missing keys.`);
        invalid++;
        continue;
      }

      valid++;
    }
  }

  console.log("\n HRV Field Validation Summary:");
  console.log(`Documents checked: ${checked}`);
  console.log(`Valid entries    : ${valid}`);
  console.log(`Invalid entries  : ${invalid}`);

  await client.close();
}

validateHRV();
