import { MongoClient } from "mongodb";
import dayjs from "dayjs";

function computeHRVMetrics(rr: number[]): { rmssd: number; sdnn: number } {
  const diffs = rr.slice(1).map((val, i) => val - rr[i]);
  const rmssd = Math.sqrt(diffs.reduce((sum, d) => sum + d ** 2, 0) / diffs.length);
  const meanRR = rr.reduce((a, b) => a + b, 0) / rr.length;
  const sdnn = Math.sqrt(rr.reduce((sum, val) => sum + (val - meanRR) ** 2, 0) / rr.length);
  return {
    rmssd: Math.round(rmssd * 100) / 100,
    sdnn: Math.round(sdnn * 100) / 100,
  };
}

function bpmToRR(hr: number[]): number[] {
  return hr.map((h) => (h > 0 ? 60000 / h : 0));
}

async function processHRV() {
  const client = await MongoClient.connect("mongodb://localhost:27017");
  const db = client.db("phr-db");
  const collection = db.collection("PHR");

  const docs = await collection.find({ "entries.type": "heartrate" }).toArray();

  for (const doc of docs) {
    const effective = new Date(doc.effectiveDatetime);
    const entries = doc.entries;
    let updated = false;

    for (const entry of entries) {
      if (entry.type !== "heartrate") continue;

      const signal: number[] = entry.signal;
      const freq = entry.samplingFrequency || 1;

      if (!signal || signal.length < 30) continue;

      const timestamps = signal.map((_, i) =>
        dayjs(effective).add(i / freq, "second").toISOString()
      );

      const hrvResults: { rmssd: number; sdnn: number; start_index: number; start_timestamp: string }[] = [];
      for (let i = 0; i <= signal.length - 30; i++) {
        const segment = signal.slice(i, i + 30);
        const rr = bpmToRR(segment);
        const metrics = computeHRVMetrics(rr);
        hrvResults.push({ ...metrics, start_index: i, start_timestamp: timestamps[i], });
      }

      entry.hrv = hrvResults;
      updated = true;
    }

    if (updated) {
      await collection.updateOne({ _id: doc._id }, { $set: { entries } });
      console.log(`[User ${doc.userId}] Updated with HRV entries.`);
    }
  }

  await client.close();
}

processHRV();
