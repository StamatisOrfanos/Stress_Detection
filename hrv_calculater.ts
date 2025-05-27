type HRVMetrics = {
  rmssd: number;
  sdnn: number;
  second: number;
  start_timestamp?: string;
};

export function computeHRVPerSecond(
  heartRates: number[], windowSize: number = 15, useJitter: boolean = false, jitterStd: number = 5.0, timestamps?: string[]): HRVMetrics[] {
  
  const rrFromHR = (hr: number): number => 60000 / hr;

  const randn = (): number => {
    let u = 0,
      v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  const addJitter = (rrs: number[], std: number): number[] => {
    return rrs.map(rr => rr + randn() * std);
  };

  const computeRR = (hrSegment: number[]): number[] => {
    const baseRR = hrSegment.map(rrFromHR);
    return useJitter ? addJitter(baseRR, jitterStd) : baseRR;
  };

  const computeMetrics = (rr: number[]): HRVMetrics => {
    if (rr.length < 2) return { rmssd: NaN, sdnn: NaN, second: -1 };
    const diffs = rr.slice(1).map((val, i) => val - rr[i]);
    const rmssd = Math.sqrt(diffs.reduce((sum, d) => sum + d * d, 0) / diffs.length);
    const mean = rr.reduce((sum, val) => sum + val, 0) / rr.length;
    const sdnn = Math.sqrt(rr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (rr.length - 1));
    return {
      rmssd: Math.round(rmssd * 100) / 100,
      sdnn: Math.round(sdnn * 100) / 100,
      second: -1
    };
  };

  const results: HRVMetrics[] = [];
  for (let i = 0; i <= heartRates.length - windowSize; i++) {
    const segment = heartRates.slice(i, i + windowSize);
    const rr = computeRR(segment);
    const metrics = computeMetrics(rr);
    metrics.second = i;
    if (timestamps && timestamps.length > i) {
      metrics.start_timestamp = timestamps[i];
    }
    results.push(metrics);
  }

  return results;
}
