type HRVMetrics = { rmssd: number; sdnn: number; window?: string; start_timestamp?: string;};
  
  export function computeHRVFromHeartRate(heartRates: number[], windowed: boolean = false, timestamps?: string[], useJitter: boolean = false,
    jitterStd: number = 5.0, upsample: boolean = false, upsampleFactor: number = 4 ): HRVMetrics | HRVMetrics[] {

    const rrFromHR = (hr: number): number => 60000 / hr;
  
    const addJitter = (rrs: number[], std: number): number[] => {
      return rrs.map(rr => rr + randn() * std);
    };
  
    const randn = (): number => {
      // Box-Muller transform for normal distribution
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    };
  
    const upsampleHR = (hr: number[], factor: number): number[] => {
      const upsampled: number[] = [];
      const n = hr.length;
      for (let i = 0; i < n - 1; i++) {
        const step = (hr[i + 1] - hr[i]) / factor;
        for (let j = 0; j < factor; j++) {
          upsampled.push(hr[i] + step * j);
        }
      }
      upsampled.push(hr[n - 1]);
      return upsampled;
    };
  
    const computeRR = (hrSegment: number[]): number[] => {
      if (upsample) {
        const interpolated = upsampleHR(hrSegment, upsampleFactor);
        return interpolated.map(rrFromHR);
      } else if (useJitter) {
        const baseRR = hrSegment.map(rrFromHR);
        return addJitter(baseRR, jitterStd);
      } else {
        return hrSegment.map(rrFromHR);
      }
    };
  
    const computeMetrics = (rr: number[]): HRVMetrics => {
      if (rr.length < 2) return { rmssd: NaN, sdnn: NaN };
      const diffs = rr.slice(1).map((val, i) => val - rr[i]);
      const rmssd =
        Math.sqrt(diffs.reduce((sum, d) => sum + d * d, 0) / diffs.length);
      const mean = rr.reduce((sum, val) => sum + val, 0) / rr.length;
      const sdnn = Math.sqrt(
        rr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / (rr.length - 1)
      );
      return {
        rmssd: Math.round(rmssd * 100) / 100,
        sdnn: Math.round(sdnn * 100) / 100,
      };
    };
  
    if (!windowed) {
      const rr = computeRR(heartRates);
      const metrics = computeMetrics(rr);
      if (timestamps && timestamps.length > 0) {
        metrics.start_timestamp = timestamps[0];
      }
      return metrics;
    } else {
      const results: HRVMetrics[] = [];
      for (let i = 0; i <= heartRates.length - 15; i += 15) {
        const segment = heartRates.slice(i, i + 15);
        const rr = computeRR(segment);
        const metrics = computeMetrics(rr);
        metrics.window = `${i}-${i + 14}`;
        if (timestamps && timestamps.length > i) {
          metrics.start_timestamp = timestamps[i];
        }
        results.push(metrics);
      }
      return results;
    }
  }
  