/**
 * PulmoVision AI - Next.js Resilient Inference Client
 */

import { PredictionResult, SampleScan } from "@/types/medical";

export async function requestPrediction(imageDataUrl: string, sample?: SampleScan | null): Promise<PredictionResult> {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_data: imageDataUrl,
      sample_id: sample ? sample.id : undefined
    })
  });

  if (!res.ok) {
    throw new Error(`Inference request failed with status ${res.status}`);
  }

  return await res.json();
}
