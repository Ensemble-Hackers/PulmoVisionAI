import { NextResponse } from "next/server";

export async function GET() {
  const pythonBase = (process.env.PYTHON_API_URL || "https://stellar-merge-pulmovision-api.hf.space").replace(/\/+$/, "");
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const res = await fetch(`${pythonBase}/api/health`, {
      signal: controller.signal
    });
    clearTimeout(timeout);
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json({
        ...data,
        status: "online",
        next_proxy: true,
        python_backend: true,
        backend_url: pythonBase
      });
    }
  } catch {
    // python server is offline
  }

  return NextResponse.json({
    status: "offline",
    system: "PulmoVision AI Next.js Edge Fallback Engine",
    version: "1.0.0",
    model_loaded: false,
    benchmark_accuracy: "90.16%",
    python_backend: false,
    backend_url: pythonBase
  });
}
