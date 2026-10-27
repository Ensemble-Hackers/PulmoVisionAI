import os
import io
import base64
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
import uvicorn

from api_server import init_models, run_inference, LOADED_MODELS, CLASSES

# Initialize Tri-Ensemble CNN models
init_models()

# Create FastAPI server
app = FastAPI(title="PulmoVision AI API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "system": "PulmoVision AI Tri-Ensemble Clinical Inference Engine",
        "version": "2.0.0",
        "models_loaded": {
            "xception": "xception" in LOADED_MODELS,
            "efficientnet": "efficientnet" in LOADED_MODELS,
            "densenet": "densenet" in LOADED_MODELS,
        },
        "total_backbones_active": len(LOADED_MODELS),
        "benchmark_accuracy": "90.16%",
        "cancer_sensitivity": "99.62%",
        "classes": CLASSES,
    }

@app.post("/api/predict")
async def predict_endpoint(req: Request):
    try:
        data = await req.json()
        image_raw = data.get("image") or data.get("image_data")
        if not image_raw:
            return JSONResponse({"error": "No image payload provided"}, status_code=400)

        if "," in image_raw:
            image_raw = image_raw.split(",", 1)[1]

        img_bytes = base64.b64decode(image_raw.strip())
        result = run_inference(img_bytes)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Gradio Interactive Interface for Direct Hugging Face UI
def gradio_predict(img):
    if img is None:
        return {"error": "Please provide an image"}
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    res = run_inference(buffered.getvalue())
    return {
        "Prediction": res["prediction"],
        "Confidence": f"{res['confidence'] * 100:.2f}%",
        "Consensus Details": res["consensus"],
        "Probabilities": res["probabilities"]
    }

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Thoracic CT Scan"),
    outputs=gr.JSON(label="Tri-Ensemble Analysis"),
    title="PulmoVision AI — Clinical Tri-Ensemble Backend",
    description="Live inference API serving Next.js Vercel frontend and standalone browser testing."
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
