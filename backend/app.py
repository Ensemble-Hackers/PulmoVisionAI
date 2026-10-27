import os
import io
import sys
import base64
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Backward compatibility shim for modern huggingface_hub removing HfFolder
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class HfFolder:
            @classmethod
            def get_token(cls):
                return os.environ.get("HF_TOKEN")
            @classmethod
            def save_token(cls, token):
                pass
            @classmethod
            def delete_token(cls):
                pass
        huggingface_hub.HfFolder = HfFolder
except Exception:
    pass

# Monkey patch gradio_client bug where boolean schema crashes get_type
try:
    import gradio_client.utils as gcu
    _orig_get_type = gcu.get_type
    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return _orig_get_type(schema)
    gcu.get_type = _safe_get_type
except Exception:
    pass

# Monkey patch Starlette TemplateResponse for Gradio 4 compatibility with modern Starlette
try:
    import starlette.templating
    _orig_template_response = starlette.templating.Jinja2Templates.TemplateResponse

    def _safe_template_response(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name = args[0]
            context = args[1]
            request = context.get("request")
            return _orig_template_response(self, request=request, name=name, context=context)
        return _orig_template_response(self, *args, **kwargs)

    starlette.templating.Jinja2Templates.TemplateResponse = _safe_template_response
except Exception:
    pass

import gradio as gr

# Hugging Face ZeroGPU compatibility hook
try:
    import spaces
except ImportError:
    class _SpacesMock:
        @staticmethod
        def GPU(fn=None, duration=60):
            if fn is None:
                def dec(f):
                    return f
                return dec
            return fn
    spaces = _SpacesMock()

# Add backend directory to sys.path so imports work anywhere
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from backend.api_server import init_models, run_inference, LOADED_MODELS, CLASSES
except ImportError:
    from api_server import init_models, run_inference, LOADED_MODELS, CLASSES  # type: ignore[import-not-found]

# Initialize Tri-Ensemble CNN models from disk / Hugging Face hub
init_models()


# Gradio visual prediction function decorated with @spaces.GPU for HF ZeroGPU
@spaces.GPU
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
        "Probabilities": res["probabilities"],
    }


# Create Gradio UI Interface
demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="pil", label="Upload Thoracic CT Scan"),
    outputs=gr.JSON(label="Tri-Ensemble Analysis"),
    title="PulmoVision AI — Clinical Tri-Ensemble Backend",
    description="Live inference API serving Next.js Vercel frontend and standalone browser testing.",
)

# Launch Gradio interface (satisfies Hugging Face ZeroGPU startup hook)
port = int(os.environ.get("PORT", 7860))
app, _, _ = demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    prevent_thread_lock=True,
)

# Add Health Check endpoints for UptimeRobot, cron jobs, and Vercel
@app.get("/api/health")
@app.get("/health")
@app.get("/ping")
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


# Add Predict REST API endpoints for Vercel Next.js client
@app.post("/predict")
@app.post("/api-predict")
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


if __name__ == "__main__":
    demo.block_thread()

