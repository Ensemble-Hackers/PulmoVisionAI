---
title: PulmoVision AI Clinical Inference API
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# PulmoVision AI — Clinical Tri-Ensemble Backend

High-performance clinical inference API for multi-backbone lung cancer detection (Xception + EfficientNetV2-S + DenseNet121 CLAHE).
Serves:
- `GET  /api/health`
- `GET  /health`
- `POST /api/predict`
