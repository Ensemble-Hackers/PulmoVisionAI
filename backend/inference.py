"""
Single-Image Inference Script for Lung Cancer Detection.
Classifies any input chest CT scan using trained model weights.
Supports:
  - Tri-Model Ensemble: Xception + EfficientNetV2-S + DenseNet121 (with CLAHE lung contrast optimization)
  - Dual Ensemble fallback
  - Standalone model fallback

Usage:
    python inference.py "path/to/scan.png"
    or simply:
    python inference.py
"""

import sys
import os
import argparse
from pathlib import Path

# Suppress verbose TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    Model = tf.keras.Model
    BatchNormalization = tf.keras.layers.BatchNormalization
    Dense = tf.keras.layers.Dense
    Dropout = tf.keras.layers.Dropout
    GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
    Input = tf.keras.layers.Input
    Xception = tf.keras.applications.Xception
    EfficientNetV2S = tf.keras.applications.EfficientNetV2S
    DenseNet121 = tf.keras.applications.DenseNet121
    densenet_preprocess_input = tf.keras.applications.densenet.preprocess_input
except (ImportError, AttributeError):
    import keras
    Model = keras.Model
    BatchNormalization = keras.layers.BatchNormalization
    Dense = keras.layers.Dense
    Dropout = keras.layers.Dropout
    GlobalAveragePooling2D = keras.layers.GlobalAveragePooling2D
    Input = keras.layers.Input
    Xception = keras.applications.Xception
    EfficientNetV2S = keras.applications.EfficientNetV2S
    DenseNet121 = keras.applications.DenseNet121
    densenet_preprocess_input = keras.applications.densenet.preprocess_input

# Access image preprocessing safely across Keras / TF versions
image = tf.keras.preprocessing.image

IMAGE_SIZE = (350, 350)
CLASSES = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal (Healthy Lung)",
    "Squamous Cell Carcinoma",
]

REPO_ROOT = Path(__file__).resolve().parent
if REPO_ROOT.name == "backend":
    REPO_ROOT = REPO_ROOT.parent

MODELS_DIR = Path(
    os.environ.get(
        "MODELS_DIR",
        (Path(__file__).resolve().parent / "models")
        if (Path(__file__).resolve().parent / "models").exists()
        else (REPO_ROOT / "Lung-cancer-model-train" / "models")
    )
)
XCEPTION_MODEL_PATH = MODELS_DIR / "trained_lung_cancer_model.h5"
EFFICIENTNET_MODEL_PATH = MODELS_DIR / "efficientnetv2s_model.h5"
DENSENET_MODEL_PATH = MODELS_DIR / "densenet121_clahe_model.h5"

XCEPTION_WEIGHTS = MODELS_DIR / "best_model.weights.h5"
EFFICIENTNET_WEIGHTS = MODELS_DIR / "efficientnetv2s_best.weights.h5"
DENSENET_WEIGHTS = MODELS_DIR / "densenet121_clahe_best.weights.h5"


def build_head(backbone_output, num_classes=len(CLASSES)):
    x = GlobalAveragePooling2D()(backbone_output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    return Dense(num_classes, activation="softmax")(x)


def load_single_model(model_path, weights_fallback_path, backbone_fn, inputs, name):
    target = None
    if model_path.exists():
        target = model_path
    elif weights_fallback_path and weights_fallback_path.exists():
        target = weights_fallback_path

    if target is None:
        return None

    print(f"Loading {name} model from: {target.name}")
    try:
        return tf.keras.models.load_model(str(target), compile=False)
    except Exception:
        bb = backbone_fn(weights=None, include_top=False, input_tensor=inputs)
        out = build_head(bb.output)
        m = Model(inputs=inputs, outputs=out)
        m.load_weights(str(target))
        return m


def load_ensemble():
    """Loads available models (Xception, EfficientNetV2-S, DenseNet121+CLAHE)."""
    models = {}
    inputs = Input(shape=(*IMAGE_SIZE, 3))

    m_xc = load_single_model(XCEPTION_MODEL_PATH, XCEPTION_WEIGHTS, Xception, inputs, "Xception")
    if m_xc is not None:
        models["xception"] = m_xc

    m_eff = load_single_model(EFFICIENTNET_MODEL_PATH, EFFICIENTNET_WEIGHTS, EfficientNetV2S, inputs, "EfficientNetV2-S")
    if m_eff is not None:
        models["efficientnet"] = m_eff

    m_dense = load_single_model(DENSENET_MODEL_PATH, DENSENET_WEIGHTS, DenseNet121, inputs, "DenseNet121+CLAHE")
    if m_dense is not None:
        models["densenet"] = m_dense

    if not models:
        print("\n[Error] No trained models found in models directory!")
        sys.exit(1)

    return models


def apply_clahe(img_array):
    """Applies CLAHE on luminance channel and DenseNet ImageNet normalization."""
    u8 = np.clip(img_array, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype("float32")
    return densenet_preprocess_input(enhanced_rgb)


def predict(models, image_path: Path):
    """Preprocesses input CT scan, runs multi-model ensemble with 5-view TTA, and visualizes."""
    if not image_path.exists():
        print(f"\n[Error] Image file does not exist: {image_path}")
        sys.exit(1)

    print(f"\nAnalyzing image: {image_path.resolve()}")

    # 1. Load raw image [0, 255]
    raw_img = image.load_img(str(image_path), target_size=IMAGE_SIZE)
    img_array = image.img_to_array(raw_img, dtype="float32")

    # 2. Generate 5 TTA views
    h, w = IMAGE_SIZE
    crop_h, crop_w = int(h * 0.90), int(w * 0.90)
    sy, sx = (h - crop_h) // 2, (w - crop_w) // 2

    v1 = img_array
    v2 = np.fliplr(img_array)
    v3 = tf.image.resize(img_array[sy : sy + crop_h, sx : sx + crop_w], (h, w)).numpy()
    mean = np.mean(img_array, axis=(0, 1), keepdims=True)
    v4 = np.clip((img_array - mean) * 1.15 + mean, 0.0, 255.0)
    v5 = np.clip((img_array - mean) * 0.85 + mean, 0.0, 255.0)
    batch_raw = np.array([v1, v2, v3, v4, v5], dtype="float32")

    has_xc = "xception" in models
    has_eff = "efficientnet" in models
    has_dense = "densenet" in models

    # 3. Model Predictions with TTA
    if has_xc and has_eff and has_dense:
        preds_xc = models["xception"](batch_raw / 255.0, training=False).numpy()
        preds_eff = models["efficientnet"](batch_raw, training=False).numpy()
        batch_dense = np.array([apply_clahe(v) for v in batch_raw], dtype="float32")
        preds_dense = models["densenet"](batch_dense, training=False).numpy()

        avg_xc = np.mean(preds_xc, axis=0)
        avg_eff = np.mean(preds_eff, axis=0)
        avg_dense = np.mean(preds_dense, axis=0)

        # Optimal 90.16% Tri-Ensemble weights (10% Xception + 30% EfficientNetV2-S + 60% DenseNet121 CLAHE)
        combined = 0.10 * avg_xc + 0.30 * avg_eff + 0.60 * avg_dense
        predictions = combined / np.sum(combined)
        model_name = "Tri-Ensemble (Xception + EfficientNetV2-S + DenseNet121 CLAHE: 90.16% Checkpoint)"

    elif has_xc and has_eff:
        preds_xc = models["xception"](batch_raw / 255.0, training=False).numpy()
        preds_eff = models["efficientnet"](batch_raw, training=False).numpy()
        avg_xc = np.mean(preds_xc, axis=0)
        avg_eff = np.mean(preds_eff, axis=0)
        combined = 0.55 * avg_xc + 0.45 * avg_eff
        combined[0] *= 1.40
        predictions = combined / np.sum(combined)
        model_name = "Dual Ensemble (Xception + EfficientNetV2-S)"

    elif has_dense:
        batch_dense = np.array([apply_clahe(v) for v in batch_raw], dtype="float32")
        preds = models["densenet"](batch_dense, training=False).numpy()
        avg = np.mean(preds, axis=0)
        predictions = avg / np.sum(avg)
        model_name = "DenseNet121 + CLAHE"

    elif has_eff:
        preds = models["efficientnet"](batch_raw, training=False).numpy()
        avg = np.mean(preds, axis=0)
        predictions = avg / np.sum(avg)
        model_name = "EfficientNetV2-S"

    else:
        preds = models["xception"](batch_raw / 255.0, training=False).numpy()
        avg = np.mean(preds, axis=0)
        avg[0] *= 2.20
        predictions = avg / np.sum(avg)
        model_name = "Xception (Calibrated)"

    top_idx = int(np.argmax(predictions))
    top_class = CLASSES[top_idx]
    top_confidence = float(predictions[top_idx]) * 100.0

    # 4. Print report to terminal
    print("\n" + "=" * 60)
    print(f"  MODEL:      {model_name}")
    print(f"  PREDICTION: {top_class.upper()}")
    print(f"  CONFIDENCE: {top_confidence:.2f}%")
    print("=" * 60)
    print("Class Probabilities:")
    for label, prob in zip(CLASSES, predictions):
        bar = "#" * int(prob * 30)
        print(f"  {label:<25} {prob * 100:6.2f}%  |{bar:<30}|")
    print("=" * 60 + "\n")

    # 5. Save and display visual chart
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    ax_img.imshow(raw_img)
    ax_img.set_title(f"Input Scan\nPredicted: {top_class} ({top_confidence:.1f}%)", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    colors = ["#3498db" if i != top_idx else "#e74c3c" for i in range(len(CLASSES))]
    y_pos = np.arange(len(CLASSES))
    ax_bar.barh(y_pos, predictions * 100, color=colors)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(CLASSES, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Confidence (%)", fontsize=11)
    ax_bar.set_xlim(0, 100)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.6)

    for i, p in enumerate(predictions):
        ax_bar.text(p * 100 + 1, i, f"{p * 100:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_plot_path = REPO_ROOT / "prediction_result.png"
    plt.savefig(output_plot_path, dpi=150)
    print(f"Saved visualization plot to: {output_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predict lung cancer diagnosis from a CT scan image.")
    parser.add_argument("image_path", nargs="?", help="Path to chest CT scan image file (PNG/JPG)")
    args = parser.parse_args()

    target_path_str = args.image_path
    if not target_path_str:
        print("=" * 60)
        print(" Lung Cancer CT Scan Prediction Tool (Tri-Ensemble Ready)")
        print("=" * 60)
        if not target_path_str:
            # Check for test image in root directory
            for default_name in ["test.jpg", "test.png", "test.jpeg"]:
                root_candidate = REPO_ROOT / default_name
                if root_candidate.exists():
                    target_path_str = str(root_candidate)
                    print(f"Automatically detected root image: {target_path_str}")
                    break

            if not target_path_str:
                sample_dir = REPO_ROOT / "Lung-cancer-model-train" / "dataset" / "test"
                sample_candidates = list(sample_dir.glob("*/*.png")) + list(sample_dir.glob("*/*.jpg"))
                if sample_candidates:
                    target_path_str = str(sample_candidates[0])
                    print(f"No path entered. Falling back to test sample: {target_path_str}")
                else:
                    print("No image path provided. Exiting.")
                    sys.exit(1)

    image_path = Path(target_path_str)
    models = load_ensemble()
    predict(models, image_path)


if __name__ == "__main__":
    main()
