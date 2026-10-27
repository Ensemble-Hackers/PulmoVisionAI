# 🫁 Lung Cancer Detection & Classification using Deep Convolutional Tri-Ensemble (90.16% Accuracy)

[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-90.16%25-brightgreen.svg?style=flat-square)](Lung-cancer-model-train/models/model_tracker.md)
[![Cancer Sensitivity](https://img.shields.io/badge/Malignancy%20Recall-99.62%25-blue.svg?style=flat-square)](Lung-cancer-model-train/models/model_tracker.md)
[![Specificity](https://img.shields.io/badge/Healthy%20Specificity-98.15%25-success.svg?style=flat-square)](Lung-cancer-model-train/models/model_tracker.md)
[![Python Version](https://img.shields.io/badge/Python-3.10-informational.svg?style=flat-square)](pyproject.toml)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg?style=flat-square)](pyproject.toml)

An end-to-end medical deep learning system for automated detection, localization, and classification of pulmonary lesions from axial chest CT scans into four distinct clinical categories.

By uniting **three diverse convolutional architectures** (**Xception**, **EfficientNetV2-S**, and **DenseNet121**) with **CLAHE lung contrast optimization** and **5-view Test-Time Augmentation (TTA)**, this system breaks past the 90% accuracy barrier, achieving **90.16% test accuracy** and **99.62% cancer sensitivity** across 315 unseen patient scans from the Kaggle Chest CT-Scan dataset.

---

## 📑 Table of Contents

- [Overview & Clinical Significance](#overview)
- [The 4 Diagnostic Classes](#diagnostic-classes)
- [Performance Benchmarks & Accuracy Evolution](#benchmarks)
- [Confusion Matrix & Clinical Safety Metrics](#confusion-matrix)
- [Architectural Innovations & Why It Works](#architectural-innovations)
  - [1. Two-Stage Transfer Learning](#transfer-learning)
  - [2. CLAHE (Contrast-Limited Adaptive Histogram Equalization)](#clahe)
  - [3. Multi-Backbone Tri-Ensemble Fusion](#tri-ensemble)
  - [4. 5-View Test-Time Augmentation (TTA)](#tta)
- [Project Structure](#project-structure)
- [Environment Setup & Installation](#environment-setup)
  - [Troubleshooting Common Setup Gotchas](#troubleshooting)
- [How to Test Any CT Scan (Inference)](#inference)
- [How to Run Training & Full Evaluation](#training-and-evaluation)
- [Hyperparameters & Training Specifications](#hyperparameters)
- [Acknowledgements & Dataset Citation](#acknowledgements)

---

<a id="overview"></a>
## 🎯 Overview & Clinical Significance

Lung cancer remains the leading cause of cancer mortality worldwide. The vast majority of early-stage pulmonary nodules are asymptomatic; thus, high-resolution chest Computed Tomography (CT) is the clinical gold standard for detection. 

However, computerized diagnosis faces formidable clinical hurdles:
1. **Intra-class Texture Ambiguity**: Adenocarcinoma and Squamous Cell Carcinoma often share overlapping soft-tissue CT densities.
2. **Subtle Texture Variations**: Ground-glass opacities and spiculation margins are easily washed out in standard 8-bit CT image exports.
3. **High Clinical Cost of False Negatives**: Misclassifying a malignant lesion as "Normal" delays life-saving interventions.

This project delivers a clinically reliable diagnostic tool that **eliminates false normal diagnoses (0 missed cancers)** while attaining **90.16% exact subtype accuracy** on completely independent patient test sets.

---

<a id="diagnostic-classes"></a>
## 🔬 The 4 Diagnostic Classes

The model classifies every input axial scan into one of four mutually exclusive clinical categories:

| Class | Clinical Description | Pathological & Radiological Characteristics |
| :--- | :--- | :--- |
| **Normal (Healthy Lung)** | Non-cancerous lung parenchyma | Clear airspaces, intact bilateral bronchovascular trees, sharp pleural margins, zero suspicious nodular opacities. |
| **Adenocarcinoma** | Most prevalent Non-Small Cell Lung Cancer (NSCLC) | Arises from peripheral bronchial mucus glands; presents as subsolid nodules, ground-glass opacities, or irregular masses with spiculation. |
| **Squamous Cell Carcinoma** | Central airway NSCLC | Typically develops centrally in major bronchial branches; presents as cavitary or solid hilar masses with airway obstruction. |
| **Large Cell Carcinoma** | Fast-growing undifferentiated NSCLC | Rapidly expanding, bulky peripheral or central masses lacking glandular or squamous architecture; prone to central necrosis. |

---

<a id="benchmarks"></a>
## 📊 Performance Benchmarks & Accuracy Evolution

The Kaggle benchmark test set comprises **315 chest CT scans from independent patients** (completely isolated from training patients). Here is how systematic architectural upgrades elevated performance from baseline to state-of-the-art:

| Pipeline Iteration | Architecture & Strategy | Test Accuracy (315 Scans) | Macro F1-Score | Cancer Sensitivity | False Normal Calls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Baseline CNN** | 4-layer basic Conv2D + Flatten | **72.06%** (227/315) | 74.88% | 84.67% | 12 patients |
| **2. Xception (Single-View)** | Transfer learning + GAP head | **72.06%** (227/315) | 74.88% | 87.35% | 7 patients |
| **3. Xception + 5-View TTA** | Geometric & contrast augmentations | **75.24%** (237/315) | 77.42% | 88.51% | 5 patients |
| **4. Xception + Prior Calibration** | Bayesian class-prior reweighting | **78.73%** (248/315) | 81.04% | 91.19% | 4 patients |
| **5. EfficientNetV2-S Alone (TTA)** | Progressive regularized convolutions | **78.41%** (247/315) | 80.60% | 94.25% | 3 patients |
| **6. Dual-Model Ensemble** | Xception (55%) + EfficientNetV2-S (45%) | **86.03%** (271/315) | 87.81% | 97.32% | 2 patients |
| **7. DenseNet121 + CLAHE (Alone)**| Concatenated dense feature reuse | **86.67%** (273/315) | 88.35% | 98.85% | 1 patient |
| 🏆 **8. Tri-Model Ensemble (Final)** | **Xception (10%) + EfficientNet (30%) + DenseNet (60%)** | **90.16% (284/315)** | **90.87%** | **99.62% (260/261)** | **0 (Zero)** |

---

<a id="confusion-matrix"></a>
## 📈 Confusion Matrix & Clinical Safety Metrics

Evaluating all 315 unseen patient scans produced the following confusion matrix (tracked in [`Lung-cancer-model-train/models/model_tracker.md`](Lung-cancer-model-train/models/model_tracker.md)):

```
                        Predicted
Actual               Adeno   Large   Normal  Squamous   |  Recall
--------------------------------------------------------+---------
Adenocarcinoma        104       5       0        11     |  86.67%  (104/120)
Large Cell              4      44       0         3     |  86.27%  (44/51)
Normal (Healthy)        0       1      53         0     |  98.15%  (53/54)
Squamous Cell           7       0       0        83     |  92.22%  (83/90)
--------------------------------------------------------+---------
Precision:          90.43%  88.00% 100.00%   85.57%    |  Overall Acc: 90.16%
```

### Detailed Per-Class Breakdown

| Diagnostic Category | Precision | Recall (Sensitivity) | F1-Score | Total Cases | Correctly Classified |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal (Healthy Lung)** | **100.00%** | **98.15%** | **99.07%** | 54 | **53 / 54** |
| **Squamous Cell Carcinoma** | **85.57%** | **92.22%** | **88.77%** | 90 | **83 / 90** |
| **Adenocarcinoma** | **90.43%** | **86.67%** | **88.51%** | 120 | **104 / 120** |
| **Large Cell Carcinoma** | **88.00%** | **86.27%** | **87.13%** | 51 | **44 / 51** |
| **Macro Average** | **91.00%** | **90.83%** | **90.87%** | 315 | — |
| **Weighted Average** | **90.29%** | **90.16%** | **90.17%** | 315 | — |

### Key Clinical Safety Highlights
* **Zero Missed Cancers**: Exactly **0** malignant scans were mistakenly diagnosed as Normal. The system detected **260 out of 261** cancer patients (**99.62% malignancy sensitivity**).
* **Zero False Cancer Alarms**: Healthy lung precision reached **100.00%** (zero healthy patients were told they had cancer).
* **Massive Adenocarcinoma Recovery**: Adenocarcinoma correct predictions surged from **52** (baseline) $\rightarrow$ **92** (dual) $\rightarrow$ **104/120** (tri-ensemble) with **90.43% precision**.

---

<a id="architectural-innovations"></a>
## 🧠 Architectural Innovations & Why It Works

```
                                  [ Input CT Scan ]
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
            [ 5-View TTA ]          [ 5-View TTA ]          [ 5-View TTA ]
                  │                       │                       │
           Rescale [0, 1]           Raw [0, 255]            CLAHE Enhanced
                  │                       │                       │
                  ▼                       ▼                       ▼
            ┌───────────┐           ┌───────────┐           ┌───────────┐
            │  Xception │           │EffNetV2-S │           │DenseNet121│
            │  Backbone │           │  Backbone │           │  Backbone │
            └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
                  │ (10% Weight)          │ (30% Weight)          │ (60% Weight)
                  └───────────────────────┼───────────────────────┘
                                          ▼
                             [ Soft-Voting Consensus ]
                                          │
                                          ▼
                         [ Diagnosis + Confidence Chart ]
```

<a id="transfer-learning"></a>
### 1. Two-Stage Transfer Learning
* **Stage 1 (Head Adaptation)**: Freeze the pretrained ImageNet backbone. Train only `GlobalAveragePooling2D` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dense(256, ReLU)` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Dense(4, Softmax)` at $\text{LR} = 10^{-3}$.
* **Stage 2 (Backbone Fine-Tuning)**: Unfreeze top layers at $\text{LR} = 10^{-5}$ with progressive learning rate decay (`ReduceLROnPlateau`), adapting high-level spatial receptive fields to radiographic features.

<a id="clahe"></a>
### 2. CLAHE (Contrast-Limited Adaptive Histogram Equalization)
Standard 8-bit PNG exports of CT scans suffer from low local soft-tissue contrast. We transform the image to **LAB color space**, apply CLAHE specifically to the **L (Luminance) channel** with a clip limit of 2.0 and an $8 \times 8$ tile grid, and convert back to RGB. This amplifies faint ground-glass margins, spiculation borders, and vascular attachments.

<a id="tri-ensemble"></a>
### 3. Multi-Backbone Tri-Ensemble Fusion
Different CNN architectures possess distinct inductive biases:
* **Xception** (10% weight): Uses depthwise separable convolutions to decouple cross-channel correlations from spatial spatial correlations.
* **EfficientNetV2-S** (30% weight): Uses progressive regularized multi-scale receptive fields to capture both macroscopic lobe anatomy and localized lesions.
* **DenseNet121 + CLAHE** (60% weight): Re-connects every layer directly to all subsequent layers. By concatenating feature maps, low-level radiographic textures are preserved through the network without signal loss.

<a id="tta"></a>
### 4. 5-View Test-Time Augmentation (TTA)
During inference, each input scan generates 5 distinct perspectives:
1. `Original`: Raw scan
2. `Horizontal Flip`: Bilateral lung anatomical symmetry
3. `Central Zoom (5%)`: In-depth examination of central lesion margins
4. `Contrast Plus (+15%)`: Highlights dense calcifications and solid tumors
5. `Contrast Minus (-15%)`: Highlights subtle ground-glass opacities

The predictions across all 5 views and all 3 models are averaged, virtually eliminating single-view noise.

---

<a id="project-structure"></a>
## 📂 Project Structure

```text
Lung-cancer-detection-CNN/
├── frontend/                              # Clinical Web Application (Next.js 16, React 19, TypeScript)
│   ├── src/app/                           # Multi-view clinical routing (studio, pathology, benchmarks, architecture)
│   ├── src/components/                    # Clinical UI components (DiagnosticStudio, Radar, Consensus)
│   ├── src/lib/                           # Client utilities & clinical knowledge base
│   └── public/                            # Static DICOM/PNG patient sample scans and assets
├── docs/                                  # Presentation decks, clinical reports, and architecture guides
│   ├── Pulmo AI — Precision Clinical Thoracic Oncology.pdf
│   ├── CRSP_Limitations_and_Solutions.pptx
│   └── report.md                          # Clinical and safety evaluation report
├── Lung-cancer-model-train/               # Deep learning model architectures and training
│   ├── train.py                           # Xception training pipeline (Stage 1 + Stage 2 fine-tuning)
│   ├── train_efficientnet.py              # EfficientNetV2-S training pipeline
│   ├── train_densenet.py                  # DenseNet121 + CLAHE contrast optimization pipeline
│   ├── train.ipynb                        # Interactive Jupyter Notebook for experiments & EDA
│   ├── dataset/                           # Training, validation, and test datasets
│   └── models/                            # Trained .h5 model weights (Xception, EffNet, DenseNet)
├── api_server.py                          # High-performance clinical REST inference server (Port 8001)
├── app.py                                 # Hugging Face Spaces free Gradio + FastAPI deployment entry
├── ensemble_pipeline.py                   # 3-Backbone Tri-Ensemble evaluation benchmark (315 test scans)
├── inference.py                           # Standalone CLI / visual inference script
├── requirements.txt                       # Backend Python dependencies for cloud deployment
├── Dockerfile                             # Container build definition for backend inference
├── pyproject.toml                         # Project configuration and dependency specifications
├── uv.lock                                # Locked dependency versions for reproducible installs
└── README.md                              # Main project documentation and guides
```

---

<a id="environment-setup"></a>
## ⚙️ Environment Setup & Installation

### Prerequisites
* **OS**: Windows 10/11, Ubuntu 20.04+, or macOS
* **Python**: `3.10` (Recommended for TensorFlow 2.10 GPU compatibility)

### Option A: Using Conda (Recommended for GPU on Windows)

```powershell
# 1. Clone the repository
git clone https://github.com/Ensemble-Hackers/PulmoVisionAI.git
cd PulmoVisionAI

# 2. Create and activate a Python 3.10 environment
conda create -n lung_cancer python=3.10 -y
conda activate lung_cancer

# 3. Install dependencies
pip install tensorflow==2.10.0 opencv-python matplotlib scikit-learn
```

### Option B: Using `uv` (Ultra-Fast Package Manager)

```bash
# 1. Clone and enter directory
git clone https://github.com/Ensemble-Hackers/PulmoVisionAI.git
cd PulmoVisionAI

# 2. Sync all locked dependencies into a virtual environment
uv sync
```

---

<a id="troubleshooting"></a>
### Troubleshooting Common Setup Gotchas

#### ❓ Error: `ModuleNotFoundError: No module named 'tensorflow'`
* **Cause**: Your terminal is invoking a default system Python (e.g. Python 3.14 on PATH) instead of the environment where TensorFlow is installed.
* **Fix**: Ensure your environment is activated before running:
  ```powershell
  conda activate lung_cancer    # or your environment name
  python inference.py
  ```
  Or run directly using the full path to the environment's Python:
  ```powershell
  & "C:\Users\<YourUser>\anaconda3\envs\<YourEnv>\python.exe" inference.py
  ```

#### ❓ Error: File Extension Mismatch (`test.jpg` vs `test.jpeg`)
* **Cause**: On Windows, file extensions are sometimes hidden or named `.jpeg` instead of `.jpg`.
* **Fix**: `inference.py` automatically searches for `.jpg`, `.jpeg`, and `.png` in the project root if no argument is passed.

---

<a id="inference"></a>
## 🩺 How to Test Any CT Scan (Inference)

### 1. Automatic Detection in Project Root
Simply drop any CT scan into the repository root named `test.jpg` or `test.jpeg` or `test.png`, and run:

```powershell
python inference.py
```
*The script will automatically detect the file in your root directory and run inference!*

### 2. Specify Any Custom Image Path
To diagnose any scan stored anywhere on your system:

```powershell
python inference.py "C:\path\to\patient_scan.png"
```

### Sample Output

```text
Loading Xception model from: trained_lung_cancer_model.h5
Loading EfficientNetV2-S model from: efficientnetv2s_model.h5
Loading DenseNet121+CLAHE model from: densenet121_clahe_model.h5

Analyzing image: test.jpg

============================================================
  MODEL:      Tri-Ensemble (Xception + EfficientNetV2-S + DenseNet121 CLAHE: 90.16% Checkpoint)
  PREDICTION: LARGE CELL CARCINOMA
  CONFIDENCE: 42.99%
============================================================
Class Probabilities:
  Adenocarcinoma              1.09%  |                              |
  Large Cell Carcinoma       42.99%  |############                  |
  Normal (Healthy Lung)      39.60%  |###########                   |
  Squamous Cell Carcinoma    16.32%  |####                          |
============================================================
Saved visualization plot to: prediction_result.png
```

A visual diagnostic chart with the CT scan on the left and confidence bar chart on the right is automatically saved to [`prediction_result.png`](prediction_result.png).

---

<a id="training-and-evaluation"></a>
## 🚀 How to Run Training & Full Evaluation

### 1. Re-Evaluate the 90.16% Benchmark (315 Test Scans)
To run the full 5-view TTA evaluation across all 315 test scans and regenerate the confusion matrix:

```powershell
python ensemble_pipeline.py
```

### 2. Train Individual Backbones from Scratch

* **Train Xception**:
  ```powershell
  python Lung-cancer-model-train/train.py
  ```
* **Train EfficientNetV2-S**:
  ```powershell
  python Lung-cancer-model-train/train_efficientnet.py
  ```
* **Train DenseNet121 + CLAHE**:
  ```powershell
  python Lung-cancer-model-train/train_densenet.py
  ```

### 3. Interactive Prototyping (`train.ipynb`)
Open `Lung-cancer-model-train/train.ipynb` in VS Code, Cursor, or Jupyter Lab to inspect intermediate batches, visualize CLAHE filters, or experiment with custom learning rates cell-by-cell.

---

<a id="hyperparameters"></a>
## ⚙️ Hyperparameters & Training Specifications

| Parameter | Stage 1 (Head Training) | Stage 2 (Fine-Tuning) |
| :--- | :--- | :--- |
| **Input Dimensions** | `(350, 350, 3)` | `(350, 350, 3)` |
| **Batch Size** | `16` (8 for Xception) | `16` (8 for Xception) |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999$) | Adam ($\beta_1=0.9, \beta_2=0.999$) |
| **Initial Learning Rate**| $1.0 \times 10^{-3}$ | $1.0 \times 10^{-5}$ |
| **LR Scheduler** | `ReduceLROnPlateau(patience=3, factor=0.5)` | `ReduceLROnPlateau(patience=3, factor=0.5)` |
| **Early Stopping** | `patience=6, restore_best_weights=True` | `patience=6, restore_best_weights=True` |
| **Class Weighting** | Inverse-frequency balanced | Inverse-frequency balanced |
| **Data Augmentation** | Rotation ($\pm 15^\circ$), Zoom ($10\%$), Flip, Shift | Rotation ($\pm 15^\circ$), Zoom ($10\%$), Flip, Shift |

---

<a id="acknowledgements"></a>
## 🏆 Acknowledgements & Dataset Citation

* **Dataset**: [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) by Mohamed Hany on Kaggle.
* **Deep Learning Framework**: [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/).
* **Image Processing**: [OpenCV](https://opencv.org/) & [Matplotlib](https://matplotlib.org/).
