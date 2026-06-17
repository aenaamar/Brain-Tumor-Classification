# Brain Tumor MRI Classification Pipeline

A multi-model deep learning and machine learning benchmark pipeline designed to classify Brain MRI scans into four pathological classes: `glioma`, `meningioma`, `notumor`, and `pituitary`.

This repository implements an empirical progression from classical statistical baselines up to specialized Deep Convolutional Neural Networks (CNNs) and State-of-the-Art (SOTA) Pre-trained Vision Transformers/Architectures utilizing transfer learning, learning rate schedulers, and inference latency profiling. A clinical-grade **Streamlit web interface** (`app.py`) provides real-time multi-model inference with Grad-CAM explainability.

---

## Architectural Overview & Pipeline Components

The pipeline is modularly structured across **six execution files**, designed to isolate experimentation frameworks:

### 1. Data Preprocessing & Validation Split
* **`split_validation.py`**: A deterministic split utility that segregates the `Testing` partition 50/50 to generate a dedicated `Validation` set dynamically. This ensures uncorrupted evaluation boundaries during deep learning hyperparameter optimization.

### 2. Baseline Architecture
* **`logistic_regression.py`**: A classical machine learning baseline. Images are scaled down to $128 \times 128$ pixels and flattened into a 1D vector of size $128 \times 128 \times 3 = 49,152$ features before fitting via a multi-class Logistic Regression estimator configured with `max_iter=500`.

### 3. Custom Deep Learning Archetype
* **`CNN.py`**: A custom 3-layer sequential Convolutional Neural Network built with PyTorch.
  * Features three staggered convolutional blocks (`32 -> 64 -> 128` channels) utilizing $3\times3$ kernels, padding, ReLU activations, and Max Pooling.
  * Fully connected head featuring an explicit structural calculation `128 * (IMG_SIZE // 8) * (IMG_SIZE // 8)` feeding into a `512` neuron hidden layer regulated by a `0.5` dropout coefficient.

### 4. Advanced SOTA Transfer Learning Frameworks
* **`resnet.py`**: A fine-tuning framework leveraging a pre-trained **ResNet50** backbone. The fully connected output layer is replaced with a custom dropout classification block (`Dropout(p=0.2)`) and multi-class Linear layer optimized using Adam ($LR = 5 \times 10^{-5}$).
* **`efficientnet.py`**: A high-efficiency, fast-inference deep architecture leveraging a pre-trained **EfficientNet-B0** classification engine. Outfitted with custom dropout regulations and early stopping mechanisms tailored for minimal hardware footprints.

### 5. NeuroScan AI — Streamlit Web Interface
* **`app.py`**: A clinical-grade interactive frontend for real-time multi-model inference and explainability.
  * Loads all four trained models simultaneously (Custom CNN, EfficientNet-B0, ResNet50, Logistic Regression) from configurable weight paths via the sidebar.
  * Accepts JPEG/PNG MRI uploads and runs parallel inference, displaying per-model predictions, confidence scores, and latency in a dark clinical UI.
  * **Grad-CAM explainability** overlays are generated for all three deep learning models, highlighting the spatial regions driving each prediction — targeting `conv3` for the Custom CNN, `features[-1]` for EfficientNet-B0, and `layer4[-1]` for ResNet50.
  * **Model Consensus** view aggregates all active model predictions and surfaces the majority vote alongside agreement count.
  * Probability comparison bar chart visualizes per-class confidence distributions across all active models simultaneously.
  * Includes a medical disclaimer footer enforcing the tool's decision-support-only scope.

---

## 📊 Deep Learning Training Configurations

| Parameter | Baseline Custom CNN | ResNet50 Transfer | EfficientNet-B0 Transfer |
| :--- | :--- | :--- | :--- |
| **Input Image Size** | $128 \times 128$ | $224 \times 224$ | $224 \times 224$ |
| **Batch Size** | 32 | 32 | 32 |
| **Max Epochs** | 10 | 40 | 40 |
| **Optimization Algorithm** | Adam ($LR = 1 \times 10^{-3}$) | Adam ($LR = 5 \times 10^{-5}$) | Adam ($LR = 5 \times 10^{-5}$) |
| **Weight Decay** | None | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **LR Scheduler** | None | `ReduceLROnPlateau` (patience=3, factor=0.5) | `ReduceLROnPlateau` (patience=3, factor=0.5) |
| **Early Stopping** | None | Patience = 10 Epochs | Patience = 5 Epochs |
| **Data Augmentation** | Resizing, Normalization | Horiz. Flip, Random Rotation ($15^{\circ}$) | Horiz. Flip, Random Rotation ($15^{\circ}$) |

---

## 🖥️ NeuroScan AI — Interface Overview

### Supported Models in UI

| Model | Weight File | Input Size | Grad-CAM Target |
| :--- | :--- | :--- | :--- |
| Custom CNN | `CNN outpput/custom_cnn.pth` | $128 \times 128$ | `conv3` |
| EfficientNet-B0 | `efficientnet_FINAL_outputs/efficientnet_best.pth` | $224 \times 224$ | `features[-1]` |
| ResNet50 | `resnet_final_outputs (1)/resnet50_best.pth` | $224 \times 224$ | `layer4[-1]` |
| Logistic Regression | `logistic_output/logistic_model.joblib` | $128 \times 128$ (flattened) | N/A |

### Launching the App

```bash
streamlit run app.py
```

The sidebar allows:
* Overriding default model weight paths via text inputs.
* Toggling individual models on/off before running inference.

### App Dependencies

The following additional packages are required for `app.py` beyond the core pipeline:

```
streamlit
opencv-python
```

Ensure these are included in `requirements.txt` or install manually:

```bash
pip install streamlit opencv-python
```

---

## Evaluation Protocol & Target Performance Metrics

To achieve operational viability, the high-capacity models target explicit performance benchmarks across unseen test environments:

1. **Overall Classification Accuracy**: Must exceed **90.00%** on the holdout test distribution.
2. **Macro F1-Score**: Must exceed **0.9000** to verify performance stability against class imbalance.
3. **Class-Specific Sensitivity (Recall)**: Must exceed **90.00%** per specific tumor variant (crucial for medical diagnostics to minimize false negatives).
4. **Inference Latency Profile**: Averaged latency per single scan must remain **< 2.0 seconds** under a sequential batch size of 1.

---

## Environment Verification & Execution Flow

### Pipeline Setup

Install dependencies directly inside your local environment or virtual shell:

```bash
pip install -r requirements.txt
```

### Execution Order

```
1. split_validation.py       # Generate validation split
2. logistic_regression.py    # Train baseline
3. CNN.py                    # Train custom CNN
4. resnet.py                 # Fine-tune ResNet50
5. efficientnet.py           # Fine-tune EfficientNet-B0
6. streamlit run app.py      # Launch NeuroScan AI interface
```

---

## ⚕️ Medical Disclaimer

NeuroScan AI is a **research and decision-support tool only**. Predictions produced by this system are **not clinical diagnoses** and must always be reviewed and confirmed by a licensed radiologist or qualified medical professional before any clinical action is taken. This system does not replace professional medical judgment.
