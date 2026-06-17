import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import os
import time
import cv2

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — clean clinical dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Root */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0f1e;
    color: #e2e8f0;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; max-width: 1300px; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0d1b3e 0%, #112240 60%, #0a192f 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-icon { font-size: 3rem; line-height: 1; }
.hero-title { font-size: 2rem; font-weight: 700; color: #64ffda; letter-spacing: -0.5px; margin: 0; }
.hero-sub { font-size: 0.9rem; color: #8892b0; margin: 0.3rem 0 0 0; font-weight: 400; }
.hero-badge {
    margin-left: auto;
    background: rgba(100,255,218,0.08);
    border: 1px solid #64ffda44;
    color: #64ffda;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    white-space: nowrap;
}

/* Cards */
.card {
    background: #0d1b3e;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    color: #64ffda;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Model selector pills */
.stMultiSelect > div > div {
    background: #112240 !important;
    border-color: #1e3a5f !important;
}

/* Result block */
.result-block {
    background: linear-gradient(135deg, #0d1b3e, #0a192f);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.model-name {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #8892b0;
    margin-bottom: 0.3rem;
}
.prediction-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.confidence-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #64ffda;
}

/* Tumor color coding */
.tumor-glioma { color: #ff6b6b; }
.tumor-meningioma { color: #ffa94d; }
.tumor-pituitary { color: #74c0fc; }
.tumor-notumor { color: #51cf66; }

/* Progress bars (confidence) */
.conf-bar-wrap { margin: 0.5rem 0; }
.conf-label { font-size: 0.75rem; color: #8892b0; display: flex; justify-content: space-between; margin-bottom: 3px; }
.conf-bar-bg { background: #1e3a5f; border-radius: 4px; height: 6px; overflow: hidden; }
.conf-bar-fill { height: 6px; border-radius: 4px; transition: width 0.4s ease; }

/* Disclaimer */
.disclaimer {
    background: rgba(255,107,107,0.06);
    border: 1px solid rgba(255,107,107,0.25);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.78rem;
    color: #ff8787;
    line-height: 1.5;
    margin-top: 1.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Metric tiles */
.metric-tile {
    background: #0d1b3e;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #64ffda;
}
.metric-label { font-size: 0.7rem; color: #8892b0; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.2rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0d1b3e; border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #8892b0; border-radius: 6px; padding: 0.5rem 1rem; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #112240 !important; color: #64ffda !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #0d1b3e;
    border: 1.5px dashed #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #64ffda22, #64ffda11);
    border: 1px solid #64ffda55;
    color: #64ffda;
    font-weight: 600;
    font-size: 0.85rem;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    width: 100%;
    letter-spacing: 0.03em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #64ffda33, #64ffda22);
    border-color: #64ffda;
    color: #fff;
}

/* Spinner */
.stSpinner > div { border-top-color: #64ffda !important; }

div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor"
}
CLASS_CSS = {
    "glioma": "tumor-glioma",
    "meningioma": "tumor-meningioma",
    "notumor": "tumor-notumor",
    "pituitary": "tumor-pituitary"
}
CLASS_COLOR = {
    "glioma": "#ff6b6b",
    "meningioma": "#ffa94d",
    "notumor": "#51cf66",
    "pituitary": "#74c0fc"
}
BAR_COLOR = {
    "glioma": "#ff6b6b",
    "meningioma": "#ffa94d",
    "notumor": "#51cf66",
    "pituitary": "#74c0fc"
}

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────
IMG_SIZE_CNN = 128
IMG_SIZE_TL  = 224

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.fc1   = nn.Linear(128 * (IMG_SIZE_CNN // 8) * (IMG_SIZE_CNN // 8), 512)
        self.fc2   = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def build_efficientnet(num_classes=4, dropout=0.2):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def build_resnet(num_classes=4, dropout=0.2):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
cnn_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_CNN, IMG_SIZE_CNN)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

tl_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_TL, IMG_SIZE_TL)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_custom_cnn(path):
    model = CustomCNN(num_classes=4)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_efficientnet(path):
    model = build_efficientnet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_resnet(path):
    model = build_resnet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_logistic(path):
    return joblib.load(path)

# ─────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────
def predict_torch(model, tensor):
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(tensor.unsqueeze(0))
        latency = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs, latency

def predict_logistic(model, pil_img):
    img = pil_img.resize((128, 128)).convert("RGB")
    arr = np.array(img).flatten().reshape(1, -1)
    t0 = time.perf_counter()
    probs = model.predict_proba(arr)[0]
    latency = (time.perf_counter() - t0) * 1000
    pred_idx = int(np.argmax(probs))
    # sklearn classes may be 0–3 mapping to CLASS_NAMES order
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs, latency

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.zero_grad()
        output = self.model(tensor.unsqueeze(0))
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_gradcam(pil_img, cam, size):
    img_np = np.array(pil_img.resize((size, size)).convert("RGB"))
    cam_resized = cv2.resize(cam, (size, size))
    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.55 * img_np + 0.45 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


def resolve_model_path(path):
    """Try the supplied path, then fallback to known model output folders."""
    if os.path.exists(path):
        return path

    base = os.path.basename(path)
    alt_dirs = [
        "CNN outpput",
        "efficientnet_FINAL_outputs",
        "resnet_final_outputs (1)",
        "logistic_output",
        "../ml_project/CNN outpput",
        "../ml_project/efficientnet_FINAL_outputs",
        "../ml_project/resnet_final_outputs (1)",
        "../ml_project/logistic_output",
        ".",
    ]

    for directory in alt_dirs:
        candidate = os.path.join(directory, base)
        if os.path.exists(candidate):
            return candidate

    return path

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="card-title">⚙️ Model Weights</div>', unsafe_allow_html=True)

    cnn_path    = st.text_input("Custom CNN (.pth)",       value="CNN outpput/custom_cnn.pth", key="cnn")
    eff_path    = st.text_input("EfficientNet-B0 (.pth)",  value="efficientnet_FINAL_outputs/efficientnet_best.pth", key="eff")
    res_path    = st.text_input("ResNet50 (.pth)",         value="resnet_final_outputs (1)/resnet50_best.pth", key="res")
    log_path    = st.text_input("Logistic Regression (.joblib)", value="logistic_output/logistic_model.joblib", key="log")

    st.markdown("---")
    st.markdown('<div class="card-title">🔬 Active Models</div>', unsafe_allow_html=True)

    use_cnn = st.checkbox("Custom CNN",           value=True)
    use_eff = st.checkbox("EfficientNet-B0",      value=True)
    use_res = st.checkbox("ResNet50",             value=True)
    use_log = st.checkbox("Logistic Regression",  value=True)

    st.markdown("---")
    st.markdown('<div class="card-title">🧬 Class Legend</div>', unsafe_allow_html=True)
    for cn, disp in CLASS_DISPLAY.items():
        color = CLASS_COLOR[cn]
        st.markdown(f'<span style="color:{color}; font-size:0.85rem;">● {disp}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.7rem; color:#4a5568; line-height:1.6;">'
        'NeuroScan AI · ITU CS Project<br>'
        'Brain Tumor MRI Classification<br>'
        'Models: CNN · EfficientNet · ResNet · LR'
        '</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-icon">🧠</div>
  <div>
    <p class="hero-title">NeuroScan AI</p>
    <p class="hero-sub">Multi-Model Brain Tumor Classification · MRI Analysis · Grad-CAM Explainability</p>
  </div>
  <div class="hero-badge">⚕ Decision Support Tool</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="card-title">📤 Upload MRI Scan</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    run_btn = st.button("▶  Run Analysis", disabled=(uploaded is None))

with col_preview:
    st.markdown('<div class="card-title">🖼 Scan Preview</div>', unsafe_allow_html=True)
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, use_container_width=True, caption=f"{uploaded.name}  ·  {pil_img.size[0]}×{pil_img.size[1]}px")
    else:
        st.markdown(
            '<div style="background:#0d1b3e; border:1px dashed #1e3a5f; border-radius:8px; height:200px; '
            'display:flex; align-items:center; justify-content:center; color:#4a5568; font-size:0.85rem;">'
            'No scan uploaded yet</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────────
if run_btn and uploaded:
    pil_img = Image.open(uploaded).convert("RGB")

    # Load selected models
    models_to_run = {}
    load_errors   = {}

    if use_cnn:
        try:
            models_to_run["Custom CNN"] = ("cnn", load_custom_cnn(resolve_model_path(cnn_path)))
        except Exception as e:
            load_errors["Custom CNN"] = str(e)

    if use_eff:
        try:
            models_to_run["EfficientNet-B0"] = ("eff", load_efficientnet(resolve_model_path(eff_path)))
        except Exception as e:
            load_errors["EfficientNet-B0"] = str(e)

    if use_res:
        try:
            models_to_run["ResNet50"] = ("res", load_resnet(resolve_model_path(res_path)))
        except Exception as e:
            load_errors["ResNet50"] = str(e)

    if use_log:
        try:
            models_to_run["Logistic Regression"] = ("log", load_logistic(resolve_model_path(log_path)))
        except Exception as e:
            load_errors["Logistic Regression"] = str(e)

    for name, err in load_errors.items():
        st.error(f"**{name}** — failed to load: {err}")

    if not models_to_run:
        st.warning("No models loaded. Check file paths in the sidebar.")
        st.stop()

    results = {}

    with st.spinner("Running inference across all models…"):
        for name, (mtype, model) in models_to_run.items():
            if mtype == "cnn":
                tensor = cnn_transform(pil_img)
                pred, conf, probs, lat = predict_torch(model, tensor)
            elif mtype == "eff":
                tensor = tl_transform(pil_img)
                pred, conf, probs, lat = predict_torch(model, tensor)
            elif mtype == "res":
                tensor = tl_transform(pil_img)
                pred, conf, probs, lat = predict_torch(model, tensor)
            elif mtype == "log":
                pred, conf, probs, lat = predict_logistic(model, pil_img)
                tensor = None

            results[name] = {
                "mtype": mtype,
                "model": model,
                "tensor": tensor if mtype != "log" else None,
                "pred": pred,
                "conf": conf,
                "probs": probs,
                "lat": lat
            }

    # ─── SUMMARY METRICS ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Summary</div>', unsafe_allow_html=True)

    n = len(results)
    mcols = st.columns(n)
    preds_list = [r["pred"] for r in results.values()]
    consensus  = max(set(preds_list), key=preds_list.count)

    for i, (name, r) in enumerate(results.items()):
        with mcols[i]:
            color = CLASS_COLOR[r["pred"]]
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-value" style="color:{color};">{CLASS_DISPLAY[r['pred']]}</div>
                <div class="metric-label">{name}</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#64ffda; margin-top:4px;">{r['conf']*100:.1f}% conf</div>
                <div style="font-size:0.68rem; color:#4a5568; margin-top:2px;">{r['lat']:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)

    # Consensus
    cons_color = CLASS_COLOR[consensus]
    agree = sum(1 for p in preds_list if p == consensus)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {cons_color}15, {cons_color}08);
                border: 1px solid {cons_color}44; border-radius: 10px;
                padding: 1rem 1.5rem; margin: 1rem 0; display:flex; align-items:center; gap:1rem;">
        <div style="font-size:2rem;">🏥</div>
        <div>
            <div style="font-size:0.68rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#8892b0;">Model Consensus</div>
            <div style="font-size:1.4rem; font-weight:700; color:{cons_color};">{CLASS_DISPLAY[consensus]}</div>
            <div style="font-size:0.78rem; color:#8892b0;">{agree}/{len(preds_list)} models agree</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── DETAILED RESULTS TABS ─────────────────────────
    st.markdown('<div class="card-title">🔍 Per-Model Details</div>', unsafe_allow_html=True)
    tabs = st.tabs(list(results.keys()))

    for tab, (name, r) in zip(tabs, results.items()):
        with tab:
            col_res, col_cam = st.columns([1, 1], gap="large")

            with col_res:
                pred_color = CLASS_COLOR[r["pred"]]
                st.markdown(f"""
                <div class="result-block">
                    <div class="model-name">{name}</div>
                    <div class="prediction-label" style="color:{pred_color};">{CLASS_DISPLAY[r['pred']]}</div>
                    <div class="confidence-score">Confidence: {r['conf']*100:.2f}%</div>
                    <div style="font-size:0.75rem; color:#4a5568; margin-top:4px;">Latency: {r['lat']:.2f} ms</div>
                </div>
                """, unsafe_allow_html=True)

                # Prob bars
                st.markdown('<div style="margin-top:1rem;">', unsafe_allow_html=True)
                for idx, cn in enumerate(CLASS_NAMES):
                    p = float(r["probs"][idx]) * 100
                    bar_color = BAR_COLOR[cn]
                    st.markdown(f"""
                    <div class="conf-bar-wrap">
                        <div class="conf-label">
                            <span>{CLASS_DISPLAY[cn]}</span>
                            <span style="font-family:'JetBrains Mono',monospace;">{p:.1f}%</span>
                        </div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{p:.1f}%; background:{bar_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_cam:
                if r["mtype"] in ("cnn", "eff", "res") and r["tensor"] is not None:
                    st.markdown('<div style="font-size:0.7rem; color:#8892b0; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem;">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
                    try:
                        model_obj = r["model"]
                        tensor    = r["tensor"]
                        pred_idx  = CLASS_NAMES.index(r["pred"])

                        # Pick target conv layer per model
                        if r["mtype"] == "cnn":
                            target_layer = model_obj.conv3
                            cam_size = IMG_SIZE_CNN
                        elif r["mtype"] == "eff":
                            target_layer = model_obj.features[-1]
                            cam_size = IMG_SIZE_TL
                        else:  # resnet
                            target_layer = model_obj.layer4[-1]
                            cam_size = IMG_SIZE_TL

                        # Need grad so temporarily enable
                        tensor_g = tensor.clone().requires_grad_(False)
                        gcam = GradCAM(model_obj, target_layer)
                        cam_map = gcam.generate(tensor_g, pred_idx)
                        overlay = overlay_gradcam(pil_img, cam_map, cam_size)
                        st.image(overlay, use_container_width=True, caption="Highlighted tumor region")

                        # Side-by-side original
                        st.image(pil_img.resize((cam_size, cam_size)), use_container_width=True, caption="Original MRI")
                    except Exception as e:
                        st.warning(f"Grad-CAM unavailable: {e}")
                else:
                    st.info("Grad-CAM is not available for Logistic Regression (no convolutional layers).")

    # ─── PROBABILITY CHART ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">📈 Probability Comparison Across Models</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1b3e")
    ax.set_facecolor("#0d1b3e")

    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / max(len(results), 1)
    colors_models = ["#64ffda", "#74c0fc", "#ffa94d", "#ff6b6b"]

    for i, (name, r) in enumerate(results.items()):
        bars = ax.bar(
            x + i * width - (len(results) - 1) * width / 2,
            r["probs"] * 100,
            width * 0.85,
            label=name,
            color=colors_models[i % len(colors_models)],
            alpha=0.85,
            edgecolor="none"
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_DISPLAY[c] for c in CLASS_NAMES], color="#8892b0", fontsize=9)
    ax.set_ylabel("Probability (%)", color="#8892b0", fontsize=9)
    ax.set_ylim(0, 110)
    ax.tick_params(colors="#8892b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    ax.yaxis.grid(True, color="#1e3a5f", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    legend = ax.legend(fontsize=8, facecolor="#0a0f1e", edgecolor="#1e3a5f", labelcolor="#8892b0")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor="#0d1b3e")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

    # ─── DISCLAIMER ────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> NeuroScan AI is a research and decision-support tool only.
        Predictions are <strong>not clinical diagnoses</strong> and must always be reviewed and confirmed by a
        licensed radiologist or qualified medical professional before any clinical action is taken.
        This system does not replace professional medical judgment.
    </div>
    """, unsafe_allow_html=True)

elif not uploaded:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color:#4a5568;">
        <div style="font-size:3rem; margin-bottom:1rem;">🩻</div>
        <div style="font-size:1rem; color:#8892b0;">Upload an MRI scan to begin analysis</div>
        <div style="font-size:0.8rem; margin-top:0.5rem;">Supports JPEG and PNG formats</div>
    </div>
    """, unsafe_allow_html=True)
