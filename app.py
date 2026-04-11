# ================= IMPORTS =================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd
import google.generativeai as genai

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="LeafSentry AI Pro",
    page_icon="🌿",
    layout="wide"
)

# ================= UI =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0b0f1a, #050816);
    color: white;
}
[data-testid="stSidebar"] {
    background: #0a0f1c;
    border-right: 1px solid #1f2a44;
}
h1, h2, h3 {
    color: #7dd3fc !important;
}
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(125,211,252,0.25);
    border-radius: 16px;
    padding: 16px;
    backdrop-filter: blur(12px);
}
</style>
""", unsafe_allow_html=True)

# ================= GEMINI (PERMANENT FIX) =================
GEMINI_OK = False
model_ai = None
GEMINI_ERROR = None

def init_gemini():
    global GEMINI_OK, model_ai, GEMINI_ERROR

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)

        if not api_key:
            GEMINI_ERROR = "Missing GEMINI_API_KEY in secrets.toml"
            return

        genai.configure(api_key=api_key)

        model_ai = genai.GenerativeModel("gemini-1.5-flash")

        # 🔥 TEST CALL (IMPORTANT)
        test = model_ai.generate_content("Say OK")

        if not test or not hasattr(test, "text"):
            GEMINI_ERROR = "Gemini test call failed"
            return

        GEMINI_OK = True
        GEMINI_ERROR = None

    except Exception as e:
        GEMINI_ERROR = str(e)
        GEMINI_OK = False


init_gemini()

# ================= TITLE =================
st.title("🌿 LeafSentry AI Pro (Stable + Gemini Fixed)")
st.caption("Production-Ready Plant Disease Detection System")

# ================= CLASSES =================
classes = ["Diseased", "Healthy"]

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= MODEL =================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = CNN()

    try:
        state_dict = torch.load("cnn.pth", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as e:
        st.sidebar.error("❌ Model load failed")
        st.sidebar.warning(str(e))
        return None

model = load_model()

# ================= AI FALLBACK =================
def offline_ai(pred, conf):
    return f"""
🔍 Offline AI Report

Prediction: {classes[pred]}
Confidence: {conf:.2f}%

Advice:
- Monitor plant health
- Maintain irrigation
- Apply preventive care if diseased
"""

# ================= GEMINI FUNCTION (FIXED) =================
def gemini_advice(pred, conf):
    if not GEMINI_OK:
        return f"""
❌ GEMINI OFFLINE

Reason:
{GEMINI_ERROR}

--- Offline Mode Active ---
{offline_ai(pred, conf)}
"""

    try:
        prompt = f"""
You are an expert agricultural AI.

Plant status: {classes[pred]} ({conf:.2f}% confidence)

Provide:
1. Meaning
2. Cause
3. Treatment
4. Prevention
"""

        response = model_ai.generate_content(prompt)
        return response.text if response else "Empty response"

    except Exception as e:
        return f"❌ Gemini runtime error: {str(e)}"

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= UPLOAD =================
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= PREDICTION =================
    with st.spinner("Analyzing plant..."):

        if model is None:
            probs = np.array([0.5, 0.5])
        else:
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0].numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100

    st.session_state.history.append({
        "Result": classes[pred],
        "Confidence": round(conf, 2)
    })

    # ================= RESULT =================
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if classes[pred] == "Diseased":
            st.error("🚨 Diseased Plant")
        else:
            st.success("🌿 Healthy Plant")

        st.progress(int(conf))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x": "Class", "y": "Confidence"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= AI REPORT =================
    st.subheader("🧠 AI Diagnosis Report")
    st.write(gemini_advice(pred, conf))

# ================= HISTORY =================
st.divider()
st.subheader("📊 Prediction History")

if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("No predictions yet.")

# ================= SIDEBAR =================
st.sidebar.title("System Status")

if model is not None:
    st.sidebar.success("CNN Model Loaded")
else:
    st.sidebar.error("Model Not Loaded")

if GEMINI_OK:
    st.sidebar.success("Gemini AI ONLINE 🧠")
else:
    st.sidebar.error("Gemini OFFLINE ❌")
    st.sidebar.warning(GEMINI_ERROR)

st.sidebar.info("System: Stable Production Build")
