# ================= IMPORTS =================
from auth import auth_page, check_auth, logout
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="LeafSentry AI",
    page_icon="🌿",
    layout="wide"
)

if not check_auth():
    auth_page()
    st.stop()

# ================= UI (MUST BE FIRST) =================
st.title("🌿 LeafSentry AI")
st.caption("CNN Plant Disease Detection System")

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
}
</style>
""", unsafe_allow_html=True)

# ================= CLASSES =================
classes = ["Diseased", "Healthy"]

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= CNN MODEL =================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
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
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ================= MODEL LOAD =================
@st.cache_resource
def load_model():
    try:
        model = CNN()
        state = torch.load("cnn.pth", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        st.sidebar.error("❌ Model Load Failed")
        st.sidebar.warning(str(e))
        return None

model = load_model()

# ================= GEMINI =================
GEMINI_OK = False
gemini_model = None
GEMINI_ERROR = None

def init_gemini():
    global GEMINI_OK, gemini_model, GEMINI_ERROR

    if not GEMINI_AVAILABLE:
        GEMINI_ERROR = "Gemini library not installed"
        return

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)

        if not api_key:
            GEMINI_ERROR = "Missing API key"
            return

        genai.configure(api_key=api_key)

        # FIXED: only working models
        models_to_try = [
            "gemini-3.0-flash",
            "gemini-3.0-pro"
        ]

        for m in models_to_try:
            try:
                temp = genai.GenerativeModel(m)
                test = temp.generate_content("test")

                if test and hasattr(test, "text"):
                    gemini_model = temp
                    GEMINI_OK = True
                    return
            except:
                continue

        GEMINI_ERROR = "No Gemini model available"
        GEMINI_OK = False

    except Exception as e:
        GEMINI_ERROR = str(e)
        GEMINI_OK = False

init_gemini()

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= OFFLINE AI =================
def offline_ai(pred, conf):
    return f"""
🔍 Offline Report

Prediction: {classes[pred]}
Confidence: {conf:.2f}%

Advice:
- Monitor plant health
- Maintain proper watering
- Check leaves regularly
"""

# ================= AI RESPONSE =================
def gemini_advice(pred, conf):
    if not GEMINI_OK:
        return f"""
❌ GEMINI OFFLINE

Reason:
{GEMINI_ERROR}

--- Offline Mode ---
{offline_ai(pred, conf)}
"""

    try:
        prompt = f"""
You are a plant disease expert.

Plant status: {classes[pred]}
Confidence: {conf:.2f}%

Give:
- Meaning
- Cause
- Treatment
- Prevention
"""
        res = gemini_model.generate_content(prompt)
        return res.text
    except Exception as e:
        return f"Gemini error: {str(e)}"

# ================= UPLOAD =================
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

    # ================= PREDICT =================
    with st.spinner("Analyzing..."):

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
        st.subheader("Prediction Result")

        if classes[pred] == "Diseased":
            st.error("🚨 Diseased Plant")
        else:
            st.success("🌿 Healthy Plant")

        st.progress(int(conf))

        fig = px.bar(x=classes, y=probs * 100)
        st.plotly_chart(fig, use_container_width=True)

    # ================= AI =================
    st.subheader("🧠 AI Diagnosis Report")
    st.write(gemini_advice(pred, conf))

# ================= HISTORY =================
st.divider()
st.subheader("📊 History")

if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("No predictions yet.")

# ================= SIDEBAR =================
st.sidebar.title("System Status")

if model:
    st.sidebar.success("CNN Loaded")
else:
    st.sidebar.error("CNN Failed")

if GEMINI_OK:
    st.sidebar.success("Gemini ONLINE 🧠")
else:
    st.sidebar.error("Gemini OFFLINE ❌")
    st.sidebar.warning(GEMINI_ERROR)

st.sidebar.info("Stable Production Build")

st.sidebar.write(f"👤 {st.session_state.user}")
if st.sidebar.button("Logout"):
    logout()
