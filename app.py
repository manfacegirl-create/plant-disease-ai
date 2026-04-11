# ================= IMPORTS =================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px
import google.generativeai as genai

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="wide"
)

# ================= CLEAN WHITE UI (FIXED CONTRAST) =================
st.markdown("""
<style>

/* APP BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #111111;
}

/* HEADER */
[data-testid="stHeader"] {
    background-color: #ffffff;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #f5f7fa;
}

/* GLOBAL TEXT */
html, body, [class*="css"] {
    color: #111111;
    font-family: Arial;
}

/* MAIN CONTAINER */
.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* CARD STYLE */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 15px;
    color: #111111;
}

/* TITLES */
h1, h2, h3 {
    color: #111111;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 10px;
}

/* BUTTONS */
.stButton>button {
    background-color: #16a34a;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# ================= GEMINI SETUP =================
GEMINI_OK = False
model_ai = None

try:
    api_key = st.secrets.get("GEMINI_API_KEY", None)

    if api_key:
        genai.configure(api_key=api_key)
        model_ai = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_OK = True

except Exception as e:
    st.warning(f"Gemini setup error: {e}")

# ================= TITLE =================
st.title("🌿 Plant Disease Detection AI")
st.caption("Clean UI + CNN Model + Gemini AI Assistant")

# ================= CLASSES =================
classes = ["🍂 Diseased", "🌱 Healthy"]

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
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================= GEMINI FUNCTION =================
def gemini_interpretation(pred_class, confidence):

    if not GEMINI_OK or model_ai is None:
        return "⚠️ Gemini is not connected. Please check API key."

    prompt = f"""
You are an agricultural AI expert.

Plant result:
- Class: {classes[pred_class]}
- Confidence: {confidence:.2f}%

Explain:
1. Meaning
2. Cause
3. Farmer advice (simple)
"""

    try:
        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"

# ================= UPLOAD =================
uploaded_file = st.file_uploader(
    "📤 Upload Plant Leaf Image",
    type=["jpg", "png", "jpeg"]
)

# ================= PREDICTION =================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # IMAGE
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    img_tensor = transform(image).unsqueeze(0)

    # PREDICT
    with st.spinner("🔍 AI analyzing plant..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100

    # RESULT
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Prediction Result")

        if pred_class == 0:
            st.error(f"🍂 Diseased ({confidence:.2f}%)")
        else:
            st.success(f"🌱 Healthy ({confidence:.2f}%)")

        st.progress(int(confidence))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x": "Class", "y": "Confidence (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)

        risk = "LOW 🌱" if pred_class == 1 else "HIGH ⚠️"
        st.info(f"AI Risk Level: {risk}")

        st.markdown('</div>', unsafe_allow_html=True)

    # GEMINI SECTION
    st.markdown("---")
    st.subheader("🧠 AI Plant Doctor Insight")

    with st.spinner("Generating advice..."):
        advice = gemini_interpretation(pred_class, confidence)

    st.markdown(f"""
    <div class="card">
    {advice}
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Upload a plant leaf image to start analysis")
