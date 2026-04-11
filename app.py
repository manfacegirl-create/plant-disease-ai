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
    page_title="Plant AI Pro",
    page_icon="🌿",
    layout="wide"
)

# ================= MODERN UI STYLE =================
st.markdown("""
<style>
body {
    background-color: #ffffff;
}

.main {
    background-color: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1, h2, h3 {
    color: #1a1a1a;
    font-family: 'Arial';
}

/* CARD STYLE */
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

/* STATUS BADGES */
.badge-healthy {
    background-color: #d1fae5;
    color: #065f46;
    padding: 8px 12px;
    border-radius: 10px;
    font-weight: bold;
}

.badge-diseased {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 8px 12px;
    border-radius: 10px;
    font-weight: bold;
}

/* BUTTON STYLE */
.stButton>button {
    background-color: #16a34a;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ================= GEMINI SETUP =================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_ai = genai.GenerativeModel("gemini-1.5-flash")
    GEMINI_OK = True
except:
    GEMINI_OK = False

# ================= TITLE =================
st.title("🌿 Plant Disease AI Pro")
st.caption("Modern AI-powered plant health analysis system")

# ================= SIDEBAR =================
st.sidebar.title("🌱 AI Control Panel")
st.sidebar.info("CNN Model Active")

# ================= CLASSES =================
classes = ["🍂 Diseased", "🌱 Healthy"]

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

@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================= GEMINI =================
def gemini_ai(pred_class, confidence):
    if not GEMINI_OK:
        return "⚠️ AI not configured"

    prompt = f"""
You are a plant doctor AI.

Class: {classes[pred_class]}
Confidence: {confidence:.2f}%

Give simple explanation + farmer advice.
"""

    try:
        return model_ai.generate_content(prompt).text
    except:
        return "AI temporarily unavailable"

# ================= UPLOAD =================
uploaded_file = st.file_uploader("📤 Upload Plant Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    # ================= IMAGE =================
    with col1:
        st.markdown("### 📷 Uploaded Image")
        st.image(image, use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with st.spinner("🔍 Analyzing plant health..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100

    # ================= RESULT CARD =================
    with col2:

        st.markdown("### 📊 Analysis Result")

        if pred_class == 0:
            st.markdown('<div class="badge-diseased">🍂 Diseased Plant</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-healthy">🌱 Healthy Plant</div>', unsafe_allow_html=True)

        st.progress(int(confidence))

        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        fig = px.bar(
            x=classes,
            y=probs * 100,
            title="Prediction Breakdown"
        )
        st.plotly_chart(fig, use_container_width=True)

        # NEW FEATURE (NO EXTRA SETUP)
        risk_level = "LOW 🌱" if confidence > 80 and pred_class == 1 else \
                     "HIGH ⚠️" if pred_class == 0 else "MEDIUM 🌿"

        st.info(f"AI Risk Level: {risk_level}")

    # ================= AI INSIGHT =================
    st.markdown("---")
    st.markdown("## 🧠 AI Plant Doctor Insight")

    with st.spinner("Generating expert advice..."):
        advice = gemini_ai(pred_class, confidence)

    st.markdown(f"""
    <div class="card">
    {advice}
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Upload a plant leaf image to start analysis")
