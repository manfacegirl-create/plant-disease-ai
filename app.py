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
    page_title="Neural Vision Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ================= FUTURISTIC UI (FROM YOUR PROMPT) =================
st.markdown("""
<style>

/* CYBER BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #050816, #000000);
    color: #ffffff;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f1a, #05070d);
    border-right: 1px solid #2b3a55;
}

/* TEXT */
html, body, [class*="css"] {
    color: #ffffff !important;
    font-family: "Segoe UI", sans-serif;
}

/* HEADINGS (NEON BLUE/PURPLE GLOW) */
h1, h2, h3 {
    color: #7dd3fc !important;
    text-shadow: 0 0 12px #3b82f6, 0 0 20px #8b5cf6;
}

/* GLASSMORPHISM CARDS */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(125,211,252,0.3);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 0 25px rgba(59,130,246,0.15);
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 18px;
    font-weight: bold;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 1px dashed #3b82f6;
    border-radius: 12px;
    padding: 15px;
}

/* PLOT */
.js-plotly-plot {
    background: transparent !important;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background-color: #3b82f6;
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
st.title("🧠 Neural Vision Dashboard")
st.caption("Futuristic Plant Disease Detection System")

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

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================= GEMINI FUNCTION =================
def gemini_advice(pred_class, confidence):

    if not GEMINI_OK:
        return "⚠️ Gemini API not configured."

    prompt = f"""
A plant leaf was classified as {classes[pred_class]} with confidence {confidence:.2f}%.
Provide:
1. Meaning
2. Possible cause
3. Farmer recommendation
"""

    try:
        return model_ai.generate_content(prompt).text
    except Exception as e:
        return f"Gemini error: {str(e)}"

# ================= UPLOAD =================
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # IMAGE PANEL
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    img_tensor = transform(image).unsqueeze(0)

    # PREDICTION
    with st.spinner("Running neural inference..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100

    # RESULT PANEL
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Engine")

        if pred == 0:
            st.error(f"DISEASE DETECTED ({conf:.2f}%)")
        else:
            st.success(f"HEALTHY ({conf:.2f}%)")

        st.progress(int(conf))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x": "Class", "y": "Confidence"}
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= GEMINI OUTPUT =================
    st.divider()
    st.subheader("🧠 Gemini AI Analysis")

    explanation = gemini_advice(pred, conf)
    st.write(explanation)

# ================= SIDEBAR =================
st.sidebar.title("System Status")
st.sidebar.write("✔ CNN Model Loaded")
st.sidebar.write("✔ Futuristic UI Active")
st.sidebar.write("✔ Gemini AI Enabled" if GEMINI_OK else "❌ Gemini OFF")
