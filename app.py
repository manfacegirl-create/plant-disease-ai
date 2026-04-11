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
    page_title="LeftSentry AI",
    page_icon="🌿",
    layout="wide"
)

# ================= CLEAN MODERN UI =================
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0b0f1a, #050816);
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
    backdrop-filter: blur(10px);
    margin-bottom: 12px;
}

.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border-radius: 10px;
    border: none;
}

[data-testid="stFileUploader"] {
    border: 1px dashed #3b82f6;
    padding: 12px;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
}

</style>
""", unsafe_allow_html=True)

# ================= GEMINI =================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_ai = genai.GenerativeModel("gemini-1.5-flash")
    GEMINI_OK = True
except:
    GEMINI_OK = False

# ================= TITLE =================
st.title("🧠 LeftSentry AI Dashboard")
st.caption("CNN Plant Disease Detection System")

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

# ================= GEMINI =================
def gemini_advice(pred, conf):

    if not GEMINI_OK:
        return "⚠️ Gemini not configured."

    try:
        prompt = f"""
A plant leaf is classified as {classes[pred]} with confidence {conf:.2f}%.
Give:
1. Meaning
2. Cause
3. Advice
"""
        return model_ai.generate_content(prompt).text
    except Exception as e:
        return f"Error: {str(e)}"

# ================= SESSION HISTORY =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= UPLOAD =================
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # IMAGE
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    img_tensor = transform(image).unsqueeze(0)

    # PREDICTION
    with st.spinner("Analyzing..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100

    # SAVE HISTORY
    st.session_state.history.append({
        "Result": classes[pred],
        "Confidence": round(conf, 2)
    })

    # RESULT PANEL
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if pred == 0 and conf > 70:
            st.error("⚠️ HIGH RISK DISEASE")
        elif pred == 0:
            st.warning("Possible Disease")
        else:
            st.success("Healthy Plant 🌱")

        st.progress(min(int(conf), 100))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x": "Class", "y": "Confidence (%)"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= GEMINI =================
    st.divider()
    st.subheader("🧠 AI Insight")

    st.write(gemini_advice(pred, conf))

# ================= HISTORY =================
st.divider()
st.subheader("📊 Prediction History")

if len(st.session_state.history) > 0:
    st.dataframe(st.session_state.history)
else:
    st.info("No predictions yet.")

# ================= SIDEBAR =================
st.sidebar.title("System Status")
st.sidebar.write("✔ CNN Model Loaded")
st.sidebar.write("✔ UI Stable Version")
st.sidebar.write("✔ Gemini Enabled" if GEMINI_OK else "❌ Gemini OFF")
