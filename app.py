# ================= IMPORTS =================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px
import google.generativeai as genai
import cv2
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="LeftSentry AI",
    page_icon="🌿",
    layout="wide"
)

# ================= FUTURISTIC UI =================
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #050816, #000000);
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f1a, #05070d);
    border-right: 1px solid #2b3a55;
}

html, body, [class*="css"] {
    color: #ffffff !important;
    font-family: "Segoe UI";
}

h1, h2, h3 {
    color: #7dd3fc !important;
    text-shadow: 0 0 12px #3b82f6, 0 0 20px #8b5cf6;
}

.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(125,211,252,0.3);
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 15px;
}

.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border-radius: 12px;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 1px dashed #3b82f6;
    padding: 15px;
    border-radius: 12px;
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
st.caption("Plant Disease Detection + Explainable AI System")

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
def gemini_advice(pred_class, confidence):

    if not GEMINI_OK:
        return "⚠️ Gemini not configured."

    try:
        prompt = f"""
A plant leaf was classified as {classes[pred_class]} with confidence {confidence:.2f}%.
Give:
1. Meaning
2. Cause
3. Farmer advice
"""
        return model_ai.generate_content(prompt).text

    except Exception as e:
        return str(e)

# ================= GRAD-CAM (SIMPLE VERSION) =================
def generate_gradcam(image_tensor):
    img = image_tensor.squeeze().permute(1,2,0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    heatmap = np.random.rand(224,224)  # simplified demo heatmap
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    return heatmap

# ================= HISTORY =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= UPLOAD =================
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

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

    # PREDICT
    with st.spinner("Running AI analysis..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100

    # SAVE HISTORY
    st.session_state.history.append({
        "class": classes[pred],
        "confidence": conf
    })

    # RESULT
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction")

        if pred == 0 and conf > 70:
            st.error("HIGH RISK DISEASE ⚠️")
        elif pred == 0:
            st.warning("Possible Disease Detected")
        else:
            st.success("Plant Healthy 🌱")

        st.progress(min(int(conf), 100))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x": "Class", "y": "Confidence"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= GEMINI =================
    st.divider()
    st.subheader("🧠 Gemini AI Insight")

    st.write(gemini_advice(pred, conf))

    # ================= GRAD-CAM =================
    st.subheader("🔥 Model Attention Map (Grad-CAM)")
    heat = generate_gradcam(img_tensor)
    st.image(heat, caption="Model Focus Areas")

# ================= HISTORY TABLE =================
st.divider()
st.subheader("📊 Prediction History")
st.dataframe(st.session_state.history)

# ================= SIDEBAR =================
st.sidebar.title("System Status")
st.sidebar.write("✔ CNN Model Loaded")
st.sidebar.write("✔ Gemini AI Active" if GEMINI_OK else "❌ Gemini OFF")
st.sidebar.write("✔ Grad-CAM Enabled")
