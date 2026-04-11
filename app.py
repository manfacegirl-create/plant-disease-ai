import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px
import google.generativeai as genai

# ================= GEMINI SETUP =================
# 🔐 PUT YOUR KEY HERE (DO NOT SHARE IT PUBLICLY)
genai.configure(api_key="PASTE_YOUR_NEW_KEY_HERE")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="wide"
)

# ================= SIDEBAR =================
st.sidebar.title("🌿 Plant AI System")
st.sidebar.info("Upload a leaf image and get instant prediction.")

st.sidebar.markdown("### Model Info")
st.sidebar.write("CNN-based classifier")
st.sidebar.write("Classes: Healthy / Diseased")

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

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["🌱 Healthy", "🍂 Diseased"]

# ================= GEMINI FUNCTION =================
def gemini_interpretation(pred_class, confidence):
    model_ai = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are an agricultural AI expert.

A plant leaf was classified as:
- Class: {classes[pred_class]}
- Confidence: {confidence:.2f}%

Give a short explanation for a farmer:
1. Meaning
2. Possible cause
3. Advice
"""

    response = model_ai.generate_content(prompt)
    return response.text

# ================= TITLE =================
st.title("🌿 Plant Disease Detection AI")
st.caption("Deep Learning CNN Model + Gemini AI Interpretation")

uploaded_file = st.file_uploader(
    "📤 Upload a leaf image (JPG, PNG, JPEG)",
    type=["jpg", "png", "jpeg"]
)

# ================= PREDICTION =================
if uploaded_file:

    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with st.spinner("🧠 AI is analyzing the leaf..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred_class = np.argmax(probs)
        confidence = probs[pred_class] * 100

    # ================= RESULT =================
    with col2:
        st.subheader("📊 Prediction Result")

        if pred_class == 1:
            st.error(f"🍂 Diseased ({confidence:.2f}%)")
        else:
            st.success(f"🌱 Healthy ({confidence:.2f}%)")

        st.progress(int(confidence))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            color=classes,
            labels={"x": "Class", "y": "Confidence (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= GEMINI AI =================
    st.divider()
    st.subheader("🧠 Gemini AI Interpretation")

    with st.spinner("🤖 Gemini is thinking..."):
        explanation = gemini_interpretation(pred_class, confidence)

    st.write(explanation)
