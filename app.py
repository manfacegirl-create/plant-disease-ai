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

# ================= GEMINI SETUP =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_ai = genai.GenerativeModel("gemini-1.5-flash")

# ================= SIDEBAR =================
st.sidebar.title("🌿 Plant AI System")
st.sidebar.info("CNN Model Only (Stable Version)")

classes = ["🌱 Healthy", "🍂 Diseased"]

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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

# ================= LOAD CNN MODEL =================
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ================= GEMINI =================
def gemini_interpretation(pred_class, confidence):
    prompt = f"""
You are an agricultural AI expert.

A plant leaf was classified as:
- Class: {classes[pred_class]}
- Confidence: {confidence:.2f}%

Give:
1. Meaning
2. Possible cause
3. Advice for farmers
"""
    response = model_ai.generate_content(prompt)
    return response.text

# ================= TITLE =================
st.title("🌿 Plant Disease Detection AI")
st.caption("CNN Model + Gemini AI (Stable Deployment Version)")

# ================= UPLOAD =================
uploaded_file = st.file_uploader(
    "📤 Upload a leaf image",
    type=["jpg", "png", "jpeg"]
)

# ================= PREDICTION =================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with st.spinner("🧠 AI analyzing..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100

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
            labels={"x": "Class", "y": "Confidence (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= GEMINI =================
    st.divider()
    st.subheader("🧠 Gemini AI Advice")

    try:
        with st.spinner("Gemini thinking..."):
            explanation = gemini_interpretation(pred_class, confidence)

        st.write(explanation)

    except Exception as e:
        st.error("Gemini failed.")
        st.code(str(e))
