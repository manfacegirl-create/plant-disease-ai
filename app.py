import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import plotly.express as px
import google.generativeai as genai
import os
import requests

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

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["CNN", "ResNet50", "Auto (Best)"]
)

st.sidebar.info("Upload a leaf image and get prediction.")

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

# ================= RESNET MODEL =================
class ResNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)

# ================= DOWNLOAD RESNET FROM GOOGLE DRIVE =================
def download_resnet():
    url = "https://drive.google.com/uc?export=download&id=1D53PoSyh3aze-EobeTpHcBJobB38xyAS"
    path = "resnet.pth"

    if not os.path.exists(path):
        with st.spinner("Downloading ResNet model..."):
            r = requests.get(url)
            with open(path, "wb") as f:
                f.write(r.content)

    return path

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    # CNN (local)
    cnn = CNN()
    cnn.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
    cnn.eval()

    # ResNet (download if needed)
    resnet = ResNetModel()
    resnet_path = download_resnet()
    resnet.load_state_dict(torch.load(resnet_path, map_location="cpu"))
    resnet.eval()

    return cnn, resnet

cnn_model, resnet_model = load_models()

# ================= MODEL SELECT =================
def get_model(choice):
    if choice == "CNN":
        return cnn_model
    elif choice == "ResNet50":
        return resnet_model
    else:
        return cnn_model  # Auto default (you can improve later)

model = get_model(model_choice)

# ================= GEMINI FUNCTION =================
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
st.caption("CNN + ResNet50 + Gemini AI (FYP Project)")

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
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

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
        st.error("Gemini failed (API issue or quota).")
        st.code(str(e))
