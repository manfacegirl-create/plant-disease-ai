# ================= IMPORTS =================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px

# NOTE: Gemini removed visually (to make it look non-AI heavy)
import google.generativeai as genai

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Plant Health Analyzer",
    page_icon="🌿",
    layout="wide"
)

# ================= BLACK & WHITE PROFESSIONAL THEME =================
st.markdown("""
<style>

/* MAIN BACKGROUND (BLACK & WHITE CLEAN) */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #000000;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #f2f2f2;
}

/* HEADER */
[data-testid="stHeader"] {
    background-color: #ffffff;
}

/* GLOBAL TEXT */
html, body, [class*="css"] {
    color: #000000;
    font-family: "Arial", sans-serif;
}

/* REMOVE AI LOOK */
h1, h2, h3 {
    color: #000000;
    font-weight: 600;
}

/* IMAGE / RESULT CARDS */
.card {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
}

/* BUTTON STYLE (MONOCHROME) */
.stButton>button {
    background-color: #000000;
    color: #ffffff;
    border-radius: 6px;
    padding: 8px 16px;
    border: none;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background-color: #000000;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("Plant Health Analyzer")
st.caption("Leaf Image Classification System")

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

# ================= UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

# ================= PREDICTION =================
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
    with st.spinner("Processing image..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100

    # RESULT
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")

        if pred_class == 0:
            st.error(f"Diseased ({confidence:.2f}%)")
        else:
            st.success(f"Healthy ({confidence:.2f}%)")

        st.progress(int(confidence))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x": "Class", "y": "Confidence"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # SIMPLE STATUS (NO AI STYLE)
        status = "LOW RISK" if pred_class == 1 else "HIGH RISK"
        st.info(f"Status: {status}")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Upload a plant leaf image to analyze.")
