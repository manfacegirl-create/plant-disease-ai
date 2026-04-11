# ================= IMPORTS =================
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Plant Health Analyzer",
    page_icon="🌿",
    layout="wide"
)

# ================= CLEAN MODERN WHITE THEME =================
st.markdown("""
<style>

/* MAIN BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #000000;
}

/* HEADER */
[data-testid="stHeader"] {
    background-color: #ffffff;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #f7f7f7;
    border-right: 1px solid #e6e6e6;
}

/* GLOBAL TEXT */
html, body, [class*="css"] {
    color: #000000 !important;
    font-family: "Arial";
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #000000 !important;
    font-weight: 600;
}

/* CARD STYLE */
.card {
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
}

/* FILE UPLOADER (IMPORTANT FIX) */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 2px dashed #000000;
    border-radius: 12px;
    padding: 20px;
}

/* UPLOADER TEXT */
[data-testid="stFileUploader"] section {
    color: #000000 !important;
}

/* BUTTONS */
.stButton>button {
    background-color: #000000;
    color: #ffffff;
    border-radius: 10px;
    padding: 8px 18px;
    border: none;
    font-weight: 500;
}

/* PROGRESS BAR */
.stProgress > div > div > div > div {
    background-color: #000000;
}

/* PLOTLY CLEAN */
.js-plotly-plot {
    background-color: #ffffff !important;
}

/* REMOVE ANY HIDDEN GREY TEXT ISSUES */
p, span, label {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🌿 Plant Health Analyzer")
st.caption("Simple Leaf Disease Classification System")

st.markdown("---")

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

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader(
    "📤 Upload a leaf image for analysis",
    type=["jpg", "png", "jpeg"]
)

# ================= PREDICTION =================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # IMAGE CARD
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    img_tensor = transform(image).unsqueeze(0)

    # PREDICT
    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100

    # RESULT CARD
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Prediction Result")

        if pred_class == 0:
            st.error(f"❌ Diseased ({confidence:.2f}%)")
        else:
            st.success(f"✅ Healthy ({confidence:.2f}%)")

        st.progress(int(confidence))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x": "Class", "y": "Confidence (%)"}
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="black"
        )

        st.plotly_chart(fig, use_container_width=True)

        # STATUS BOX
        if pred_class == 1:
            st.success("Status: LOW RISK 🌱")
        else:
            st.error("Status: HIGH RISK ⚠️")

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("📌 Please upload a leaf image to begin analysis.")
