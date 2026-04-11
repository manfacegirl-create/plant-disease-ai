import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px

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

# ================= TITLE =================
st.title("🌿 Plant Disease Detection AI")
st.caption("Deep Learning CNN Model for Leaf Classification")

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

    # preprocessing
    img_tensor = transform(image).unsqueeze(0)

    with st.spinner("🧠 AI is analyzing the leaf..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred_class = np.argmax(probs)
        confidence = probs[pred_class] * 100

    # ================= RESULT PANEL =================
    with col2:
        st.subheader("📊 Prediction Result")

        if pred_class == 1:
            st.error(f"🍂 Diseased ({confidence:.2f}%)")
        else:
            st.success(f"🌱 Healthy ({confidence:.2f}%)")

        st.progress(int(confidence))

        st.markdown("### Confidence Breakdown")

        fig = px.bar(
            x=classes,
            y=probs * 100,
            color=classes,
            labels={"x": "Class", "y": "Confidence (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= EXTRA INFO =================
    st.divider()

    st.subheader("🧠 AI Interpretation")
    if pred_class == 1:
        st.warning("The model detected possible disease symptoms in the leaf image.")
    else:
        st.info("The leaf appears healthy with no strong disease indicators.")
