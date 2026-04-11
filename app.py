# ================= IMPORTS =================
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd
import google.generativeai as genai
import io

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="LeafSentry AI",
    page_icon="🌿",
    layout="wide"
)

# ================= FUTURISTIC UI =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0b0f1a, #050816);
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
    backdrop-filter: blur(12px);
    margin-bottom: 12px;
}

.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border-radius: 10px;
    border: none;
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
st.title("🌿 LeafSentry AI Pro")
st.caption("Next-Gen Plant Disease Intelligence System")

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
        self.features = nn.Sequential(
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
        x = self.features(x)
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

# ================= FALLBACK KNOWLEDGE BASE =================
disease_info = {
    "Diseased": {
        "meaning": "Plant shows signs of infection or stress.",
        "cause": "Fungal, bacterial, or environmental stress.",
        "advice": "Remove infected leaves, apply treatment, improve airflow."
    },
    "Healthy": {
        "meaning": "Plant is in good condition.",
        "cause": "No disease detected.",
        "advice": "Maintain sunlight, watering, and soil quality."
    }
}

# ================= GRAD-CAM =================
def generate_gradcam(model, image_tensor):
    image_tensor.requires_grad = True

    output = model(image_tensor)
    pred = output.argmax(dim=1)

    output[0, pred].backward()

    gradients = image_tensor.grad.data
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activation = image_tensor.detach()

    for i in range(activation.shape[1]):
        activation[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    return heatmap.numpy()

# ================= GEMINI =================
def gemini_advice(pred, conf):
    if not GEMINI_OK:
        info = disease_info[classes[pred]]
        return f"""
🔍 Offline AI Insight:

Meaning: {info['meaning']}
Cause: {info['cause']}
Advice: {info['advice']}
"""

    try:
        prompt = f"""
A plant leaf is classified as {classes[pred]} with confidence {conf:.2f}%.
Give:
- Meaning
- Cause
- Treatment advice
"""
        return model_ai.generate_content(prompt).text
    except:
        return "AI temporarily unavailable."

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Control Panel")
show_gradcam = st.sidebar.toggle("Show Grad-CAM", True)
show_chat = st.sidebar.toggle("AI Chat Mode", False)

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

    # PREDICT
    with st.spinner("Analyzing plant health..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100

    # HISTORY
    st.session_state.history.append({
        "Result": classes[pred],
        "Confidence": round(conf, 2)
    })

    # RESULT
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction")

        if pred == 0 and conf > 70:
            st.error("🚨 High Disease Risk")
        elif pred == 0:
            st.warning("Possible Disease Detected")
        else:
            st.success("Healthy Plant 🌱")

        st.progress(int(conf))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x": "Class", "y": "Confidence"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ================= GRAD-CAM =================
    if show_gradcam:
        st.subheader("🔥 Model Attention (Grad-CAM)")
        heatmap = generate_gradcam(model, img_tensor)

        st.image(heatmap, caption="Where AI is looking", clamp=True)

    # ================= AI INSIGHT =================
    st.subheader("🧠 AI Diagnosis Report")
    st.write(gemini_advice(pred, conf))

    # ================= CHAT MODE =================
    if show_chat:
        st.subheader("💬 Ask AI About This Plant")

        user_q = st.text_input("Ask a question (e.g. How to cure it?)")

        if user_q:
            try:
                response = model_ai.generate_content(
                    f"Plant status: {classes[pred]} ({conf:.2f}%). Question: {user_q}"
                )
                st.info(response.text)
            except:
                st.error("Chat AI unavailable")

# ================= HISTORY =================
st.divider()
st.subheader("📊 Prediction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Report", csv, "leaf_history.csv", "text/csv")
else:
    st.info("No predictions yet.")

# ================= SIDEBAR STATUS =================
st.sidebar.markdown("### System Status")
st.sidebar.success("CNN Model Loaded")
st.sidebar.success("UI Pro Mode Active")
st.sidebar.success("Gemini Enabled" if GEMINI_OK else "Offline Mode")
