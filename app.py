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
    page_title="Plant AI System",
    page_icon="🌿",
    layout="wide"
)

# ================= CLEAN WHITE BRUTALIST UI =================
st.markdown("""
<style>

/* BACKGROUND */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
    color: #000000;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #f5f5f5;
    border-right: 2px solid #000000;
}

/* TEXT */
html, body, [class*="css"] {
    color: #000000 !important;
    font-family: "Arial", sans-serif;
}

/* HEADINGS */
h1, h2, h3 {
    color: #000000 !important;
    font-weight: 700;
}

/* CARDS */
.card {
    background: #ffffff;
    border: 2px solid #000000;
    padding: 15px;
    border-radius: 0px;
    margin-bottom: 15px;
}

/* BUTTONS */
.stButton>button {
    background-color: #000000;
    color: #ffffff;
    border-radius: 0px;
    padding: 10px 18px;
    font-weight: bold;
}

/* UPLOADER */
[data-testid="stFileUploader"] {
    border: 2px dashed #000000;
    padding: 15px;
    background-color: #ffffff;
}

/* PLOT */
.js-plotly-plot {
    background-color: #ffffff !important;
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
st.title("Plant AI System Dashboard")
st.caption("CNN Classifier + AI Prompt Generator Tools")

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
        return "Gemini not configured."

    prompt = f"""
A plant leaf was classified as {classes[pred_class]} with confidence {confidence:.2f}%.
Give simple farmer advice.
"""
    try:
        return model_ai.generate_content(prompt).text
    except:
        return "Gemini error."

# ================= PROMPT GENERATORS =================
def dashboard_prompt():
    return """
Generate a React/Tailwind component for my model performance dashboard.
Use a sharp black & white brutalist UI style.
Include cards, bold typography, and grid layout.
"""

def confusion_matrix_prompt():
    return """
Show me a confusion matrix for my Random Forest model.
Use grayscale heatmap with strong black text labels and clean scientific styling.
"""

def logo_prompt():
    return """
Describe a DALL-E prompt for a minimalist ML logo in black & white vector style.
Make it modern, geometric, and clean.
"""

# ================= UI =================
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Input Image")
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    img_tensor = transform(image).unsqueeze(0)

    with st.spinner("Analyzing..."):
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred]) * 100

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction")

        if pred == 0:
            st.error(f"Diseased ({conf:.2f}%)")
        else:
            st.success(f"Healthy ({conf:.2f}%)")

        st.progress(int(conf))

        fig = px.bar(
            x=classes,
            y=probs*100,
            labels={"x":"Class","y":"Confidence"}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("AI Advice")
    st.write(gemini_advice(pred, conf))

# ================= PROMPT TOOL SECTION =================
st.divider()
st.subheader("Prompt Generator Tools (ML Design Assistant)")

colA, colB, colC = st.columns(3)

with colA:
    if st.button("Dashboard Prompt"):
        st.code(dashboard_prompt(), language="text")

with colB:
    if st.button("Confusion Matrix Prompt"):
        st.code(confusion_matrix_prompt(), language="text")

with colC:
    if st.button("ML Logo Prompt"):
        st.code(logo_prompt(), language="text")
