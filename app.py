import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
st.set_page_config(page_title="Plant Disease AI", layout="wide")

# ================= MODEL =================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128,2)

    def forward(self,x):
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
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

classes = ["Diseased", "Healthy"]

# ================= GRAD-CAM =================
def gradcam(image_tensor):
    image_tensor.requires_grad = True
    output = model(image_tensor)

    pred_class = output.argmax()
    output[0, pred_class].backward()

    gradients = image_tensor.grad[0].numpy()
    heatmap = np.mean(gradients, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    return heatmap

# ================= UI =================
st.title("🌿 AI Plant Disease Detection System")
st.markdown("Upload a leaf image and let AI detect if it's **Healthy or Diseased**")

col1, col2 = st.columns(2)

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    result = classes[pred.item()]
    confidence = conf.item() * 100

    # ================= RESULT =================
    with col2:
        st.subheader("📊 Prediction Result")

        if result == "Diseased":
            st.error(f"⚠️ Diseased ({confidence:.2f}%)")
        else:
            st.success(f"✅ Healthy ({confidence:.2f}%)")

        # Probability bar chart
        st.subheader("Confidence Distribution")
        fig, ax = plt.subplots()
        ax.bar(classes, probs[0].numpy()*100)
        ax.set_ylabel("Confidence (%)")
        st.pyplot(fig)

    # ================= GRAD-CAM =================
    st.subheader("🔍 Disease Highlight (Grad-CAM Approximation)")

    heatmap = gradcam(img_tensor.clone())

    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.axis('off')

    st.pyplot(plt)