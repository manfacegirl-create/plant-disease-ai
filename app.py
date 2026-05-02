# ================= IMPORTS =================
import streamlit as st
import sqlite3
import bcrypt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd

# ================= GEMINI IMPORT =================
try:
    from google import genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# ================= PAGE CONFIG =================
st.set_page_config(page_title="LeafSentry AI", page_icon="🌿", layout="wide")

# ================= DATABASE =================
def connect_db():
    return sqlite3.connect("users.db", check_same_thread=False)

conn = connect_db()
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password BLOB
)
""")
conn.commit()

# ================= PASSWORD =================
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def strong_password(password):
    return (
        len(password) >= 6 and
        any(c.isdigit() for c in password) and
        any(c.isalpha() for c in password)
    )

# ================= AUTH =================
def signup_user(username, password):
    try:
        hashed = hash_password(password)
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    data = c.fetchone()
    if data:
        return check_password(password, data[0])
    return False

def check_auth():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

# ================= AUTH UI =================
def auth_page():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #5f6dfc, #d946ef);
    }
    .card {
        background: white;
        border-radius: 20px;
        padding: 40px;
        max-width: 500px;
        margin: auto;
        margin-top: 80px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🔐 LeafSentry Access")

    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])

    # LOGIN
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    # SIGNUP
    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            if not strong_password(p):
                st.warning("Weak password")
            else:
                if signup_user(u, p):
                    st.success("Account created")
                else:
                    st.error("Username exists")

    # RESET
    with tab3:
        u = st.text_input("Username")
        p = st.text_input("New Password", type="password")
        if st.button("Reset Password"):
            if not strong_password(p):
                st.warning("Weak password")
            else:
                c.execute("SELECT * FROM users WHERE username=?", (u,))
                if c.fetchone():
                    new_hash = hash_password(p)
                    c.execute("UPDATE users SET password=? WHERE username=?", (new_hash, u))
                    conn.commit()
                    st.success("Password updated")
                else:
                    st.error("User not found")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= AUTH CHECK =================
if not check_auth():
    auth_page()
    st.stop()

# ================= UI =================
st.title("🌿 LeafSentry AI")
st.caption("CNN Plant Disease Detection System")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #0b0f1a, #050816);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================= MODEL =================
classes = ["Diseased", "Healthy"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

@st.cache_resource
def load_model():
    try:
        model = CNN()
        model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
        model.eval()
        return model
    except:
        return None

model = load_model()

# ================= GEMINI =================
GEMINI_OK = False
gemini_client = None

if GEMINI_AVAILABLE:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        if api_key:
            gemini_client = genai.Client(api_key=api_key)
            GEMINI_OK = True
    except:
        pass

def offline_ai(pred, conf):
    return f"{classes[pred]} ({conf:.2f}%) - Basic care recommended."

def gemini_advice(pred, conf):
    if not GEMINI_OK:
        return offline_ai(pred, conf)

    prompt = f"Plant is {classes[pred]} with {conf:.2f}% confidence. Give treatment."
    try:
        res = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return res.text
    except:
        return offline_ai(pred, conf)

# ================= UPLOAD =================
file = st.file_uploader("Upload Leaf Image", type=["jpg","png"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)

    tensor = transform(img).unsqueeze(0)

    if model:
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].numpy()
    else:
        probs = np.array([0.5, 0.5])

    pred = int(np.argmax(probs))
    conf = float(probs[pred]) * 100

    st.subheader(classes[pred])
    st.progress(int(conf))

    st.plotly_chart(px.bar(x=classes, y=probs * 100), use_container_width=True)

    st.subheader("🧠 AI Diagnosis")
    st.write(gemini_advice(pred, conf))

# ================= SIDEBAR =================
st.sidebar.write(f"👤 {st.session_state.user}")
if st.sidebar.button("Logout"):
    logout()
