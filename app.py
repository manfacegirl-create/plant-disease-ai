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

# ================= GEMINI =================
try:
    from google import genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# ================= PAGE =================
st.set_page_config(page_title="LeafSentry AI", page_icon="🌿", layout="wide")

# ================= DATABASE =================
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password BLOB
)
""")
conn.commit()

# ================= PASSWORD =================
def hash_password(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt())

def check_password(pw, hashed):
    return bcrypt.checkpw(pw.encode(), hashed)

def strong_password(pw):
    return len(pw) >= 6 and any(c.isdigit() for c in pw) and any(c.isalpha() for c in pw)

# ================= AUTH =================
def signup(u, p):
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (u, hash_password(p)))
        conn.commit()
        return True
    except:
        return False

def login(u, p):
    c.execute("SELECT password FROM users WHERE username=?", (u,))
    data = c.fetchone()
    return data and check_password(p, data[0])

def check_auth():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in

def logout():
    st.session_state.logged_in = False
    st.rerun()

# ================= UI STYLE =================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #020617, #000000);
    color: #e0f2fe;
}

/* Neon glow titles */
h1, h2, h3 {
    color: #38bdf8 !important;
    text-shadow: 0 0 10px #38bdf8;
}

/* Glass card */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 0 20px rgba(56,189,248,0.2);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
    color: white;
    border-radius: 12px;
    border: none;
    box-shadow: 0 0 10px #38bdf8;
}

/* Inputs */
input {
    background-color: #020617 !important;
    color: #38bdf8 !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ================= AUTH UI =================
def auth_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🔐 LeafSentry Access")

    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Reset"])

    # LOGIN
    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", key="login_btn"):
            if login(u, p):
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    # SIGN UP
    with tab2:
        u = st.text_input("New Username", key="signup_user")
        p = st.text_input("New Password", type="password", key="signup_pass")

        if st.button("Create Account", key="signup_btn"):
            if not strong_password(p):
                st.warning("Weak password")
            elif signup(u, p):
                st.success("Account created")
            else:
                st.error("Username exists")

    # RESET
    with tab3:
        u = st.text_input("Username", key="reset_user")
        p = st.text_input("New Password", type="password", key="reset_pass")

        if st.button("Reset Password", key="reset_btn"):
            if not strong_password(p):
                st.warning("Weak password")
            else:
                c.execute("SELECT * FROM users WHERE username=?", (u,))
                if c.fetchone():
                    c.execute("UPDATE users SET password=? WHERE username=?",
                              (hash_password(p), u))
                    conn.commit()
                    st.success("Password updated")
                else:
                    st.error("User not found")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= AUTH CHECK =================
if not check_auth():
    auth_page()
    st.stop()

# ================= MAIN APP =================
st.title("🌿 LeafSentry AI")
st.caption("Neural Plant Disease Detection")

if st.sidebar.button("Logout"):
    logout()

# ================= MODEL =================
classes = ["Diseased", "Healthy"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
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
        return self.fc(x.view(x.size(0), -1))

@st.cache_resource
def load_model():
    try:
        m = CNN()
        m.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
        m.eval()
        return m
    except:
        return None

model = load_model()

# ================= GEMINI =================
GEMINI_OK = False
client = None

if GEMINI_AVAILABLE:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            client = genai.Client(api_key=key)
            GEMINI_OK = True
    except:
        pass

def ai_advice(pred, conf):
    if not GEMINI_OK:
        return "Basic care: monitor plant, water properly."

    prompt = f"Plant is {classes[pred]} ({conf:.2f}%). Give treatment."
    try:
        r = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return r.text
    except:
        return "AI unavailable."

# ================= UPLOAD =================
file = st.file_uploader("Upload Leaf Image", type=["jpg","png"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)

    x = transform(img).unsqueeze(0)

    if model:
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].numpy()
    else:
        probs = np.array([0.5, 0.5])

    pred = int(np.argmax(probs))
    conf = float(probs[pred]) * 100

    st.subheader(f"{classes[pred]} ({conf:.2f}%)")
    st.progress(int(conf))

    st.plotly_chart(px.bar(x=classes, y=probs*100), use_container_width=True)

    st.subheader("🧠 AI Diagnosis")
    st.write(ai_advice(pred, conf))
