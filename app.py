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
st.set_page_config(
    page_title="LeafSentry AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

/* ================= GLOBAL ================= */

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #07120d;
    color: #ecfdf5;
}

/* Remove Streamlit default */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ================= NAVBAR ================= */

.navbar {
    width: 100%;
    padding: 18px 40px;
    background: rgba(0,0,0,0.45);
    border-bottom: 1px solid rgba(34,197,94,0.2);
    position: sticky;
    top: 0;
    z-index: 999;
    backdrop-filter: blur(10px);

    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 28px;
    font-weight: 700;
    color: #4ade80;
}

.nav-links {
    display: flex;
    gap: 25px;
}

.nav-links a {
    color: #dcfce7;
    text-decoration: none;
    font-weight: 500;
    transition: 0.3s;
}

.nav-links a:hover {
    color: #4ade80;
}

/* ================= HERO ================= */

.hero {
    padding: 90px 60px;
    border-radius: 25px;
    margin-top: 20px;

    background:
    linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.7)),
    url('https://images.unsplash.com/photo-1466692476868-aef1dfb1e735?q=80&w=2070&auto=format&fit=crop');

    background-size: cover;
    background-position: center;

    border: 1px solid rgba(74,222,128,0.2);
}

.hero-title {
    font-size: 65px;
    font-weight: 800;
    color: white;
    line-height: 1.1;
}

.hero-sub {
    font-size: 20px;
    color: #d1fae5;
    max-width: 700px;
    margin-top: 20px;
}

.hero-btn {
    display: inline-block;
    margin-top: 30px;
    padding: 14px 30px;
    border-radius: 12px;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    color: white !important;
    text-decoration: none;
    font-weight: bold;
    box-shadow: 0 0 20px rgba(74,222,128,0.4);
}

/* ================= SECTION ================= */

.section-title {
    font-size: 38px;
    color: #4ade80;
    margin-top: 60px;
    margin-bottom: 20px;
    font-weight: 700;
}

/* ================= CARDS ================= */

.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(74,222,128,0.15);
    border-radius: 20px;
    padding: 30px;
    transition: 0.3s;
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-5px);
    border: 1px solid rgba(74,222,128,0.4);
    box-shadow: 0 0 30px rgba(74,222,128,0.15);
}

.feature-icon {
    font-size: 40px;
}

.feature-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 10px;
    color: #4ade80;
}

/* ================= AUTH ================= */

.auth-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,222,128,0.2);
    border-radius: 25px;
    padding: 40px;
    margin-top: 50px;
    backdrop-filter: blur(12px);
}

/* ================= BUTTONS ================= */

.stButton>button {
    width: 100%;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #16a34a, #22c55e);
    color: white;
    font-weight: bold;
    height: 50px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 0 15px rgba(74,222,128,0.4);
}

/* ================= INPUTS ================= */

.stTextInput input {
    background-color: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* ================= FOOTER ================= */

.footer {
    margin-top: 80px;
    padding: 40px;
    border-top: 1px solid rgba(74,222,128,0.1);
    text-align: center;
    color: #a7f3d0;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================
st.markdown("""
<div class="navbar">
    <div class="logo">🌿 LeafSentry AI</div>

    <div class="nav-links">
        <a href="#">Home</a>
        <a href="#">Plant</a>
        <a href="#">Blog</a>
        <a href="#">Privacy Policy</a>
        <a href="#">Contact Us</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= AUTH PAGE =================
def auth_page():

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            Smart Plant Disease<br>
            Detection System
        </div>

        <div class="hero-sub">
            Detect unhealthy leaves instantly using deep learning,
            AI diagnosis, and real-time plant analysis.
            Built for modern agriculture and smart farming.
        </div>

        <a class="hero-btn" href="#">
            🌱 Start Detecting
        </a>
    </div>
    """, unsafe_allow_html=True)

    # FEATURES
    st.markdown('<div class="section-title">Why Choose LeafSentry?</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">AI Diagnosis</div>
            <p>
            Uses neural networks to identify leaf diseases instantly
            with intelligent treatment recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Detection</div>
            <p>
            Upload an image and get real-time plant health analysis
            in seconds.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌿</div>
            <div class="feature-title">Healthy Farming</div>
            <p>
            Prevent crop loss and improve farming efficiency using
            smart monitoring.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # AUTH BOX
    st.markdown('<div class="auth-box">', unsafe_allow_html=True)

    st.markdown("## 🔐 Account Access")

    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Reset Password"])

    # LOGIN
    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if login(u, p):
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid username or password")

    # SIGNUP
    with tab2:
        u = st.text_input("Create Username", key="signup_user")
        p = st.text_input("Create Password", type="password", key="signup_pass")

        if st.button("Create Account"):
            if not strong_password(p):
                st.warning("Password must contain letters and numbers.")
            elif signup(u, p):
                st.success("Account created successfully")
            else:
                st.error("Username already exists")

    # RESET
    with tab3:
        u = st.text_input("Username", key="reset_user")
        p = st.text_input("New Password", type="password", key="reset_pass")

        if st.button("Reset Password"):
            if not strong_password(p):
                st.warning("Weak password")
            else:
                c.execute("SELECT * FROM users WHERE username=?", (u,))
                if c.fetchone():
                    c.execute(
                        "UPDATE users SET password=? WHERE username=?",
                        (hash_password(p), u)
                    )
                    conn.commit()
                    st.success("Password updated")
                else:
                    st.error("User not found")

    st.markdown('</div>', unsafe_allow_html=True)

    # BLOG SECTION
    st.markdown('<div class="section-title">Latest Plant Blogs</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)

    blogs = [
        (
            "🍂 Common Tomato Diseases",
            "Learn how to identify yellow leaves, fungal spots, and root infections."
        ),
        (
            "🌱 Smart Irrigation Tips",
            "Prevent overwatering and improve plant growth using modern techniques."
        ),
        (
            "🦠 AI in Agriculture",
            "Discover how machine learning is transforming plant disease detection."
        )
    ]

    for col, blog in zip([b1, b2, b3], blogs):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-title">{blog[0]}</div>
                <p>{blog[1]}</p>
            </div>
            """, unsafe_allow_html=True)

    # FOOTER
    st.markdown("""
    <div class="footer">
        © 2026 LeafSentry AI • Smart Agriculture Platform • Privacy Policy • Contact Us
    </div>
    """, unsafe_allow_html=True)

# ================= AUTH CHECK =================
if not check_auth():
    auth_page()
    st.stop()

# ================= MAIN APP =================
st.markdown("""
<div class="hero">
    <div class="hero-title">
        🌿 Plant Disease Detection
    </div>

    <div class="hero-sub">
        Upload a leaf image and let AI analyze your plant health instantly.
    </div>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🌿 LeafSentry")
    st.write(f"Welcome, {st.session_state.user}")
    if st.button("Logout"):
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
        model = CNN()
        model.load_state_dict(torch.load("cnn.pth", map_location="cpu"))
        model.eval()
        return model
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
        return """
        Basic Care Recommendation:
        - Ensure proper watering
        - Remove infected leaves
        - Place under sunlight
        - Monitor regularly
        """

    prompt = f"""
    Plant condition: {classes[pred]}
    Confidence: {conf:.2f}%

    Give short treatment advice.
    """

    try:
        r = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )

        return r.text

    except:
        return "AI service unavailable."

# ================= DETECTION UI =================
st.markdown('<div class="section-title">Upload Plant Image</div>', unsafe_allow_html=True)

file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if file:

    col1, col2 = st.columns([1,1])

    img = Image.open(file)

    with col1:
        st.image(img, use_container_width=True)

    x = transform(img).unsqueeze(0)

    if model:
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].numpy()
    else:
        probs = np.array([0.5, 0.5])

    pred = int(np.argmax(probs))
    conf = float(probs[pred]) * 100

    with col2:

        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">
                Prediction Result
            </div>

            <h1 style="color:#4ade80;">
                {classes[pred]}
            </h1>

            <p>
                Confidence Score: {conf:.2f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(conf))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x":"Class", "y":"Confidence"}
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">🧠 AI Treatment Advice</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="feature-card">
        {ai_advice(pred, conf)}
    </div>
    """, unsafe_allow_html=True)
