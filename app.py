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

# ================= CONFIG =================
st.set_page_config(
    page_title="LeafSentry AI",
    page_icon="🌿",
    layout="wide"
)

# ================= STATE =================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

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

# ================= AUTH =================
def hash_password(p):
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt())

def check_password(p, h):
    return bcrypt.checkpw(p.encode(), h)

def signup(u, p):
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (u, hash_password(p)))
        conn.commit()
        return True
    except:
        return False

def login(u, p):
    c.execute("SELECT password FROM users WHERE username=?", (u,))
    data = c.fetchone()
    return data and check_password(p, data[0])

def strong_password(p):
    return len(p) >= 6 and any(i.isdigit() for i in p) and any(i.isalpha() for i in p)

# ================= 🌿 PLANT GREEN THEME CSS =================
st.markdown("""
<style>

/* BACKGROUND (SOFT FARM GREEN) */
.stApp {
    background: linear-gradient(
        180deg,
        #eaf7ee 0%,
        #dff3e4 40%,
        #cdebd6 100%
    );
    color: #12301d;
}

/* HIDE STREAMLIT */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ================= NAVBAR ================= */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 12px 18px;

    background: rgba(34, 85, 55, 0.92);
    backdrop-filter: blur(12px);

    border-bottom: 2px solid #2f6f46;

    position: sticky;
    top: 0;
    z-index: 999;
}

/* LOGO */
.logo {
    font-size: 22px;
    font-weight: 900;
    color: #eafff0;
}

/* BUTTONS */
.stButton > button {
    background: #2f6f46;
    color: #eafff0;

    border: 1px solid #4caf72;
    padding: 9px 14px;

    border-radius: 12px;
    font-weight: 700;

    transition: 0.25s;
    width: 100%;
}

/* HOVER */
.stButton > button:hover {
    background: #3f8f5c;
    box-shadow: 0 0 12px rgba(76, 175, 114, 0.5);
    transform: translateY(-2px);
}

/* ACTIVE BUTTON */
.active-btn > button {
    background: #a6e3b7 !important;
    color: #12301d !important;
    font-weight: 900;
}

/* ================= HERO ================= */
.hero {
    padding: 100px 60px;
    border-radius: 25px;
    margin-top: 20px;

    background:
    linear-gradient(rgba(0, 50, 20, 0.55), rgba(0, 20, 10, 0.75)),
    url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
}

/* BIG TITLE */
.hero-title {
    font-size: clamp(48px, 6vw, 92px);
    font-weight: 1000;
    color: #eafff0;
}

.hero-sub {
    font-size: 20px;
    color: #d7f7df;
    max-width: 750px;
    margin-top: 10px;
}

/* ================= CARDS ================= */
.card {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #bde7c8;
    border-radius: 18px;
    padding: 22px;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(46, 125, 70, 0.25);
}

.card-title {
    color: #2f6f46;
    font-size: 22px;
    font-weight: 900;
}

/* INPUT */
.stTextInput input {
    background: #f4fff7 !important;
    border: 1px solid #bde7c8 !important;
    border-radius: 10px !important;
    color: #12301d !important;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"] {
    background: #f6fff8;
    border: 1px solid #bde7c8;
    padding: 20px;
    border-radius: 15px;
}

/* FOOTER */
.footer {
    text-align: center;
    padding: 30px;
    color: #2f6f46;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================
st.markdown('<div class="navbar">', unsafe_allow_html=True)
st.markdown('<div class="logo">🌿 LeafSentry AI</div>', unsafe_allow_html=True)

cols = st.columns(6, gap="small")
pages = ["Home", "Plant", "Blog", "Privacy", "Contact", "Login"]

for i, p in enumerate(pages):
    with cols[i]:
        cls = "active-btn" if st.session_state.page == p else ""
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        if st.button(p, key=p):
            st.session_state.page = p
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ================= HOME =================
if st.session_state.page == "Home":

    st.markdown("""
<div class="hero">
    <div class="hero-title">LeafSentry AI</div>
    <div class="hero-sub">
        Smart AI system for plant disease detection and crop health monitoring.
    </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""<div class="card"><div class="card-title">🌿 Monitoring</div>Detect plant health instantly.</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="card"><div class="card-title">⚡ Speed</div>Fast AI predictions.</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""<div class="card"><div class="card-title">🧠 AI Model</div>Deep learning classification.</div>""", unsafe_allow_html=True)

# ================= OTHER PAGES =================
elif st.session_state.page == "Plant":
    st.title("🌱 Plant Information")

elif st.session_state.page == "Blog":
    st.title("📰 Blog")

elif st.session_state.page == "Privacy":
    st.title("🔒 Privacy Policy")

elif st.session_state.page == "Contact":
    st.title("📞 Contact Us")

# ================= LOGIN =================
elif st.session_state.page == "Login":

    st.title("🔐 Login System")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(u, p):
                st.session_state.logged_in = True
                st.success("Logged in")
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("New Username")
        np = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if strong_password(np):
                if signup(nu, np):
                    st.success("Account created")
                else:
                    st.error("User already exists")
            else:
                st.warning("Weak password")

# ================= ML =================
elif st.session_state.page == "ML":

    if not st.session_state.logged_in:
        st.warning("Please login first")
        st.stop()

    st.title("🧠 Disease Detection AI")

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
            return self.fc(self.net(x).view(x.size(0), -1))

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

    img = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if img:
        image = Image.open(img)
        st.image(image, use_container_width=True)

        x = transform(image).unsqueeze(0)

        if model:
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].numpy()
        else:
            probs = np.array([0.5, 0.5])

        pred = np.argmax(probs)

        st.success(f"{classes[pred]} ({probs[pred]*100:.2f}%)")

        fig = px.bar(x=classes, y=probs * 100)
        st.plotly_chart(fig, use_container_width=True)
