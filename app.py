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

# ================= DB =================
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

# ================= PRO CSS (SAFE STREAMLIT ONLY) =================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #07140c, #020705);
    color: white;
}

/* REMOVE STREAMLIT CLUTTER */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ================= NAVBAR (ONE LINE PRO UI) ================= */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 12px 20px;

    background: rgba(10, 35, 18, 0.75);
    backdrop-filter: blur(14px);

    border-bottom: 1px solid rgba(34, 197, 94, 0.4);
    position: sticky;
    top: 0;
    z-index: 999;
}

/* LOGO */
.logo {
    font-size: 20px;
    font-weight: 900;
    color: #22c55e;
    text-shadow: 0 0 12px #22c55e;
}

/* NAV ROW = SINGLE LINE */
.navbar-container {
    display: flex;
    gap: 8px;
}

/* BUTTON FIX (NO HTML TAGS) */
.stButton > button {
    background: transparent;
    color: #d1fae5;
    border: 1px solid rgba(34,197,94,0.35);

    padding: 9px 14px;
    border-radius: 10px;

    font-weight: 700;
    transition: 0.25s;
    width: 100%;
}

/* HOVER */
.stButton > button:hover {
    background: rgba(34,197,94,0.15);
    box-shadow: 0 0 10px #22c55e;
    transform: translateY(-2px);
}

/* ACTIVE STATE */
.active-btn > button {
    background: #22c55e !important;
    color: black !important;
    box-shadow: 0 0 15px #22c55e;
}

/* HERO TITLE (PRO LEVEL SCALING) */
.hero-title {
    font-size: clamp(50px, 6vw, 95px);
    font-weight: 1000;
    color: #4ade80;
    text-shadow: 0 0 25px #22c55e;
    line-height: 1.05;
}

.hero-sub {
    font-size: 20px;
    color: #d1fae5;
    max-width: 750px;
    margin-top: 10px;
}

/* CARD UI */
.card {
    background: rgba(16, 24, 39, 0.85);
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 18px;
    padding: 22px;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 0 18px rgba(34,197,94,0.25);
}

.card-title {
    color: #4ade80;
    font-size: 22px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR (NO HTML OUTPUT BUG FIXED) =================
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
<div style="padding:90px 40px;">
    <div class="hero-title">LeafSentry AI</div>
    <div class="hero-sub">
        Professional AI-powered plant disease detection system.
    </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""<div class="card"><div class="card-title">🌿 Monitoring</div>Real-time plant health analysis.</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="card"><div class="card-title">⚡ Speed</div>Instant AI predictions.</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""<div class="card"><div class="card-title">🧠 AI Model</div>Deep CNN classification system.</div>""", unsafe_allow_html=True)

# ================= OTHER PAGES =================
elif st.session_state.page == "Plant":
    st.title("🌱 Plant Info")

elif st.session_state.page == "Blog":
    st.title("📰 Blog")

elif st.session_state.page == "Privacy":
    st.title("🔒 Privacy Policy")

elif st.session_state.page == "Contact":
    st.title("📞 Contact")

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
                st.error("Invalid login")

    with tab2:
        nu = st.text_input("New Username")
        np = st.text_input("New Password", type="password")

        if st.button("Create"):
            if strong_password(np):
                if signup(nu, np):
                    st.success("Account created")
                else:
                    st.error("User exists")
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
