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

# ================= PAGE CONFIG =================
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
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def strong_password(password):
    return len(password) >= 6 and any(i.isdigit() for i in password) and any(i.isalpha() for i in password)

def signup(username, password):
    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login(username, password):
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    data = c.fetchone()
    return data and check_password(password, data[0])

# ================= CSS (GREEN NEON UI) =================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #07140c, #020705);
    color: white;
}

/* HIDE STREAMLIT UI */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ================= NAVBAR ================= */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 12px 25px;

    background: rgba(10, 40, 20, 0.65);
    backdrop-filter: blur(12px);

    border-bottom: 1px solid rgba(34, 197, 94, 0.4);

    position: sticky;
    top: 0;
    z-index: 999;
}

/* TITLE IN NAVBAR */
.logo {
    font-size: 24px;
    font-weight: 900;
    color: #22c55e;
    text-shadow: 0 0 10px #22c55e;
}

/* BUTTON ROW */
.nav-buttons {
    display: flex;
    gap: 10px;
}

/* NAV BUTTONS */
.nav-btn button {
    background: transparent;
    border: 1px solid rgba(34,197,94,0.4);
    color: #d1fae5;
    padding: 10px 16px;
    border-radius: 12px;
    font-weight: 700;
    transition: 0.3s;
}

/* HOVER */
.nav-btn button:hover {
    background: rgba(34,197,94,0.2);
    box-shadow: 0 0 12px #22c55e;
    transform: translateY(-2px);
}

/* ACTIVE */
.active button {
    background: #22c55e !important;
    color: black !important;
    box-shadow: 0 0 15px #22c55e;
}

/* HERO */
.hero {
    padding: 120px 60px;
    border-radius: 25px;
    margin-top: 20px;

    background:
    linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.85)),
    url("https://images.unsplash.com/photo-1466692476868-aef1dfb1e735?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
}

/* BIG TITLE FIX */
.hero-title {
    font-size: clamp(48px, 6vw, 90px);
    font-weight: 1000;
    color: #4ade80;
    text-shadow: 0 0 25px #22c55e;
    line-height: 1.1;
}

.hero-sub {
    font-size: 22px;
    max-width: 700px;
    color: #d1fae5;
    margin-top: 15px;
}

/* CARDS */
.card {
    background: rgba(16, 24, 39, 0.85);
    border: 1px solid #1f5134;
    border-radius: 20px;
    padding: 25px;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 0 20px rgba(34,197,94,0.3);
}

.card-title {
    color: #4ade80;
    font-size: 24px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR (NO PAGE SWITCHING, STATE ONLY) =================
def nav_button(label, page):
    active = "active" if st.session_state.page == page else ""
    st.markdown(f'<div class="nav-btn {active}">', unsafe_allow_html=True)
    if st.button(label, key=page):
        st.session_state.page = page
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="navbar">
    <div class="logo">🌿 LeafSentry AI</div>
    <div class="nav-buttons">
""", unsafe_allow_html=True)

cols = st.columns(6, gap="small")

with cols[0]:
    nav_button("Home", "Home")
with cols[1]:
    nav_button("Plant", "Plant")
with cols[2]:
    nav_button("Blog", "Blog")
with cols[3]:
    nav_button("Privacy", "Privacy")
with cols[4]:
    nav_button("Contact", "Contact")
with cols[5]:
    nav_button("Login", "Login")

st.markdown("</div></div>", unsafe_allow_html=True)

# ================= HOME =================
if st.session_state.page == "Home":

    st.markdown("""
<div class="hero">
    <div class="hero-title">
        LeafSentry AI
    </div>

    <div class="hero-sub">
        Smart plant disease detection powered by AI and deep learning.
    </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""<div class="card"><div class="card-title">🌿 Monitoring</div>Detect plant health instantly.</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="card"><div class="card-title">⚡ Fast AI</div>Instant leaf analysis.</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""<div class="card"><div class="card-title">🧠 Deep Learning</div>AI disease classification.</div>""", unsafe_allow_html=True)

# ================= OTHER PAGES =================
elif st.session_state.page == "Plant":
    st.title("🌱 Plant Info")

elif st.session_state.page == "Blog":
    st.title("📰 Blog")

elif st.session_state.page == "Privacy":
    st.title("🔒 Privacy Policy")

elif st.session_state.page == "Contact":
    st.title("📞 Contact Us")

# ================= LOGIN =================
elif st.session_state.page == "Login":

    st.title("🔐 Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(u, p):
                st.session_state.logged_in = True
                st.success("Logged in")
            else:
                st.error("Wrong credentials")

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

    img = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

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

        fig = px.bar(x=classes, y=probs*100)
        st.plotly_chart(fig, use_container_width=True)
