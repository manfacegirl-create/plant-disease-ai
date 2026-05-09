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

# ================= ROUTING (NEW SYSTEM) =================
query_params = st.query_params
page = query_params.get("page", "Home")

if isinstance(page, list):
    page = page[0]

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

# ================= CSS (NEON GLASS UI) =================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #0b1a12, #050a07);
    color: white;
}

/* HIDE STREAMLIT */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ================= GLASS NAVBAR ================= */
.navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 12px 20px;

    background: rgba(10, 25, 18, 0.6);
    backdrop-filter: blur(12px);

    border-bottom: 1px solid rgba(34, 197, 94, 0.3);
}

/* NAV LINKS */
.nav-links {
    display: flex;
    gap: 10px;
}

/* NAV BUTTON STYLE */
.nav-item {
    padding: 10px 18px;
    border-radius: 12px;
    text-decoration: none;
    color: #d1fae5;
    font-weight: 600;

    transition: 0.3s;
}

/* HOVER */
.nav-item:hover {
    background: rgba(34, 197, 94, 0.15);
    box-shadow: 0 0 10px #22c55e;
}

/* ACTIVE PAGE */
.active {
    background: #22c55e;
    color: black !important;
    box-shadow: 0 0 15px #22c55e;
}

/* MOBILE HAMBURGER */
.menu {
    display: none;
}

/* HERO */
.hero {
    background:
    linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.8)),
    url("https://images.unsplash.com/photo-1466692476868-aef1dfb1e735?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
    padding: 100px 60px;
    border-radius: 25px;
    margin-top: 20px;
}

.hero-title {
    font-size: 70px;
    font-weight: 900;
    color: #4ade80;
    text-shadow: 0 0 20px #22c55e;
}

.hero-sub {
    font-size: 22px;
    color: #d1fae5;
    max-width: 700px;
}

/* CARDS */
.card {
    background: rgba(16, 24, 39, 0.8);
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
    font-weight: 700;
}

/* MOBILE */
@media (max-width: 768px) {
    .nav-links {
        display: none;
    }

    .menu {
        display: block;
        color: white;
        font-size: 24px;
    }
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR (GLASS + ACTIVE STATE) =================
nav_items = ["Home", "Plant", "Blog", "Privacy", "Contact", "Login"]

st.markdown('<div class="navbar">', unsafe_allow_html=True)

st.markdown("🌿 <b>LeafSentry AI</b>", unsafe_allow_html=True)

links_html = '<div class="nav-links">'

for item in nav_items:
    active_class = "active" if page == item else ""
    links_html += f'<a class="nav-item {active_class}" href="?page={item}">{item}</a>'

links_html += "</div>"

st.markdown(links_html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ================= ROUTE SYNC =================
st.session_state.page = page

# ================= HOME =================
if page == "Home":

    st.markdown("""
<div class="hero">
<div class="hero-title">LeafSentry AI</div>
<div class="hero-sub">
Smart plant disease detection powered by AI & deep learning.
</div>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
<div class="card"><div class="card-title">🌿 Monitoring</div><p>Detect plant health instantly.</p></div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="card"><div class="card-title">⚡ Fast AI</div><p>Instant predictions from images.</p></div>
""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div class="card"><div class="card-title">🧠 Deep Learning</div><p>Neural network classification.</p></div>
""", unsafe_allow_html=True)

# ================= PLANT =================
elif page == "Plant":
    st.title("🌱 Plant Info")

# ================= BLOG =================
elif page == "Blog":
    st.title("📰 Blog")

# ================= PRIVACY =================
elif page == "Privacy":
    st.title("🔒 Privacy Policy")

# ================= CONTACT =================
elif page == "Contact":
    st.title("📞 Contact")

# ================= LOGIN =================
elif page == "Login":

    st.title("🔐 Login System")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(u, p):
                st.session_state.logged_in = True
                st.success("Login success")
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
elif page == "ML":

    if not st.session_state.logged_in:
        st.warning("Login required")
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

    img = st.file_uploader("Upload Leaf Image", type=["png","jpg","jpeg"])

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

# ================= FOOTER =================
st.markdown("""
<div style="text-align:center; padding:30px; color:#86efac;">
© 2026 LeafSentry AI • Built with Streamlit
</div>
""", unsafe_allow_html=True)
