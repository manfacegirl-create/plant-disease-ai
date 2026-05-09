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
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def strong_password(password):
    return (
        len(password) >= 6
        and any(i.isdigit() for i in password)
        and any(i.isalpha() for i in password)
    )

# ================= AUTH =================
def signup(username, password):
    try:
        c.execute(
            "INSERT INTO users VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except:
        return False

def login(username, password):
    c.execute(
        "SELECT password FROM users WHERE username=?",
        (username,)
    )
    data = c.fetchone()

    if data:
        return check_password(password, data[0])

    return False

# ================= SESSION =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ================= CSS =================
st.markdown("""
<style>

.stApp{
    background:#08130d;
    color:white;
}

/* HIDE STREAMLIT UI */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* REMOVE GAP BETWEEN NAV COLUMNS */
div[data-testid="column"]{
    padding:0px !important;
    margin:0px !important;
}

/* NAV BUTTONS - CONNECTED BAR */
div.stButton > button{
    width:100%;
    height:65px;
    background:#15803d;
    color:white;
    border:none;
    border-radius:0px;
    font-size:16px;
    font-weight:700;
    transition:0.25s;
}

/* HOVER */
div.stButton > button:hover{
    background:#e5e7eb;
    color:#15803d;
    transform:scale(1.02);
}

/* HERO */
.hero{
    background:
    linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.75)),
    url("https://images.unsplash.com/photo-1466692476868-aef1dfb1e735?q=80&w=2070&auto=format&fit=crop");

    background-size:cover;
    background-position:center;

    padding:100px 70px;
    border-radius:25px;
    margin-top:20px;
    margin-bottom:30px;
}

.hero-title{
    font-size:70px;
    font-weight:900;
}

.hero-sub{
    font-size:22px;
    color:#d1fae5;
    margin-top:20px;
    max-width:700px;
}

/* CARDS */
.card{
    background:#101827;
    border:1px solid #1f5134;
    border-radius:20px;
    padding:30px;
    margin-top:20px;
}

.card-title{
    color:#4ade80;
    font-size:28px;
    font-weight:700;
}

/* INPUT */
.stTextInput input{
    background:#0b1d13 !important;
    color:white !important;
    border:1px solid #1f5134 !important;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"]{
    background:#101827;
    border:1px solid #1f5134;
    padding:20px;
    border-radius:15px;
}

/* FOOTER */
.footer{
    text-align:center;
    margin-top:60px;
    padding:30px;
    color:#a7f3d0;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR (FIXED) =================
cols = st.columns(6, gap="small")

nav_items = [
    ("Home", "Home"),
    ("Plant", "Plant"),
    ("Blog", "Blog"),
    ("Privacy Policy", "Privacy"),
    ("Contact Us", "Contact"),
    ("Login", "Login")
]

for i, (label, page) in enumerate(nav_items):
    with cols[i]:
        if st.button(label, key=page):
            st.session_state.page = page

# ================= HOME =================
if st.session_state.page == "Home":

    st.markdown("""
<div class="hero">

<div class="hero-title">
LeafSentry AI
</div>

<div class="hero-sub">
Smart plant disease detection powered by deep learning and AI.
</div>

</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
<div class="card">
<div class="card-title">🌿 Plant Monitoring</div>
<p>Detect unhealthy plants instantly.</p>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="card">
<div class="card-title">⚡ Fast Detection</div>
<p>Upload leaf images for instant AI prediction.</p>
</div>
""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div class="card">
<div class="card-title">🧠 Deep Learning</div>
<p>Neural network disease classification.</p>
</div>
""", unsafe_allow_html=True)

# ================= PLANT =================
elif st.session_state.page == "Plant":
    st.title("🌱 Plant Information")

# ================= BLOG =================
elif st.session_state.page == "Blog":
    st.title("📰 Blog")

# ================= PRIVACY =================
elif st.session_state.page == "Privacy":
    st.title("🔒 Privacy Policy")

# ================= CONTACT =================
elif st.session_state.page == "Contact":
    st.title("📞 Contact Us")

# ================= LOGIN =================
elif st.session_state.page == "Login":

    st.title("🔐 Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login Account"):
            if login(username, password):
                st.session_state.logged_in = True
                st.session_state.page = "ML"
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid login")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if strong_password(new_pass):
                if signup(new_user, new_pass):
                    st.success("Account Created")
                else:
                    st.error("Username exists")
            else:
                st.warning("Weak password")

# ================= ML =================
elif st.session_state.page == "ML":

    if not st.session_state.logged_in:
        st.warning("Please login first")
        st.session_state.page = "Login"
        st.rerun()

    st.title("🧠 Plant Disease Detection")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Home"
        st.rerun()

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

    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)

        c1, c2 = st.columns(2)

        with c1:
            st.image(image, use_container_width=True)

        x = transform(image).unsqueeze(0)

        if model:
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].numpy()
        else:
            probs = np.array([0.5, 0.5])

        pred = np.argmax(probs)
        conf = probs[pred] * 100

        with c2:
            st.markdown(f"""
<div class="card">
<div class="card-title">Prediction</div>
<h1>{classes[pred]}</h1>
<h3>{conf:.2f}%</h3>
</div>
""", unsafe_allow_html=True)

            st.progress(int(conf))

        fig = px.bar(x=classes, y=probs * 100)
        st.plotly_chart(fig, use_container_width=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
© 2026 LeafSentry AI
</div>
""", unsafe_allow_html=True)
