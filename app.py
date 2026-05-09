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

# ================= PAGE =================
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
def hash_password(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt())

def check_password(pw, hashed):
    return bcrypt.checkpw(pw.encode(), hashed)

def strong_password(pw):
    return len(pw) >= 6 and any(i.isdigit() for i in pw)

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

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ================= STYLE =================
st.markdown("""
<style>

.stApp{
    background:#08130d;
    color:white;
}

/* NAVBAR */
.navbar{
    background:#0b1d13;
    padding:18px;
    border-radius:14px;
    border:1px solid #1f5134;
    margin-bottom:25px;
}

.navtitle{
    font-size:34px;
    font-weight:800;
    color:#4ade80;
}

.hero{
    background:
    linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.7)),
    url('https://images.unsplash.com/photo-1466692476868-aef1dfb1e735?q=80&w=2070&auto=format&fit=crop');

    background-size:cover;
    background-position:center;

    padding:90px;
    border-radius:25px;
    border:1px solid #1f5134;
}

.hero-title{
    font-size:70px;
    font-weight:900;
    color:white;
}

.hero-sub{
    font-size:20px;
    color:#d1fae5;
    margin-top:15px;
    max-width:700px;
}

/* CARDS */
.card{
    background:#102117;
    border:1px solid #1f5134;
    border-radius:20px;
    padding:30px;
    margin-top:20px;
    transition:0.3s;
}

.card:hover{
    transform:translateY(-5px);
    box-shadow:0 0 20px rgba(74,222,128,0.2);
}

.card-title{
    color:#4ade80;
    font-size:26px;
    font-weight:700;
}

/* BUTTONS */
.stButton>button{
    width:100%;
    height:50px;
    border-radius:12px;
    border:none;
    background:linear-gradient(90deg,#16a34a,#22c55e);
    color:white;
    font-weight:700;
}

/* INPUT */
.stTextInput input{
    background:#0b1d13 !important;
    color:white !important;
    border:1px solid #1f5134 !important;
}

/* FOOTER */
.footer{
    text-align:center;
    margin-top:80px;
    color:#a7f3d0;
    padding:30px;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================
st.markdown("""
<div class="navbar">
    <div class="navtitle">🌿 LeafSentry AI</div>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(6)

with nav1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"

with nav2:
    if st.button("🌱 Plant"):
        st.session_state.page = "Plant"

with nav3:
    if st.button("📰 Blog"):
        st.session_state.page = "Blog"

with nav4:
    if st.button("🔒 Privacy Policy"):
        st.session_state.page = "Privacy"

with nav5:
    if st.button("📞 Contact Us"):
        st.session_state.page = "Contact"

with nav6:
    if st.button("🧠 ML Project"):
        st.session_state.page = "ML"

# ================= AUTH PAGE =================
if not st.session_state.logged_in:

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            Smart Plant Disease Detection
        </div>

        <div class="hero-sub">
            AI powered agriculture system for disease identification,
            healthy farming and smart monitoring.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🔐 Login System")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login(u, p):
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid account")

    with tab2:
        su = st.text_input("New Username")
        sp = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if strong_password(sp):
                if signup(su, sp):
                    st.success("Account Created")
                else:
                    st.error("Username already exists")
            else:
                st.warning("Password too weak")

    st.stop()

# ================= HOME =================
if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            Welcome to LeafSentry AI
        </div>

        <div class="hero-sub">
            Detect plant diseases instantly using deep learning
            and intelligent crop analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🌿 Healthy Crops</div>
            <p>Improve farming quality with AI monitoring.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">⚡ Fast Detection</div>
            <p>Get results in seconds after uploading.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">🧠 Smart AI</div>
            <p>Powered by neural network prediction models.</p>
        </div>
        """, unsafe_allow_html=True)

# ================= PLANT PAGE =================
elif st.session_state.page == "Plant":

    st.title("🌱 Plant Information")

    st.markdown("""
    <div class="card">
        <div class="card-title">Common Plant Diseases</div>

        <ul>
            <li>Leaf Spot</li>
            <li>Powdery Mildew</li>
            <li>Root Rot</li>
            <li>Bacterial Wilt</li>
            <li>Rust Fungus</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ================= BLOG PAGE =================
elif st.session_state.page == "Blog":

    st.title("📰 Plant Blog")

    for i in range(1,4):
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Blog Article {i}</div>
            <p>
            Learn modern agriculture techniques and disease prevention
            strategies for healthy farming.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ================= PRIVACY =================
elif st.session_state.page == "Privacy":

    st.title("🔒 Privacy Policy")

    st.markdown("""
    <div class="card">
        We do not share your uploaded images or account data.
        Your information stays securely stored locally.
    </div>
    """, unsafe_allow_html=True)

# ================= CONTACT =================
elif st.session_state.page == "Contact":

    st.title("📞 Contact Us")

    st.markdown("""
    <div class="card">
        📧 Email: leafsentry@gmail.com

        🌐 Website: www.leafsentryai.com

        📱 Phone: +60 12-345 6789
    </div>
    """, unsafe_allow_html=True)

# ================= ML PAGE =================
elif st.session_state.page == "ML":

    st.title("🧠 Leaf Disease Detection")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

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
            model.load_state_dict(
                torch.load("cnn.pth", map_location="cpu")
            )
            model.eval()
            return model
        except:
            return None

    model = load_model()

    # ================= UPLOAD =================
    uploaded = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        img = Image.open(uploaded)

        col1, col2 = st.columns(2)

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
            <div class="card">
                <div class="card-title">
                    Prediction
                </div>

                <h1>{classes[pred]}</h1>

                <h3>{conf:.2f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(conf))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={"x":"Class","y":"Confidence"}
        )

        st.plotly_chart(fig, use_container_width=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
© 2026 LeafSentry AI • Smart Agriculture Platform
</div>
""", unsafe_allow_html=True)
