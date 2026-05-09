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

# ================= SESSION STATE =================
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

# ================= AUTH FUNCTIONS =================
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def signup(username, password):
    try:
        c.execute(
            "INSERT INTO users VALUES (?,?)",
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

def strong_password(password):
    return (
        len(password) >= 6
        and any(i.isdigit() for i in password)
        and any(i.isalpha() for i in password)
    )

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* ================= GLOBAL ================= */

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* BACKGROUND */
.stApp {
    background: linear-gradient(
        180deg,
        #eaf7ee 0%,
        #dff3e4 50%,
        #cdebd6 100%
    );

    color: #12301d;
}

/* HIDE STREAMLIT DEFAULT */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* ================= NAVBAR ================= */

.navbar-wrap {

    background: rgba(34, 85, 55, 0.95);

    padding-top: 18px;
    padding-bottom: 18px;
    padding-left: 30px;
    padding-right: 30px;

    border-radius: 0px;

    margin-bottom: 30px;
}

/* LOGO */
.logo {

    font-size: 34px;

    font-weight: 900;

    color: white;

    margin-bottom: 18px;
}

/* FIX COLUMN SPACING */
div[data-testid="stHorizontalBlock"] {
    gap: 0.8rem !important;
}

/* BUTTONS */
.stButton > button {

    width: 100%;

    background: transparent;

    color: #18442a;

    border: 2px solid #48a868;

    border-radius: 14px;

    padding-top: 12px;
    padding-bottom: 12px;

    font-weight: 700;

    transition: 0.25s;
}

/* HOVER */
.stButton > button:hover {

    background: #48a868;

    color: white;

    transform: translateY(-2px);

    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}

/* ACTIVE BUTTON */
.active-btn button {

    background: #48a868 !important;

    color: white !important;
}

/* ================= HERO ================= */

.hero {

    margin-top: 10px;

    padding-top: 120px;
    padding-bottom: 120px;

    padding-left: 55px;
    padding-right: 55px;

    border-radius: 28px;

    background:
    linear-gradient(
        rgba(0,50,20,0.55),
        rgba(0,20,10,0.75)
    ),

    url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;

    background-position: center;

    margin-bottom: 28px;
}

/* HERO TITLE */
.hero-title {

    font-size: clamp(58px, 7vw, 95px);

    font-weight: 1000;

    color: #ffffff;

    line-height: 1.1;
}

/* HERO SUBTEXT */
.hero-sub {

    font-size: 22px;

    color: #ddffe5;

    max-width: 760px;

    margin-top: 16px;
}

/* ================= CARDS ================= */

.card {

    background: rgba(255,255,255,0.85);

    border: 1px solid #bde7c8;

    border-radius: 22px;

    padding: 28px;

    transition: 0.3s;

    min-height: 180px;
}

/* CARD HOVER */
.card:hover {

    transform: translateY(-6px);

    box-shadow: 0px 10px 22px rgba(46,125,70,0.25);
}

/* CARD TITLE */
.card-title {

    color: #2f6f46;

    font-size: 28px;

    font-weight: 900;

    margin-bottom: 14px;
}

/* CARD TEXT */
.card-text {

    font-size: 17px;

    color: #23452e;

    line-height: 1.7;
}

/* ================= INPUTS ================= */

.stTextInput input {

    background: #f4fff7 !important;

    border: 1px solid #bde7c8 !important;

    border-radius: 10px !important;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"] {

    background: #f6fff8;

    border: 1px solid #bde7c8;

    padding: 20px;

    border-radius: 15px;
}

/* TITLES */
h1, h2, h3 {

    color: #1e5631;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================

st.markdown(
    '<div class="navbar-wrap">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="logo">🌿 LeafSentry AI</div>',
    unsafe_allow_html=True
)

pages = [
    "Home",
    "Plant",
    "Blog",
    "Privacy",
    "Contact",
    "Login"
]

cols = st.columns(6)

for i, page in enumerate(pages):

    with cols[i]:

        active = st.session_state.page == page

        if active:
            st.markdown(
                '<div class="active-btn">',
                unsafe_allow_html=True
            )

        if st.button(page, key=page):
            st.session_state.page = page

        if active:
            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)

# ================= HOME PAGE =================

if st.session_state.page == "Home":

    st.markdown("""
    <div class="hero">

        <div class="hero-title">
            LeafSentry AI
        </div>

        <div class="hero-sub">
            Smart AI system for plant disease detection
            and crop health monitoring.
        </div>

    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # CARD 1
    with c1:

        st.markdown("""
        <div class="card">

            <div class="card-title">
                🌿 Monitoring
            </div>

            <div class="card-text">
                Detect plant diseases instantly using
                advanced AI-powered crop monitoring.
            </div>

        </div>
        """, unsafe_allow_html=True)

    # CARD 2
    with c2:

        st.markdown("""
        <div class="card">

            <div class="card-title">
                ⚡ Speed
            </div>

            <div class="card-text">
                Ultra-fast predictions with real-time
                plant health analysis and insights.
            </div>

        </div>
        """, unsafe_allow_html=True)

    # CARD 3
    with c3:

        st.markdown("""
        <div class="card">

            <div class="card-title">
                🧠 AI Model
            </div>

            <div class="card-text">
                Deep learning classification model
                trained for accurate disease detection.
            </div>

        </div>
        """, unsafe_allow_html=True)

# ================= PLANT PAGE =================

elif st.session_state.page == "Plant":

    st.title("🌱 Plant Information")

    st.write(
        "Upload plant images and analyze crop health."
    )

    uploaded = st.file_uploader(
        "Upload Plant Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded:

        image = Image.open(uploaded)

        st.image(
            image,
            caption="Uploaded Plant",
            use_container_width=True
        )

        st.success(
            "AI analysis ready."
        )

# ================= BLOG PAGE =================

elif st.session_state.page == "Blog":

    st.title("📰 Blog")

    st.write(
        "Latest agricultural AI news and updates."
    )

# ================= PRIVACY PAGE =================

elif st.session_state.page == "Privacy":

    st.title("🔒 Privacy Policy")

    st.write(
        "Your uploaded images and data are protected."
    )

# ================= CONTACT PAGE =================

elif st.session_state.page == "Contact":

    st.title("📞 Contact Us")

    st.write(
        "Email: support@leafsentry.ai"
    )

# ================= LOGIN PAGE =================

elif st.session_state.page == "Login":

    st.title("🔐 Login System")

    tab1, tab2 = st.tabs([
        "Login",
        "Sign Up"
    ])

    # LOGIN TAB
    with tab1:

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login User"):

            if login(username, password):

                st.session_state.logged_in = True

                st.success(
                    "Login successful."
                )

            else:

                st.error(
                    "Invalid username or password."
                )

    # SIGNUP TAB
    with tab2:

        new_user = st.text_input(
            "New Username"
        )

        new_pass = st.text_input(
            "New Password",
            type="password"
        )

        if st.button("Create Account"):

            if strong_password(new_pass):

                if signup(new_user, new_pass):

                    st.success(
                        "Account created successfully."
                    )

                else:

                    st.error(
                        "Username already exists."
                    )

            else:

                st.warning(
                    "Password must contain letters, numbers, and be at least 6 characters."
                )
