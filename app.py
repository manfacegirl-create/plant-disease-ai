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

# ================= 🌿 CLEAN PLANT THEME CSS =================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(180deg, #eaf7ee 0%, #dff3e4 50%, #cdebd6 100%);
    color: #12301d;
}

/* STREAMLIT CLEANUP */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ================= NAVBAR ================= */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 14px 20px;
    background: rgba(34, 85, 55, 0.92);

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
    border-radius: 12px;

    padding: 9px 14px;
    font-weight: 700;

    width: 100%;
    transition: 0.25s;
}

/* HOVER */
.stButton > button:hover {
    background: #3f8f5c;
    box-shadow: 0 0 10px rgba(76,175,114,0.4);
    transform: translateY(-2px);
}

/* ACTIVE */
.active-btn > button {
    background: #a6e3b7 !important;
    color: #12301d !important;
    font-weight: 900;
}

/* ================= HERO (FIXED POSITION) ================= */
.hero {
    margin-top: 30px;   /* 🔥 FIXED SPACING */
    padding: 95px 55px;

    border-radius: 25px;

    background:
    linear-gradient(rgba(0,50,20,0.55), rgba(0,20,10,0.75)),
    url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
}

/* BIG TITLE FIX */
.hero-title {
    font-size: clamp(50px, 6vw, 92px);
    font-weight: 1000;
    color: #eafff0;
    line-height: 1.1;
}

/* SUBTITLE */
.hero-sub {
    font-size: 20px;
    color: #d7f7df;
    max-width: 750px;
    margin-top: 12px;
}

/* ================= CARDS ================= */
.card {
    background: rgba(255,255,255,0.80);
    border: 1px solid #bde7c8;
    border-radius: 18px;
    padding: 22px;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(46,125,70,0.25);
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

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR (FIXED STRUCTURE) =================
col_logo, c1, c2, c3, c4, c5, c6 = st.columns([2,1,1,1,1,1,1])

with col_logo:
    st.markdown('<div class="logo">🌿 LeafSentry AI</div>', unsafe_allow_html=True)

pages = ["Home", "Plant", "Blog", "Privacy", "Contact", "Login"]
cols = [c1, c2, c3, c4, c5, c6]

for i, p in enumerate(pages):
    with cols[i]:
        if st.session_state.page == p:
            st.markdown('<div class="active-btn">', unsafe_allow_html=True)
            if st.button(p, key=p):
                st.session_state.page = p
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            if st.button(p, key=p):
                st.session_state.page = p

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
