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

# ================= 🌿 CLEAN UI THEME =================
st.markdown("""
<style>

/* ================= BACKGROUND ================= */
.stApp {
    background: linear-gradient(180deg, #eaf7ee 0%, #dff3e4 50%, #cdebd6 100%);
    color: #0b1f14;
}

/* HIDE STREAMLIT DEFAULT UI */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* ================= NAVBAR (FIXED CONTRAST) ================= */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 14px 22px;

    background: #1f4d33;   /* solid no transparency */

    border-bottom: 2px solid #2f6f46;

    position: sticky;
    top: 0;
    z-index: 999;
}

/* LOGO */
.logo {
    font-size: 24px;
    font-weight: 900;
    color: #ffffff;

    text-shadow: 0px 2px 6px rgba(0,0,0,0.35);
    white-space: nowrap;
}

/* BUTTONS */
.stButton > button {
    background: #ffffff;
    color: #1f4d33;

    border: 2px solid #4caf72;
    border-radius: 12px;

    padding: 8px 14px;
    font-weight: 700;

    width: 100%;

    transition: 0.25s;
}

/* HOVER */
.stButton > button:hover {
    background: #d9f7e2;
    color: #12301d;

    transform: translateY(-2px);
    box-shadow: 0 0 12px rgba(76,175,114,0.35);
}

/* ACTIVE BUTTON */
.active-btn > button {
    background: #4caf72 !important;
    color: #ffffff !important;

    font-weight: 900;
    border: 2px solid #ffffff;
}

/* ================= HERO ================= */
.hero {
    margin-top: 25px;
    padding: 95px 55px;
    border-radius: 25px;

    background:
    linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.65)),
    url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
}

/* HERO TEXT */
.hero-title {
    font-size: clamp(50px, 6vw, 92px);
    font-weight: 1000;
    color: #ffffff;

    text-shadow: 0px 4px 10px rgba(0,0,0,0.6);
}

.hero-sub {
    font-size: 20px;
    color: #eafff0;

    text-shadow: 0px 2px 6px rgba(0,0,0,0.5);
    max-width: 750px;
    margin-top: 12px;
}

/* ================= CARDS ================= */
.card {
    background: #ffffff;
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
    color: #1f4d33;
    font-size: 22px;
    font-weight: 900;
}

/* INPUT FIX */
.stTextInput input {
    background: #ffffff !important;
    border: 1px solid #bde7c8 !important;
    border-radius: 10px !important;
    color: #0b1f14 !important;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================
st.markdown('<div class="navbar">', unsafe_allow_html=True)

st.markdown('<div class="logo">🌿 LeafSentry AI</div>', unsafe_allow_html=True)

pages = ["Home", "Plant", "Blog", "Privacy", "Contact", "Login"]
cols = st.columns(len(pages))

for i, p in enumerate(pages):
    with cols[i]:
        if st.session_state.page == p:
            if st.button(p, key=f"active_{p}"):
                st.session_state.page = p
        else:
            if st.button(p, key=p):
                st.session_state.page = p

st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="card"><div class="card-title">🌿 Monitoring</div>Detect plant health instantly.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="card-title">⚡ Speed</div>Fast AI predictions.</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card"><div class="card-title">🧠 AI Model</div>Deep learning classification.</div>', unsafe_allow_html=True)

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
