import streamlit as st
import sqlite3
import bcrypt

# ================= DATABASE =================
def connect_db():
    return sqlite3.connect("users.db", check_same_thread=False)

conn = connect_db()
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

# ================= VALIDATION =================
def strong_password(password):
    return (
        len(password) >= 6 and
        any(c.isdigit() for c in password) and
        any(c.isalpha() for c in password)
    )

# ================= AUTH FUNCTIONS =================
def signup_user(username, password):
    try:
        hashed = hash_password(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    data = c.fetchone()

    if data:
        stored_hash = data[0]
        if check_password(password, stored_hash):
            return True

    return False

# ================= UI =================
def auth_page():
    st.markdown("""
    <style>
    .auth-box {
        background: rgba(255,255,255,0.05);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(125,211,252,0.2);
        backdrop-filter: blur(10px);
        max-width: 400px;
        margin: auto;
        margin-top: 80px;
    }
    .title {
        text-align: center;
        font-size: 28px;
        color: #7dd3fc;
        margin-bottom: 10px;
    }
   
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-box">', unsafe_allow_html=True)
    st.markdown('<div class="title">🔐 LeafSentry Access</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # ================= LOGIN =================
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", use_container_width=True):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success("Access Granted 🚀")
                st.rerun()
            else:
                st.error("Invalid username or password")

    # ================= SIGNUP =================
    with tab2:
        new_user = st.text_input("New Username", key="signup_user")
        new_pass = st.text_input("New Password", type="password", key="signup_pass")

        if st.button("Create Account", use_container_width=True):
            if not strong_password(new_pass):
                st.warning("Password must be at least 6 characters with letters and numbers")
            else:
                if signup_user(new_user, new_pass):
                    st.success("Account created! You can login now.")
                else:
                    st.error("Username already exists")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= SESSION =================
def check_auth():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()
