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

# ================= AUTH =================
def signup_user(username, password):
    try:
        hashed = hash_password(password)
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except:
        return False

def login_user(username, password):
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    data = c.fetchone()

    if data:
        if check_password(password, data[0]):
            return True
    return False

# ================= SESSION =================
def check_auth():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

# ================= UI =================
def auth_page():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #5f6dfc, #d946ef);
    }
    .card {
        background: white;
        border-radius: 20px;
        padding: 40px;
        max-width: 500px;
        margin: auto;
        margin-top: 80px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🔐 LeafSentry Access")

    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])

    # ===== LOGIN =====
    with tab1:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", key="login_btn"):
            if login_user(u, p):
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    # ===== SIGNUP =====
    with tab2:
        u = st.text_input("New Username", key="signup_user")
        p = st.text_input("New Password", type="password", key="signup_pass")

        if st.button("Create Account", key="signup_btn"):
            if not strong_password(p):
                st.warning("Weak password")
            else:
                if signup_user(u, p):
                    st.success("Account created")
                else:
                    st.error("Username exists")

    # ===== RESET =====
    with tab3:
        u = st.text_input("Username", key="reset_user")
        p = st.text_input("New Password", type="password", key="reset_pass")

        if st.button("Reset Password", key="reset_btn"):
            if not strong_password(p):
                st.warning("Weak password")
            else:
                c.execute("SELECT * FROM users WHERE username=?", (u,))
                if c.fetchone():
                    new_hash = hash_password(p)
                    c.execute("UPDATE users SET password=? WHERE username=?", (new_hash, u))
                    conn.commit()
                    st.success("Password updated")
                else:
                    st.error("User not found")

    st.markdown('</div>', unsafe_allow_html=True)
