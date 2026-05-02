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
    st.set_page_config(page_title="LeafSentry Access", layout="wide")

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

    # LOGIN
    with tab1:
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(user, pw):
                st.session_state.logged_in = True
                st.session_state.user = user
                st.success("Access Granted 🚀")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # SIGNUP
    with tab2:
        new_user = st.text_input("New Username")
        new_pw = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if not strong_password(new_pw):
                st.warning("Weak password")
            else:
                if signup_user(new_user, new_pw):
                    st.success("Account created!")
                else:
                    st.error("Username exists")

    # FORGOT PASSWORD
    with tab3:
        user = st.text_input("Username")
        new_pw = st.text_input("New Password", type="password")

        if st.button("Reset Password"):
            if not strong_password(new_pw):
                st.warning("Weak password")
            else:
                c.execute("SELECT * FROM users WHERE username=?", (user,))
                if c.fetchone():
                    new_hash = hash_password(new_pw)
                    c.execute("UPDATE users SET password=? WHERE username=?", (new_hash, user))
                    conn.commit()
                    st.success("Password reset successful")
                else:
                    st.error("User not found")

    st.markdown('</div>', unsafe_allow_html=True)
