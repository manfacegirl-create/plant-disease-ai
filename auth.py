import streamlit as st
import sqlite3
import hashlib

# ================= DATABASE =================
def connect_db():
    return sqlite3.connect("users.db", check_same_thread=False)

conn = connect_db()
c = conn.cursor()

# Create table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
conn.commit()

# ================= PASSWORD HASH =================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ================= SIGNUP =================
def signup_user(username, password):
    try:
        hashed = hash_password(password)
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except:
        return False

# ================= LOGIN =================
def login_user(username, password):
    hashed = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed))
    return c.fetchone()

# ================= UI =================
def auth_page():
    st.title("🔐 LeafSentry Authentication")

    menu = ["Login", "Sign Up"]
    choice = st.radio("Select Option", menu)

    if choice == "Login":
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            result = login_user(username, password)
            if result:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.success(f"Welcome {username} 👋")
                st.rerun()
            else:
                st.error("Invalid credentials")

    elif choice == "Sign Up":
        st.subheader("Create Account")

        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Sign Up"):
            if signup_user(new_user, new_pass):
                st.success("Account created! You can login now.")
            else:
                st.error("Username already exists")

# ================= SESSION CHECK =================
def check_auth():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    return st.session_state.logged_in

# ================= LOGOUT =================
def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()