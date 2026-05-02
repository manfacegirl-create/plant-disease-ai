def auth_page():
    st.set_page_config(page_title="LeafSentry Access", layout="wide")

    # ===== STYLE =====
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #5f6dfc, #d946ef);
    }

    .container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }

    .card {
        background: white;
        border-radius: 20px;
        width: 900px;
        height: 480px;
        display: flex;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    }

    .left {
        width: 45%;
        background: #f3f4f6;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }

    .right {
        width: 55%;
        padding: 40px;
    }

    .title {
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
    }

    .subtitle {
        font-size: 14px;
        color: gray;
        margin-bottom: 20px;
    }

    .login-btn button {
        background: #55b043;
        color: white;
        border-radius: 25px;
        font-weight: bold;
    }

    .forgot {
        font-size: 13px;
        color: gray;
        cursor: pointer;
    }

    </style>
    """, unsafe_allow_html=True)

    # ===== LAYOUT =====
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    left, right = st.columns([1, 1.2])

    # ===== LEFT =====
    with left:
        st.markdown('<div class="left">', unsafe_allow_html=True)

        # 👉 replace with your real logo
        st.image("https://via.placeholder.com/150", width=150)

        st.markdown("<h2 style='color:#ec4899;'>LeafSentry</h2>", unsafe_allow_html=True)
        st.markdown("<h1 style='color:#f97316;'>Access</h1>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ===== RIGHT =====
    with right:
        st.markdown('<div class="right">', unsafe_allow_html=True)

        st.markdown('<div class="title">🔐 LeafSentry Access</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Secure login to your dashboard</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])

        # ===== LOGIN =====
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")

            st.markdown("<div class='forgot'>Use 'Forgot Password' tab if needed</div>", unsafe_allow_html=True)

            if st.button("LOGIN", use_container_width=True):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.success("Access Granted 🚀")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        # ===== SIGN UP =====
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

        # ===== FORGOT PASSWORD =====
        with tab3:
            reset_user = st.text_input("Enter Username", key="reset_user")
            new_password = st.text_input("New Password", type="password", key="reset_pass")

            if st.button("Reset Password", use_container_width=True):
                if not strong_password(new_password):
                    st.warning("Weak password")
                else:
                    c.execute("SELECT * FROM users WHERE username=?", (reset_user,))
                    if c.fetchone():
                        new_hash = hash_password(new_password)
                        c.execute("UPDATE users SET password=? WHERE username=?", (new_hash, reset_user))
                        conn.commit()
                        st.success("Password updated successfully ✅")
                    else:
                        st.error("User not found")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
