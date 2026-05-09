# ================= NAVIGATION BAR =================
st.markdown("""
<style>

/* ================= NAVBAR ================= */

.nav-container{
    background:#15803d;
    padding:0px;
    border-radius:0px;
    margin-bottom:25px;
    border:none;
}

div.stButton > button{
    width:100%;
    height:70px;
    background:#15803d;
    color:white;
    border:none;
    border-radius:0px;
    font-size:24px;
    font-weight:600;
    transition:0.3s;
}

div.stButton > button:hover{
    background:#e5e7eb;
    color:#15803d;
}

.login-btn button{
    background:#166534 !important;
}

</style>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "show_login" not in st.session_state:
    st.session_state.show_login = False

# ================= MENU BAR =================
st.markdown('<div class="nav-container">', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1.5,1.2,1])

with col1:
    if st.button("Home"):
        st.session_state.page = "Home"

with col2:
    if st.button("Plant"):
        st.session_state.page = "Plant"

with col3:
    if st.button("Blog"):
        st.session_state.page = "Blog"

with col4:
    if st.button("Privacy Policy"):
        st.session_state.page = "Privacy"

with col5:
    if st.button("Contact Us"):
        st.session_state.page = "Contact"

with col6:
    if st.button("Login"):
        st.session_state.show_login = True

st.markdown('</div>', unsafe_allow_html=True)

# ================= LOGIN PAGE =================
if st.session_state.show_login:

    st.markdown("""
    <div style="
        background:#102117;
        padding:40px;
        border-radius:20px;
        border:1px solid #1f5134;
        margin-top:40px;
    ">
        <h1 style="color:#4ade80;">
            🔐 Login To LeafSentry AI
        </h1>

        <p style="color:white;">
            Access your AI Plant Disease Detection dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login Account"):

            if login(username, password):
                st.session_state.logged_in = True
                st.success("Login Successful")
                st.rerun()

            else:
                st.error("Invalid Username or Password")

    with tab2:

        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Create Account"):

            if strong_password(new_pass):

                if signup(new_user, new_pass):
                    st.success("Account Created")

                else:
                    st.error("Username Already Exists")

            else:
                st.warning("Password must contain numbers and letters")

    st.stop()
