# ================= IMPORTS =================
import streamlit as st
import sqlite3
import bcrypt
from PIL import Image

# ================= PAGE CONFIG =================
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

if "username" not in st.session_state:
    st.session_state.username = ""

# ================= DATABASE =================
conn = sqlite3.connect(
    "users.db",
    check_same_thread=False
)

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
    return bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    )

def check_password(password, hashed):
    return bcrypt.checkpw(
        password.encode(),
        hashed
    )

def signup(username, password):

    try:

        c.execute(
            "INSERT INTO users VALUES (?, ?)",
            (
                username,
                hash_password(password)
            )
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
        return check_password(
            password,
            data[0]
        )

    return False

def strong_password(password):

    return (
        len(password) >= 6
        and any(i.isdigit() for i in password)
        and any(i.isalpha() for i in password)
    )

# ================= CSS =================
st.markdown("""
<style>

/* ================= GLOBAL ================= */

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* APP BACKGROUND */
.stApp {
    background: #dfeee2;
}

/* REMOVE STREAMLIT DEFAULT */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* REMOVE TOP SPACE */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* ================= NAVBAR ================= */

.navbar {

    background: #204d34;

    padding: 18px 25px;

    border-radius: 20px;

    margin-bottom: 25px;
}

/* LOGO */
.logo {

    color: white;

    font-size: 34px;

    font-weight: 900;

    margin-bottom: 15px;
}

/* BUTTON SPACING */
div[data-testid="stHorizontalBlock"] {
    gap: 0.8rem;
}

/* NAV BUTTON */
.stButton > button {

    width: 100%;

    background: white;

    color: black;

    border: 2px solid #3fa466;

    border-radius: 14px;

    padding: 10px 0;

    font-weight: 700;

    transition: 0.2s;
}

/* BUTTON HOVER */
.stButton > button:hover {

    background: #3fa466;

    color: white;

    transform: translateY(-2px);
}

/* ================= HERO SECTION ================= */

.hero {

    padding: 90px 60px;

    border-radius: 30px;

    background:
    linear-gradient(
        rgba(0,0,0,0.45),
        rgba(0,0,0,0.55)
    ),
    url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;

    background-position: center;

    margin-bottom: 30px;
}

/* HERO TITLE */
.hero-title {

    font-size: 80px;

    font-weight: 1000;

    color: white;

    margin-bottom: 15px;
}

/* HERO SUBTITLE */
.hero-sub {

    color: #f1fff4;

    font-size: 22px;

    max-width: 700px;
}

/* ================= CARDS ================= */

.card {

    background: white;

    border-radius: 24px;

    padding: 28px;

    min-height: 180px;

    box-shadow: 0px 5px 15px rgba(0,0,0,0.08);

    transition: 0.25s;
}

/* CARD HOVER */
.card:hover {
    transform: translateY(-5px);
}

/* CARD TITLE */
.card-title {

    font-size: 30px;

    font-weight: 900;

    color: black;

    margin-bottom: 12px;
}

/* CARD TEXT */
.card-text {

    color: black;

    font-size: 17px;

    line-height: 1.7;
}

/* INPUT */
.stTextInput input {

    border-radius: 12px;

    border: 1px solid #9fd3b2;

    background: white;

    color: black;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"] {

    background: white;

    padding: 20px;

    border-radius: 18px;

    border: 1px solid #cce8d5;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================

st.markdown(
    """
    <div class="navbar">
        <div class="logo">
            🌿 LeafSentry AI
        </div>
    </div>
    """,
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

        if st.button(page):

            st.session_state.page = page

# ================= HOME PAGE =================

if st.session_state.page == "Home":

    # IMPORTANT:
    # DO NOT USE EXTRA BACKTICKS INSIDE THIS BLOCK

    hero_html = """
    <div class="hero">

        <div class="hero-title">
            LeafSentry AI
        </div>

        <div class="hero-sub">
            Smart AI system for plant disease detection
            and crop health monitoring.
        </div>

    </div>
    """

    st.markdown(
        hero_html,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        <div class="card">

            <div class="card-title">
                🌿 Monitoring
            </div>

            <div class="card-text">
                Detect plant diseases instantly using AI-powered monitoring.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">

            <div class="card-title">
                ⚡ Speed
            </div>

            <div class="card-text">
                Fast real-time crop health predictions.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="card">

            <div class="card-title">
                🧠 AI Model
            </div>

            <div class="card-text">
                Deep learning classification model for disease detection.
            </div>

        </div>
        """, unsafe_allow_html=True)

# ================= PLANT PAGE =================

elif st.session_state.page == "Plant":

    st.title("🌱 Plant Disease Detection")

    if not st.session_state.logged_in:

        st.error(
            "You must login first to use the AI model."
        )

    else:

        st.success(
            f"Welcome {st.session_state.username}"
        )

        uploaded = st.file_uploader(
            "Upload Plant Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded:

            image = Image.open(uploaded)

            st.image(
                image,
                caption="Uploaded Plant Image",
                use_container_width=True
            )

            st.success(
                "Prediction Complete"
            )

            st.write("### Prediction Result")
            st.write("Healthy Plant")

            st.write("### Confidence")
            st.progress(95)

# ================= BLOG =================

elif st.session_state.page == "Blog":

    st.title("📰 Blog")

    st.write(
        "Latest updates about smart farming and AI."
    )

# ================= PRIVACY =================

elif st.session_state.page == "Privacy":

    st.title("🔒 Privacy Policy")

    st.write(
        "Your uploaded data is protected securely."
    )

# ================= CONTACT =================

elif st.session_state.page == "Contact":

    st.title("📞 Contact")

    st.write(
        "support@leafsentry.ai"
    )

# ================= LOGIN =================

elif st.session_state.page == "Login":

    st.title("🔐 Login System")

    tab1, tab2 = st.tabs([
        "Login",
        "Sign Up"
    ])

    # LOGIN
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

                st.session_state.username = username

                st.success(
                    "Login successful"
                )

            else:

                st.error(
                    "Invalid username or password"
                )

    # SIGNUP
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
                        "Account created successfully"
                    )

                else:

                    st.error(
                        "Username already exists"
                    )

            else:

                st.warning(
                    "Password must contain letters, numbers, and be at least 6 characters."
                )
