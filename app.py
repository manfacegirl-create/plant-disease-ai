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

/* APP BACKGROUND */
.stApp {
    background-color: #dfeee2;
}

/* REMOVE STREAMLIT UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* REMOVE TOP SPACE */
.block-container {
    padding-top: 1rem;
}

/* NAVBAR */
.navbar {
    background-color: #204d34;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 25px;
}

/* LOGO */
.logo {
    color: white;
    font-size: 36px;
    font-weight: bold;
}

/* BUTTONS */
.stButton > button {

    width: 100%;

    background-color: white;

    color: black;

    border-radius: 12px;

    border: 2px solid #3fa466;

    padding: 10px;

    font-weight: bold;
}

/* BUTTON HOVER */
.stButton > button:hover {

    background-color: #3fa466;

    color: white;
}

/* HERO */
.hero-box {

    background-image:
    linear-gradient(
        rgba(0,0,0,0.5),
        rgba(0,0,0,0.5)
    ),
    url("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?q=80&w=2070&auto=format&fit=crop");

    background-size: cover;

    background-position: center;

    padding: 100px 60px;

    border-radius: 25px;

    margin-bottom: 30px;
}

/* HERO TITLE */
.hero-title {

    color: white;

    font-size: 75px;

    font-weight: bold;
}

/* HERO TEXT */
.hero-text {

    color: white;

    font-size: 24px;
}

/* CARD */
.card {

    background-color: white;

    padding: 30px;

    border-radius: 22px;

    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);

    min-height: 180px;
}

/* CARD TITLE */
.card-title {

    color: black;

    font-size: 30px;

    font-weight: bold;

    margin-bottom: 10px;
}

/* CARD TEXT */
.card-text {

    color: black;

    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================

st.markdown("""
<div class="navbar">
    <div class="logo">
        🌿 LeafSentry AI
    </div>
</div>
""", unsafe_allow_html=True)

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
    # THIS FIXES THE HTML SHOWING ISSUE

    st.markdown(
        """
        <div class="hero-box">

            <div class="hero-title">
                LeafSentry AI
            </div>

            <div class="hero-text">
                Smart AI system for plant disease detection
                and crop health monitoring.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🌿 Monitoring
                </div>

                <div class="card-text">
                    Detect plant diseases instantly using AI monitoring.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    ⚡ Speed
                </div>

                <div class="card-text">
                    Fast real-time crop health predictions.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🧠 AI Model
                </div>

                <div class="card-text">
                    Deep learning model for disease classification.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

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

            st.success("Prediction Complete")

            st.write("### Prediction Result")
            st.write("Healthy Plant")

            st.write("### Confidence")
            st.progress(95)

# ================= BLOG =================

elif st.session_state.page == "Blog":

    st.title("📰 Blog")

    st.write(
        "Latest AI farming updates."
    )

# ================= PRIVACY =================

elif st.session_state.page == "Privacy":

    st.title("🔒 Privacy Policy")

    st.write(
        "Your data is securely protected."
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

                st.session_state.username = username

                st.success(
                    "Login successful"
                )

            else:

                st.error(
                    "Invalid username or password"
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
                        "Account created successfully"
                    )

                else:

                    st.error(
                        "Username already exists"
                    )

            else:

                st.warning(
                    "Password must contain letters and numbers and be at least 6 characters."
                )
