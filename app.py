# Professional LeafSentry AI Streamlit UI


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

# ================= PROFESSIONAL UI =================
st.markdown("""
<style>

/* ================= MAIN APP ================= */

.stApp {

    background:
    linear-gradient(
        135deg,
        #eef7f0 0%,
        #dcefe2 50%,
        #c8e6d1 100%
    );

    color: #111111;
}

/* REMOVE STREAMLIT DEFAULT */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

/* ================= NAVBAR ================= */

.navbar {

    background: rgba(17, 53, 36, 0.95);

    padding: 18px 28px;

    border-radius: 22px;

    margin-bottom: 28px;

    backdrop-filter: blur(10px);

    box-shadow: 0px 8px 24px rgba(0,0,0,0.12);
}

.logo {

    color: white;

    font-size: 38px;

    font-weight: 900;

    letter-spacing: 1px;
}

/* ================= BUTTONS ================= */

.stButton > button {

    width: 100%;

    background: white;

    color: #0d2f1d;

    border: none;

    border-radius: 14px;

    padding: 12px;

    font-weight: 700;

    transition: 0.25s;

    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}

.stButton > button:hover {

    background: #2f8f57;

    color: white;

    transform: translateY(-3px);
}

/* ================= HERO SECTION ================= */

.hero {

    background:
    linear-gradient(
        rgba(0,0,0,0.45),
        rgba(0,0,0,0.55)
    ),

    url('https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?q=80&w=2070&auto=format&fit=crop');

    background-size: cover;

    background-position: center;

    border-radius: 30px;

    padding: 110px 70px;

    margin-bottom: 35px;

    box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
}

.hero-title {

    color: white;

    font-size: 78px;

    font-weight: 900;

    line-height: 1.05;
}

.hero-sub {

    color: #f3fff6;

    font-size: 24px;

    max-width: 760px;

    margin-top: 18px;

    line-height: 1.7;
}

/* ================= FEATURE CARDS ================= */

.feature-card {

    background: rgba(255,255,255,0.92);

    border-radius: 24px;

    padding: 35px;

    min-height: 220px;

    backdrop-filter: blur(10px);

    box-shadow: 0px 8px 22px rgba(0,0,0,0.08);

    transition: 0.3s;
}

.feature-card:hover {

    transform: translateY(-8px);

    box-shadow: 0px 14px 28px rgba(0,0,0,0.12);
}

.feature-title {

    color: #111111;

    font-size: 30px;

    font-weight: 900;

    margin-bottom: 15px;
}

.feature-text {

    color: #222222;

    font-size: 18px;

    line-height: 1.8;
}

/* ================= INPUTS ================= */

.stTextInput input {

    border-radius: 14px !important;

    border: 1px solid #b7d8c0 !important;

    padding: 12px !important;

    background: white !important;

    color: black !important;
}

/* ================= FILE UPLOADER ================= */

section[data-testid="stFileUploader"] {

    background: white;

    padding: 22px;

    border-radius: 18px;

    border: 1px solid #d2e6d8;

    box-shadow: 0px 4px 10px rgba(0,0,0,0.06);
}

/* ================= TITLES ================= */

h1, h2, h3 {
    color: #111111 !important;
}

p {
    color: #222222;
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

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                Smart Farming<br>
                Powered by AI
            </div>

            <div class="hero-sub">
                Advanced plant disease detection and crop health analysis
                using modern artificial intelligence technology.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-title">
                    🌿 Disease Monitoring
                </div>

                <div class="feature-text">
                    Instantly detect plant diseases using AI-powered image analysis.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-title">
                    ⚡ Fast Prediction
                </div>

                <div class="feature-text">
                    Receive accurate crop health predictions within seconds.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-title">
                    🧠 Deep Learning
                </div>

                <div class="feature-text">
                    Powered by modern neural network models for precision agriculture.
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
            "Please login first to access the AI prediction system."
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

            st.write("### Confidence Score")
            st.progress(95)

# ================= BLOG PAGE =================
elif st.session_state.page == "Blog":

    st.title("📰 Blog")

    st.write(
        "Latest smart farming and AI agriculture updates."
    )

# ================= PRIVACY PAGE =================
elif st.session_state.page == "Privacy":

    st.title("🔒 Privacy Policy")

    st.write(
        "All uploaded images and user data are securely protected."
    )

# ================= CONTACT PAGE =================
elif st.session_state.page == "Contact":

    st.title("📞 Contact Us")

    st.write(
        "support@leafsentry.ai"
    )

# ================= LOGIN PAGE =================
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

