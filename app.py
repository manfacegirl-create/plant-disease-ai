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

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="LeafSentry AI",
    page_icon="🌿",
    layout="wide"
)

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

# ================= PASSWORD =================
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def strong_password(password):
    return (
        len(password) >= 6
        and any(i.isdigit() for i in password)
        and any(i.isalpha() for i in password)
    )

# ================= AUTH =================
def signup(username, password):

    try:
        c.execute(
            "INSERT INTO users VALUES (?, ?)",
            (username, hash_password(password))
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
        return check_password(password, data[0])

    return False

# ================= SESSION =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ================= CSS =================
# ================= NAVBAR CSS =================
st.markdown("""
<style>

/* NAVBAR CONTAINER */
.navbar{
    background:#15803d;
    padding:0px;
    margin-bottom:25px;
    border-radius:0px;
    overflow:hidden;
}

/* REMOVE COLUMN GAPS */
div[data-testid="column"]{
    padding:0 !important;
}

/* NAV BUTTONS */
div.stButton > button{
    width:100%;
    height:75px;

    background:#15803d;
    color:white;

    border:none;
    border-radius:0px;

    font-size:20px;
    font-weight:700;

    transition:0.3s;

    margin:0 !important;
}

/* HOVER */
div.stButton > button:hover{
    background:#e5e7eb;
    color:#15803d;
}

/* ACTIVE EFFECT */
div.stButton > button:focus{
    background:#e5e7eb !important;
    color:#15803d !important;
    box-shadow:none !important;
}

</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================
st.markdown('<div class="navbar">', unsafe_allow_html=True)

nav1, nav2, nav3, nav4, nav5, nav6 = st.columns(
    [1,1,1,1.5,1.3,1]
)

with nav1:
    if st.button("Home"):
        st.session_state.page = "Home"

with nav2:
    if st.button("Plant"):
        st.session_state.page = "Plant"

with nav3:
    if st.button("Blog"):
        st.session_state.page = "Blog"

with nav4:
    if st.button("Privacy Policy"):
        st.session_state.page = "Privacy"

with nav5:
    if st.button("Contact Us"):
        st.session_state.page = "Contact"

with nav6:
    if st.button("Login"):
        st.session_state.page = "Login"

st.markdown('</div>', unsafe_allow_html=True)

# ================= HOME PAGE =================
if st.session_state.page == "Home":

    st.markdown("""
<div class="hero">

<div class="hero-title">
LeafSentry AI
</div>

<div class="hero-sub">
Smart plant disease detection powered by deep learning
and artificial intelligence.
</div>

</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
<div class="card">

<div class="card-title">
🌿 Plant Monitoring
</div>

<p>
Detect unhealthy plants instantly using image analysis.
</p>

</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="card">

<div class="card-title">
⚡ Fast Detection
</div>

<p>
Upload leaf images and receive instant AI predictions.
</p>

</div>
""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div class="card">

<div class="card-title">
🧠 Deep Learning
</div>

<p>
Neural network powered disease classification system.
</p>

</div>
""", unsafe_allow_html=True)

# ================= PLANT PAGE =================
elif st.session_state.page == "Plant":

    st.title("🌱 Plant Information")

    st.markdown("""
<div class="card">

<div class="card-title">
Common Plant Diseases
</div>

<ul>
<li>Leaf Spot</li>
<li>Powdery Mildew</li>
<li>Root Rot</li>
<li>Rust Fungus</li>
<li>Bacterial Wilt</li>
</ul>

</div>
""", unsafe_allow_html=True)

# ================= BLOG PAGE =================
elif st.session_state.page == "Blog":

    st.title("📰 Blog")

    st.markdown("""
<div class="card">

<div class="card-title">
How AI Helps Modern Farming
</div>

<p>
Artificial intelligence helps farmers detect plant diseases faster
and improve crop quality.
</p>

</div>
""", unsafe_allow_html=True)

# ================= PRIVACY PAGE =================
elif st.session_state.page == "Privacy":

    st.title("🔒 Privacy Policy")

    st.markdown("""
<div class="card">

<p>
Your uploaded images and personal information remain private
and securely stored.
</p>

</div>
""", unsafe_allow_html=True)

# ================= CONTACT PAGE =================
elif st.session_state.page == "Contact":

    st.title("📞 Contact Us")

    st.markdown("""
<div class="card">

<p>
📧 Email: leafsentry@gmail.com
</p>

<p>
📱 Phone: +60 123-456-789
</p>

</div>
""", unsafe_allow_html=True)

# ================= LOGIN PAGE =================
elif st.session_state.page == "Login":

    st.markdown("""
<div class="card">

<div class="card-title">
🔐 Login To LeafSentry AI
</div>

<p>
Access your AI plant disease detection dashboard.
</p>

</div>
""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:

        username = st.text_input("Username")

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login Account"):

            if login(username, password):

                st.session_state.logged_in = True
                st.session_state.page = "ML"

                st.success("Login Successful")

                st.rerun()

            else:
                st.error("Invalid Username or Password")

    with tab2:

        new_user = st.text_input("Create Username")

        new_pass = st.text_input(
            "Create Password",
            type="password"
        )

        if st.button("Create Account"):

            if strong_password(new_pass):

                if signup(new_user, new_pass):

                    st.success("Account Created Successfully")

                else:
                    st.error("Username Already Exists")

            else:
                st.warning(
                    "Password must contain letters and numbers"
                )

# ================= ML PAGE =================
elif st.session_state.page == "ML":

    if not st.session_state.logged_in:

        st.warning("Please login first.")

        st.session_state.page = "Login"

        st.rerun()

    st.title("🧠 Plant Disease Detection")

    if st.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.page = "Home"

        st.rerun()

    # MODEL
    classes = ["Diseased", "Healthy"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    class CNN(nn.Module):

        def __init__(self):

            super().__init__()

            self.net = nn.Sequential(

                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.AdaptiveAvgPool2d(1)

            )

            self.fc = nn.Linear(64, 2)

        def forward(self, x):

            x = self.net(x)

            return self.fc(
                x.view(x.size(0), -1)
            )

    @st.cache_resource
    def load_model():

        try:

            model = CNN()

            model.load_state_dict(
                torch.load(
                    "cnn.pth",
                    map_location="cpu"
                )
            )

            model.eval()

            return model

        except:
            return None

    model = load_model()

    uploaded = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                use_container_width=True
            )

        x = transform(image).unsqueeze(0)

        if model:

            with torch.no_grad():

                probs = torch.softmax(
                    model(x),
                    dim=1
                )[0].numpy()

        else:

            probs = np.array([0.5, 0.5])

        pred = int(np.argmax(probs))

        conf = float(probs[pred]) * 100

        with col2:

            st.markdown(f"""
<div class="card">

<div class="card-title">
Prediction Result
</div>

<h1>
{classes[pred]}
</h1>

<h3>
{conf:.2f}% Confidence
</h3>

</div>
""", unsafe_allow_html=True)

            st.progress(int(conf))

        fig = px.bar(
            x=classes,
            y=probs * 100,
            labels={
                "x":"Class",
                "y":"Confidence"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ================= FOOTER =================
st.markdown("""
<div class="footer">
© 2026 LeafSentry AI • Smart Agriculture Platform
</div>
""", unsafe_allow_html=True)
