# Professional LeafSentry AI Streamlit UI


# ================= IMPORTS =================
import streamlit as st
import sqlite3
import bcrypt
from PIL import Image

# ================= PAGE CONFIG =================from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

STREAMLIT_URL = os.environ.get("STREAMLIT_URL", "http://localhost:8501")

PLANTS = [
    {
        "id": 1, "name": "Mango", "sci": "Mangifera indica", "region": "Tropical Asia",
        "img": "https://images.unsplash.com/photo-1553279768-865429fa0078?q=60&w=600&auto=format&fit=crop",
        "desc": "A tropical stone fruit tree native to South Asia, widely cultivated for its sweet fruit. Susceptible to Anthracnose and Powdery Mildew affecting total fruit yield and leaf integrity.",
        "diseases": [0, 1, 2, 3, 4]
    },
    {
        "id": 2, "name": "Arjun", "sci": "Terminalia arjuna", "region": "Indian Subcontinent",
        "img": "https://images.unsplash.com/photo-1585320806297-9794b3e4eeae?q=60&w=600&auto=format&fit=crop",
        "desc": "A medicinal tree of great importance in Ayurvedic medicine. Vulnerable to leaf gall and rust spots that damage structural foliage.",
        "diseases": [5, 6, 7, 8]
    },
    {
        "id": 3, "name": "Guava", "sci": "Psidium guajava", "region": "Tropical America",
        "img": "https://images.unsplash.com/photo-1534531173927-aeb928d54385?q=60&w=600&auto=format&fit=crop",
        "desc": "An evergreen shrub or small tree grown for edible fruits. Prone to Anthracnose, Algal Leaf Spot, and Canker infections.",
        "diseases": [9, 10, 11, 12]
    },
    {
        "id": 4, "name": "Neem", "sci": "Azadirachta indica", "region": "Indian Subcontinent",
        "img": "https://images.unsplash.com/photo-1605000797499-95a51c5269ae?q=60&w=600&auto=format&fit=crop",
        "desc": "A deeply revered medicinal tree renowned for pest-repellent properties. Occasionally burdened by Powdery Mildew or Bacterial Leaf Spot in humid monsoon periods.",
        "diseases": [13, 14]
    }
]

DISEASES = [
    # Mango Diseases
    {"id": 0, "plant_id": 1, "plant": "Mango", "name": "Anthracnose", "severity": "high",
     "parts": ["Leaves", "Fruits", "Twigs"],
     "desc": "Caused by Colletotrichum gloeosporioides. Manifests as dark brown spots expanding rapidly under rain.",
     "symptoms": ["Dark brown, irregular spots on leaves", "Premature leaf fall", "Black sunken decay spots on fruits"],
     "causes": ["Fungal pathogen spores", "Excessive humidity and rainfall", "Poor canopy airflow"],
     "treatments": ["Apply copper-based fungicides before monsoons", "Prune overlapping branches for aeration",
                    "Remove fallen leaves promptly"]},
    {"id": 1, "plant_id": 1, "plant": "Mango", "name": "Powdery Mildew", "severity": "medium",
     "parts": ["Leaves", "Flowers"],
     "desc": "Oidium mangiferae infection coating new tissue in dusty white powder, causing blossom drop.",
     "symptoms": ["White powdery crust on flower panicles and leaves", "Crinkling of young leaves",
                  "Drying and shedding of blossoms"],
     "causes": ["Fungal spores traveling via wind", "Cool dry nights followed by warm humid days"],
     "treatments": ["Spray wettable sulfur early in the flowering cycle", "Keep orchard clean of dead organic debris"]},
    {"id": 2, "plant_id": 1, "plant": "Mango", "name": "Bacterial Canker", "severity": "critical",
     "parts": ["Leaves", "Fruits", "Stems"],
     "desc": "Xanthomonas campestris infection causing water-soaked spots that split bark and lesion fruit surface.",
     "symptoms": ["Water-soaked lesions surrounded by yellow halos", "Gummy exudate from stem cracks",
                  "Star-shaped cracks on dark fruit lesions"],
     "causes": ["Bacterial pathogen entering through wounds", "Windblown rain splash transport"],
     "treatments": ["Spray Streptomycin Sulfate or copper oxychloride", "Sterilize pruning shears between trees"]},
    {"id": 3, "plant_id": 1, "plant": "Mango", "name": "Sooty Mold", "severity": "low", "parts": ["Leaves", "Twigs"],
     "desc": "Black fungal layer thriving on honeydew excreted by sucking insects. Blocks photosynthesis but doesn't attack tissue directly.",
     "symptoms": ["Velvety black layer covering leaf upper surface", "Presence of aphids, scale insects, or mealybugs"],
     "causes": ["Secondary fungal growth on insect sugary honeydew", "Neglected insect management"],
     "treatments": ["Spray starch solution or mild systemic insecticide to wipe insects",
                    "Wash foliage down with water jets"]},
    {"id": 4, "plant_id": 1, "plant": "Mango", "name": "Die Back", "severity": "critical",
     "parts": ["Twigs", "Branches"],
     "desc": "Lasiodiplodia theobromae causing systemic drying from tips downward, fatal if unchecked.",
     "symptoms": ["Drying of twigs from top downwards", "Discoloration and browning of leaves on affected branch",
                  "Internal bark turning brown"],
     "causes": ["Fungal entry through physical injuries", "Tree stress from drought or nutrient starvation"],
     "treatments": ["Prune twigs 3 inches below infected green margin", "Coat cut ends with copper paste fungicide"]},

    # Arjun Diseases
    {"id": 5, "plant_id": 2, "plant": "Arjun", "name": "Leaf Gall", "severity": "low", "parts": ["Leaves"],
     "desc": "Insect-induced abnormalities resulting in nodular green/purple swelling structures over the leaves.",
     "symptoms": ["Wart-like raised galls on leaf surface", "Deformation of young leaves when infestation is heavy"],
     "causes": ["Egg-laying activities and feeding by psyllid insects", "Tissue hypersensitivity response"],
     "treatments": ["Remove heavily galled leaves manually", "Spray systemic insecticide during spring new flush"]},
    {"id": 6, "plant_id": 2, "plant": "Arjun", "name": "Rust Spots", "severity": "medium", "parts": ["Leaves"],
     "desc": "Puccinia family infection producing tiny orange powdery pustules that exhaust the tree's resources.",
     "symptoms": ["Powdery orange-brown bumps on lower leaf surface", "Yellow chlorotic halos on upper leaf surface"],
     "causes": ["Airborne rust fungal spores", "Persistent morning dew on leaf canopy"],
     "treatments": ["Spray Mancozeb or Propiconazole", "Clear ground weeds to drop ambient humidity"]},
    {"id": 7, "plant_id": 2, "plant": "Arjun", "name": "Leaf Spot", "severity": "medium", "parts": ["Leaves"],
     "desc": "Cercospora fungal attack resulting in circular, target-board-pattern spots that diminish cosmetic/medicinal harvest value.",
     "symptoms": ["Small circular spots with grey centers and dark margins",
                  "Coalescing patches causing leaf necrosis"],
     "causes": ["Fungal spores overwintering on ground leaves", "Warm, rain-splashed environments"],
     "treatments": ["Apply carbendazim protective fungicide", "Maintain spaced tree boundaries"]},
    {"id": 8, "plant_id": 2, "plant": "Arjun", "name": "Powdery Mildew", "severity": "low", "parts": ["Leaves"],
     "desc": "White mycelial web spreading across young arjun trees, mostly superficial but stunts youth cycles.",
     "symptoms": ["Thin white powdery patches across leaf tops", "Slight curling or slowing of new shoot expansions"],
     "causes": ["Fungal spore colonization in shady microclimates"],
     "treatments": ["Increase sunlight exposure if possible", "Apply neem oil or sulfur spray formulations"]},

    # Guava Diseases
    {"id": 9, "plant_id": 3, "plant": "Guava", "name": "Anthracnose", "severity": "high", "parts": ["Leaves", "Fruits"],
     "desc": "Glomerella cingulata attacking green and ripe guava fruits, leaving them mummified and unmarketable.",
     "symptoms": ["Sunken dark brown spots on fruit developing pink spore masses", "Pinhead brown spots on leaves"],
     "causes": ["Fungal pathogen active in warm rain spells", "Mummified fruits left hanging on tree branches"],
     "treatments": ["Spray Copper Oxychloride regularly post-fruit set", "Prune lower branches touching the wet soil"]},
    {"id": 10, "plant_id": 3, "plant": "Guava", "name": "Algal Leaf Spot", "severity": "low", "parts": ["Leaves"],
     "desc": "Cephaleuros virescens parasitic green alga causing orange velvety rust-like spots. Uncommon true parasite.",
     "symptoms": ["Velvety, rust-colored or orange circular spots on leaves",
                  "Mild scratchy look on older lower foliage"],
     "causes": ["Parasitic alga favored by wet weather and crowded orchards",
                "Poor nutritional fertilizer feeding schedules"],
     "treatments": ["Apply protective copper hydroxide spray treatment",
                    "Improve general tree vigor via balanced NPK feeds"]},
    {"id": 11, "plant_id": 3, "plant": "Guava", "name": "Canker", "severity": "high", "parts": ["Fruits", "Twigs"],
     "desc": "Pestalotiopsis psidii resulting in rough corky eruptive circular lesions strictly destroying fresh fruit skin quality.",
     "symptoms": ["Corky, raised round spots on guava fruits", "Cracking of fruit skin under severe pressure"],
     "causes": ["Fungal wound invader utilizing insect puncture holes"],
     "treatments": ["Control fruit-boring insects proactively", "Spray systemic triazole fungicides early stage"]},
    {"id": 12, "plant_id": 3, "plant": "Guava", "name": "Wilt", "severity": "critical",
     "parts": ["Roots", "Vascular System"],
     "desc": "Fusarium oxysporum soil born pathogen clogging vascular routes. Leaves turn yellow, dry, and drop within weeks. Fatal.",
     "symptoms": ["Yellowing and wilting of leaves from top branches down",
                  "Complete leaf drop followed by branch death", "Root blackening"],
     "causes": ["Soil-dwelling fungal pathogen attacking root cortex", "Waterlogging in heavy clay soils"],
     "treatments": ["Strictly solarize nursery bed soil",
                    "Inject carbendazim or remove/burn dead infected trees to protect neighbors"]},

    # Neem Diseases
    {"id": 13, "plant_id": 4, "plant": "Neem", "name": "Powdery Mildew", "severity": "low", "parts": ["Leaves"],
     "desc": "Erysiphe species covering antimicrobial neem leaflets, proving nature's pesticide still gets infected.",
     "symptoms": ["Patchy white flour-like dusting over narrow leaflets", "Premature casting of infected shade leaves"],
     "causes": ["Fungal adaptation utilizing cool, shaded high-density growth zones"],
     "treatments": ["Prune surrounding thick wild brush", "Apply potassium bicarbonate or sulfur sprays"]},
    {"id": 14, "plant_id": 4, "plant": "Neem", "name": "Bacterial Leaf Spot", "severity": "medium", "parts": ["Leaves"],
     "desc": "Pseudomonas azadirachtae creating angular small dark brown water lesions bounded tightly by leaf veins.",
     "symptoms": ["Angular dark brown or black lesions on leaflets",
                  "Yellow chlorotic halo framing the dark rectangles"],
     "causes": ["Bacterial migration inside open leaf stomata during long damp storms"],
     "treatments": ["Avoid overhead irrigation systems", "Spray copper fungicides at first onset of spotting"]}
]


@app.route("/")
def home():
    return render_template("index.html", plants=PLANTS, diseases=DISEASES,
                           streamlit_url=STREAMLIT_URL, page="home")


@app.route("/plants")
def plants():
    return render_template("plants.html", plants=PLANTS, streamlit_url=STREAMLIT_URL, page="plants")


@app.route("/plants/<int:plant_id>")
def plant_detail(plant_id):
    plant = next((p for p in PLANTS if p["id"] == plant_id), None)
    if not plant:
        return "Plant not found", 404

    plant_diseases = [d for d in DISEASES if d["id"] in plant["diseases"]]
    return render_template("plant_detail.html", plant=plant, diseases=plant_diseases,
                           streamlit_url=STREAMLIT_URL, page="plants")


@app.route("/diseases")
def diseases():
    plant_filter = request.args.get("plant", "")
    severity_filter = request.args.get("severity", "")
    search = request.args.get("search", "").strip().lower()

    filtered = DISEASES
    if plant_filter:
        filtered = [d for d in filtered if d["plant"].lower() == plant_filter.lower()]
    if severity_filter:
        filtered = [d for d in filtered if d["severity"].lower() == severity_filter.lower()]

    if search:
        def matches_search(d):
            if search in d["name"].lower() or search in d["desc"].lower():
                return True
            if any(search in symptom.lower() for symptom in d.get("symptoms", [])):
                return True
            if any(search in cause.lower() for cause in d.get("causes", [])):
                return True
            return False

        filtered = [d for d in filtered if matches_search(d)]

    return render_template("diseases.html", diseases=filtered, plant_filter=plant_filter,
                           severity_filter=severity_filter, search=request.args.get("search", ""),
                           streamlit_url=STREAMLIT_URL, page="diseases")


@app.route("/diseases/<int:disease_id>")
def disease_detail(disease_id):
    disease = next((d for d in DISEASES if d["id"] == disease_id), None)
    if not disease:
        return "Disease not found", 404
    plant = next((p for p in PLANTS if p["id"] == disease["plant_id"]), None)
    return render_template("disease_detail.html", disease=disease,
                           plant=plant, streamlit_url=STREAMLIT_URL, page="diseases")


@app.route("/health-check")
def health_check():
    plant_id = request.args.get("plant", "")
    return render_template("health_check.html", plants=PLANTS,
                           plant_id=plant_id, streamlit_url=STREAMLIT_URL, page="health-check")


@app.route("/about")
def about():
    return render_template("about.html", streamlit_url=STREAMLIT_URL, page="about")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    success = False
    if request.method == "POST":
        success = True
    return render_template("contact.html", success=success,
                           streamlit_url=STREAMLIT_URL, page="contact")


@app.route("/api/plants")
def api_plants():
    return jsonify(PLANTS)


@app.route("/api/diseases")
def api_diseases():
    return jsonify(DISEASES)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
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

