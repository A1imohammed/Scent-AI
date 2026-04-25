import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# ------------------ STYLING ------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: white;
    }
    h1, h2, h3 {
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

TRAIN_FILE = "training_data.json"
MEMORY_FILE = "user_memory.json"

# ------------------ MODEL ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------ TRAIN FUNCTION ------------------
def train_model(train_data):
    if len(train_data) < 2:
        return None

    texts = [d["notes"] for d in train_data]
    labels = [d["label"] for d in train_data]

    if len(set(labels)) < 2:
        return None

    X_train = model.encode(texts)

    clf = LogisticRegression()
    clf.fit(X_train, labels)

    return clf

# ------------------ USER LOGIN ------------------
username = st.text_input("Enter your username")
if not username:
    st.stop()

# ------------------ LOAD MEMORY ------------------
try:
    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
except:
    data = {"users": {}}

if username not in data["users"]:
    data["users"][username] = {"liked_notes": []}

if "liked_notes" not in st.session_state:
    st.session_state.liked_notes = data["users"][username]["liked_notes"]

# ------------------ LOAD TRAIN DATA ------------------
try:
    with open(TRAIN_FILE, "r") as f:
        train_data = json.load(f)
except:
    train_data = []

clf = train_model(train_data)

# ------------------ HEADER ------------------
st.image("logo.png", width=120)

st.markdown("""
# Scent Genie  
### Discover your signature fragrance
""")

st.write(f"Logged in as: {username}")
st.write("Preferences:", st.session_state.liked_notes)

# ------------------ LOAD DATA ------------------
try:
    df = pd.read_csv("perfumes.csv")
except:
    st.error("Missing perfumes.csv file")
    st.stop()

df.columns = df.columns.str.strip().str.lower()
st.write("Total fragrances:", len(df))

# ------------------ FILTER UI ------------------
col1, col2, col3 = st.columns(3)

with col1:
    fragrance_type = st.selectbox("Type", ["All", "designer", "niche", "arabic"])

with col2:
    occasion = st.selectbox("Occasion", ["Any", "Daily", "Date", "Office", "Party"])

with col3:
    weather = st.selectbox("Weather", ["Any", "Hot", "Cold"])

user_input = st.text_input("Describe what you like:")

# ------------------ FILTERING ------------------
df_filtered = df.copy()

if fragrance_type != "All":
    df_filtered = df_filtered[df_filtered["type"] == fragrance_type]

if occasion == "Date":
    df_filtered = df_filtered[df_filtered["category"].str.contains("sweet|spicy", case=False)]
elif occasion == "Office":
    df_filtered = df_filtered[df_filtered["category"].str.contains("fresh", case=False)]
elif occasion == "Party":
    df_filtered = df_filtered[df_filtered["projection"].str.contains("strong", case=False)]

if weather == "Hot":
    df_filtered = df_filtered[df_filtered["category"].str.contains("fresh|citrus", case=False)]
elif weather == "Cold":
    df_filtered = df_filtered[df_filtered["category"].str.contains("warm|spicy|oud", case=False)]

if df_filtered.empty:
    st.warning("No perfumes match your filters.")
    st.stop()

# ------------------ EMBEDDINGS ------------------
X = model.encode(df_filtered["notes"].tolist())

# ------------------ RECOMMEND ------------------
if user_input:
    with st.spinner("Finding your perfect scent..."):

        combined_input = user_input + " " + " ".join(st.session_state.liked_notes)
        user_vec = model.encode([combined_input])

        scores = np.dot(X, user_vec.T).flatten()

        # ML boost
        if clf:
            pred_scores = clf.predict_proba(X)[:, 1]
            scores = scores + pred_scores

        top = scores.argsort()[-3:][::-1]

    st.success("✨ Recommendations tailored for you")

    for i in top:
        perfume = df_filtered.iloc[i]

        # MATCH REASON
        perfume_notes = perfume["notes"].lower()
        user_words = combined_input.lower().split()
        matched_words = [w for w in user_words if w in perfume_notes]
        reason = ", ".join(matched_words[:3]) if matched_words else "general match"

        st.markdown(f"""
        <div style="
            background-color:#1e293b;
            padding:20px;
            border-radius:15px;
            margin-bottom:15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
        ">
            <h3 style="color:#fbbf24;">{perfume['name']}</h3>
            <p>🧴 <b>Type:</b> {perfume['type']}</p>
            <p>🌿 <b>Notes:</b> {perfume['notes']}</p>
            <p>⏱ <b>Longevity:</b> {perfume['longevity']} hrs</p>
            <p>📢 <b>Projection:</b> {perfume['projection']}</p>
            <p style="color:#22c55e;">✔ Matches: {reason}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            like = st.button(f"👍 Like {perfume['name']}", key=f"like_{i}")

        with col2:
            dislike = st.button(f"👎 Dislike {perfume['name']}", key=f"dislike_{i}")

        if like or dislike:
            entry = {
                "user": username,
                "notes": perfume["notes"],
                "label": 1 if like else 0
            }

            train_data.append(entry)

            with open(TRAIN_FILE, "w") as f:
                json.dump(train_data, f)

        if like:
            st.session_state.liked_notes.append(perfume["notes"])

        if dislike:
            st.session_state.liked_notes.append("avoid_" + perfume["notes"])

        # SAVE MEMORY
        data["users"][username]["liked_notes"] = st.session_state.liked_notes

        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f)

        st.divider()

# ------------------ RESET ------------------
if st.button("Reset Preferences"):
    st.session_state.liked_notes = []
    data["users"][username]["liked_notes"] = []

    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f)