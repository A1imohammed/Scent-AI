import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv
import urllib.parse
from duckduckgo_search import DDGS

load_dotenv()

# ------------------ CONFIG ------------------
WHATSAPP_NUMBER = "12345678900"  # 🔥 CHANGE THIS

# ------------------ STYLING ------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #020617);
        color: white;
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

# ------------------ IMAGE FETCH ------------------
@st.cache_data(show_spinner=False)
def fetch_perfume_image(name, brand):
    query = f"{brand} {name} perfume bottle"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=3))
        if results:
            return results[0]["image"]
    except:
        pass
    return "https://via.placeholder.com/200?text=No+Image"

# ------------------ WHATSAPP LINK ------------------
def generate_whatsapp_link(perfume):
    message = f"""
Hi! I'm interested in this fragrance:

Name: {perfume['name']}
Brand: {perfume['brand']}
Type: {perfume['type']}
Notes: {perfume['notes']}

Can you share price and availability?
"""
    encoded = urllib.parse.quote(message)
    return f"https://wa.me/{6172304375}?text={encoded}"

# ------------------ TRAIN MODEL ------------------
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

# ------------------ MEMORY ------------------
try:
    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
except:
    data = {"users": {}}

if username not in data["users"]:
    data["users"][username] = {"liked_notes": []}

if "liked_notes" not in st.session_state:
    st.session_state.liked_notes = data["users"][username]["liked_notes"]

# ------------------ TRAIN DATA ------------------
try:
    with open(TRAIN_FILE, "r") as f:
        train_data = json.load(f)
except:
    train_data = []

clf = train_model(train_data)

# ------------------ HEADER ------------------
st.image("logo.png", width=150)
st.title("Scent Genie")
st.write(f"Logged in as: {username}")

# ------------------ DATA ------------------
df = pd.read_csv("perfumes.csv")
df.columns = df.columns.str.strip().str.lower()

# ------------------ FILTERS ------------------
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

        if clf:
            scores += clf.predict_proba(X)[:, 1]

        top = scores.argsort()[-3:][::-1]

    st.subheader("Top Recommendations")

    for i in top:
        perfume = df_filtered.iloc[i]

        # MATCH REASON
        perfume_notes = perfume["notes"].lower()
        user_words = combined_input.lower().split()
        matched = [w for w in user_words if w in perfume_notes]
        reason = ", ".join(matched[:3]) if matched else "general match"

        # IMAGE
        image_url = perfume.get("image_url", "")
        if not image_url or not isinstance(image_url, str):
            image_url = fetch_perfume_image(perfume["name"], perfume["brand"])

        # CARD
        with st.container():
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(image_url, use_container_width=True)

            with col2:
                st.markdown(f"""
                <h3 style="color:#fbbf24;">{perfume['name']}</h3>
                <p style="color:#94a3b8;">{perfume['brand']}</p>
                <p>🧴 {perfume['type']}</p>
                <p>🌿 {perfume['notes']}</p>
                <p>⏱ {perfume['longevity']} hrs</p>
                <p>📢 {perfume['projection']}</p>
                <p style="color:#22c55e;">✔ Match: {reason}</p>
                """, unsafe_allow_html=True)

                # BUTTONS
                c1, c2 = st.columns(2)

                with c1:
                    like = st.button("👍 Like", key=f"like_{i}")

                with c2:
                    dislike = st.button("👎 Dislike", key=f"dislike_{i}")

                # BUY BUTTON (FIXED)
                buy_link = generate_whatsapp_link(perfume)

                st.markdown(f"""
                <a href="{buy_link}" target="_blank">
                    <button style="
                        width:100%;
                        padding:10px;
                        margin-top:10px;
                        background-color:#22c55e;
                        color:white;
                        border:none;
                        border-radius:8px;
                        font-size:16px;
                        cursor:pointer;
                    ">
                        🛒 Buy via WhatsApp
                    </button>
                </a>
                """, unsafe_allow_html=True)

        # ------------------ LOGIC ------------------
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

        data["users"][username]["liked_notes"] = st.session_state.liked_notes

        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f)

# ------------------ RESET ------------------
if st.button("Reset Preferences"):
    st.session_state.liked_notes = []
    data["users"][username]["liked_notes"] = []

    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f)