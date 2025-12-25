import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spotify Listening Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- DARK MODE + STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: #FAFAFA;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #39FF14;
}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Spotify Listening Behavior Dashboard")
st.caption("EDA + Gradient Boosting ML | Ethical & Realistic Evaluation")

# ---------------- FILE UPLOADER (FIX) ----------------
uploaded_file = st.file_uploader(
    "Upload your Spotify StreamingHistory_music_0.json file",
    type=["json"]
)

if uploaded_file is None:
    st.info("Please upload your Spotify streaming history JSON file to view the dashboard.")
    st.stop()

# ---------------- LOAD DATA ----------------
data = json.loads(uploaded_file.read().decode("utf-8"))
df = pd.DataFrame(data)

# ---------------- CLEANING ----------------
df['endTime'] = pd.to_datetime(df['endTime'])
df['minutesPlayed'] = df['msPlayed'] / 60000
df = df[df['minutesPlayed'] > 0.5]

# ---------------- FEATURES ----------------
df['hour'] = df['endTime'].dt.hour
df['day'] = df['endTime'].dt.day_name()

# Target (defined by hour, NOT used as feature)
df['time_period'] = df['hour'].apply(
    lambda x: 1 if 6 <= x < 18 else 0   # 1 = Day, 0 = Night
)

# ---------------- KPIs ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎵 Total Listens", len(df))
col2.metric("⏱️ Total Minutes", round(df['minutesPlayed'].sum(), 1))
col3.metric("🌙 Night %", round((df['time_period'] == 0).mean()*100, 1))
col4.metric("🎤 Unique Artists", df['artistName'].nunique())

st.divider()

# ---------------- SINGLE GRID (ALL VISUALS) ----------------
fig, axs = plt.subplots(2, 3, figsize=(16, 8))
fig.patch.set_facecolor('#0E1117')
neon = "#39FF14"

# Top Artists
top_artists = (
    df.groupby('artistName')['minutesPlayed']
      .sum().sort_values(ascending=False).head(8)
)
axs[0,0].barh(top_artists.index, top_artists.values, color=neon)
axs[0,0].set_title("Top Artists", color=neon)

# Listening by Hour
df['hour'].value_counts().sort_index().plot(
    ax=axs[0,1], color=neon
)
axs[0,1].set_title("Listening by Hour", color=neon)

# Listening by Day
df['day'].value_counts().plot(
    kind='bar', ax=axs[0,2], color=neon
)
axs[0,2].set_title("Listening by Day", color=neon)

# Session Length Distribution
axs[1,0].hist(df['minutesPlayed'], bins=20, color=neon)
axs[1,0].set_title("Session Length Distribution", color=neon)

# Day vs Night Split
pd.Series(df['time_period']).map({1:"Day",0:"Night"}).value_counts().plot(
    kind='bar', ax=axs[1,1], color=neon
)
axs[1,1].set_title("Day vs Night Usage", color=neon)

# Empty cell
axs[1,2].axis('off')

for ax in axs.flat:
    ax.set_facecolor('#0E1117')
    ax.tick_params(colors='white')

st.pyplot(fig)

st.divider()

# ---------------- ML SECTION ----------------
st.subheader("🤖 ML Prediction: Day vs Night Listening")

X = df[['minutesPlayed']]
y = df['time_period']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

st.metric("Gradient Boosting Accuracy", f"{accuracy}%")

# ---------------- CLASSIFICATION REPORT ----------------
report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.subheader("📊 Model Evaluation Table")
st.dataframe(report_df.style.format("{:.2f}"))

st.markdown("""
**Model summary**
- Algorithm: Gradient Boosting Classifier  
- Feature used: Listening session duration  
- Target: Day vs Night listening  

The modest accuracy reflects ethical evaluation without data leakage.
""")

st.caption("Built with Python • Pandas • Matplotlib • Scikit-learn • Streamlit")
