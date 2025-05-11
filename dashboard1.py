import streamlit as st
from bertopic import BERTopic
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from streamlit_autorefresh import st_autorefresh

# Auto-refresh every 1 minute
st_autorefresh(interval=60 * 1000, key="datarefresh")

# Load model and docs
try:
    topic_model: BERTopic = joblib.load("model/topic_model.pkl")
    docs = joblib.load("model/docs.pkl")
except FileNotFoundError:
    st.error("Model or document file not found. Please run fetch_data.py and train_model.py first.")
    st.stop()

# Generate topics and frequencies
topics, _ = topic_model.transform(docs)
freq_df = topic_model.get_topic_info()

# Filter out -1 topic (outliers)
freq_df = freq_df[freq_df.Topic != -1]

# Auto-generate simple topic names
def generate_clean_names(model, max_words=3):
    topic_names = {}
    for topic_id in model.get_topics():
        if topic_id == -1:
            continue
        top_words = [word for word, _ in model.get_topic(topic_id)[:max_words]]
        clean_name = " ".join(top_words).title()
        topic_names[topic_id] = clean_name
    return topic_names

topic_names = generate_clean_names(topic_model)

# --- Sidebar with Trending Topics ---
st.sidebar.title("🔥 Trending Topics")
selected_topic = None

for topic_id, name in topic_names.items():
    if st.sidebar.button(name):
        selected_topic = topic_id

# --- Main Area ---
st.title("📰 Real-Time Trend Dashboard")
if selected_topic is None:
    st.subheader("⬅️ Click a trending topic to view related posts")
else:
    st.subheader(f"📌 Related Posts for: {topic_names[selected_topic]}")
    topic_docs = [doc for i, doc in enumerate(docs) if topics[i] == selected_topic]
    
    if topic_docs:
        for doc in topic_docs:
            st.markdown(f"- {doc}")
    else:
        st.info("No related posts found.")

# Optional: Display count of posts per topic (just for info)
st.markdown("---")
st.caption("🔁 This dashboard auto-refreshes every 60 seconds.")
