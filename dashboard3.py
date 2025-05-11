import streamlit as st
from bertopic import BERTopic
import joblib
from streamlit_autorefresh import st_autorefresh
from streamlit.components.v1 import html
from keybert import KeyBERT
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer


# Initialize KeyBERT
kw_model = KeyBERT()

# Function for smart topic naming
def generate_smart_topic_names(model, docs, topics, max_words=4):
    topic_names = {}
    for topic_id in model.get_topics():
        if topic_id == -1:
            continue
        topic_docs = [docs[i] for i, t in enumerate(topics) if t == topic_id]
        if topic_docs:
            keywords = kw_model.extract_keywords(" ".join(topic_docs), keyphrase_ngram_range=(1, 3), stop_words='english', top_n=1)
            topic_names[topic_id] = keywords[0][0].title() if keywords else f"Topic {topic_id}"
        else:
            topic_names[topic_id] = f"Topic {topic_id}"
    return topic_names

# Load model and docs
try:
    topic_model: BERTopic = joblib.load("model/topic_model.pkl")
    docs = joblib.load("model/docs.pkl")
except FileNotFoundError:
    st.error("Model or document file not found. Please run fetch_data.py and train_model.py first.")
    st.stop()

# Get topics and frequencies
topics, _ = topic_model.transform(docs)
freq_df = topic_model.get_topic_info()
freq_df = freq_df[freq_df.Topic != -1].sort_values("Count", ascending=False).head(20)

# Generate smarter topic names using KeyBERT
topic_names = generate_smart_topic_names(topic_model, docs, topics)

# Set up modern layout
st.set_page_config(page_title="Trending Topics", layout="wide")

st.markdown("""
    <style>
    .card {
        padding: 1.2rem;
        background: linear-gradient(to right, #232526, #414345);
        color: white;
        border-radius: 14px;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .card:hover {
        background-color: #2e2e40;
        transform: translateY(-3px);
    }
    .section-header {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f1f1f1;
        margin: 1rem 0;
    }
    .grid-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Modern Real-Time Trending Dashboard")
st.caption("Auto-refreshes every 60 seconds | Click a topic to see related posts")

# Trending topic selection
st.sidebar.title("🔥 Top Trending Topics")

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

for topic_id, name in topic_names.items():
    if st.sidebar.button(name, key=f"sidebar_button_{topic_id}"):
        st.session_state.selected_topic = topic_id

selected_topic = st.session_state.selected_topic

# Show selected topic
topic_display = st.container()

if selected_topic is not None:
    topic_display.markdown(f"<div class='section-header'>📌 Related Posts for: {topic_names[selected_topic]}</div>", unsafe_allow_html=True)
    topic_docs = [doc for i, doc in enumerate(docs) if topics[i] == selected_topic]

    if topic_docs:
        for doc in topic_docs:
            topic_display.markdown(f"<div class='card'>{doc}</div>", unsafe_allow_html=True)
    else:
        topic_display.info("No related posts found.")

    # Add trend visualization
    st.subheader("📈 Topic Trend Over Time")
    timestamps = joblib.load("model/timestamps.pkl")
    time_data = pd.DataFrame({"timestamp": timestamps, "topic": topics})
    time_data = time_data[time_data.topic == selected_topic]
    time_data['timestamp'] = pd.to_datetime(time_data['timestamp'])
    time_data['hour'] = time_data['timestamp'].dt.floor('H')
    hourly_counts = time_data.groupby('hour').size().reset_index(name='count')

    fig = px.line(hourly_counts, x='hour', y='count', markers=True,
                  labels={'hour': 'Time', 'count': 'Mentions'},
                  title=f"Trend of '{topic_names[selected_topic]}' over time")
    st.plotly_chart(fig, use_container_width=True)

else:
    topic_display.markdown("<div class='section-header'>⬅️ Select a trending topic to view related posts</div>", unsafe_allow_html=True)
    topic_display.markdown("<div class='grid-container'>", unsafe_allow_html=True)
    for topic_id, name in topic_names.items():
        topic_display.markdown(f"<div class='card'>{name}</div>", unsafe_allow_html=True)
    topic_display.markdown("</div>", unsafe_allow_html=True)
