import streamlit as st
from bertopic import BERTopic
import joblib
from streamlit_autorefresh import st_autorefresh
from streamlit.components.v1 import html
from keybert import KeyBERT
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="Trending Topics", layout="wide")
# Auto-refresh every 1 minute
st_autorefresh(interval=60 * 1000, key="datarefresh")

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
freq_df = freq_df[freq_df.Topic != -1].sort_values("Count", ascending=False).head(25)

# Generate smarter topic names using KeyBERT
kw_model = KeyBERT(SentenceTransformer('all-MiniLM-L6-v2'))

def generate_enhanced_topic_names(model, topic_ids, docs, n_words=3, n_grams=2):
    topic_names = {}
    
    for topic_id in topic_ids:
        # Extract top words from the topic
        topic_words = [word for word, _ in model.get_topic(topic_id)[:15]]  # Use more top words

        # Combine the top words into a phrase to generate more meaningful topic names
        topic_word_str = " ".join(topic_words)
        
        # Use a TF-IDF model to get the most relevant terms in the context of the documents
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, n_grams))
        tfidf_matrix = tfidf_vectorizer.fit_transform([topic_word_str] + docs)
        terms = np.array(tfidf_vectorizer.get_feature_names_out())
        scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

        # Get the top n words based on TF-IDF scores
        top_n_indices = scores.argsort()[-n_words:][::-1]
        tfidf_keywords = terms[top_n_indices]

        # Use KeyBERT to get semantically relevant keywords from the documents
        semantically_relevant_keywords = kw_model.extract_keywords(" ".join(docs), keyphrase_ngram_range=(1, n_grams), stop_words='english', top_n=3)
        
        # Combine both sources of keywords: TF-IDF-based and KeyBERT-based
        final_keywords = list(set(tfidf_keywords).union([keyword[0] for keyword in semantically_relevant_keywords]))

        # Join keywords into a meaningful topic name (up to 'n_words' words)
        topic_name = " ".join(final_keywords[:n_words]) if final_keywords else topic_word_str.title()
        
        # Store the topic name
        topic_names[topic_id] = topic_name.title()
    
    return topic_names

topic_names = generate_enhanced_topic_names(topic_model, freq_df.Topic.values, docs)

# Set up modern layout
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

# Display topics as buttons
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
