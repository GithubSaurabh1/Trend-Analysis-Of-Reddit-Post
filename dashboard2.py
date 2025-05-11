import streamlit as st
from bertopic import BERTopic
import joblib
from streamlit_autorefresh import st_autorefresh
from keybert import KeyBERT

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
freq_df = freq_df[freq_df.Topic != -1]

# Generate improved topic names using KeyBERT
kw_model = KeyBERT()
topic_labels = {}

for topic_id in topic_model.get_topics().keys():
    if topic_id == -1:
        continue
    topic_docs = [doc for i, doc in enumerate(docs) if topics[i] == topic_id][:10]
    combined_docs = " ".join(topic_docs)
    keywords = kw_model.extract_keywords(combined_docs, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
    label = " | ".join([kw[0].title() for kw in keywords])
    topic_labels[topic_id] = label

# Assign the labels to the model
topic_model.set_topic_labels(topic_labels)
topic_names = topic_labels

# Trending header
st.markdown("""
    <style>
    .trending-topic {
        padding: 1rem;
        background-color: #f0f2f6;
        color: #111 !important; /*Ensure text is visible */
        border-radius: 12px;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .trending-topic:hover {
        background-color: #e6eaf1;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

st.title("📰 Real-Time Trend Dashboard")
st.caption("🔁 This dashboard auto-refreshes every 60 seconds.")

# Trending topic selection
selected_topic = None

st.sidebar.title("🔥 Trending Topics")
for topic_id, name in topic_names.items():
    if st.sidebar.button(name):
        selected_topic = topic_id

# Show topic posts
if selected_topic is not None:
    st.subheader(f"📌 Related Posts for: {topic_names[selected_topic]}")
    topic_docs = [doc for i, doc in enumerate(docs) if topics[i] == selected_topic]

    if topic_docs:
        for doc in topic_docs:
            st.markdown(f"<div class='trending-topic'>{doc}</div>", unsafe_allow_html=True)
    else:
        st.info("No related posts found.")
else:
    st.subheader("⬅️ Click a trending topic to view related posts")
    for topic_id, name in topic_names.items():
        st.markdown(f"<div class='trending-topic'>{name}</div>", unsafe_allow_html=True)
