from bertopic import BERTopic
import joblib
import os

# --- Load documents fetched from Reddit + News ---
docs_path = "model/docs.pkl"

if not os.path.exists(docs_path):
    raise FileNotFoundError(f"❌ '{docs_path}' not found. Please run fetch_data.py first.")

docs = joblib.load(docs_path)

# --- Train BERTopic model ---
topic_model = BERTopic(language="english")
topics, _ = topic_model.fit_transform(docs)

# --- Save the trained model ---
joblib.dump(topic_model, "model/topic_model.pkl")
print("✅ Trained BERTopic model saved to model/topic_model.pkl")

#Load and save timestamps (now from Reddit)
timestamsps=joblib.load("model/timestamps.pkl")
joblib.dump(timestamsps,"model/timestamps.pkl")
