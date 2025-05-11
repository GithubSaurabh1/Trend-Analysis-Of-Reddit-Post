from bertopic import BERTopic
import pandas as pd
import praw
import joblib

# Step 1: Reddit API setup
reddit = praw.Reddit(
    client_id="Qz6zD8uzIYQPKVCdTKdgyQ",
    client_secret="NeSaY9Qq120d8HOsaXz1HYw4iftVwg",
    user_agent="topic_model by u/According_Housing437"
)

# Step 2: Fetch post titles from a subreddit
print("Fetching data from r/technology...")
subreddit = reddit.subreddit("technology")
titles = [post.title for post in subreddit.hot(limit=100)]

# Step 3: Train the BERTopic model
print("Training BERTopic model...")
topic_model = BERTopic(language="english")
topics, probs = topic_model.fit_transform(titles)

# Step 4: Show top topics
print("\nTop Topics:")
print(topic_model.get_topic_info().head())

# Step 5: Save the model and data
topic_model.save("bertopic_reddit_model")
df = pd.DataFrame({"title": titles, "topic": topics})
df.to_csv("reddit_titles_with_topics.csv", index=False)

import joblib
import os

os.makedirs("model", exist_ok=True)  # This will create the folder if it doesn't exist

# Assume topic_model and docs are already created
joblib.dump(topic_model, "model/topic_model.pkl")
joblib.dump(titles, "model/docs.pkl")


print("\n✅ Model trained and saved as 'bertopic_reddit_model'")
print("✅ Data saved to 'reddit_titles_with_topics.csv'")
