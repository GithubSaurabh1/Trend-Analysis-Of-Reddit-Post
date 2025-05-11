import praw
import joblib
import os
import requests
import datetime

# --- Reddit API setup ---
reddit = praw.Reddit(
    client_id="Qz6zD8uzIYQPKVCdTKdgyQ",
    client_secret="NeSaY9Qq120d8HOsaXz1HYw4iftVwg",
    user_agent="topic_model by u/According_Housing437"
)

# --- NewsAPI setup ---
newsapi_key = "fc6afd07287a4609b77417698858ae66"  # 🔑 Get from https://newsapi.org

# --- Subreddits to fetch from ---
subreddits = ["worldnews", "technology", "Apple", "CryptoCurrency", "MachineLearning", "politics"]

# --- Fetch Reddit posts ---
reddit_docs = []
timestamps=[]
for sub in subreddits:
    subreddit =reddit.subreddit(sub)
    
    for post in subreddit.hot(limit=500):
        if not post.stickied:
            reddit_docs.append(post.title + " " + post.selftext)
            timestamps.append(datetime.datetime.fromtimestamp(post.created_utc))
print(f"✅ Fetched {len(reddit_docs)} Reddit posts.")

# --- Fetch NewsAPI headlines ---
# def fetch_news_articles(api_key, query="technology", page_size=50):
#     url = f"https://newsapi.org/v2/top-headlines?q={query}&language=en&pageSize={page_size}&apiKey={api_key}"
#     response = requests.get(url)
#     articles = response.json().get("articles", [])
#     return [article["title"] for article in articles if article.get("title")]

# news_topics = ["technology", "business", "science", "world", "sports", "entertainment"]
# news_docs = []

# for topic in news_topics:
#     news_docs.extend(fetch_news_articles(newsapi_key, query=topic))

# placeholder_time = datetime.datetime.now()
# news_timestamps = [placeholder_time] * len(news_docs)
# print(f"✅ Fetched {len(news_docs)} news headlines.")

# --- Combine and Save ---
# all_docs = reddit_docs + news_docs
# all_timestamps = timestamps + news_timestamps 
# assert len(all_docs) == len(all_timestamps), f"Length mismatch: {len(all_docs)} docs vs {len(all_timestamps)} timestamps"

# --- Save both
os.makedirs("model", exist_ok=True)
joblib.dump(reddit_docs, "model/docs.pkl")
joblib.dump(timestamps,"model/timestamps.pkl")

# joblib.dump(reddit_docs, "model/docs.pkl")
# joblib.dump(timestamps,"model/timestamps.pkl")

print(f"📦 Total documents saved: {len(reddit_docs)} to model/docs.pkl")
print(f"📦 Total documents saved: {len(timestamps)} to model/timestamps.pkl")
