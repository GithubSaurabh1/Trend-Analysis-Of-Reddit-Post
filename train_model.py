from bertopic import BERTopic
import praw
import datetime as dt
import pandas as pd
from psaw import PushshiftAPI


#Reddit API setup

reddit=praw.Reddit(
    client_id="Qz6zD8uzIYQPKVCdTKdgyQ",     #Reddit API Clinet ID 
    client_secret="NeSaY9Qq120d8HOsaXz1HYw4iftVwg",#Reddit API Clinet secret
    user_agent="topic_model by u/According_Housing37"
)

api = PushshiftAPI(reddit)

#FETECH REDDIT POSTS FROM MULTIPLE SUBREDDITS

subreddits=["technology","worldnews","science","politics","Artificl Intelligence"]
posts=[]

for sub in subreddits:
    print(f"Fetching from r/{sub}...")
    submissions=api.search_submissions(
        subreddits=sub,
        limit=200,
        after=int(dt.datetime(2024,1,1).timestamp())
    )
    
    for submission in submission:
        if hasattr(submission,'title'):
            posts.append(submission.title)
            
            
print(f"Total post collected :{len(posts)}")


#TRAIN BERTopic
topic_model=BERTopic(language="english")
topics,probs=topic_model.fit_transform(posts)

#Save Model

topic_model.save("bertropic_model")
print("Model Trained and saved to 'betropic_model/'")