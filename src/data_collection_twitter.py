import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Twitter API v2 credentials (replace with yours)
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAK2s2AEAAAAATNGeEJ33f2o399oQ%2FHcqmUCFBMk%3DhNElCGN6Hc2rNPvHPgw8oQPCPQUW2piLNK7tjaLihJETMXEpO0"  # Or use OAuth 2.0

# Authenticate
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    wait_on_rate_limit=True
)

# Search parameters
query = "NIFTY OR BANKNIFTY -is:retweet lang:en"
max_results = 100  # Adjust based on your access level

# Fetch tweets
tweets = client.search_recent_tweets(
    query=query,
    max_results=max_results,
    tweet_fields=['created_at', 'text']
)

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
tweet_data = []
if tweets.data is not None:
    for tweet in tweets.data:
        sentiment = analyzer.polarity_scores(tweet.text)['compound']
        tweet_data.append({
            'created_at': tweet.created_at,
            'text': tweet.text,
            'sentiment': sentiment
        })

# Save to CSV
df = pd.DataFrame(tweet_data)
df.to_csv("data/raw/twitter_nifty_sentiment.csv", index=False)
print(df.head())
