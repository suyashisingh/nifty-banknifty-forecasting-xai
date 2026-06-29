import os
import sys
import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'reddit_nifty_sentiment.csv')

# --- Reddit credentials ---
REDDIT_CLIENT_ID     = 'YOUR_REDDIT_CLIENT_ID'
REDDIT_CLIENT_SECRET = 'YOUR_REDDIT_CLIENT_SECRET'
REDDIT_USER_AGENT    = 'sentiment_analyzer/0.1 by suyashisingh'

# --- Collection parameters ---
SUBREDDIT   = 'IndiaInvestments'
QUERY       = 'NIFTY'
TIME_FILTER = 'all'
POST_LIMIT  = 50

# --- FinBERT model ---
FINBERT_MODEL = 'yiyanghkust/finbert-tone'

FINBERT_LABEL_MAP = {
    'Positive': +1.0,
    'Negative': -1.0,
    'Neutral':   0.0,
}

PROGRESS_EVERY = 100   # print a progress line every N comments


def load_finbert():
    """Load FinBERT pipeline once at startup. Returns None on failure."""
    try:
        from transformers import pipeline
        print(f'Loading FinBERT model: {FINBERT_MODEL} ...')
        pipe = pipeline(
            'text-classification',
            model=FINBERT_MODEL,
            truncation=True,
            max_length=512,
        )
        print('FinBERT loaded successfully.')
        return pipe
    except Exception as exc:
        print(f'WARNING: Could not load FinBERT ({exc}). Will use VADER-only fallback.')
        return None


def finbert_score(pipe, text: str) -> float | None:
    """
    Run FinBERT on text. Returns weighted numeric score or None on failure.
    Score = direction (+1/-1/0) * confidence (0-1), range -1 to +1.
    """
    try:
        result = pipe(text[:512])[0]          # truncate at char level as safety
        label  = result['label']              # 'Positive' / 'Negative' / 'Neutral'
        conf   = float(result['score'])
        return FINBERT_LABEL_MAP.get(label, 0.0) * conf
    except Exception:
        return None


def combined_score(vader: float, finbert: float | None) -> tuple[float, bool]:
    """
    Returns (combined_score, used_dual_model).
    Falls back to VADER alone if finbert is None.
    """
    if finbert is None:
        return vader, False
    return 0.5 * vader + 0.5 * finbert, True


if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

    # Load models
    vader_analyzer = SentimentIntensityAnalyzer()
    finbert_pipe   = load_finbert()

    # Connect to Reddit
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    print(f'\nFetching r/{SUBREDDIT} | query="{QUERY}" | '
          f'time_filter={TIME_FILTER} | limit={POST_LIMIT}')

    data             = []
    total_comments   = 0
    finbert_success  = 0

    for submission in reddit.subreddit(SUBREDDIT).search(
        QUERY, time_filter=TIME_FILTER, limit=POST_LIMIT
    ):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            text = comment.body

            vader  = vader_analyzer.polarity_scores(text)['compound']
            fb     = finbert_score(finbert_pipe, text) if finbert_pipe else None
            score, dual = combined_score(vader, fb)

            if dual:
                finbert_success += 1

            data.append({
                'post_title':    submission.title,
                'comment':       text,
                'vader_score':   vader,
                'finbert_score': fb if fb is not None else vader,
                'sentiment':     score,
                'created_utc':   comment.created_utc,
            })

            total_comments += 1
            if total_comments % PROGRESS_EVERY == 0:
                fb_rate = finbert_success / total_comments * 100
                print(f'  Processed {total_comments} comments | '
                      f'FinBERT success rate so far: {fb_rate:.1f}%')

    if not data:
        print('No comments collected. Exiting.')
        sys.exit(0)

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_PATH, index=False)

    fb_rate_final = finbert_success / total_comments * 100

    print(f'\n{"=" * 60}')
    print('REDDIT COLLECTION — FINAL SUMMARY')
    print(f'{"=" * 60}')
    print(f'Total comments processed : {total_comments}')
    print(f'FinBERT success (dual)   : {finbert_success} ({fb_rate_final:.1f}%)')
    print(f'VADER fallback           : {total_comments - finbert_success} '
          f'({100 - fb_rate_final:.1f}%)')
    print(f'Sentiment — mean : {df["sentiment"].mean():.4f}')
    print(f'Sentiment — min  : {df["sentiment"].min():.4f}')
    print(f'Sentiment — max  : {df["sentiment"].max():.4f}')
    print(f'Output           : {OUTPUT_PATH}')
    print(f'Rows saved       : {len(df)}')
    print(f'{"=" * 60}')
