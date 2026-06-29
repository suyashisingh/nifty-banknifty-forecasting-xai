import os
import sys
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'reddit_nifty_sentiment.csv')

FINBERT_MODEL = 'ProsusAI/finbert'
FINBERT_LABEL_MAP = {
    'positive': +1.0,
    'negative': -1.0,
    'neutral':   0.0,
}
PROGRESS_EVERY = 200


# --- STEP 1: Load existing CSV ---
print(f'Loading {DATA_PATH} ...')
if not os.path.exists(DATA_PATH):
    print(f'ERROR: File not found: {DATA_PATH}')
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f'Loaded {len(df)} rows. Columns: {list(df.columns)}')

if 'comment' not in df.columns:
    print('ERROR: "comment" column not found in CSV.')
    sys.exit(1)

if 'sentiment' not in df.columns:
    print('ERROR: "sentiment" column not found in CSV.')
    sys.exit(1)

df = df.rename(columns={'sentiment': 'vader_score'})


# --- STEP 2: Load FinBERT ---
print(f'\nLoading FinBERT model: {FINBERT_MODEL} ...')
try:
    from transformers import pipeline as hf_pipeline
    finbert = hf_pipeline(
        'text-classification',
        model=FINBERT_MODEL,
        truncation=True,
        max_length=512,
    )
    print('FinBERT loaded successfully.')
except Exception as exc:
    print(f'ERROR: Failed to load FinBERT: {exc}')
    sys.exit(1)


# --- STEP 3: Score each comment ---
print(f'\nScoring {len(df)} comments with FinBERT ...')

finbert_scores  = []
finbert_success = 0
finbert_fallback = 0

for i, row in enumerate(df.itertuples(index=False), 1):
    text       = str(row.comment)
    vader_val  = float(row.vader_score)

    try:
        result = finbert(text[:512])[0]
        label  = result['label']
        conf   = float(result['score'])
        fb_val = FINBERT_LABEL_MAP.get(label, 0.0) * conf
        finbert_scores.append(fb_val)
        finbert_success += 1
    except Exception:
        finbert_scores.append(vader_val)
        finbert_fallback += 1

    if i % PROGRESS_EVERY == 0:
        pct = finbert_success / i * 100
        print(f'  [{i}/{len(df)}] FinBERT success so far: {pct:.1f}%')

df['finbert_score'] = finbert_scores


# --- STEP 4: Compute combined score ---
df['sentiment'] = 0.5 * df['vader_score'] + 0.5 * df['finbert_score']


# --- STEP 5: Save back ---
out_cols = ['post_title', 'comment', 'vader_score', 'finbert_score', 'sentiment', 'created_utc']
missing  = [c for c in out_cols if c not in df.columns]
if missing:
    print(f'WARNING: Expected output columns missing: {missing} — saving all available columns.')
    df.to_csv(DATA_PATH, index=False)
else:
    df[out_cols].to_csv(DATA_PATH, index=False)

total = len(df)
fb_pct  = finbert_success  / total * 100
fal_pct = finbert_fallback / total * 100

print(f'\n{"=" * 60}')
print('REDDIT FINBERT RESCORING — FINAL SUMMARY')
print(f'{"=" * 60}')
print(f'Total comments rescored   : {total}')
print(f'FinBERT success           : {finbert_success} ({fb_pct:.1f}%)')
print(f'VADER fallback            : {finbert_fallback} ({fal_pct:.1f}%)')
print(f'Sentiment before (VADER)  : '
      f'mean={df["vader_score"].mean():.4f}  '
      f'min={df["vader_score"].min():.4f}  '
      f'max={df["vader_score"].max():.4f}')
print(f'Sentiment after (combined): '
      f'mean={df["sentiment"].mean():.4f}  '
      f'min={df["sentiment"].min():.4f}  '
      f'max={df["sentiment"].max():.4f}')
print(f'Output saved to           : {DATA_PATH}')
print(f'{"=" * 60}')
