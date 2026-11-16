"""Train a small, reproducible demo model, or supply a labeled CSV."""
import argparse
import csv
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'sentiment.joblib'


def train(data_path=ROOT / 'data' / 'reviews.csv', output=MODEL_PATH):
    with Path(data_path).open(encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows or any(not r.get('text', '').strip() or r.get('label') not in {'positive', 'negative', 'neutral'} for r in rows):
        raise ValueError('CSV must contain nonempty text and positive/negative/neutral label columns.')
    if {r['label'] for r in rows} != {'positive', 'negative', 'neutral'}:
        raise ValueError('Training data must include all three sentiment labels.')
    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, C=4)),
    ])
    model.fit([r['text'] for r in rows], [r['label'] for r in rows])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'reviews.csv')
    args = parser.parse_args()
    train(args.data)
    print(f'Model saved to {MODEL_PATH}')
