"""Download TweetEval sentiment splits and evaluate before saving the replacement."""
import csv
import json
import urllib.request

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

from train import ROOT, MODEL_PATH, train

BASE = 'https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/'
LABELS = {'0': 'negative', '1': 'neutral', '2': 'positive'}


def download(split):
    destination = ROOT / 'data' / 'tweeteval'
    destination.mkdir(parents=True, exist_ok=True)
    parts = []
    for kind in ('text', 'labels'):
        path = destination / f'{split}_{kind}.txt'
        if not path.exists():
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(BASE + path.name, timeout=30) as response:
                path.write_bytes(response.read())
        parts.append(path.read_text(encoding='utf-8').splitlines())
    texts, labels = parts
    if len(texts) != len(labels):
        raise ValueError('Text and label counts differ')
    return texts, [LABELS[label] for label in labels]


if __name__ == '__main__':
    train_text, train_labels = download('train')
    test_text, test_labels = download('test')
    csv_path = ROOT / 'data' / 'tweeteval' / 'train.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['text', 'label'])
        writer.writerows(zip(train_text, train_labels))
    # Always compare to the original demo, including on a clean checkout.
    baseline_path = MODEL_PATH.with_name('baseline.joblib')
    baseline = train(output=baseline_path)
    baseline_predictions = baseline.predict(test_text)
    candidate_path = MODEL_PATH.with_name('candidate.joblib')
    model = train(csv_path, candidate_path)
    predictions = model.predict(test_text)
    report = {
        'dataset': BASE,
        'training_examples': len(train_text),
        'test_examples': len(test_text),
        'baseline_accuracy': accuracy_score(test_labels, baseline_predictions),
        'baseline_macro_f1': f1_score(test_labels, baseline_predictions, average='macro'),
        'accuracy': accuracy_score(test_labels, predictions),
        'macro_f1': f1_score(test_labels, predictions, average='macro'),
        'classification_report': classification_report(test_labels, predictions, output_dict=True),
    }
    if report['macro_f1'] <= report['baseline_macro_f1']:
        raise RuntimeError('Candidate did not improve held-out macro F1')
    candidate_path.replace(MODEL_PATH)
    baseline_path.unlink(missing_ok=True)
    (ROOT / 'models' / 'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
