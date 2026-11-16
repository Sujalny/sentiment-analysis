# Sentiment Studio

A dark-themed local Flask application that classifies English text as positive, negative, or neutral using a Scikit-learn TF-IDF + Logistic Regression pipeline persisted with Joblib.

## Run on Windows

```powershell
cd sentiment-analysis
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_benchmark.py
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000. The first training run downloads the dataset and may take a few minutes. The downloaded dataset, generated model, and local logs are excluded from the repository. If the saved model is missing but the full training CSV is cached, the app trains it at startup; otherwise it asks you to run the benchmark training command. Debug mode is off and the development server binds only to localhost.

If the Windows Python launcher does not detect your installation, replace `py -3` in the environment creation command with the full path to your Python executable. On this computer that is `C:\Users\sujal\AppData\Local\Programs\Python\Python313\python.exe` (invoke it with PowerShell's `&` operator).

## API

`POST /api/predict` with `Content-Type: application/json`:

```json
{"text": "I love this product"}
```

Returns `sentiment` and `scores` for all three labels. Text must contain 1–5,000 characters after trimming; invalid input returns 400 and oversized requests return 413.

## Train with your own data

Provide a UTF-8 CSV with `text,label` columns and all three labels (`positive`, `negative`, `neutral`):

```powershell
.\.venv\Scripts\python.exe train.py --data path\to\reviews.csv
```

Restart the app after training. Only load Joblib files you trust: loading an artifact can execute Python code.

The saved model now uses 45,615 training tweets from [TweetEval](https://github.com/cardiffnlp/tweeteval). On its separate 12,284-example test split, accuracy improved from 44.0% to 59.2%, and macro F1 from 0.401 to 0.574. Full per-class metrics are in `models/evaluation.json`. This is still an experimental baseline, not a high-accuracy general-purpose analyzer. Tweet performance does not establish performance on reviews or other domains. Scores are uncalibrated. Sarcasm and mixed sentiment remain difficult.

Reproduce dataset downloading, training, and held-out evaluation with `python train_benchmark.py`. The test split is used for reporting, not fitting; no hyperparameter search was performed. Downloads are cached in `data/tweeteval`. The original `data/reviews.csv` is retained only as a small test fixture. Running plain `train.py` still trains that fixture; use `train_benchmark.py` for the full model. Follow the source repository's dataset terms and attribution for redistribution.

The API returns `uncertain: true` when the top probability is below 0.60 or the top-two margin is below 0.20. This display heuristic is not a calibration guarantee.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Checks cover model serialization, example predictions, the homepage, input validation, and request size limits. Example predictions are smoke checks on bundled training examples, not a measure of generalization.

Text is processed by your Flask server and is not stored by the application. The stylesheet optionally loads Google Fonts; system fonts are used offline. Use a production WSGI server and suitable deployment controls if exposing the application beyond your computer.
