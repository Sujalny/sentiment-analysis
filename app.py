"""Local Flask sentiment analysis application."""
import joblib
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from train import MODEL_PATH, ROOT, train


def create_app(model=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 32 * 1024
    if model is None:
        # Only load the trusted, locally generated artifact.
        dataset = ROOT / 'data' / 'tweeteval' / 'train.csv'
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
        elif dataset.exists():
            model = train(dataset)
        else:
            raise RuntimeError('Model missing. Run python train_benchmark.py first to download data and train the full model.')

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.post('/api/predict')
    def predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get('text'), str):
            return jsonify(error='Provide a JSON object with a text string.'), 400
        text = data['text'].strip()
        if not text or len(text) > 5000:
            return jsonify(error='Enter between 1 and 5,000 characters.'), 400
        probabilities = model.predict_proba([text])[0]
        scores = {label: round(float(value), 6) for label, value in zip(model.classes_, probabilities)}
        ranked = sorted(scores.values(), reverse=True)
        uncertain = ranked[0] < 0.60 or ranked[0] - ranked[1] < 0.20
        return jsonify(sentiment=max(scores, key=scores.get), scores=scores, uncertain=uncertain)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify(error='Request is too large.'), 413

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=5000)
