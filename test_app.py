import unittest

from app import create_app
from train import train
from tempfile import TemporaryDirectory
from pathlib import Path
import joblib


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        path = Path(cls.directory.name) / 'model.joblib'
        train(output=path)
        cls.client = create_app(joblib.load(path)).test_client()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_home(self):
        self.assertEqual(self.client.get('/').status_code, 200)

    def test_predictions(self):
        for text, expected in [('I love this product it is wonderful', 'positive'), ('Poor quality and a complete waste of money', 'negative'), ('The package arrived on Tuesday', 'neutral')]:
            response = self.client.post('/api/predict', json={'text': text})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json['sentiment'], expected)
            self.assertAlmostEqual(sum(response.json['scores'].values()), 1, places=5)

    def test_invalid_input(self):
        for payload in [{}, [], {'text': 12}, {'text': '  '}, {'text': 'a' * 5001}]:
            self.assertEqual(self.client.post('/api/predict', json=payload).status_code, 400)
        self.assertEqual(self.client.post('/api/predict', data='bad', content_type='application/json').status_code, 400)

    def test_request_limit(self):
        self.assertEqual(self.client.post('/api/predict', json={'text': 'a' * 40000}).status_code, 413)


if __name__ == '__main__':
    unittest.main()
