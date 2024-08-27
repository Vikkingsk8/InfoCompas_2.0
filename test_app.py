import unittest
from app import app  # Импортируйте ваше Flask-приложение
import pandas as pd
import os

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_download_pdf(self):
        response = self.app.get('/download_pdf')
        self.assertEqual(response.status_code, 200)

    def test_chat_endpoint(self):
        data = {'question': 'привет'}
        response = self.app.post('/chat', json=data)
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('answer', json_response)
        self.assertEqual(json_response['answer'], 'Привет! Чем могу помочь?')

    def test_load_suggestions(self):
        response = self.app.get('/load_suggestions')
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('suggestions', json_response)
        self.assertIsInstance(json_response['suggestions'], list)

    def test_chat_endpoint_with_short_question(self):
        data = {'question': ''}
        response = self.app.post('/chat', json=data)
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('answer', json_response)
        self.assertEqual(json_response['answer'], 'Пожалуйста, задайте более конкретный вопрос.')

    def test_chat_endpoint_with_relevant_links(self):
        data = {'question': 'что ты умеешь'}
        response = self.app.post('/chat', json=data)
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('answer', json_response)
        self.assertIn('links', json_response)
        self.assertIsInstance(json_response['links'], list)

if __name__ == '__main__':
    unittest.main()