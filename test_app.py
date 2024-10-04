# Авторы: Ермилов В.В., Файбисович В.А.
import unittest
from app import app
import pandas as pd
import os
from config import Config

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

    def test_chat_endpoint_with_conversational_response(self):
        data = {'question': 'привет'}
        response = self.app.post('/chat', json=data)
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('answer', json_response)
        self.assertEqual(json_response['answer'], 'Привет! Чем могу помочь?')

    def test_chat_endpoint_with_invalid_question(self):
        data = {'question': 'xyz'}
        response = self.app.post('/chat', json=data)
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('answer', json_response)
        self.assertIn('Извините, я не совсем понял ваш вопрос', json_response['answer'])

    def test_feedback_endpoint(self):
        data = {
            'question': 'Тестовый вопрос',
            'answer': 'Тестовый ответ',
            'type': 'like',
            'comment': 'Тестовый комментарий'
        }
        response = self.app.post('/feedback', json=data)
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('message', json_response)
        self.assertEqual(json_response['message'], 'Спасибо за ваш отзыв!')

    def test_like_endpoint(self):
        data = {
            'question': 'Тестовый вопрос',
            'answer': 'Тестовый ответ'
        }
        response = self.app.post('/like', json=data)
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('message', json_response)
        self.assertEqual(json_response['message'], 'Лайк получен')

    def test_dislike_endpoint(self):
        data = {
            'question': 'Тестовый вопрос',
            'answer': 'Тестовый ответ'
        }
        response = self.app.post('/dislike', json=data)
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('message', json_response)
        self.assertEqual(json_response['message'], 'Дизлайк получен')

    def test_analytics_data_endpoint(self):
        response = self.app.get('/analytics_data')
        self.assertEqual(response.status_code, 200)
        json_response = response.get_json()
        self.assertIn('total_queries', json_response)
        self.assertIn('successful_queries', json_response)
        self.assertIn('success_rate', json_response)
        self.assertIn('top_questions', json_response)
        self.assertIn('feedback_distribution', json_response)
        self.assertIn('queries_over_time', json_response)
        self.assertIn('queries_by_day', json_response)
        self.assertIn('queries_by_week', json_response)
        self.assertIn('queries_by_month', json_response)

if __name__ == '__main__':
    unittest.main()