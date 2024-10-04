# Авторы: Ермилов В.В., Файбисович В.А.
from locust import HttpUser, task, between
import random

class ChatUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def chat(self):
        questions = [
            "Как тебя зовут?",
            "Что ты умеешь?",
            "Как дела?",
            "Привет",
            "Как найти информацию о продукте?"
        ]
        question = random.choice(questions)
        self.client.post("/chat", json={"question": question})