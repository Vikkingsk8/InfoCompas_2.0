from flask import Flask, request, jsonify, render_template, send_file, session
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os
import logging
import fitz  # PyMuPDF
import csv

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')  # Используйте переменные окружения для секретных ключей

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Конфигурация
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CACHE_DIR = os.path.join(BASE_DIR, 'cache')
    CACHE_FILE = os.getenv('CACHE_FILE', os.path.join(CACHE_DIR, 'embeddings_cache.npy'))
    EXCEL_PATH = os.getenv('EXCEL_PATH', os.path.join(DATA_DIR, 'ответы.xlsx'))
    LINKS_PATH = os.getenv('LINKS_PATH', os.path.join(DATA_DIR, 'links.xlsx'))
    PDF_PATH = os.getenv('PDF_PATH', os.path.join(DATA_DIR, 'instruction.pdf'))
    FEEDBACK_FILE = os.path.join(DATA_DIR, 'feedback.csv')

# Загрузка токенизатора и модели из Hugging Face
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

# Функция для получения эмбеддинга
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Проверка наличия кэша и загрузка его
def load_embeddings_cache():
    if os.path.exists(Config.CACHE_FILE):
        try:
            return np.load(Config.CACHE_FILE, allow_pickle=True).item()
        except Exception as e:
            logging.error(f"Error loading embeddings cache: {e}")
    return {}

embeddings_cache = load_embeddings_cache()

# Загрузка данных из Excel
def load_excel_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_excel(path)
    return df

def preprocess_excel_data(df, column_name):
    df = df.assign(Текст_вопроса=df[column_name].str.split('?')).explode('Текст_вопроса')
    df['Текст_вопроса'] = df['Текст_вопроса'].str.lower().str.strip()
    return df

df_answers = load_excel_data(Config.EXCEL_PATH)
df_answers = preprocess_excel_data(df_answers, 'Текст вопроса')

df_links = load_excel_data(Config.LINKS_PATH)
df_links = preprocess_excel_data(df_links, 'Вопрос')

# Получение эмбеддингов для вопросов
start_time = time.time()
df_answers['embedding'] = df_answers['Текст_вопроса'].apply(lambda x: embeddings_cache.get(x) if x in embeddings_cache else get_embedding(x))
end_time = time.time()
logging.info(f"Время получения эмбеддингов: {end_time - start_time} секунд")

# Сохранение эмбеддингов в кэш
for idx, row in df_answers.iterrows():
    embeddings_cache[row['Текст_вопроса']] = row['embedding']
np.save(Config.CACHE_FILE, embeddings_cache)

# Функция для поиска наиболее похожего вопроса
def find_most_similar_question(query_embedding, df):
    similarities = cosine_similarity([query_embedding], df['embedding'].tolist())
    most_similar_index = np.argmax(similarities)
    return df.iloc[most_similar_index]['Текст ответа']

# Функция для поиска похожих вопросов в файле links
def find_relevant_links(user_question, threshold=0.3):
    links_questions = df_links['Текст_вопроса'].tolist()
    links_vectorizer = TfidfVectorizer()
    links_tfidf_matrix = links_vectorizer.fit_transform(links_questions)
    user_vector = links_vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, links_tfidf_matrix).flatten()
    relevant_indices = [i for i, sim in enumerate(similarity) if sim > threshold]
    return df_links.iloc[relevant_indices]

# Словарь с разговорными фразами и ответами
conversational_responses = {
    "привет": "Привет! Чем могу помочь?",
    "как дела": "У меня все отлично, спасибо! А у вас?",
    "что ты умеешь": "Я могу отвечать на ваши вопросы на основе предоставленных данных.",
    "как тебя зовут": "Меня зовут ИнфоКомпас. Я здесь, чтобы помочь вам найти информацию."
}

@app.route('/download_pdf')
def download_pdf():
    return send_file(Config.PDF_PATH, as_attachment=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_question = request.json.get('question')
        if 'last_link_question' in session and session['last_link_question'] == user_question:
            return jsonify({'answer': f"Вы нажали на ссылку с вопросом: {user_question}", 'feedback': False})
        if len(user_question.strip()) < 2:
            return jsonify({'answer': "Пожалуйста, задайте более конкретный вопрос.", 'feedback': False})
        
        user_question_lower = user_question.lower()
        
        # Проверка на разговорные фразы
        if user_question_lower in conversational_responses:
            return jsonify({'answer': conversational_responses[user_question_lower], 'feedback': False})
        
        query_embedding = get_embedding(user_question_lower)
        
        # Поиск ответа в Excel
        answer = find_most_similar_question(query_embedding, df_answers)
        
        # Поиск похожих вопросов в файле links
        links = find_relevant_links(user_question_lower)
        formatted_links = [{'question': row['Текст_вопроса'], 'url': row['Ссылка']} for _, row in links.iterrows() if row['Ссылка']]
        
        return jsonify({'answer': answer, 'feedback': True, 'links': formatted_links})
    except Exception as e:
        logging.error(f"Error in /chat: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/load_suggestions', methods=['GET'])
def load_suggestions():
    suggestions = [q.strip().rstrip('?') for question in df_answers['Текст_вопроса'].tolist() for q in question.split('?') if q.strip()]
    suggestions = [s[0].upper() + s[1:] if s else s for s in suggestions]
    return jsonify({'suggestions': suggestions})

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        question = data.get('question')
        answer = data.get('answer')
        feedback = data.get('feedback')  # 'like' or 'dislike'

        if not question or not answer or feedback not in ['like', 'dislike']:
            return jsonify({'error': 'Invalid feedback data'}), 400

        # Save feedback to the database or file
        save_feedback(question, answer, feedback)

        return jsonify({'message': 'Feedback received'}), 200
    except Exception as e:
        logging.error(f"Error in /feedback: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

def save_feedback(question, answer, feedback):
    if not os.path.exists(Config.FEEDBACK_FILE):
        with open(Config.FEEDBACK_FILE, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['question', 'answer', 'likes', 'dislikes'])

    feedback_data = load_feedback()
    found = False
    for row in feedback_data:
        if row[0] == question and row[1] == answer:
            if feedback == 'like':
                row[2] = int(row[2]) + 1
            elif feedback == 'dislike':
                row[3] = int(row[3]) + 1
            found = True
            break

    if not found:
        feedback_data.append([question, answer, 1 if feedback == 'like' else 0, 1 if feedback == 'dislike' else 0])

    with open(Config.FEEDBACK_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(feedback_data)

def load_feedback():
    feedback_data = []
    if os.path.exists(Config.FEEDBACK_FILE):
        with open(Config.FEEDBACK_FILE, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                feedback_data.append(row)
    return feedback_data

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)