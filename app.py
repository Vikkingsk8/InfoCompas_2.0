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

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')  # Используйте переменные окружения для секретных ключей

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Конфигурация
class Config:
    CACHE_FILE = os.getenv('CACHE_FILE', 'embeddings_cache.npy')
    EXCEL_PATH = os.getenv('EXCEL_PATH', 'ответы.xlsx')  # Use environment variable
    LINKS_PATH = os.getenv('LINKS_PATH', 'links.xlsx')  # Use environment variable
    PDF_PATH = os.getenv('PDF_PATH', 'instruction.pdf')  # Use environment variable

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
    suggestions = [q.strip() + '?' for question in df_answers['Текст_вопроса'].tolist() for q in question.split('?') if q.strip()]
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)