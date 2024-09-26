from flask import Flask, request, jsonify, render_template, session, send_file
from flask_caching import Cache
import tempfile
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import logging
import time
import re
from collections import defaultdict, Counter
import datetime
import requests
from dashboard import create_dash_app
from config import Config
import io
import sqlite3

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')
dash_app = create_dash_app(app, routes_pathname_prefix='/dashboard/')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Настройка кэширования
cache = Cache(app)

# Глобальные переменные для отслеживания метрик
total_queries = 0
successful_queries = 0
query_history = []
feedback_history = []

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def load_embeddings_cache():
    if os.path.exists(Config.CACHE_FILE):
        try:
            return np.load(Config.CACHE_FILE, allow_pickle=True).item()
        except Exception as e:
            logging.error(f"Ошибка загрузки кэша эмбеддингов: {e}")
    return {}

embeddings_cache = load_embeddings_cache()

def load_excel_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_excel(path)
    return df

def preprocess_excel_data(df, column_name):
    df = df.assign(Текст_вопроса=df[column_name].str.split('?')).explode('Текст_вопроса')
    df['Текст_вопроса'] = df['Текст_вопроса'].str.lower().str.strip().replace('ё', 'е').astype(str)
    return df

def preprocess_user_question(question):
    return question.lower().strip().replace('ё', 'е')

def get_embedding(text):
    if not isinstance(text, str):
        raise ValueError(f"text input must be of type `str`, got {type(text)}")
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

df_answers = load_excel_data(Config.EXCEL_PATH)
df_answers = preprocess_excel_data(df_answers, 'Текст вопроса')

df_links = load_excel_data(Config.LINKS_PATH)
df_links = preprocess_excel_data(df_links, 'Вопрос')

df_fork = load_excel_data(Config.CROSSROAD_FILE)
df_fork = preprocess_excel_data(df_fork, 'Вопрос')

start_time = time.time()
df_answers['embedding'] = df_answers['Текст_вопроса'].apply(
    lambda x: embeddings_cache.get(x) if x in embeddings_cache else get_embedding(x) if isinstance(x, str) else None
)
end_time = time.time()
logging.info(f"Время получения эмбеддингов: {end_time - start_time} секунд")

if not os.path.exists(Config.CACHE_DIR):
    os.makedirs(Config.CACHE_DIR)

for idx, row in df_answers.iterrows():
    embeddings_cache[row['Текст_вопроса']] = row['embedding']
np.save(Config.CACHE_FILE, embeddings_cache)

# Подготовка данных для BM25
tokenized_corpus = [doc.split() for doc in df_answers['Текст_вопроса']]
bm25 = BM25Okapi(tokenized_corpus)

# Список вопросительных слов и фраз для игнорирования
question_words = set(['что', 'такое', 'это', 'почему', 
                      'расскажи', 'как', 'что значит', 'кто', 'где', 'когда', 
                      'сколько', 'какой', 'какая', 'какие', 'чем', 'зачем', 
                      'дел ам', 'дел', 'где найти', 'найти', 'как найти', 
                      'как это сделать', 'что делать'])

# Создаем индекс при загрузке данных
question_index = defaultdict(set)

def preprocess_text(text):
    # Замена "ё" на "е"
    text = text.replace('ё', 'е').replace('Ё', 'Е')
    # Удаление знаков препинания и приведение к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Токенизация текста
    words = text.split()
    # Удаление вопросительных слов и стоп-слов
    filtered_words = [word for word in words if word not in question_words]
    return set(filtered_words)  # Возвращаем множество слов

# Предобработка и индексация вопросов из базы данных
for idx, row in df_answers.iterrows():
    if isinstance(row['Текст вопроса'], str):
        questions = row['Текст вопроса'].split('?')
    else:
        print(f"Предупреждение: 'Текст вопроса' не является строкой для индекса {idx}: {row['Текст вопроса']}")
        questions = []

    for question in questions:
        preprocessed_question = preprocess_text(question)
        for word in preprocessed_question:
            question_index[word].add(idx)

def check_question_validity(user_question):
    preprocessed_user_question = preprocess_text(user_question)
    
    # Проверяем, остались ли какие-либо слова после предобработки
    if len(preprocessed_user_question) == 0:
        return False, None
    
    # Проверяем, есть ли хотя бы одно слово из вопроса пользователя в нашей базе данных
    for word in preprocessed_user_question:
        if word in question_index:
            return True, user_question  # Возвращаем оригинальный вопрос пользователя
    
    return False, None

def find_best_answer(query, embedding_weight=0.7, levenshtein_weight=0.15, bm25_weight=0.15):
    if not isinstance(query, str):
        logging.warning(f"Получен неверный тип данных для query: {type(query)}. Преобразование в строку.")
        query = str(query)
    
    query_embedding = get_embedding(query)
    
    # Остальной код функции...
    
    # Расчет сходства эмбеддингов
    embedding_similarities = cosine_similarity([query_embedding], df_answers['embedding'].tolist())[0]
    
    # Расчет расстояния Левенштейна
    levenshtein_similarities = np.array([fuzz.ratio(query, q) / 100 for q in df_answers['Текст_вопроса']])
    
    # Расчет BM25
    bm25_scores = np.array(bm25.get_scores(query.split()))
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)  # Нормализация
    
    # Комбинирование оценок с весами
    combined_scores = (
        embedding_weight * embedding_similarities +
        levenshtein_weight * levenshtein_similarities +
        bm25_weight * bm25_scores
    )
    
    best_match_index = np.argmax(combined_scores)
    best_match_score = combined_scores[best_match_index]
    
    # Возвращаем лучший ответ, если оценка выше порога
    if best_match_score > 0.5:  # Порог можно настроить
        return df_answers.iloc[best_match_index]['Текст ответа'], df_answers.iloc[best_match_index]['Текст_вопроса']
    else:
        return None, None
    

def find_relevant_links(user_question, threshold=0.3):
    user_question = preprocess_user_question(user_question)
    links_questions = df_links['Текст_вопроса'].tolist()
    links_vectorizer = TfidfVectorizer()
    links_tfidf_matrix = links_vectorizer.fit_transform(links_questions)
    user_vector = links_vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, links_tfidf_matrix).flatten()
    relevant_indices = [i for i, sim in enumerate(similarity) if sim > threshold]
    return df_links.iloc[relevant_indices]

def find_relevant_fork_links(user_question, threshold=0.3):
    user_question = preprocess_user_question(user_question)
    fork_questions = df_fork['Вопрос'].tolist()
    fork_vectorizer = TfidfVectorizer()
    fork_tfidf_matrix = fork_vectorizer.fit_transform(fork_questions)
    user_vector = fork_vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, fork_tfidf_matrix).flatten()
    relevant_indices = [i for i, sim in enumerate(similarity) if sim > threshold]
    return df_fork.iloc[relevant_indices]

conversational_responses = {
    "привет": "Привет! Чем могу помочь?",
    "как дела": "У меня все отлично, спасибо! А у вас?",
    "что ты умеешь": "Я могу отвечать на ваши вопросы на основе предоставленных данных.",
    "как тебя зовут": "Меня зовут ИнфоКомпас. Я здесь, чтобы помочь вам найти информацию."
}

def load_initial_questions():
    try:
        df = pd.read_excel(Config.CROSSROAD_FILE)
        questions = df['Вопрос'].tolist()[:6]
        return questions
    except Exception as e:
        logging.error(f"Ошибка при загрузке начальных вопросов: {e}")
        return []

def init_db():
    if not os.path.exists(Config.DATA_DIR):
        os.makedirs(Config.DATA_DIR)

    if not os.path.exists(Config.SQLITE_DB_PATH):
        conn = sqlite3.connect(Config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP,
                question TEXT,
                success BOOLEAN
            )
        ''')
        cursor.execute('''
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP,
                question TEXT,
                answer TEXT,
                feedback_type TEXT,
                comment TEXT
            )
        ''')
        conn.commit()
        conn.close()

init_db()


@app.route('/download_pdf')
def download_pdf():
    cached_pdf_path = cache.get('cached_pdf_path')

    if cached_pdf_path is None or not os.path.exists(cached_pdf_path):
        logger.info("Кэшированный файл не найден. Создаем новый временный файл.")
        # Если кэшированный файл не существует, создаем новый временный файл в CACHE_DIR
        cache_file_name = 'cached_instruction.pdf'
        cached_pdf_path = os.path.join(Config.CACHE_DIR, cache_file_name)

        # Проверяем, существует ли директория CACHE_DIR, и создаем её, если нет
        if not os.path.exists(Config.CACHE_DIR):
            os.makedirs(Config.CACHE_DIR)

        # Копируем содержимое оригинального PDF в кэшированный файл
        with open(Config.PDF_PATH, 'rb') as original_file, open(cached_pdf_path, 'wb') as cached_file:
            original_content = original_file.read()
            cached_file.write(original_content)
            logger.info(f"Содержимое оригинального файла прочитано и записано в кэшированный файл. Размер: {len(original_content)} байт")

        cache.set('cached_pdf_path', str(cached_pdf_path), timeout=3600)  # Кэшируем строку с путем к файлу
        logger.info(f"Файл PDF успешно кэширован. Путь: {cached_pdf_path}")
    else:
        logger.info(f"Используем кэшированный файл. Путь: {cached_pdf_path}")

    if not os.path.exists(cached_pdf_path):
        logger.error("Кэшированный файл не найден.")
        return "File not found", 404

    logger.info(f"Отправляем кэшированный файл. Путь: {cached_pdf_path}")
    return send_file(cached_pdf_path, as_attachment=False)

@app.route('/load_suggestions', methods=['GET'])
def load_suggestions():
    suggestions = [q.strip().rstrip('?') for question in df_answers['Текст_вопроса'].tolist() for q in question.split('?') if q.strip()]
    suggestions = [s[0].upper() + s[1:] if s else s for s in suggestions]
    return jsonify({'suggestions': suggestions})

@app.route('/log', methods=['POST'])
def log():
    data = request.json
    message = data.get('message')
    logging.debug(f"Client log: {message}")
    return jsonify({'status': 'ok'}), 200

@app.route('/')
def index():
    initial_questions = load_initial_questions()
    logging.debug("Rendering index.html with initial questions")
    return render_template('index.html', initial_questions=initial_questions)

@app.route('/chat', methods=['POST'])
def chat():
    global total_queries, successful_queries, query_history
    try:
        user_question = request.json.get('question')
        user_question = preprocess_user_question(user_question)
        if 'last_link_question' in session and session['last_link_question'] == user_question:
            return jsonify({'answer': f"Вы нажали на ссылку с вопросом: {user_question}", 'feedback': False})
        if len(user_question.strip()) < 2:
            return jsonify({'answer': "Пожалуйста, задайте более конкретный вопрос.", 'feedback': False})
        
        user_question_lower = user_question.lower()
        
        if user_question_lower in conversational_responses:
            return jsonify({'answer': conversational_responses[user_question_lower], 'feedback': False})
        
        # Проверка валидности вопроса
        is_valid, validated_question = check_question_validity(user_question)
        
        logging.info(f"Вопрос пользователя: {user_question}")
        logging.info(f"Валидность вопроса: {is_valid}")
        
        total_queries += 1
        query_time = datetime.datetime.now()
        
        conn = sqlite3.connect(Config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        if not is_valid:
            query_history.append({'time': query_time, 'question': user_question, 'success': False})
            cursor.execute('''
                INSERT INTO queries (time, question, success)
                VALUES (?, ?, ?)
            ''', (query_time, user_question, False))
            conn.commit()
            conn.close()
            return jsonify({'answer': "Извините, я не совсем понял ваш вопрос. Пожалуйста, перефразируйте его или задайте более конкретный вопрос.", 'feedback': False})
        
        # Если вопрос валидный, ищем ответ на оригинальный вопрос пользователя
        answer, matched_question = find_best_answer(validated_question)
        
        logging.info(f"Найденный ответ: {answer}")
        logging.info(f"Совпавший вопрос: {matched_question}")
        
        if answer is None:
            query_history.append({'time': query_time, 'question': user_question, 'success': False})
            cursor.execute('''
                INSERT INTO queries (time, question, success)
                VALUES (?, ?, ?)
            ''', (query_time, user_question, False))
            conn.commit()
            conn.close()
            return jsonify({'answer': "Извините, я не могу найти подходящий ответ на ваш вопрос. Пожалуйста, перефразируйте его или задайте другой вопрос.", 'feedback': False})
        
        successful_queries += 1
        query_history.append({'time': query_time, 'question': user_question, 'success': True})
        cursor.execute('''
            INSERT INTO queries (time, question, success)
            VALUES (?, ?, ?)
        ''', (query_time, user_question, True))
        conn.commit()
        conn.close()
        
        links = find_relevant_links(user_question_lower)
        fork_links = find_relevant_fork_links(user_question_lower)
        
        formatted_links = [{'question': row['Текст_вопроса'], 'url': row['Ссылка'], 'type': 'link'} for _, row in links.iterrows() if row['Ссылка']]
        
        # Изменение здесь: используем множество для удаления дубликатов
        unique_fork_questions = set(row['Вопрос'] for _, row in fork_links.iterrows())
        formatted_fork_links = [{'question': q, 'url': None, 'type': 'fork'} for q in unique_fork_questions]
        
        all_links = formatted_fork_links + formatted_links
        
        return jsonify({'answer': answer, 'feedback': True, 'links': all_links, 'matched_question': matched_question})
    except Exception as e:
        logging.error(f"Ошибка в /chat: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    global feedback_history
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    feedback_type = data.get('type')
    comment = data.get('comment', '')

    if not all([question, answer, feedback_type]):
        return jsonify({'error': 'Не все необходимые данные предоставлены'}), 400

    try:
        conn = sqlite3.connect(Config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (time, question, answer, feedback_type, comment)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.datetime.now(), question, answer, feedback_type, comment))
        conn.commit()
        conn.close()
        
        feedback_history.append({
            'time': datetime.datetime.now(),
            'question': question,
            'answer': answer,
            'feedback_type': feedback_type,
            'comment': comment
        })
        
        return jsonify({'message': 'Спасибо за ваш отзыв!'}), 200
    except Exception as e:
        logging.error(f"Ошибка при сохранении обратной связи: {e}")
        return jsonify({'error': 'Ошибка при сохранении обратной связи'}), 500

def save_feedback(question, answer, feedback):
    if not os.path.exists(Config.FEEDBACK_FILE):
        df = pd.DataFrame(columns=['question', 'answer', 'likes', 'dislikes'])
        df.to_excel(Config.FEEDBACK_FILE, index=False)

    df = pd.read_excel(Config.FEEDBACK_FILE)
    row = df[(df['question'] == question) & (df['answer'] == answer)]

    if not row.empty:
        row_index = row.index[0]
        if feedback == 'like':
            df.at[row_index, 'likes'] += 1
        elif feedback == 'dislike':
            df.at[row_index, 'dislikes'] += 1
    else:
        new_row = pd.DataFrame({'question': [question], 'answer': [answer], 'likes': [1 if feedback == 'like' else 0], 'dislikes': [1 if feedback == 'dislike' else 0]})
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_excel(Config.FEEDBACK_FILE, index=False)

@app.route('/like', methods=['POST'])
def like():
    try:
        data = request.json
        question = data.get('question')
        answer = data.get('answer')

        if not question or not answer:
            return jsonify({'error': 'Неверные данные обратной связи'}), 400

        save_feedback(question, answer, 'like')
        return jsonify({'message': 'Лайк получен'}), 200
    except Exception as e:
        logging.error(f"Ошибка в /like: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/dislike', methods=['POST'])
def dislike():
    try:
        data = request.json
        question = data.get('question')
        answer = data.get('answer')

        if not question or not answer:
            return jsonify({'error': 'Неверные данные обратной связи'}), 400

        save_feedback(question, answer, 'dislike')
        return jsonify({'message': 'Дизлайк получен'}), 200
    except Exception as e:
        logging.error(f"Ошибка в /dislike: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/dashboard/')
def render_dashboard():
    return dash_app.index()

@app.route('/analytics_data', methods=['GET'])
def get_analytics_data():
    try:
        conn = sqlite3.connect(Config.SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM queries')
        total_queries = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM queries WHERE success = 1')
        successful_queries = cursor.fetchone()[0]
        
        success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
        
        cursor.execute('SELECT question, COUNT(*) as count FROM queries GROUP BY question ORDER BY count DESC LIMIT 10')
        top_questions = cursor.fetchall()
        
        cursor.execute('SELECT feedback_type, COUNT(*) as count FROM feedback GROUP BY feedback_type')
        feedback_distribution = dict(cursor.fetchall())
        
        cursor.execute('SELECT time, success FROM queries ORDER BY time')
        query_history = cursor.fetchall()
        
        queries_over_time = {}
        queries_by_day = {}
        queries_by_week = {}
        queries_by_month = {}
        
        for time, success in query_history:
            time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            time_str = time.strftime('%Y-%m-%d %H:%M:%S')
            day_str = time.strftime('%Y-%m-%d')
            week_str = time.strftime('%Y-%W')
            month_str = time.strftime('%Y-%m')
            
            if time_str not in queries_over_time:
                queries_over_time[time_str] = 0
            queries_over_time[time_str] += 1
            
            if day_str not in queries_by_day:
                queries_by_day[day_str] = 0
            queries_by_day[day_str] += 1
            
            if week_str not in queries_by_week:
                queries_by_week[week_str] = 0
            queries_by_week[week_str] += 1
            
            if month_str not in queries_by_month:
                queries_by_month[month_str] = 0
            queries_by_month[month_str] += 1
        
        conn.close()
        
        return jsonify({
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': success_rate,
            'top_questions': top_questions,
            'feedback_distribution': feedback_distribution,
            'queries_over_time': queries_over_time,
            'queries_by_day': queries_by_day,
            'queries_by_week': queries_by_week,
            'queries_by_month': queries_by_month
        })
    except Exception as e:
        app.logger.error(f"Ошибка в /analytics_data: {str(e)}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)