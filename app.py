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
from fuzzywuzzy import process

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')

logging.basicConfig(level=logging.INFO)

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CACHE_DIR = os.path.join(BASE_DIR, 'cache')
    CACHE_FILE = os.getenv('CACHE_FILE', os.path.join(CACHE_DIR, 'embeddings_cache.npy'))
    EXCEL_PATH = os.getenv('EXCEL_PATH', os.path.join(DATA_DIR, 'ответы.xlsx'))
    LINKS_PATH = os.getenv('LINKS_PATH', os.path.join(DATA_DIR, 'links.xlsx'))
    PDF_PATH = os.getenv('PDF_PATH', os.path.join(DATA_DIR, 'instruction.pdf'))
    FEEDBACK_FILE = os.path.join(DATA_DIR, 'feedback.xlsx')
    KEYWORDS_FILE = os.path.join(DATA_DIR, 'key_words.xlsx')

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
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
    df['Текст_вопроса'] = df['Текст_вопроса'].str.lower().str.strip()
    return df

df_answers = load_excel_data(Config.EXCEL_PATH)
df_answers = preprocess_excel_data(df_answers, 'Текст вопроса')

df_links = load_excel_data(Config.LINKS_PATH)
df_links = preprocess_excel_data(df_links, 'Вопрос')

# Загрузка ключевых слов
df_keywords = pd.read_excel(Config.KEYWORDS_FILE)
keywords = df_keywords['Ключевые слова'].tolist()

start_time = time.time()
df_answers['embedding'] = df_answers['Текст_вопроса'].apply(lambda x: embeddings_cache.get(x) if x in embeddings_cache else get_embedding(x))
end_time = time.time()
logging.info(f"Время получения эмбеддингов: {end_time - start_time} секунд")

# Сохранение кэша эмбеддингов
for idx, row in df_answers.iterrows():
    embeddings_cache[row['Текст_вопроса']] = row['embedding']
np.save(Config.CACHE_FILE, embeddings_cache)

def check_keywords(query, keywords, threshold=80):
    query_words = query.lower().split()
    for word in query_words:
        matches = process.extractBests(word, keywords, score_cutoff=threshold)
        if matches:
            return True
    return False

def find_most_similar_question(query_embedding, query, df, embedding_threshold=0.7, levenshtein_threshold=80):
    if not check_keywords(query, keywords):
        return None
    
    similarities = cosine_similarity([query_embedding], df['embedding'].tolist())
    max_similarity = np.max(similarities)
    
    if max_similarity >= embedding_threshold:
        most_similar_index = np.argmax(similarities)
        return df.iloc[most_similar_index]['Текст ответа']
    else:
        # Если не найдено достаточно похожих вопросов по эмбеддингам, используем расстояние Левенштейна
        questions = df['Текст_вопроса'].tolist()
        closest_match, score = process.extractOne(query, questions)
        if score >= levenshtein_threshold:
            closest_match_index = questions.index(closest_match)
            return df.iloc[closest_match_index]['Текст ответа']
        else:
            return None

def find_relevant_links(user_question, threshold=0.3):
    links_questions = df_links['Текст_вопроса'].tolist()
    links_vectorizer = TfidfVectorizer()
    links_tfidf_matrix = links_vectorizer.fit_transform(links_questions)
    user_vector = links_vectorizer.transform([user_question])
    similarity = cosine_similarity(user_vector, links_tfidf_matrix).flatten()
    relevant_indices = [i for i, sim in enumerate(similarity) if sim > threshold]
    return df_links.iloc[relevant_indices]

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
        
        if user_question_lower in conversational_responses:
            return jsonify({'answer': conversational_responses[user_question_lower], 'feedback': False})
        
        query_embedding = get_embedding(user_question_lower)
        
        answer = find_most_similar_question(query_embedding, user_question_lower, df_answers)
        
        if answer is None:
            return jsonify({'answer': "Извините, я не могу найти подходящий ответ на ваш вопрос. Пожалуйста, перефразируйте его или задайте другой вопрос.", 'feedback': False})
        
        links = find_relevant_links(user_question_lower)
        formatted_links = [{'question': row['Текст_вопроса'], 'url': row['Ссылка']} for _, row in links.iterrows() if row['Ссылка']]
        
        return jsonify({'answer': answer, 'feedback': True, 'links': formatted_links})
    except Exception as e:
        logging.error(f"Ошибка в /chat: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

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
            return jsonify({'error': 'Неверные данные обратной связи'}), 400

        save_feedback(question, answer, feedback)

        return jsonify({'message': 'Обратная связь получена'}), 200
    except Exception as e:
        logging.error(f"Ошибка в /feedback: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)