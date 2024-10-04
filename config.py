# Авторы: Ермилов В.В., Файбисович В.А.
import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CACHE_DIR = os.path.join(BASE_DIR, 'cache')
    CACHE_FILE = os.getenv('CACHE_FILE', os.path.join(CACHE_DIR, 'embeddings_cache.npy'))
    EXCEL_PATH = os.getenv('EXCEL_PATH', os.path.join(DATA_DIR, 'ответы.xlsx'))
    LINKS_PATH = os.getenv('LINKS_PATH', os.path.join(DATA_DIR, 'links.xlsx'))
    PDF_PATH = os.getenv('PDF_PATH', os.path.join(DATA_DIR, 'instruction.pdf'))
    FEEDBACK_FILE = os.path.join(DATA_DIR, 'feedback.xlsx')
    CROSSROAD_FILE = os.path.join(DATA_DIR, 'развилка.xlsx')

    CACHE_TYPE = 'filesystem'
    CACHE_DIR = CACHE_DIR  # Директория для хранения кэша
    CACHE_DEFAULT_TIMEOUT = 3600  # Время жизни кэша в секундах (1 час)

    # Путь к базе данных SQLite
    SQLITE_DB_PATH = os.path.join(DATA_DIR, 'dashboard_data.db')