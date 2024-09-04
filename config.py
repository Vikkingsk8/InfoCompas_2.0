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