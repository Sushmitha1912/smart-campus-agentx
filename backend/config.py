import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
MEMORY_DB_PATH = BASE_DIR / "campus_memory.db"
TFIDF_CACHE_PATH = BASE_DIR / "kb_tfidf_cache.pkl"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    print("[WARNING] GROQ_API_KEY not set. Create a .env file with your Groq API key.")