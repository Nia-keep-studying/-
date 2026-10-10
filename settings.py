"""Shared project paths and configuration; no model or database is loaded here."""
import os
from pathlib import Path


from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


DATABASE_PATH = project_path(os.getenv("DATABASE_PATH", "data/orders_sqlmodel.db"))
MODEL_CACHE_DIR = project_path(
    os.getenv("MODEL_CACHE_DIR", os.getenv("HF_HOME", ".model_cache"))
)
# Set the cache before importing either model library.
os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)

ORDER_API_BASE_URL = os.getenv("ORDER_API_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

# Existing experimental settings, not universally optimal thresholds.
VECTOR_TOP_K = 5
MIN_SCORE = 0.4
RERANK_TOP_K = 3
LOGIN_SESSION_TTL_SECONDS = 60 * 60
