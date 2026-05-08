from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Root of the repo (two levels up from this file: src/doubleml_rag/config.py)
_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = _ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


class Settings:
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    voyage_api_key: str = os.environ.get("VOYAGE_API_KEY", "")

    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR


settings = Settings()
