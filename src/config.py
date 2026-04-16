# -*- coding: utf-8 -*-
"""
Centralized configuration — loads secrets from .env securely.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_mp_api_key() -> str:
    """Return Materials Project API key or raise with helpful message."""
    key = os.getenv("MP_API_KEY", "").strip()
    if not key or key == "PASTE_YOUR_KEY_HERE":
        raise ValueError(
            "Materials Project API key not configured.\n"
            "1. Get a free key at https://next-gen.materialsproject.org/api\n"
            "2. Paste it in .env:  MP_API_KEY=your_key_here"
        )
    return key
