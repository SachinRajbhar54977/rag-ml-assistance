# src/rag_assistant/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from a .env file (if it exists) into the process environment.
# This runs once, when this module is first imported.
load_dotenv()

# Base directory of the project — three levels up from this file
# (config.py -> rag_assistant -> src -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
print(f"PROJECT_ROOT = {PROJECT_ROOT}")

# RAW_DATA_DIR: where PDFs live.
# os.getenv(key, default) returns the env var if set, otherwise the default.
# We fall back to a path relative to the project root so the project works
# out of the box even without a .env file.
RAW_DATA_DIR = (PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")).resolve()
PROCESSED_DATA_DIR = (PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")).resolve()

print(f"RAW_DATA_DIR = {RAW_DATA_DIR}")

# LOG_LEVEL: default to INFO if not set.
# .upper() guards against someone writing "debug" instead of "DEBUG" in .env.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
print(f"LOG_LEVEL = {LOG_LEVEL}")

# Validate LOG_LEVEL against the values Python's logging module actually understands.
# Without this, a typo like "INOF" in .env would silently break logger.py later
# (logging.getLevelName("INOF") doesn't raise — it just returns a nonsense value).
_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
if LOG_LEVEL not in _VALID_LEVELS:
    raise ValueError(
        f"Invalid LOG_LEVEL '{LOG_LEVEL}' in environment. "
        f"Must be one of {_VALID_LEVELS}."
    )