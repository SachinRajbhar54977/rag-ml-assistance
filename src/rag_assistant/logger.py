# src/rag_assistant/logger.py
import logging
from rag_assistant.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given name.
    Safe to call multiple times with the same name — will not
    attach duplicate handlers.
    """
    logger = logging.getLogger(name)

    # Guard against duplicate handlers. If this logger was already
    # configured (e.g. get_logger("rag_assistant.ingestion") called
    # twice, or the module re-imported), skip re-adding a handler —
    # otherwise every log line prints once per attached handler.
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    handler = logging.StreamHandler()  # prints to console (stderr by default)
    handler.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent this logger's messages from also being handled by the
    # root logger (which can otherwise cause duplicate output if
    # something else configures logging.basicConfig()).
    logger.propagate = False

    return logger