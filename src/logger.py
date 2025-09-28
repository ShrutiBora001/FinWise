
"""
Logger utility for consistent logging across modules.
"""
import logging


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # Prevent duplicate handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
