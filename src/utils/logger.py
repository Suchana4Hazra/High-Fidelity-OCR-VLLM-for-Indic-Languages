"""Simple logger utilities used by the demo application."""
import logging
from typing import Optional


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def setup_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    # Avoid adding multiple handlers if called repeatedly
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        root.addHandler(stream)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            root.addHandler(fh)
    root.setLevel(level)
    return root
