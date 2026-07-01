"""Centralized logger factory for the entire pipeline.

Usage:
    from finetune_sherpa_vi.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Hello")
"""

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Track whether root logger has been configured already
_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once. Call this from main.py."""
    global _configured
    if _configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "requests", "youtube_dl", "yt_dlp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger. Lazily sets up basic config if not done yet."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
