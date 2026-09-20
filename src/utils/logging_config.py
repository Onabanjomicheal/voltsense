"""Standard logging setup: console + rotating file handler, consistent format
across training and serving. Import get_logger(__name__) instead of using print()."""
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

_configured = False


def _configure_once():
    global _configured
    if _configured:
        return
    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "battery_soc.log"), maxBytes=5_000_000, backupCount=3
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_once()
    return logging.getLogger(name)
