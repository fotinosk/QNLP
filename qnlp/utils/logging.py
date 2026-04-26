import datetime
import logging
import sys
from pathlib import Path


def setup_logger(
    log_path: str | Path = "./runs/logs",
    log_name: str = "train_logger",
    ts_string: str | None = None,
    console: bool = True,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs if root logger is configured

    if logger.hasHandlers():
        logger.handlers.clear()

    if not ts_string:
        ts_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    full_log_path = log_dir / f"{log_name}_{ts_string}.log"

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(full_log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def get_log_file_path(logger: logging.Logger) -> str | None:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None
