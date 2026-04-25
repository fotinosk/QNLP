import datetime
import logging
import os


def setup_logger(log_path="./runs/logs", log_name="train_logger", ts_string: str | None = None):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    if not ts_string:
        ts_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_log_path = os.path.join(log_path, f"{log_name}_{ts_string}.log")

    os.makedirs(os.path.dirname(full_log_path), exist_ok=True)

    # File handler
    fh = logging.FileHandler(full_log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_log_file_path(logger):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None
