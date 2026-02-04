import logging
import datetime
import os

def setup_logger(log_path="./runs/logs", log_name="train_logger"):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    ts_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    full_log_path = os.path.join(log_path, f"{log_name}_{ts_string}.log")

    # File handler
    fh = logging.FileHandler(full_log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
