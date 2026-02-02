import logging

def setup_logger(log_path):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
