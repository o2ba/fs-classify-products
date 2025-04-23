import logging
import sys


def get_logger(name: str = "ml_project") -> logging.Logger:
    """
    Initializes and returns a simple logger that outputs to the console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is initialized more than once
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger