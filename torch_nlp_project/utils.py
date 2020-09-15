import logging


def get_logger(file_path: str):
    """
    get logger that logging to std.out and file on 'file_path'.
    :param file_path: File path to save log.
    """
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(file_handler)

    return logger
