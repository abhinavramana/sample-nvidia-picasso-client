import logging


LOGGING_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARN,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "WARNING": logging.WARNING,
}
LOG_LEVEL_TO_SET = LOGGING_LEVEL_MAP["INFO"]


def get_logger_for_file(name):
    file_logger = logging.getLogger(name)
    file_logger.setLevel(LOG_LEVEL_TO_SET)
    return file_logger