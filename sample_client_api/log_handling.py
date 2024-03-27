import logging
import os


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
    """
    This is required because we cannot use logging.basicConfig as it messes with opentelemetry. But somehow
    default log level is warn, so we manually have to set this everywhere to be INFO
    """
    file_logger = logging.getLogger(name)
    file_logger.setLevel(LOG_LEVEL_TO_SET)
    return file_logger