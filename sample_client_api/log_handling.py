import logging
import os
from opentelemetry.trace import Status, StatusCode, get_current_span


class LogErrorHandler(logging.Handler):
    def emit(self, record):
        span = get_current_span()
        got_exception = False
        if span is not None:
            if record.exc_info is not None:
                exc_type, exc_value, tb = record.exc_info
                if exc_value is not None:
                    span.record_exception(exc_value)
                    got_exception = True

            if record.levelno >= logging.ERROR or got_exception:
                span.set_status(Status(StatusCode.ERROR, record.getMessage()))


LOG_ERROR_HANDLER = LogErrorHandler()

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
    if (
        os.getenv("OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED", "false").lower()
        == "true"
    ):
        file_logger.addHandler(LOG_ERROR_HANDLER)
    file_logger.setLevel(LOG_LEVEL_TO_SET)
    return file_logger