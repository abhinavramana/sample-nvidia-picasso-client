import logging
import os
import nest_asyncio
from opentelemetry.instrumentation.logging import LoggingInstrumentor

"""
***************************************************************************************************
ALERT: THIS FILE SHOULD NEVER CONTAIN ANY PACKAGE IMPORTS

This should never be taken out of the basic init because this is the first thing that should
be initialized even before the imports
***************************************************************************************************

These functions do nothing if the root logger already has handlers configured, unless the keyword
argument *force* is set to `True`
"""
LoggingInstrumentor().instrument(set_logging_format=True)
# Disabling basicConfig to allow OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
if (
    os.getenv("OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED", "false").lower()
    == "false"
):
    logging.basicConfig(
        level=logging.INFO, force=True
    )  # Should happen before any logging
logging.info("Initialized the logger...")
# nest_asyncio should happen before any asyncio stuff (event loop, fastapi etc.) happens
nest_asyncio.apply()
