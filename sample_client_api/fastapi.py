from datetime import datetime

from fastapi import FastAPI
from starlette.middleware import Middleware
from sample_client_api.log_handling import get_logger_for_file
from sample_client_api import config
from sample_client_apiapi.nvidia_dispatcher import nvidia_dispatcher
from sample_client_api.bootup.nvidia_objects import IMMUTABLE_BOOTUP_MANAGER
from sample_client_api.log_secret_configs import log_environment_configs
from sample_client_api.middleware import LoggingMiddleware

logger = get_logger_for_file(__name__)


def add_routes_to_app(app: FastAPI) -> FastAPI:
    # Add all necessary routers with the correct prefix
    app.include_router(nvidia_dispatcher, prefix="/api/nvidia_dispatch")

    # After all the variable configs would be good to see all routes
    url_list = [{"path": route.path, "name": route.name} for route in app.routes]
    logger.info(f"Initialized routes : {url_list}")
    return app


def get_basic_fastapi_app() -> FastAPI:
    # Decide if the current environment is production or testing
    opts = {
        "docs_url": config.API_DOCS_ENDPOINT,
        "redoc_url": None,
    }
    logger.info("Initializing objects before app boots up ...")
    IMMUTABLE_BOOTUP_MANAGER.perform_bootup()
    logger.info("Startup objects initialized ...")
    fast_app = FastAPI(**opts)
    logger.info("FastAPI is up and running ...")
    log_environment_configs()
    fast_app.add_event_handler(event_type="shutdown",
                               func=IMMUTABLE_BOOTUP_MANAGER.perform_shutdown)
    return fast_app


app = get_basic_fastapi_app()
# Should happen after the basic bootup and before middleware management
add_routes_to_app(app)

"""
https://github.com/SigNoz/signoz/issues/1692: The add_middleware adds the middleware at the beginning (or top) of list,
leading to a situation where logging middleware gets processed before the OpenTelemetry middleware. Since the trace is
not started yet, you see the empty context. I added this workaround to push the logging middleware to the bottom, so
it gets processed later when there is trace context
"""
app.user_middleware.append(Middleware(LoggingMiddleware))
app.middleware_stack = app.build_middleware_stack()
logger.info("Initialized Middlewares...")


@app.get("/")
async def health_check():
    return {
        "now": datetime.utcnow(),
        "service": "nvidia-picasso",
    }
