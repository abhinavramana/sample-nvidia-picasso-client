import json
import time
from typing import Any, Dict, Optional
from fastapi import Request
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse
from sample_client_api.log_handling import get_logger_for_file

logger = get_logger_for_file(__name__)


def get_stats_json(
    start_time: float, request: Request, response_status_code: Optional[int] = None
) -> Dict[str, Any]:
    run_time = time.time() - start_time
    try:
        payload = request.state._state.get("payload")
        if payload is not None:
            payload = payload.decode("utf-8")
    except Exception as ex:
        payload = f"could-not-decode-payload: {str(ex)}"
    # Place the necessary stats into the logs as a JSON object, avoid the base route on the app
    stats_json = {
        "path": f"{request.method}-{request.url.path}",
        "payload": payload,
        "run_time": run_time,
        "headers": request.headers.items(),
        "response_status_code": response_status_code,
    }
    return stats_json


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Record the actual processing time for the request
        start_time = time.time()
        try:
            response = await call_next(request)
            response_status_code = response.status_code
            stats_json = get_stats_json(start_time, request, response_status_code)
            if request.url.path != "/":
                # JSON dump the data here before logging so that de-serialization is more well-defined
                # if the logs need to be parsed for insights.
                logger.info(json.dumps(stats_json))
        except ClientDisconnect:
            # We don't want to handle this and logs get cluttered
            stats_json = get_stats_json(
                start_time, request, status.HTTP_418_IM_A_TEAPOT
            )
            stats_json["error_reason"] = "ClientDisconnect"
            logger.info(json.dumps(stats_json))
            response = JSONResponse(stats_json, status.HTTP_418_IM_A_TEAPOT)
        except Exception as ex:
            # Since this was an unexpected exception
            stats_json = get_stats_json(
                start_time, request, status.HTTP_500_INTERNAL_SERVER_ERROR
            )
            json_response_body = {"detail": repr(ex)}
            logger.info(json.dumps(stats_json))
            logger.error(json.dumps(json_response_body), exc_info=True)
            response = JSONResponse(
                json_response_body, status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        return response
