from fastapi import Request
from fastapi.routing import APIRoute
from starlette.datastructures import Headers
from starlette.responses import Response
from typing import Any, Callable, Coroutine


def always_json_parsing_adjustment(request: Request):
    # Changes in FastAPI 0.65.3 only interpret the body of a request as JSON if
    # either no content-type header or a json-related header is provided. The client
    # seems to provide an `application/x-www-form-encoded` header although the byte
    # stream is JSON (erroneous). This worked in FastAPI 0.63.0 since content-type was
    # never checked but to be able to upgrade the version while not breaking compatibility
    # for old apps, we explicitly eliminate content-type header to force JSON parsing
    new_header = [
        (key, value)
        for key, value in request.headers.raw
        if key.decode("latin-1") != "content-type"
    ]
    request._headers = Headers(raw=new_header)


class WOMBOAPIRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request):
            # Any custom logic that is specific to WOMBO's API routes
            # Manage the content-type header for backward compatibility
            always_json_parsing_adjustment(request)
            return await original_route_handler(request)

        return custom_route_handler
