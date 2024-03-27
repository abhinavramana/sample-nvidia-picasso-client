from typing import Any, Callable
from fastapi import APIRouter as FastAPIRouter
from fastapi.types import DecoratedCallable

from sample_client_api.custom_api_route import WOMBOAPIRoute


class WOMBOAPIRouter(FastAPIRouter):
    def __init__(self, **kwargs):
        if "route_class" not in kwargs:
            kwargs["route_class"] = WOMBOAPIRoute
        super(WOMBOAPIRouter, self).__init__(**kwargs)

    def api_route(
        self, path: str, *, include_in_schema: bool = True, **kwargs: Any
    ) -> Callable[[DecoratedCallable], DecoratedCallable]:
        no_slash_path = path[:-1] if path.endswith("/") else path
        add_no_slash_path = super().api_route(
            no_slash_path, include_in_schema=False, **kwargs
        )
        slash_path = f"{no_slash_path}/"
        add_slash_path = super().api_route(
            slash_path, include_in_schema=include_in_schema, **kwargs
        )

        def decorator(func: DecoratedCallable) -> DecoratedCallable:
            add_no_slash_path(func)
            return add_slash_path(func)

        return decorator
