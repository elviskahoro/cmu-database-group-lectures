# Type stubs for the slice of FastAPI used by lecture_search.search_api
# (deploy-only dep, absent from the dev venv; see tach.toml [external]).

from collections.abc import Callable
from typing import TypeVar

_F = TypeVar("_F", bound=Callable[..., object])

class FastAPI:
    def __init__(
        self,
        *,
        title: str = ...,
        docs_url: str | None = ...,
        redoc_url: str | None = ...,
        openapi_url: str | None = ...,
    ) -> None: ...
    def add_middleware(
        self,
        middleware_class: type,
        *,
        allow_origins: list[str] = ...,
        allow_methods: list[str] = ...,
        allow_headers: list[str] = ...,
    ) -> None: ...
    def post(self, path: str, *, response_model: None = ...) -> Callable[[_F], _F]: ...
    def get(self, path: str, *, response_model: None = ...) -> Callable[[_F], _F]: ...

class Request:
    async def body(self) -> bytes: ...
