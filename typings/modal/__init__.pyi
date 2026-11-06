# Type stubs for the slice of Modal used by lecture_search.search_api
# (deploy-only dep, absent from the dev venv; see tach.toml [external]).

from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T", bound=type)
_F = TypeVar("_F", bound=Callable[..., object])

class Image:
    @classmethod
    def debian_slim(cls, python_version: str = ...) -> Image: ...
    def pip_install(self, *packages: str) -> Image: ...
    def add_local_python_source(self, *modules: str) -> Image: ...

class Secret:
    @classmethod
    def from_name(cls, name: str) -> Secret: ...

class App:
    def __init__(self, name: str = ...) -> None: ...
    def cls(
        self,
        *,
        image: Image,
        secrets: list[Secret],
        cpu: float,
        memory: int,
        max_containers: int,
        scaledown_window: int,
    ) -> Callable[[_T], _T]: ...

def concurrent(*, max_inputs: int) -> Callable[[_T], _T]: ...
def enter() -> Callable[[_F], _F]: ...
def asgi_app() -> Callable[[_F], _F]: ...
