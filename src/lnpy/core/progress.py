"""Routines to work with progress bar."""

# pyright: reportMissingImports=false
from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

from lnpy.options import OPTIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any, Literal

    from .typing_compat import TypeVar

    T = TypeVar("T")


# * TQDM setup ----------------------------------------------------------------
HAS_TQDM = find_spec("tqdm")


@lru_cache
def _get_tqdm_default() -> Callable[..., Any]:
    if HAS_TQDM:
        import tqdm as tqdm_

        try:
            from IPython.core.getipython import (  # ty: ignore[unresolved-import,unused-ignore-comment]  # pyrefly: ignore[missing-import]
                get_ipython,
            )

            p = get_ipython()
            if p is not None and p.has_trait("kernel"):
                from tqdm.notebook import tqdm as tqdm_default

                return tqdm_default
            return cast("Callable[..., Any]", tqdm_.tqdm)
        except ImportError:
            return cast("Callable[..., Any]", tqdm_.tqdm)
    else:

        def wrapper(seq: Iterable[T], *args: Any, **kwargs: Any) -> Iterable[T]:  # noqa: ARG001
            return seq

        return wrapper


def tqdm(seq: Iterable[T], *args: Any, **kwargs: Any) -> Iterable[T]:
    opt = OPTIONS["tqdm_bar"]
    if HAS_TQDM:
        import tqdm as tqdm_

        func: Any
        if opt == "text":
            func = tqdm_.tqdm
        elif opt == "notebook":
            func = tqdm_.tqdm_notebook
        else:
            func = _get_tqdm_default()
    else:
        func = _get_tqdm_default()
    return cast("Iterable[T]", func(seq, *args, **kwargs))


def get_tqdm(
    seq: Iterable[T],
    len_min: int | Literal["tqdm_len_calc", "tqdm_len_build"],
    leave: bool | None = None,
    **kwargs: Any,
) -> Iterable[T]:
    n = kwargs.get("total")
    if isinstance(len_min, str):
        len_min = OPTIONS[len_min]

    if n is None:
        seq = tuple(seq)
        n = len(seq)

    if HAS_TQDM and OPTIONS["tqdm_use"] and n >= len_min:
        if leave is None:
            leave = OPTIONS["tqdm_leave"]
        seq = tqdm(seq, leave=leave, **kwargs)
    return seq


def get_tqdm_calc(
    seq: Iterable[T], leave: bool | None = None, **kwargs: Any
) -> Iterable[T]:
    return get_tqdm(seq, len_min="tqdm_len_calc", leave=leave, **kwargs)


def get_tqdm_build(
    seq: Iterable[T], leave: bool | None = None, **kwargs: Any
) -> Iterable[T]:
    return get_tqdm(seq, len_min="tqdm_len_build", leave=leave, **kwargs)
