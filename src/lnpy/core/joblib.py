"""Routines interacting with joblib"""

from __future__ import annotations

from importlib.util import find_spec
from itertools import starmap
from typing import TYPE_CHECKING, cast

from lnpy.options import OPTIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from typing import Any, Literal

    from .typing_compat import TypeVar

    R = TypeVar("R")


HAS_JOBLIB: bool = find_spec("joblib") is not None


def _use_joblib(
    len_key: Literal["joblib_len_calc", "joblib_len_build"],
    *,
    use_joblib: bool = True,
    total: int,
) -> bool:
    return (
        use_joblib
        and HAS_JOBLIB
        and OPTIONS["joblib_use"]
        and total >= OPTIONS[len_key]
    )


def _parallel(seq: Iterable[Any]) -> Iterator[Any]:
    import joblib

    return cast(
        "Iterator[Any]",
        joblib.Parallel(
            return_as="generator",
            n_jobs=OPTIONS["joblib_n_jobs"],
            backend=OPTIONS["joblib_backend"],
            **OPTIONS["joblib_kws"],
        )(seq),
    )


def parallel_map_build(
    func: Callable[..., R],
    items: Iterable[Any],
    *args: Any,
    total: int,
    **kwargs: Any,
) -> Iterator[R]:

    if _use_joblib("joblib_len_build", total=total):
        import joblib

        return _parallel(joblib.delayed(func)(x, *args, **kwargs) for x in items)
    return (func(x, *args, **kwargs) for x in items)


def _func_call(x: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    return x(*args, **kwargs)


def parallel_map_call(
    items: Sequence[Callable[..., Any]],
    *args: Any,
    total: int,
    **kwargs: Any,
) -> Iterator[Any]:
    if _use_joblib("joblib_len_calc", total=total):
        import joblib

        return _parallel(joblib.delayed(_func_call)(x, *args, **kwargs) for x in items)
    return (x(*args, **kwargs) for x in items)


def parallel_map_attr(
    attr: str,
    items: Sequence[Any],
    total: int,
) -> Iterator[Any]:
    from operator import attrgetter

    func = attrgetter(attr)

    if _use_joblib("joblib_len_calc", total=total):
        import joblib

        return _parallel(joblib.delayed(func)(x) for x in items)
    return (func(x) for x in items)


def parallel_map_func_starargs(
    func: Callable[..., R],
    items: Iterable[Any],
    total: int,
) -> Iterator[R]:
    items = tuple(items)

    if _use_joblib("joblib_len_calc", total=total):
        import joblib

        return _parallel(starmap(joblib.delayed(func), items))
    return starmap(func, items)
