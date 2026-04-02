"""Routines interacting with joblib"""

from __future__ import annotations

from importlib.util import find_spec
from itertools import starmap
from typing import TYPE_CHECKING

from lnpy.options import OPTIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import Any, Literal

    from .typing_compat import TypeVar

    R = TypeVar("R")


HAS_JOBLIB = find_spec("joblib")


def _use_joblib(
    items: Sequence[Any],
    len_key: Literal["joblib_len_calc", "joblib_len_build"],
    use_joblib: bool = True,
    total: int | None = None,
) -> bool:
    if use_joblib and HAS_JOBLIB and OPTIONS["joblib_use"]:
        if total is None:
            total = len(items)
        return total >= OPTIONS[len_key]
    return False


def _parallel(seq: Iterable[Any]) -> list[Any]:
    import joblib

    return joblib.Parallel(  # type: ignore[no-any-return]  # pyright: ignore[reportReturnType]
        n_jobs=OPTIONS["joblib_n_jobs"],
        backend=OPTIONS["joblib_backend"],
        **OPTIONS["joblib_kws"],
    )(seq)


def parallel_map_build(
    func: Callable[..., R], items: Iterable[Any], *args: Any, **kwargs: Any
) -> list[R]:

    items = tuple(items)
    if _use_joblib(items, "joblib_len_build"):
        import joblib

        return _parallel(joblib.delayed(func)(x, *args, **kwargs) for x in items)
    return [func(x, *args, **kwargs) for x in items]


def _func_call(x: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    return x(*args, **kwargs)


def parallel_map_call(
    items: Sequence[Callable[..., Any]],
    use_joblib: bool,  # noqa: ARG001
    *args: Any,
    **kwargs: Any,
) -> list[Any]:
    if _use_joblib(items, "joblib_len_calc"):
        import joblib

        return _parallel(joblib.delayed(_func_call)(x, *args, **kwargs) for x in items)
    return [x(*args, **kwargs) for x in items]


def parallel_map_attr(attr: str, use_joblib: bool, items: Sequence[Any]) -> list[Any]:  # noqa: ARG001
    from operator import attrgetter

    func = attrgetter(attr)

    if _use_joblib(items, "joblib_len_calc"):
        import joblib

        return _parallel(joblib.delayed(func)(x) for x in items)
    return [func(x) for x in items]


def parallel_map_func_starargs(
    func: Callable[..., R],
    use_joblib: bool,  # noqa: ARG001
    items: Iterable[Any],
    total: int | None = None,
) -> list[R]:
    items = tuple(items)

    if _use_joblib(items, "joblib_len_calc", total=total):
        import joblib

        return _parallel(starmap(joblib.delayed(func), items))
    return list(starmap(func, items))
