"""
Options (:mod:`~lnpy.options`)
==============================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType
    from typing import Any

    ValidatorFunc = Callable[[Any], bool]


class Options(TypedDict, total=False):
    """Options."""

    tqdm_use: bool
    tqdm_len_calc: int
    tqdm_len_build: int
    tqdm_leave: bool
    tqdm_bar: str

    joblib_use: bool
    joblib_n_jobs: int
    joblib_backend: str | None
    joblib_kws: dict[str, Any]
    joblib_len_calc: int
    joblib_len_build: int


class OptionsReq(TypedDict, total=True):
    """Options with required parameters."""

    tqdm_use: bool
    tqdm_len_calc: int
    tqdm_len_build: int
    tqdm_leave: bool
    tqdm_bar: str

    joblib_use: bool
    joblib_n_jobs: int
    joblib_backend: str | None
    joblib_kws: dict[str, Any]
    joblib_len_calc: int
    joblib_len_build: int


class Validators(TypedDict):
    """Validators."""

    tqdm_use: ValidatorFunc
    tqdm_len_calc: ValidatorFunc
    tqdm_len_build: ValidatorFunc
    tqdm_leave: ValidatorFunc
    tqdm_bar: ValidatorFunc

    joblib_use: ValidatorFunc
    joblib_n_jobs: ValidatorFunc
    joblib_backend: ValidatorFunc
    joblib_kws: ValidatorFunc
    joblib_len_calc: ValidatorFunc
    joblib_len_build: ValidatorFunc


OPTIONS: OptionsReq = {
    "tqdm_use": True,
    "tqdm_len_calc": 100,
    "tqdm_len_build": 500,
    "tqdm_leave": False,
    "tqdm_bar": "default",
    "joblib_use": True,
    "joblib_n_jobs": -1,
    "joblib_backend": None,
    "joblib_kws": {},
    "joblib_len_calc": 200,
    "joblib_len_build": 500,
}


def _isbool(x: Any) -> bool:
    return isinstance(x, bool)


def _isint(x: Any) -> bool:
    return isinstance(x, int)


def _isstr(x: Any) -> bool:
    return isinstance(x, str)


def _isdict(x: Any) -> bool:
    return isinstance(x, dict)


def _isstr_or_none(x: Any) -> bool:
    return x is None or _isstr(x)


_VALIDATORS: Validators = {
    "tqdm_use": _isbool,
    "tqdm_len_calc": _isint,
    "tqdm_len_build": _isint,
    "tqdm_leave": _isbool,
    "tqdm_bar": lambda x: x in {"default", "text", "notebook"},
    "joblib_use": _isbool,
    "joblib_n_jobs": _isint,
    "joblib_backend": _isstr_or_none,
    "joblib_kws": _isdict,
    "joblib_len_calc": _isint,
    "joblib_len_build": _isint,
}

_SETTERS: dict[str, Any] = {}


def _apply_update(options_dict: Options) -> None:
    for k, v in options_dict.items():
        if k in _SETTERS:
            _SETTERS[k](v)
    OPTIONS.update(options_dict)


class set_options:  # noqa: N801
    """
    Set options for xarray in a controlled context.
    Currently supported options:

    * `tqdm_use` : if `True`, use progress bar where appropriate
    * `tqdm_len_calc` : min length for using bar in calculations of properties
    * `tqdm_len_build` : min length for building Collection objects
    * `tqdm_leave` : if True, leave bar after execution.  Default=False
    * `joblib_use` : if `True`, use joblib
    * `joblib_n_jobs` : number of processors to use, default=all processors
    * `joblib_backend` : backend to use.  Default='multiprocessing'.
    * `joblib_kws` : dictionary of arguments to joblib.Parallel.
    * `joblib_len_calc` : min length to use parallel in calculations
    * `joblib_len_build` : min length to use parallel in build

    Examples
    --------
    You can use ``set_options`` either as a context manager:

    >>> import lnpy
    >>> with lnpy.set_options(use_tqdm=True, tqdm_min_len_calc=50):  # doctest: +SKIP
    ...     c.xge.betaOmega()


    Or to set global options:

    >>> _ = lnpy.set_options(tqdm_len_calc=50)
    """

    def __init__(self, **kwargs: Any) -> None:
        self.old: Options = {}
        for k, v in cast("Options", kwargs).items():
            if k not in OPTIONS:
                msg = f"argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}"
                raise ValueError(msg)
            if k in _VALIDATORS and not _VALIDATORS[k](v):  # type: ignore[literal-required]  # ty: ignore[invalid-key]
                msg = f"option {k!r} given an invalid value: {v!r}"
                raise ValueError(msg)
            self.old[k] = OPTIONS[k]  # type: ignore[literal-required]  # ty: ignore[invalid-key]
        _apply_update(cast("Options", kwargs))

    def __enter__(self) -> None:
        return

    def __exit__(
        self,
        type: type[BaseException] | None,  # noqa: A002
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _apply_update(self.old)
