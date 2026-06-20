"""Validations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, cast

import numpy as np
import pandas as pd
import xarray as xr

from .typing import NDArrayAny
from .typing_compat import TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import UnionType
    from typing import TypeGuard

    from .typing_compat import TypeIs

T = TypeVar("T")


class Validator(Generic[T]):
    """Generic validator"""

    def __init__(
        self, type_: type[T] | UnionType, type_name: str | None = None
    ) -> None:
        self.type_ = type_

        if type_name is None:
            type_name = getattr(type_, "__name__", str(self.type_))
        self.type_name = type_name

    def typeis(self, val: object) -> TypeIs[T]:
        return isinstance(val, self.type_)

    def typeguard(self, val: object) -> TypeGuard[T]:
        return isinstance(val, self.type_)

    def validate(self, val: object) -> T:
        if self.typeis(val):
            return val

        msg = f"Type {type(val)} != {self.type_name}"
        raise TypeError(msg)

    __call__ = validate


class _Validators:
    ndarray = Validator[NDArrayAny](np.ndarray)
    dataarray = Validator(xr.DataArray)
    dataset = Validator(xr.Dataset)
    xarray = Validator[xr.DataArray | xr.Dataset](xr.DataArray | xr.Dataset)
    series = Validator(pd.Series)
    dataframe = Validator(pd.DataFrame)
    maskedarray = Validator[np.ma.MaskedArray[Any, np.dtype[Any]]](np.ma.MaskedArray)

    @staticmethod
    def as_str_or_iterable(x: str | Iterable[str]) -> list[str]:
        """Convert str or iterable of string to list of str"""
        if isinstance(x, str):
            return [x]
        return list(x)

    @staticmethod
    def as_sequence(iterable: Iterable[T]) -> Sequence[T]:
        if isinstance(iterable, Sequence):
            return cast("Sequence[T]", iterable)
        return list(iterable)

    @staticmethod
    def as_list(iterable: Iterable[T]) -> list[T]:
        if not isinstance(iterable, list):
            return list(iterable)
        return iterable


validate = _Validators()
