"""
Typing definitions for :mod:`lnpy`
==================================
"""
# ruff: noqa: UP007
# pylint: disable=consider-alternative-union-syntax

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
)

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from .typing_compat import Concatenate, ParamSpec, TypeAlias, TypeVar

if TYPE_CHECKING:
    # Note: use fully qualified names
    import lnpy.lnpidata
    import lnpy.lnpiseries
    from lnpy.combine.grouper import IndexedGrouper
    from lnpy.ensembles import CanonicalEnsemble, GrandCanonicalEnsemble  # noqa: F401


__all__ = [
    "C_Ensemble",
    "EnsembleT",
    "FuncT",
    "IndexingInt",
    "MaskConvention",
    "NDArrayAny",
    "P",
    "PeakError",
    "PeakStyle",
    "PhasesFactorySignature",
    "R",
    "T",
    "TagPhasesSignature",
    "XArrayLike",
]

# * NDArray
NDArrayAny: TypeAlias = NDArray[Any]
"""Alias for simple :class:`numpy.typing.NDArray[Any]`"""
NDArrayInt = NDArray[np.int64]
NDArrayBool: TypeAlias = NDArray[np.bool_]


# * TypeVars
EnsembleT = TypeVar("EnsembleT", "GrandCanonicalEnsemble", "CanonicalEnsemble")
"""TypeVar for Ensemble."""
FloatT = TypeVar("FloatT", np.float32, np.float64, default=Any)  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
GenArrayT = TypeVar("GenArrayT", NDArray[Any], xr.DataArray)
GenArrayOrSeriesT = TypeVar(
    "GenArrayOrSeriesT", NDArray[Any], xr.DataArray, "pd.Series[Any]"
)
SeriesOrDataArrayT = TypeVar("SeriesOrDataArrayT", "pd.Series[Any]", xr.DataArray)
FrameOrDatasetT = TypeVar("FrameOrDatasetT", pd.DataFrame, xr.Dataset)
DataT = TypeVar("DataT", xr.DataArray, xr.Dataset)
FrameOrDataT = TypeVar("FrameOrDataT", pd.DataFrame, xr.DataArray, xr.Dataset)
DataAnyT = TypeVar(
    "DataAnyT",
    NDArray[Any],
    "pd.Series[Any]",
    pd.DataFrame,
    xr.DataArray,
    xr.Dataset,
)

# * Decorating
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
C_Ensemble: TypeAlias = Callable[Concatenate[EnsembleT, P], R]

FuncT = TypeVar("FuncT", bound=Callable[..., Any])
NumbaType = Any


# * Callables
TagPhasesSignature = Callable[
    [Sequence["lnpy.lnpidata.lnPiMasked"]], "Sequence[int] | NDArrayAny"
]
"""Signature for tag_phases function."""

PhasesFactorySignature = Callable[..., "lnpy.lnpiseries.lnPiCollection"]
"""Signature for phases_factory function."""


# * Literals
PeakStyle = Literal["indices", "mask", "marker"]
PeakError = Literal["ignore", "raise", "warn"]

MaskConvention: TypeAlias = 'Literal["image", "masked"] | bool'
"""Convention for boolean masks."""

Casting = Literal["no", "equiv", "safe", "same_kind", "unsafe"]
MissingCoreDimOptions = Literal["raise", "copy", "drop"]


# * Aliases
IndexingInt: TypeAlias = Union[
    int,
    "np.int_",
    "np.integer[Any]",
    "np.unsignedinteger[Any]",
    "np.signedinteger[Any]",
    "np.int8",
]

XArrayLike: TypeAlias = Union[ArrayLike, xr.DataArray]
IndexIterScalar: TypeAlias = Union[str, bytes, bool, int, float]
Scalar: TypeAlias = IndexIterScalar

AxisReduce: TypeAlias = int
DimsReduce: TypeAlias = Union[Hashable, Collection[Hashable]]

ApplyUFuncKwargs: TypeAlias = Mapping[str, Any]
KeepAttrs: TypeAlias = Union[
    Literal["drop", "identical", "no_conflicts", "drop_conflicts", "override"],
    bool,
    None,
]

IndexAny: TypeAlias = "pd.Index[Any]"

# * IndexedGrouper
Groups: TypeAlias = Union[Sequence[Any], NDArrayAny, IndexAny, pd.MultiIndex]
FactoryIndexedGrouperTypes: TypeAlias = Union[
    str,
    Iterable[str],
    Mapping[str, Any],
    "IndexedGrouper",
]
