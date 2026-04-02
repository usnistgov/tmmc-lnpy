"""Array utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

from .validate import is_dataset, is_ndarray

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike

    from .typing import NDArrayAny


_ALLOWED_FLOAT_DTYPES = {np.dtype(np.float32), np.dtype(np.float64)}


@overload
def select_dtype(
    x: xr.Dataset,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike | None,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None: ...
@overload
def select_dtype(
    x: xr.DataArray | ArrayLike,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike | None,
) -> np.dtype[np.float32] | np.dtype[np.float64]: ...
@overload
def select_dtype(
    x: xr.Dataset | xr.DataArray | ArrayLike,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike | None,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None: ...


def select_dtype(
    x: ArrayLike | xr.DataArray | xr.Dataset,
    *,
    out: NDArrayAny | xr.DataArray | None,
    dtype: DTypeLike | None,
) -> np.dtype[np.float32] | np.dtype[np.float64] | None:  # DTypeLikeArg[Any]:
    """
    Select a dtype from, in order, out, dtype, or passed array.

    If pass in a Dataset, return dtype
    """
    if is_dataset(x):
        if dtype is None:
            return dtype
        dtype = np.dtype(dtype)
    elif out is not None:
        dtype = out.dtype
    elif dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = getattr(x, "dtype", np.dtype(np.float64))

    if dtype in _ALLOWED_FLOAT_DTYPES:
        return dtype  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    msg = f"{dtype=} not supported.  dtype must be conformable to float32 or float64."
    raise ValueError(msg)


def asarray_maybe_recast(
    data: ArrayLike, dtype: DTypeLike | None = None, recast: bool = False
) -> NDArrayAny:
    """Perform asarray with optional recast to `dtype` if not already an array."""
    if is_ndarray(data):
        if recast and dtype is not None:
            return np.asarray(data, dtype=dtype)
        return data
    return np.asarray(data, dtype=dtype)


# * Filling -------------------------------------------------------------------
def ffill(arr: NDArrayAny, axis: int = -1, limit: int | None = None) -> NDArrayAny:
    import bottleneck

    limit_ = limit if limit is not None else arr.shape[axis]
    return bottleneck.push(arr, n=limit_, axis=axis)  # type: ignore[no-any-return]


def bfill(arr: NDArrayAny, axis: int = -1, limit: int | None = None) -> NDArrayAny:
    """Inverse of ffill"""
    import bottleneck

    # work around for bottleneck 178
    limit_ = limit if limit is not None else arr.shape[axis]

    arr = np.flip(arr, axis=axis)
    # fill
    arr = bottleneck.push(arr, axis=axis, n=limit_)
    # reverse back to original
    return np.flip(arr, axis=axis)


# * Helpers
def array_to_scalar(x: float | NDArrayAny) -> float:
    """
    Convert array to scalar.

    If `x` is an ndarray, convert to float of 0-d,
    or extract first element (of flattened array )
    if N-d.
    """
    if isinstance(x, np.ndarray):
        return x.flat[0]  # type: ignore[no-any-return]
    return x
