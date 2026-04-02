"""
Utility functions (:mod:`~lnpy.core.utils`)
===========================================
"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

from .mask import mask_change_convention

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
    from typing import Any

    import xarray as xr
    from numpy.typing import ArrayLike, NDArray

    from lnpy.lnpidata import lnPiMasked

    from .typing import MaskConvention, NDArrayAny
    from .typing_compat import TypeVar

    T = TypeVar("T")


# * pandas stuff --------------------------------------------------------------
def allbut(levels: Iterable[Hashable], *names: Hashable) -> list[Hashable]:
    name_set = set(names)
    return [item for item in levels if item not in name_set]


# * Utilities -----------------------------------------------------------------
def get_lnz_iter(lnz: Iterable[float | None], x: ArrayLike) -> NDArrayAny:
    """
    Create a lnz_iter object for varying a single lnz

    Parameters
    ----------
    lnz : list
        list with one element equal to None.  This is the component which will be varied
        For example, lnz=[lnz0,None,lnz2] implies use values of lnz0,lnz2 for components 0 and 2, and
        vary component 1

    x : array
        values to insert for variable component

    Returns
    -------
    ndarray
        Shape ``(len(x),len(lnz))``.
        array with rows [lnz0,lnz1,lnz2]
    """
    x = np.asarray(x)
    z = np.zeros_like(x)

    return np.array([x if m is None else z + m for m in lnz]).T


def distance_matrix(
    mask: ArrayLike, convention: MaskConvention = "image"
) -> NDArray[np.float64]:
    """
    Create matrix of distances from elements of mask
    to nearest background point

    Parameters
    ----------
    mask : array-like
        image mask
    convention : str or bool, default='image'
        mask convention

    Returns
    -------
    distance : ndarray
        Same shape as mask.
        Distance from possible feature elements to background


    See Also
    --------
    ~scipy.ndimage.distance_transform_edt
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.asarray(mask, dtype=bool)
    mask = mask_change_convention(mask, convention_in=convention, convention_out=True)

    # pad mask
    # add padding to end of matrix in each dimension
    ndim = mask.ndim
    pad_width = ((0, 1),) * ndim
    mask = np.pad(mask, pad_width=pad_width, mode="constant", constant_values=False)

    # distance filter
    dist = distance_transform_edt(mask)

    # remove padding
    s = (slice(None, -1),) * ndim
    return dist[s]


def lnpimasked_to_dataset(
    data: lnPiMasked, keys: Sequence[str] = ("lnpi", "PE")
) -> xr.Dataset:
    """
    Convert a :class:`~lnpy.lnpidata.lnPiMasked` object into as :class:`~xarray.Dataset`.

    Parameters
    ----------
    data : lnPiMasked

    Returns
    -------
    output : Dataset
    """
    return data.xce.table(keys=keys, default_keys=None)


def dataset_to_lnpimasked(
    ds: xr.Dataset,
    lnpi_name: str = "lnpi",
    pe_name: str = "PE",
    extra_kws: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> lnPiMasked:
    """
    Convert a :class:`~xarray.Dataset` to a :class:`~lnpy.lnpidata.lnPiMasked` object.

    Parameters
    ----------
    ds : :class:`~xarray.Dataset`
    lnpi_name, pe_name : str
        Names of 'lnPi' and 'PE' parameters
    extra_kws : mapping, optional
        parameter `extra_kws`.  Note that if `pe_name` found in `ds`, then will add it to `extra_kws`.

    Returns
    -------
    lnpi : lnPiMasked
    """
    from lnpy.lnpidata import lnPiMasked

    data = ds[lnpi_name]
    extra_kws = {} if extra_kws is None else dict(extra_kws)
    if pe_name in ds:
        extra_kws[pe_name] = ds[pe_name].to_numpy()

    return lnPiMasked.from_dataarray(da=data, extra_kws=extra_kws, **kwargs)


# * Helpers -------------------------------------------------------------------
def peek_at(iterable: Iterable[T]) -> tuple[T, Iterator[T]]:
    """Returns the first value from iterable, as well as a new iterator with
    the same content as the original iterable
    """
    gen = iter(iterable)
    peek = next(gen)
    return peek, chain([peek], gen)
