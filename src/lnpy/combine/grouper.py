"""Routines for working with indexed/grouped data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from lnpy.core.validate import is_dataframe, is_xarray
from lnpy.core.xr_utils import select_axis_dim

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence
    from typing import Any

    import xarray as xr

    from lnpy.core.typing import (
        AxisReduce,
        DimsReduce,
        FactoryIndexedGrouperTypes,
        Groups,
        IndexAny,
        NDArrayAny,
        NDArrayInt,
    )
    from lnpy.core.typing_compat import Self


# * Factor functions ----------------------------------------------------------
def factor_by(
    by: Groups,
    sort: bool = True,
) -> tuple[list[Any] | IndexAny | pd.MultiIndex, NDArrayInt]:
    """
    Factor by to codes and groups.

    Parameters
    ----------
    by : sequence
        Values to group by. Negative or ``None`` values indicate to skip this
        value. Note that if ``by`` is a pandas :class:`pandas.Index` object,
        missing values should be marked with ``None`` only.
    sort : bool, default=True
        If ``True`` (default), sort ``groups``.
        If ``False``, return groups in order of first appearance.

    Returns
    -------
    groups : list or :class:`pandas.Index`
        Unique group names (excluding negative or ``None`` Values.)
    codes : ndarray of int
        Indexer into ``groups``.


    Examples
    --------
    >>> by = [1, 1, 0, -1, 0, 2, 2]
    >>> groups, codes = factor_by(by, sort=False)
    >>> groups
    [1, 0, 2]
    >>> codes
    array([ 0,  0,  1, -1,  1,  2,  2])

    Note that with sort=False, groups are in order of first appearance.

    >>> groups, codes = factor_by(by)
    >>> groups
    [0, 1, 2]
    >>> codes
    array([ 1,  1,  0, -1,  0,  2,  2])

    This also works for sequences of non-intengers.

    >>> by = ["a", "a", None, "c", "c", -1]
    >>> groups, codes = factor_by(by)
    >>> groups
    ['a', 'c']
    >>> codes
    array([ 0,  0, -1,  1,  1, -1])

    And for :class:`pandas.Index` objects

    >>> import pandas as pd
    >>> by = pd.Index(["a", "a", None, "c", "c", None])
    >>> groups, codes = factor_by(by)
    >>> groups
    Index(['a', 'c'], dtype='str')
    >>> codes
    array([ 0,  0, -1,  1,  1, -1])

    """
    from pandas import factorize

    # filter None and negative -> None
    by_: Groups = (
        by
        if isinstance(by, pd.Index)
        else np.fromiter(
            (None if isinstance(x, (int, np.integer)) and x < 0 else x for x in by),
            dtype=object,
        )
    )

    codes, groups = factorize(by_, sort=sort)  # type: ignore[arg-type]

    codes = codes.astype(np.int64)
    if isinstance(by_, (pd.Index, pd.MultiIndex)):
        if not isinstance(groups, (pd.Index, pd.MultiIndex)):  # type: ignore[unreachable] # pragma: no cover
            msg = f"{type(groups)=} should be instance of pd.Index"
            raise TypeError(msg)
        groups.names = by_.names  # type: ignore[unreachable]
        return groups, codes

    return list(groups), codes


def factor_by_to_index(
    by: Groups,
    sort: bool = True,
    **kwargs: Any,
) -> tuple[list[Any] | IndexAny | pd.MultiIndex, NDArrayInt, NDArrayInt, NDArrayInt]:
    """
    Transform group_idx to quantities to be used with :func:`reduce_data_indexed`.

    Parameters
    ----------
    by: array-like
        Values to factor.
    **kwargs
        Extra arguments to :func:`numpy.argsort`

    Returns
    -------
    groups : list or :class:`pandas.Index`
        Unique groups in `group_idx` (excluding Negative or ``None`` values in
        ``group_idx`` if ``exclude_negative`` is ``True``).
    index : ndarray
        Indexing array. ``index[start[k]:end[k]]`` are the index with group
        ``groups[k]``.
    start : ndarray
        See ``index``
    end : ndarray
        See ``index``.

    See Also
    --------
    reduce_data_indexed
    factor_by

    Examples
    --------
    >>> factor_by_to_index([0, 1, 0, 1])
    ([0, 1], array([0, 2, 1, 3]), array([0, 2]), array([2, 4]))

    >>> factor_by_to_index(["a", "b", "a", "b"])
    (['a', 'b'], array([0, 2, 1, 3]), array([0, 2]), array([2, 4]))

    Also, missing values (None or negative) are excluded:

    >>> factor_by_to_index([None, "a", None, "b"])
    (['a', 'b'], array([1, 3]), array([0, 1]), array([1, 2]))

    You can also pass :class:`pandas.Index` objects:

    >>> factor_by_to_index(pd.Index([None, "a", None, "b"], name="my_index"))
    (Index(['a', 'b'], dtype='str', name='my_index'), array([1, 3]), array([0, 1]), array([1, 2]))

    """
    # factorize by to groups and codes
    groups, codes = factor_by(by, sort=sort)

    # exclude missing
    keep = codes >= 0
    if not np.all(keep):
        index = np.where(keep)[0]
        codes = codes[keep]
    else:
        index = None

    if sort:
        indexes_sorted = np.argsort(codes, **kwargs)
        group_idx_sorted = codes[indexes_sorted]
    else:
        indexes_sorted = np.arange(len(codes))
        group_idx_sorted = codes

    _groups, n_start, count = np.unique(
        group_idx_sorted, return_index=True, return_counts=True
    )
    n_end = n_start + count

    if index is not None:
        indexes_sorted = index[indexes_sorted]

    return groups, indexes_sorted, n_start, n_end


# * Indexer class -------------------------------------------------------------
class IndexedGrouper:
    """
    Indexed grouping wrapper.

    This wraps the properties `group`, `index`, `start`, and `end`, and associated
    constructors.
    """

    def __init__(
        self,
        index: Sequence[int],
        start: Sequence[int],
        end: Sequence[int],
        groups: Any = None,
    ) -> None:
        self.index: NDArrayInt = np.asarray(index, dtype=np.int64)
        self.start: NDArrayInt = np.asarray(start, dtype=np.int64)
        self.end: NDArrayInt = np.asarray(end, dtype=np.int64)
        self.groups = groups

    @classmethod
    def from_group(cls, group: Groups, sort: bool = True, **kwargs: Any) -> Self:
        """Create object from single `group` array"""
        kwargs.setdefault("kind", "stable")
        groups, index, start, end = factor_by_to_index(group, sort=sort, **kwargs)
        return cls(index=index, start=start, end=end, groups=groups)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    @classmethod
    def from_groups(
        cls,
        *groups: Groups,
        names: Hashable | Sequence[Hashable] | None = None,
        sort: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create object from multiple `group` arrays"""
        idx = (
            pd.Index(groups[0], name=names)
            if len(groups) == 1
            else pd.MultiIndex.from_arrays(groups, names=names)  # type: ignore[arg-type, unused-ignore]  # pyright: ignore[reportArgumentType]
        )

        return cls.from_group(idx, sort=sort, **kwargs)

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame | xr.DataArray | xr.Dataset,
        keys: str | Iterable[str],
        sort: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create object from data object and group variables/columns"""
        if isinstance(keys, str):
            keys = [keys]
        return cls.from_groups(*(data[k] for k in keys), sort=sort, **kwargs)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    @classmethod
    def from_size(
        cls,
        data: NDArrayAny | pd.Series[Any] | pd.DataFrame | xr.DataArray | xr.Dataset,
        dim: DimsReduce | None = None,
        axis: AxisReduce = -1,
    ) -> Self:
        """Create a grouper for whole object"""
        if is_xarray(data):
            axis, dim = select_axis_dim(data, axis, dim)
            size = data.sizes[dim]  # type: ignore[index]
        elif is_dataframe(data):
            size = len(data)
        else:
            size = data.shape[axis]

        return cls(
            index=range(size),
            start=[0],
            end=[size],
        )


def factory_indexed_grouper(
    grouper: FactoryIndexedGrouperTypes | None = None,
    *,
    # From data and keys
    data: NDArrayAny | pd.Series[Any] | pd.DataFrame | xr.DataArray | xr.Dataset,
    keys: str | Iterable[str] | None = None,
    dim: DimsReduce | None = None,
    axis: AxisReduce = -1,
    # From groups
    group: Groups | None = None,
    groups: Iterable[Groups] | None = None,
    # From index/start/end
    index: Sequence[int] | None = None,
    start: Sequence[int] | None = None,
    end: Sequence[int] | None = None,
    **kwargs: Any,
) -> IndexedGrouper:
    """
    Factory for ``IndexedGrouper`` object.

    The order or evaluations is:

    #. ``index``, ``start``, ``end``
    #. ``group``
    #. ``groups``
    #. ``keys``
    #.  Default ``from_size`` grouper.

    Parameters
    ----------
    grouper : int or str or iterable of str or mapping or IndexedGrouper
        Parameter meanings are as follows

        #. str or iterable of str : ``factory_indexed_grouper(data=data, keys=grouper, **kwargs)``
        #. mapping : return ``factory_indexed_grouper(**grouper, data=data, **kwargs)

    data : mapping
        Mapping object (DataFrame, DataArray, or Dataset).
    keys : str or iterable of str
        Variables/columns from ``data`` to group by.
    dim : str, optional
        Dimension to sample.
    axis : int, default=-1
        Axis to sample.
    group : array-like
        Group by these values.
    groups : sequence of array-like
        Group by these arrays.
    index, start, end : array-like
        Parameters to ``IndexedGrouper``.
    **kwargs
        Extra arguments to constructors

    """
    if grouper is None:
        if start is not None and end is not None:
            if index is None:
                index = range(end[-1])
            return IndexedGrouper(index, start, end)

        if group is not None:
            return IndexedGrouper.from_group(group, **kwargs)

        if groups is not None:
            return IndexedGrouper.from_groups(*groups, **kwargs)

        if (is_xarray(data) or is_dataframe(data)) and keys is not None:
            return IndexedGrouper.from_data(data=data, keys=keys, **kwargs)

        return IndexedGrouper.from_size(data=data, axis=axis, dim=dim)

    # passing grouper
    if isinstance(grouper, IndexedGrouper):
        return grouper

    if isinstance(grouper, Mapping):
        return factory_indexed_grouper(
            data=data, axis=axis, dim=dim, **grouper, **kwargs
        )
    return factory_indexed_grouper(
        data=data, axis=axis, dim=dim, keys=grouper, **kwargs
    )
