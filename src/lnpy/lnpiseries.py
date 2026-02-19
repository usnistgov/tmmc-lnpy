"""
Collection of lnPi objects (:mod:`~lnpy.lnpiseries`)
====================================================
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, overload
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from module_utilities import cached

from .core.docstrings import docfiller
from .core.joblib import parallel_map_build as parallel_map
from .core.mask import labels_to_masks, masks_to_labels

# lazy loads
from .core.progress import get_tqdm_build as get_tqdm
from .core.utils import peek_at
from .extensions import AccessorMixin

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Hashable,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )
    from typing import (
        Any,
        Final,
        Literal,
    )

    from numpy.typing import ArrayLike, DTypeLike
    from pandas.core.groupby.generic import SeriesGroupBy

    from . import ensembles, lnpienergy, stability
    from .core.typing import IndexingInt, NDArrayAny, Scalar
    from .core.typing_compat import IndexAny, Self
    from .lnpidata import lnPiMasked


# Accessors
class _CallableResult:
    def __init__(self, parent: lnPiCollection, func: Callable[..., Any]) -> None:
        functools.update_wrapper(self, func)

        self._parent = parent
        self._func = func

    def __call__(self, *args: Any, **kwargs: Any) -> lnPiCollection:
        return self._parent.new_like(self._func(*args, **kwargs))


class _Groupby:
    def __init__(self, parent: lnPiCollection, group: SeriesGroupBy[Any, Any]) -> None:
        self._parent = parent
        self._group = group

    def __iter__(self) -> Iterator[tuple[Any, lnPiCollection]]:
        return ((meta, self._parent.new_like(x)) for meta, x in self._group)

    def __getattr__(self, attr: str) -> _CallableResult | lnPiCollection:
        if hasattr(self._group, attr):
            out = getattr(self._group, attr)
            if callable(out):
                return _CallableResult(self._parent, out)
            return self._parent.new_like(out)

        msg = f"no attribute {attr} in groupby"
        raise AttributeError(msg)


# @SeriesWrapper.decorate_accessor("loc")
class _LocIndexer:
    """
    Indexer by value.

    See :attr:`pandas.Series.loc`
    """

    def __init__(self, parent: lnPiCollection) -> None:
        self._parent = parent
        self._loc = self._parent._series.loc

    @overload
    def __getitem__(self, idx: Scalar | tuple[Scalar, ...]) -> lnPiMasked: ...

    @overload
    def __getitem__(
        self,
        idx: list[Scalar] | IndexAny | slice | Callable[[pd.Series[Any]], Any],
    ) -> lnPiCollection: ...

    @overload
    def __getitem__(self, idx: Any) -> lnPiMasked | lnPiCollection: ...

    def __getitem__(self, idx: Any) -> lnPiMasked | lnPiCollection:
        out = self._loc[idx]
        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        return out  # type: ignore[no-any-return]

    def __setitem__(
        self, idx: Any, values: lnPiMasked | pd.Series[Any] | Sequence[lnPiMasked]
    ) -> None:
        self._parent._series.loc[idx] = values


# @SeriesWrapper.decorate_accessor("iloc")
class _iLocIndexer:  # noqa: N801
    """
    Indexer by position.

    See :attr:`pandas.Series.iloc`
    """

    def __init__(self, parent: lnPiCollection) -> None:
        self._parent = parent
        self._iloc = self._parent._series.iloc

    @overload
    def __getitem__(self, idx: IndexingInt) -> lnPiMasked: ...

    @overload
    def __getitem__(self, idx: Sequence[int] | IndexAny | slice) -> lnPiCollection: ...

    @overload
    def __getitem__(self, idx: Any) -> lnPiMasked | lnPiCollection: ...

    def __getitem__(self, idx: Any) -> lnPiMasked | lnPiCollection:
        out = self._iloc[idx]
        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        return out  # type: ignore[no-any-return]

    def __setitem__(
        self, idx: Any, values: lnPiMasked | pd.Series[Any] | Sequence[lnPiMasked]
    ) -> None:
        self._parent._series.iloc[idx] = values


# @SeriesWrapper.decorate_accessor("query")
class _Query:
    """
    Select values by string query.

    See :meth:`pandas.DataFrame.query`
    """

    def __init__(self, parent: lnPiCollection) -> None:
        self._parent = parent
        self._frame: pd.DataFrame = self._parent.index.to_frame().reset_index(drop=True)

    def __call__(self, expr: str, **kwargs: Any) -> lnPiCollection:
        idx = self._frame.query(expr, **kwargs).index
        return self._parent.iloc[idx]  # type: ignore[no-any-return]


# @SeriesWrapper.decorate_accessor("zloc")
class _LocIndexer_unstack_zloc:  # noqa: N801
    """positional indexer for everything but phase"""

    def __init__(
        self,
        parent: lnPiCollection,
        level: Hashable | Sequence[Hashable] = ("phase",),
    ) -> None:
        self._parent = parent
        self._level = level
        self._loc = self._parent._series.unstack(self._level).iloc  # noqa: PD010

    def __getitem__(self, idx: Any) -> lnPiCollection:
        out = self._loc[idx]
        if isinstance(out, pd.DataFrame):
            out = out.stack(self._level, future_stack=True)  # noqa: PD013
        out = out.dropna()

        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        else:
            msg = "unknown indexer for zloc"
            raise TypeError(msg)
        return out


# @SeriesWrapper.decorate_accessor("mloc")
class _LocIndexer_unstack_mloc:  # noqa: N801
    """indexer with pandas index"""

    def __init__(
        self,
        parent: lnPiCollection,
        level: Hashable | Sequence[Hashable] = ("phase",),
    ) -> None:
        self._parent = parent
        self._level = level
        self._index = self._parent.index

        self._index_names = set(self._index.names)
        self._loc = self._parent._series.iloc

    def _get_loc_idx(self, idx: pd.MultiIndex | IndexAny) -> Any:
        index = self._index
        if isinstance(idx, pd.MultiIndex):
            # names in idx and
            drop: list[Hashable] = list(self._index_names - set(idx.names))
            index = index.droplevel(drop)
            # reorder idx
            idx = idx.reorder_levels(index.names)
        else:
            drop = list(set(index.names) - {idx.name})
            index = index.droplevel(drop)
        return index.get_indexer_for(idx)

    def __getitem__(self, idx: pd.MultiIndex | IndexAny) -> lnPiCollection:
        indexer = self._get_loc_idx(idx)
        out = self._loc[indexer]

        if isinstance(out, pd.Series):
            out = self._parent.new_like(out)
        else:
            msg = "unknown indexer for mloc"
            raise TypeError(msg)
        return out


class lnPiCollection(AccessorMixin):  # noqa: PLR0904, N801
    r"""
    Wrapper around :class:`pandas.Series` for collection of :class:`~lnpy.lnpidata.lnPiMasked` objects.


    Parameters
    ----------
    data : sequence of lnPiMasked
        :math:`\ln \Pi(N)` instances to consider.
    index : array-like, pandas.Index, pandas.MultiIndex, optional
        Index to apply to Series.
    xarray_output : bool, default=True
        If True, then wrap lnPiCollection outputs in :class:`~xarray.DataArray`
    concat_dim : str, optional
        Name of dimensions to concat results along.
        Also Used by :class:`~lnpy.ensembles.GrandCanonicalEnsemble`.
    concat_coords : string, optional
        parameters `coords `to :func:`xarray.concat`
    unstack : bool, default=True
        If True, then outputs will be unstacked using :meth:`xarray.DataArray.unstack`
    single_state : bool, default=True
        If True, verify that all data has same shape, and value of `state_kws`.
        That is, all ``lnpi`` are for a single state.
    *args **kwargs
        Extra arguments to Series constructor

    """

    _concat_dim = "sample"
    _concat_coords = "different"
    _use_joblib = True
    _xarray_output = True
    _xarray_unstack = True
    _xarray_dot_kws: Final = {"optimize": "optimal"}
    _use_cache = True

    def __init__(
        self,
        data: Self | Sequence[lnPiMasked] | pd.Series[Any],
        index: ArrayLike | IndexAny | pd.MultiIndex | None = None,
        xarray_output: bool = True,
        concat_dim: str | None = None,
        concat_coords: str | None = None,
        unstack: bool = True,
        name: Hashable | None = None,
        base_class: str | type = "first",
        dtype: DTypeLike | None = None,
    ) -> None:
        if concat_dim is not None:
            self._concat_dim = concat_dim
        if concat_coords is not None:
            self._concat_coords = concat_coords
        if xarray_output is not None:
            self._xarray_output = xarray_output
        if unstack is not None:
            self._xarray_unstack = unstack

        if isinstance(data, self.__class__):
            x = data
            data = x.s

        self._base_class = base_class
        self._verify = self._base_class is not None

        series: pd.Series[Any] = pd.Series(  # type: ignore[misc]  # pyright: ignore[reportCallIssue]
            data=data,  # type: ignore[arg-type,unused-ignore]  # pyright: ignore[reportArgumentType]
            index=index,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            dtype=dtype,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            name=name,
        )
        self._verify_series(series)
        self._series = series
        self._cache: dict[str, Any] = {}

    def _verify_series(self, series: pd.Series[Any]) -> None:
        if not self._verify:
            return

        series = series.dropna()

        if series.empty:
            msg = "No non null values in series"
            raise ValueError(msg)

        base_class = self._base_class
        if isinstance(base_class, str) and base_class.lower() == "first":
            base_class = type(series.iloc[0])
        if isinstance(base_class, str):
            raise TypeError

        for d in series:
            if not issubclass(type(d), base_class):
                # print("series", series)  # noqa: ERA001
                # print("series.iloc[0]", series.dropna().iloc[0])  # noqa: ERA001
                # print("series", d)  # noqa: ERA001
                msg = f"all elements must be of type {base_class}.  found {type(d)}."
                raise TypeError(msg)

        # lnpy
        first = series.iloc[0]
        state_kws = first.state_kws
        shape = first.shape

        for lnpi in series:
            if lnpi.state_kws != state_kws or lnpi.shape != shape:
                raise ValueError
            # would like to do this, but
            # fails for parallel builds
            # assert lnpi._base is _base

    def new_like(
        self,
        data: Sequence[lnPiMasked] | pd.Series[Any] | None = None,
        index: ArrayLike | IndexAny | pd.MultiIndex | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create new object with optional new data/index"""
        if data is None:
            data = self.s

        return type(self)(
            data=data,
            index=index,
            xarray_output=self._xarray_output,
            concat_dim=self._concat_dim,
            concat_coords=self._concat_coords,
            unstack=self._xarray_unstack,
            **kwargs,
        )

    # ** Series Specific
    @property
    def series(self) -> pd.Series[Any]:
        """View of the underlying :class:`pandas.Series`"""
        return self._series

    @series.setter
    def series(self, series: pd.Series[Any]) -> None:
        self._cache = {}
        self._verify_series(series)
        self._series = series

    @property
    def s(self) -> pd.Series[Any]:
        """Alias to :meth:`series`"""
        return self.series

    def __iter__(self) -> Iterator[lnPiMasked]:
        return iter(self._series)

    @property
    def values(self) -> NDArrayAny:
        """Series values"""
        return self._series.to_numpy()

    @property
    def items(self) -> NDArrayAny:
        """Alias to :attr:`values`"""
        return self.values

    @property
    def index(self) -> IndexAny:
        """Series index"""
        return self._series.index

    @property
    def name(self) -> Hashable:
        """Series name"""
        return self._series.name

    def copy(self) -> Self:
        return type(self)(data=self.s, base_class=self._base_class)

    def _wrapped_pandas_method(
        self, mtd: str, *args: Any, wrap: bool = False, **kwargs: Any
    ) -> lnPiMasked | pd.Series[Any] | Self:
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(self._series, mtd)(*args, **kwargs)
        if wrap and isinstance(val, pd.Series):
            val = self.new_like(val)
        return val  # type: ignore[no-any-return]

    def xs(
        self,
        key: Hashable | Sequence[Hashable],
        axis: int = 0,
        level: Hashable | Sequence[Hashable] | None = None,
        drop_level: bool = False,
        wrap: bool = True,
    ) -> Self | pd.Series[Any] | lnPiMasked:
        """Interface to :meth:`pandas.Series.xs`"""
        return self._wrapped_pandas_method(
            "xs", wrap=wrap, key=key, axis=axis, level=level, drop_level=drop_level
        )

    def __getitem__(self, key: Any) -> Self | lnPiMasked:
        """Interface to :meth:`pandas.Series.__getitem__`"""
        return self._wrapped_pandas_method("__getitem__", wrap=True, key=key)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    def __setitem__(
        self, idx: Any, values: lnPiMasked | Sequence[lnPiMasked] | pd.Series[Any]
    ) -> None:
        """Interface to :meth:`pandas.Series.__setitem__`"""
        self._series[idx] = values

    def __len__(self) -> int:
        return len(self.s)

    def append(
        self,
        to_append: pd.Series[Any] | Self,
        ignore_index: bool = False,
        verify_integrity: bool = True,
        concat_kws: Mapping[str, Any] | None = None,
        inplace: bool = False,
    ) -> Self:
        """
        Interface to :func:`pandas.concat`


        Parameters
        ----------
        to_append : object
            Object to append
        ignore_index : bool, default=False
        verify_integrity : bool, default=True
        concat_kws : mapping, optional
            Extra arguments to
        inplace : bool, default=False

        See Also
        --------
        pandas.concat
        """
        if hasattr(to_append, "series") and isinstance(to_append.series, pd.Series):
            series = to_append.series
        elif isinstance(to_append, pd.Series):
            series = to_append
        else:
            msg = f"Unknown to append type={type(to_append)}"
            raise ValueError(msg)

        if concat_kws is None:
            concat_kws = {}

        s = pd.concat(
            (self.series, series),
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            **concat_kws,
        )

        if inplace:
            self.series = s
            return self

        return self.new_like(s)

    def droplevel(self, level: int | Hashable | Sequence[int | Hashable]) -> Self:
        """
        New object with dropped level

        See Also
        --------
        pandas.Series.droplevel
        """
        return self.new_like(self._series.droplevel(level=level, axis=0))  # type: ignore[arg-type,unused-ignore]

    def apply(
        self,
        func: Callable[..., Any],
        convert_dtype: bool = True,
        args: tuple[Any, ...] = (),
        wrap: bool = False,
        **kwds: Any,
    ) -> Self | pd.Series[Any]:
        """Interface to :meth:`pandas.Series.apply`"""
        return self._wrapped_pandas_method(  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
            "apply",
            wrap=wrap,
            func=func,
            convert_dtype=convert_dtype,
            args=args,
            **kwds,
        )

    def sort_index(self, *args: Any, **kwargs: Any) -> Self:
        """Interface to :meth:`pandas.Series.sort_index`"""
        return self._wrapped_pandas_method("sort_index", *args, wrap=True, **kwargs)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    @overload
    def groupby(
        self,
        by: Hashable | Sequence[Hashable] = ...,
        *,
        level: int | Hashable | Sequence[int | Hashable] | None = ...,
        as_index: bool = ...,
        sort: bool = ...,
        group_keys: bool = ...,
        observed: bool = ...,
        dropna: bool = ...,
        wrap: Literal[True],
    ) -> _Groupby: ...

    @overload
    def groupby(
        self,
        by: Hashable | Sequence[Hashable] = ...,
        *,
        level: int | Hashable | Sequence[int | Hashable] | None = ...,
        as_index: bool = ...,
        sort: bool = ...,
        group_keys: bool = ...,
        observed: bool = ...,
        dropna: bool = ...,
        wrap: bool,
    ) -> SeriesGroupBy[Any, Any] | _Groupby: ...

    def groupby(
        self,
        by: Hashable | Sequence[Hashable] | None = None,
        *,
        level: int | Hashable | Sequence[int | Hashable] | None = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = False,
        dropna: bool = True,
        wrap: bool = False,
    ) -> SeriesGroupBy[Any, Any] | _Groupby:
        """
        Wrapper around :meth:`pandas.Series.groupby`.

        Parameters
        ----------
        wrap : bool, default=False
            if True, try to wrap output in class of self

        See Also
        --------
        pandas.Series.groupby
        """
        group = self.s.groupby(  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
            by=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

        if wrap:
            return _Groupby(self, group)
        return group  # type: ignore[no-any-return]

    @overload
    def groupby_allbut(
        self,
        drop: Hashable | list[Hashable] | tuple[Hashable],
        *,
        wrap: Literal[False] = ...,
        **kwargs: Any,
    ) -> SeriesGroupBy[Any, Any]: ...

    @overload
    def groupby_allbut(
        self,
        drop: Hashable | list[Hashable] | tuple[Hashable],
        *,
        wrap: Literal[True],
        **kwargs: Any,
    ) -> _Groupby: ...

    @overload
    def groupby_allbut(
        self,
        drop: Hashable | list[Hashable] | tuple[Hashable],
        *,
        wrap: bool,
        **kwargs: Any,
    ) -> SeriesGroupBy[Any, Any] | _Groupby: ...

    def groupby_allbut(
        self,
        drop: Hashable | list[Hashable] | tuple[Hashable],
        *,
        wrap: bool = False,
        **kwargs: Any,
    ) -> SeriesGroupBy[Any, Any] | _Groupby:
        """Groupby all but columns in drop"""
        from .core.utils import allbut

        if isinstance(drop, list):
            drop = tuple(drop)
        elif not isinstance(drop, tuple):
            drop = (drop,)

        by = allbut(self.index.names, *drop)

        # To suppress annoying errors.
        if len(by) == 1:
            return self.groupby(by=by[0], wrap=wrap, **kwargs)
        return self.groupby(by=by, wrap=wrap, **kwargs)

    @classmethod
    def _concat_to_series(
        cls,
        objs: Iterable[Self]
        | Iterable[pd.Series[Any]]
        | Mapping[Hashable, Self]
        | Mapping[Hashable, pd.Series[Any]],
        **concat_kws: Any,
    ) -> pd.Series[Any]:
        from collections.abc import Mapping

        if isinstance(objs, Mapping):
            out = {}
            remap = None
            for k in objs:
                v = objs[k]  # pyright: ignore[reportArgumentType]
                if remap is None:
                    remap = bool(isinstance(v, cls))
                if remap:
                    out[k] = v._series
                else:
                    out[k] = v
            objs = out  # pylint: disable=redefined-variable-type
        else:
            first, objs = peek_at(objs)  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
            if isinstance(first, cls):
                objs = (x._series for x in objs)  # pyright: ignore[reportAttributeAccessIssue]

        return pd.concat(objs, **concat_kws)  # type: ignore[arg-type] # pyright: ignore[reportCallIssue, reportArgumentType]

    def concat_like(
        self,
        objs: Sequence[Self]
        | Sequence[pd.Series[Any]]
        | Mapping[Hashable, Self]
        | Mapping[Hashable, pd.Series[Any]],
        **concat_kws: Any,
    ) -> Self:
        """Concat a sequence of objects like `self`"""
        s = self._concat_to_series(objs, **concat_kws)
        return self.new_like(s)

    @classmethod
    def concat(
        cls,
        objs: Sequence[Self]
        | Sequence[pd.Series[Any]]
        | Mapping[Hashable, Self]
        | Mapping[Hashable, pd.Series[Any]],
        *args: Any,
        concat_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create collection from sequence of objects"""
        if concat_kws is None:
            concat_kws = {}
        s = cls._concat_to_series(objs, **concat_kws)
        return cls(s, *args, **kwargs)

    # Note: use property(cached.meth(func)) here
    # normal cache.prop has some logic issues
    # with this pattern.

    @cached.prop
    def loc(self) -> _LocIndexer:
        return _LocIndexer(self)

    @cached.prop
    def iloc(self) -> _iLocIndexer:
        return _iLocIndexer(self)

    @cached.prop
    def query(self) -> _Query:
        return _Query(self)

    @cached.prop
    def zloc(self) -> _LocIndexer_unstack_zloc:
        return _LocIndexer_unstack_zloc(self)

    @cached.prop
    def mloc(self) -> _LocIndexer_unstack_mloc:
        return _LocIndexer_unstack_mloc(self)

    # ** lnPi Specific
    @cached.prop
    def _lnz_series(self) -> pd.Series[Any]:
        return self._series.apply(lambda x: x.lnz)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"<class {self.__class__.__name__}>\n{self._lnz_series!r}"

    def __str__(self) -> str:
        return str(self._lnz_series)

    @property
    def state_kws(self) -> dict[str, Any]:
        """state_kws from first :class:`~lnpy.lnpidata.lnPiMasked`"""
        return self.iloc[0].state_kws

    @property
    def nlnz(self) -> int:
        """Number of unique lnzs"""
        return len(self.index.droplevel("phase").drop_duplicates())

    @cached.prop
    def index_frame(self) -> pd.DataFrame:
        """
        Values (from :class:`xarray.DataArray`) for each sample.

        includes a column 'lnz_index' which is the unique lnz values
        regardless of phase
        """
        sample_frame = (
            self.index.droplevel("phase")
            .drop_duplicates()
            .to_frame()
            .assign(lnz_sample=lambda x: np.arange(len(x)))["lnz_sample"]
        )
        return (
            self.index.to_frame()
            .reset_index("phase", drop=True)[["phase"]]
            .assign(lnz_index=lambda x: sample_frame[x.index])
            .reset_index()
        )

    @overload
    def _get_lnz(
        self, component: int, *, iloc: int = ..., zloc: int | None = ...
    ) -> float: ...

    @overload
    def _get_lnz(
        self, component: None = ..., *, iloc: int = ..., zloc: int | None = ...
    ) -> NDArrayAny: ...

    def _get_lnz(
        self, component: int | None = None, *, iloc: int = 0, zloc: int | None = None
    ) -> float | NDArrayAny:
        """
        Helper function to.
        returns self.iloc[idx].lnz[component]
        """
        v: lnPiMasked = (
            self.zloc[zloc]._series if zloc is not None else self._series
        ).iloc[iloc]
        lnz = v.lnz
        if component is not None:
            lnz = lnz[component]
        return lnz

    def _get_level(self, level: str = "phase") -> IndexAny:
        """Return level values from index"""
        index = self.index
        if isinstance(index, pd.MultiIndex):
            level_idx = index.names.index(level)
            index = index.levels[level_idx]
        return index

    def get_index_level(self, level: str = "phase") -> IndexAny:
        """Get index values for specified level"""
        return self._get_level(level=level)

    # @cached.prop
    @property
    def _nrec(self) -> int:
        return len(self._series)

    def _lnpi_tot(self, fill_value: float | None = None) -> NDArrayAny:
        # new method
        # this is no faster than the original
        # but makes clear where the time is being spent
        first = self.iloc[0]
        n = len(self)
        out = np.empty((n, *first.shape), dtype=first.dtype)
        seq = get_tqdm((x.filled(fill_value) for x in self), total=n)
        for i, x in enumerate(seq):
            out[i, ...] = x
        return out

    @property
    def _lnz_tot(self) -> NDArrayAny:
        return np.stack([x.lnz for x in self])

    @property
    def lnz(self) -> NDArrayAny:
        return self._lnz_tot

    def _pi_params(
        self, fill_value: float | None = None
    ) -> tuple[NDArrayAny, NDArrayAny, NDArrayAny]:
        first = self.iloc[0]
        n = len(self)

        pi_norm = np.empty((n, *first.shape), dtype=first.dtype)
        pi_sum = np.empty(n, dtype=first.dtype)
        lnpi_zero = np.empty(n, dtype=first.dtype)

        seq = get_tqdm(
            (x._pi_params(fill_value) for x in self), total=n, desc="pi_norm"
        )

        for i, (pi_norm_, pi_sum_, lnpi_zero_) in enumerate(seq):
            pi_norm[i, ...] = pi_norm_
            pi_sum[i, ...] = pi_sum_
            lnpi_zero[i, ...] = lnpi_zero_
        return pi_norm, pi_sum, lnpi_zero

    def wrap_list_results(
        self, items: Sequence[Any] | Sequence[xr.DataArray]
    ) -> Sequence[Any] | xr.DataArray:
        """Utility to wrap output in :class:xarray.DataArray"""
        if self._xarray_output and isinstance(items[0], xr.DataArray):
            return xr.concat(items, self.index, coords=self._concat_coords)
        return items

    ##################################################
    # Constructors
    @classmethod
    def from_list(
        cls,
        items: Sequence[lnPiMasked],
        index: Iterable[int] | NDArrayAny,
        **kwargs: Any,
    ) -> Self:
        """
        Create collection from list of :class:`~lnpy.lnpidata.lnPiMasked` objects.

        Parameters
        ----------
        items : sequence of lnPiMasked
            Sequence of lnPi
        index : sequence
            Sequence of phases ID for each lnPi
        **kwargs :
            Extra keyword arguments to `cls`

        Returns
        -------
        lnPiCollection
        """
        table = pd.DataFrame(
            [lnpi._index_dict(phase) for lnpi, phase in zip(items, index, strict=True)]
        )
        new_index = pd.MultiIndex.from_frame(table)
        return cls(data=items, index=new_index, **kwargs)

    @classmethod
    def from_builder(
        cls,
        lnzs: Sequence[float] | NDArrayAny,
        # TODO(wpk): make better type for build_phases.
        build_phases: Callable[..., tuple[list[lnPiMasked], NDArrayAny]],
        ref: lnPiMasked | None = None,
        build_kws: Mapping[str, Any] | None = None,
        nmax: int | None = None,
        base_class: str | type = "first",
        **kwargs: Any,
    ) -> Self:
        """
        Build collection from scalar builder

        Parameters
        ----------
        lnzs : sequence of float
            One dimensional array of lnz value for the varying component.
        ref : lnPiMasked
            lnpi_phases to reweight to get list of lnpi's
        build_phases : callable
            Typically one of `PhaseCreator.build_phases_mu` or `PhaseCreator.build_phases_dmu`
        build_kws : optional
            optional arguments to `build_phases`

        Returns
        -------
        lnPiCollection


        See Also
        --------
        ~lnpy.segment.PhaseCreator.build_phases
        ~lnpy.segment.PhaseCreator.build_phases_mu
        ~lnpy.segment.PhaseCreator.build_phases_dmu

        """
        if build_kws is None:
            build_kws = {}

        build_kws = dict(build_kws, phases_factory=False)
        seq = get_tqdm(lnzs, desc="build")

        items: list[lnPiMasked] = []
        index: list[int] = []
        for data, idx in parallel_map(
            build_phases, seq, ref=ref, nmax=nmax, **build_kws
        ):
            items += data
            index += list(idx)
        return cls.from_list(items, index, base_class=base_class, **kwargs)

    ################################################################################
    # dataarray io
    def to_dataarray(
        self, dtype: DTypeLike | None = None, reset_index: bool = True
    ) -> xr.DataArray:
        """
        Convert collection to a :class:`~xarray.DataArray`

        Parameters
        ----------
        dtype : `numpy.dtype`, optional
            Default to `numpy.uint8`.
        reset_index : bool, default=True
        """
        dtype = dtype or np.uint8

        labels = []
        indexes = []

        for _, g in self.groupby_allbut("phase"):
            indexes.append(g.index[[0]])

            features = np.array(g.index.get_level_values("phase")) + 1
            masks = [x.mask for x in g]
            labels.append(
                masks_to_labels(masks, features=features, convention=False, dtype=dtype)
            )

        index = indexes[0].append(indexes[1:])

        data = np.stack(labels)

        out = (
            xr.DataArray(
                data,
                dims=self.xge.dims_rec + self.xge.dims_n,
                name="labels",
            )
            .assign_coords(**{self._concat_dim: index, **self.state_kws})
            .assign_attrs(**self.xge._standard_attrs)
        )

        if reset_index:
            out = out.reset_index(self._concat_dim)

        return out

    @classmethod
    @docfiller.decorate
    def from_labels(
        cls,
        ref: lnPiMasked,
        labels: Sequence[NDArrayAny],
        lnzs: Sequence[float | NDArrayAny] | NDArrayAny,
        features: Sequence[int] | NDArrayAny | None = None,
        include_boundary: bool = False,
        labels_kws: Mapping[str, Any] | None = None,
        check_features: bool = True,
        **kwargs: Any,
    ) -> Self:
        r"""
        Create from reference :class:`~lnpy.lnpidata.lnPiMasked` and labels array


        Parameters
        ----------
        ref : lnPiMasked
        labels : sequence of ndarray of int
            Each ``labels[i]`` is a labels array for each value of ``lnzs[i]``.
            That is, the labels for different phases at a given value of `lnz`.
        lnzs : sequence
            Each lnzs[i] will be passed to ``ref.reweight``.
        {features}
        {include_boundary}
        labels_kws : mapping, optional
        {check_features}
        **kwargs
            Extra arguments past to :meth:`from_list`

        See Also
        --------
        ~lnpy.core.mask.labels_to_masks
        from_list
        """
        if labels_kws is None:
            labels_kws = {}
        if len(labels) != len(lnzs):
            raise ValueError

        items = []
        indexes = []

        for label, lnz in zip(labels, lnzs, strict=True):
            lnpi = ref.reweight(lnz)

            masks, features_tmp = labels_to_masks(
                labels=label,
                features=features,
                include_boundary=include_boundary,
                convention=False,
                check_features=check_features,
                **labels_kws,
            )

            index = list(np.array(features_tmp) - 1)
            items += lnpi.list_from_masks(masks, convention=False)
            indexes += index

        return cls.from_list(items=items, index=indexes, **kwargs)

    @classmethod
    @docfiller.decorate
    def from_dataarray(
        cls,
        ref: lnPiMasked,
        da: xr.DataArray,
        grouper: Hashable = "sample",
        include_boundary: bool = False,
        labels_kws: Mapping[str, Any] | None = None,
        features: Sequence[int] | NDArrayAny | None = None,
        check_features: bool = True,
        **kwargs: Any,
    ) -> Self:
        """
        Create a collection from DataArray of labels

        Parameters
        ----------
        ref : lnPiMasked
        da : DataArray or int
            Labels.
        grouper : Hashable
            Name of dimension(s) to group along to give a single label array
        {features}
        {check_features}


        See Also
        --------
        from_labels
        """
        labels = []
        lnzs = []

        for _, g in da.groupby(grouper):  # type: ignore[arg-type, unused-ignore]  # pyright: ignore[reportArgumentType]
            lnzs.append(np.array([g.coords[k] for k in da.attrs["dims_lnz"]]))
            labels.append(g.values)

        return cls.from_labels(
            ref=ref,
            labels=labels,
            lnzs=lnzs,
            labels_kws=labels_kws,
            features=features,
            include_boundary=include_boundary,
            check_features=check_features,
            **kwargs,
        )  # yapf: disable

    @cached.prop
    @docfiller.decorate
    def xge(self) -> ensembles.GrandCanonicalEnsemble:
        """{accessor.xge}"""
        from .ensembles import GrandCanonicalEnsemble

        return GrandCanonicalEnsemble(self)

    @cached.prop
    def wfe(self) -> lnpienergy.wFreeEnergyCollection:
        """Accessor to :class:`~lnpy.lnpienergy.wFreeEnergyPhases` from :attr:`wfe_phases`."""
        from .lnpienergy import wFreeEnergyCollection

        return wFreeEnergyCollection(self)

    @cached.prop
    def wfe_phases(self) -> lnpienergy.wFreeEnergyPhases:
        """Accessor to :class:`~lnpy.lnpienergy.wFreeEnergyCollection` from :attr:`wfe`."""
        from .lnpienergy import wFreeEnergyPhases

        return wFreeEnergyPhases(self)

    @property
    def wlnPi(self) -> lnpienergy.wFreeEnergyCollection:  # noqa: N802
        """
        Deprecated accessor to :class:`~lnpy.lnpienergy.wFreeEnergyCollection` from :attr:`wlnPi`.

        Alias to :attr:`wfe`
        """
        warn(
            "Using `wlnPi` accessor is deprecated.  Please use `wfe` accessor instead",
            stacklevel=1,
        )
        return self.wfe

    @property
    def wlnPi_single(self) -> lnpienergy.wFreeEnergyPhases:  # noqa: N802
        """
        Deprecated accessor to :class:`~lnpy.lnpienergy.wFreeEnergyPhases` from :attr:`wlnPi_single`.

        Alias to :attr:`wfe_phases`
        """
        warn(
            "Using `wlnPi_single is deprecated.  Please use `self.wfe_phases` instead",
            stacklevel=1,
        )
        return self.wfe_phases

    @cached.prop
    def spinodal(self) -> stability.Spinodals:
        """Accessor to :class:`~lnpy.stability.Spinodals`"""
        from .stability import Spinodals

        return Spinodals(self)

    @cached.prop
    def binodal(self) -> stability.Binodals:
        """Accessor to :class:`~lnpy.stability.Binodals`"""
        from .stability import Binodals

        return Binodals(self)

    def stability_append(
        self,
        other: Self | None,
        append: bool = True,
        sort: bool = True,
        copy_stability: bool = True,
    ) -> Self:
        if (not append) and (not copy_stability):
            msg = "one of append or copy_stability must be True"
            raise ValueError(msg)

        if other is None:
            other = self
        spin = other.spinodal
        bino = other.binodal
        if append:
            new = self.append(spin.appender).append(bino.appender)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
            if sort:
                new = new.sort_index()
        else:
            new = self.copy()
        if copy_stability:
            # TODO(wpk): fix this hack
            new._cache["spinodal"] = spin
            new._cache["binodal"] = bino
        return new
