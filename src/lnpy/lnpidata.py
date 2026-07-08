"""
lnPi data classes and routines (:mod:`~lnpy.lnpidata`)
======================================================
"""

################################################################################
# Delayed
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast, overload

import attrs
import attrs.validators as av
import numpy as np
import pandas as pd
import xarray as xr
from module_utilities import cached

from .core import validate
from .core._attrs_utils import MyAttrsMixin, convert_mapping_or_none_to_dict
from .core.compat import copy_if_needed
from .core.docstrings import docfiller
from .core.mask import labels_to_masks, masks_change_convention
from .core.typing_compat import override
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
    from pathlib import Path
    from typing import Any, ClassVar, Concatenate, ParamSpec, SupportsFloat

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from . import ensembles
    from .core.typing import (
        InterpolationMethods,
        MaskConvention,
        NDArrayAny,
        OptionalKwsAny,
    )
    from .core.typing_compat import Self

    P = ParamSpec("P")


# * Utilities -----------------------------------------------------------------
def _get_n_ranges(shape: tuple[int, ...], dtype: DTypeLike | None) -> list[NDArrayAny]:
    return [np.arange(s, dtype=dtype) for s in shape]


@lru_cache(maxsize=20)
def _get_shift(
    shape: tuple[int, ...], dlnz: tuple[float, ...], dtype: DTypeLike | None
) -> NDArrayAny:
    shift = np.zeros([], dtype=dtype)
    for _i, (nr, m) in enumerate(
        zip(_get_n_ranges(shape=shape, dtype=dtype), dlnz, strict=True)
    ):
        shift = np.add.outer(shift, nr * m)
    return shift


@lru_cache(maxsize=20)
def _get_data(base: lnPiArray, dlnz: tuple[float, ...]) -> NDArrayAny:
    if all(x == 0 for x in dlnz):  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
        return base.data
    return _get_shift(base.data.shape, dlnz, base.data.dtype) + base.data


def _get_maskedarray(
    base: lnPiArray, self: lnPiMasked, dlnz: tuple[float, ...]
) -> np.ma.MaskedArray[Any, np.dtype[Any]]:
    return np.ma.MaskedArray(
        _get_data(base, dlnz),
        mask=self.mask,
        fill_value=base.fill_value,
    )


@lru_cache(maxsize=20)
def _get_filled(
    base: lnPiArray,
    self: lnPiMasked,
    dlnz: tuple[float, ...],
    fill_value: float | None = None,
) -> NDArray[Any]:
    return _get_maskedarray(base, self, dlnz).filled(fill_value)


# * lnPiArray -----------------------------------------------------------------
def _convert_lnz(lnz: ArrayLike) -> NDArray[np.float64]:
    return np.atleast_1d(lnz).astype(np.float64)


def _convert_fill_value(fill_value: SupportsFloat | None) -> float:
    if fill_value is None:
        return np.nan
    return float(fill_value)


def _validate_data(self_: Any, attribute: Any, data: NDArrayAny) -> None:  # noqa: ARG001
    if data.ndim != len(self_.lnz):
        msg = f"Length of {self_.lnz=} must be {data.ndim}"
        raise ValueError(msg)


@docfiller.decorate
@attrs.frozen(
    eq=False, kw_only=True, init=False
)  # use eq=False to make hashable by object
class lnPiArray(MyAttrsMixin):  # noqa: N801
    """
    Wrapper on lnPi lnPiArray

    Parameters
    ----------
    {lnz}
    {data}
    {state_kws}
    {extra_kws}
    {fill_value}
    {copy}
    """

    copy: bool | None = None
    lnz: NDArray[np.float64] = attrs.field(converter=_convert_lnz)
    data: NDArrayAny = attrs.field(validator=_validate_data)
    state_kws: dict[str, Any] = attrs.field(
        factory=dict, converter=convert_mapping_or_none_to_dict
    )
    extra_kws: dict[str, Any] = attrs.field(
        factory=dict, converter=convert_mapping_or_none_to_dict
    )
    fill_value: float = attrs.field(default=np.nan, converter=_convert_fill_value)

    if TYPE_CHECKING:
        # Lie to make pyright/pyrefly/ty happy
        def __attrs_init__(self, **kwargs: Any) -> None: ...

    def __init__(
        self,
        lnz: float | np.floating[Any] | ArrayLike,
        data: ArrayLike,
        state_kws: OptionalKwsAny = None,
        extra_kws: OptionalKwsAny = None,
        fill_value: float | None = np.nan,
        copy: bool | None = None,
    ) -> None:
        # NOTE: maybe use view?
        data = np.array(data, copy=copy_if_needed(copy))
        data.flags.writeable = False

        self.__attrs_init__(
            lnz=lnz,
            data=data,
            state_kws=state_kws,
            extra_kws=extra_kws,
            fill_value=fill_value,
        )

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[NDArrayAny, P], NDArrayAny],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self: ...
    @overload
    def pipe(
        self,
        func: Callable[..., NDArrayAny],
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...

    def pipe(
        self,
        func: Callable[..., NDArrayAny],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """Apply numpy function to underlying data."""
        return self.new_like(data=func(self.data, *args, **kwargs))


# * Masked lnPi object --------------------------------------------------------
def _validate_mask(self_: Any, attribute: Any, mask: NDArray[np.bool_]) -> None:  # noqa: ARG001
    if mask.shape != self_.base.data.shape:
        msg = f"{mask.shape=} must be {self_.base.data.shape}."
        raise ValueError(msg)


@docfiller.decorate  # noqa: PLR0904
@attrs.define(frozen=True, eq=False, init=False)
class lnPiMasked(AccessorMixin, MyAttrsMixin):  # noqa: N801
    """
    Masked array like wrapper for lnPi data.

    This is the basic data structure for storing the output from a single TMMC simulation.

    Parameters
    ----------
    {lnz}
    {base}
    {mask_masked}
    {copy}

    See Also
    --------
    numpy.ma.MaskedArray

    Notes
    -----
    Note that in most cases, :class:`lnPiMasked` should not be called directly.
    Rather, a constructor like :meth:`from_data` should be used to construct the object.

    Note the the value of `lnz` is the value to reweight to.

    Basic terminology:

    * T : temperature.
    * k : Boltzmann's constant.
    * beta : Inverse temperature `= 1/(k T)`.
    * mu : chemical potential.
    * lnz : log of activity `= ln(z)`.
    * z : activity `= beta * mu`.
    * lnPi : log of macrostate distribution.
    """

    lnz: NDArray[np.float64] = attrs.field(converter=_convert_lnz)
    base: lnPiArray = attrs.field(validator=av.instance_of(lnPiArray))
    mask: NDArray[np.bool_] = attrs.field(validator=_validate_mask)

    _dlnz: tuple[float, ...] = attrs.field(init=False, repr=False)
    _cache: dict[str, Any] = attrs.field(
        factory=dict[str, "Any"], init=False, repr=False
    )
    _DataClass: ClassVar[type[lnPiArray]] = lnPiArray

    if TYPE_CHECKING:
        # Lie to make pyright/pyrefly/ty happy
        def __attrs_init__(self, **kwargs: Any) -> None: ...

    def __init__(
        self,
        lnz: ArrayLike,
        base: lnPiArray,
        mask: ArrayLike | None = None,
        copy: bool | None = None,
    ) -> None:
        mask = (
            np.full_like(base.data, fill_value=False, dtype=np.bool_)
            if mask is None
            else np.asarray(mask, copy=copy_if_needed(copy), dtype=np.bool_)
        )

        self.__attrs_init__(
            lnz=lnz,
            base=base,
            mask=mask,
        )

        object.__setattr__(
            self,
            "_dlnz",
            tuple(
                0 if np.equal(lnz, base_lnz) else float(lnz - base_lnz)
                for lnz, base_lnz in zip(self.lnz, self.base.lnz, strict=True)
            ),
        )

    @classmethod
    def from_data(
        cls,
        data: NDArrayAny,
        lnz: float | ArrayLike,
        lnz_data: float | ArrayLike | None = None,
        mask: NDArrayAny | None = None,
        state_kws: dict[str, Any] | None = None,
        extra_kws: dict[str, Any] | None = None,
        fill_value: float | None = None,
        copy: bool | None = None,
    ) -> Self:
        """
        Create :class:`lnPiMasked` object from raw data.

        Parameters
        ----------
        lnz : float or sequence of float
            Value of `lnz` to reweight data to.
        lnz_data : float or sequence of float, optional
            Value of `lnz` at which `data` was collected.
            Defaults to ``lnz``.
        {data}
        {mask_masked}
        {state_kws}
        {extra_kws}
        {fill_value}
        {copy}

        Returns
        -------
        out : lnPiMasked
        """
        fill_value = fill_value or np.nan

        if lnz_data is None:
            lnz_data = lnz

        base = cls._DataClass(
            lnz=lnz_data,
            data=data,
            state_kws=state_kws,
            extra_kws=extra_kws,
            fill_value=fill_value,
            copy=copy,
        )
        return cls(lnz=lnz, base=base, mask=mask, copy=copy)

    def as_pure(self, keepdims: bool = False) -> Iterator[Self]:
        """
        Iterator of pure component objects

        Parameters
        ----------
        keepdims: bool, default=False
            If ``True``, keep the reduced dimensions (i.e., `ndim` is
            unchanged). Otherwise return `1d` objects.

        Yields
        ------
        lnpi_component: lnPiMasked
            Pure component lnPiMasked


        Example
        -------
        >>> import numpy as np
        >>> import lnpy
        >>> ref = lnpy.lnPiMasked.from_data(
        ...     data=np.arange(9).reshape(3, 3), lnz=[0, 2], lnz_data=[0, 2]
        ... )
        >>> ref
        <lnPi(lnz=[0. 2.])>
        >>> ref.data
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
        >>> pures = list(ref.as_pure())
        >>> pures[0]
        <lnPi(lnz=[0.])>
        >>> pures[0].data
        array([0, 3, 6])
        >>> pures[1]
        <lnPi(lnz=[2.])>
        >>> pures[1].data
        array([0, 1, 2])
        """
        for index in range(self.ndim):
            zero = [0] if keepdims else 0
            slc = tuple(slice(None) if i == index else zero for i in range(self.ndim))

            if keepdims:
                lnz = np.full_like(self.lnz, fill_value=-np.inf)
                lnz[index] = self.lnz[index]

                base_lnz = np.full_like(self.lnz, fill_value=-np.inf)
                base_lnz[index] = self.base.lnz[index]

            else:
                lnz = self.lnz[index]
                base_lnz = self.base.lnz[index]

            yield type(self)(
                lnz=lnz,
                base=self.base.new_like(
                    lnz=base_lnz,
                    data=self.base.data[slc],
                ),
            )

    @property
    def dtype(self) -> np.dtype[Any]:
        """Type (dtype) of underling data"""
        return self.base.data.dtype

    def _clear_cache(self) -> None:
        self._cache.clear()

    @property
    def state_kws(self) -> dict[str, Any]:
        """State variables."""
        return self.base.state_kws

    @property
    def extra_kws(self) -> dict[str, Any]:
        """Extra parameters."""
        return self.base.extra_kws

    @property
    def ma(self) -> np.ma.MaskedArray[Any, np.dtype[Any]]:
        """Masked array view of data reweighted data"""
        return _get_maskedarray(self.base, self, self._dlnz)

    def filled(self, fill_value: float | None = None) -> NDArrayAny:
        """Filled view or reweighted data"""
        return _get_filled(self.base, self, self._dlnz, fill_value)

    @property
    def data(self) -> NDArrayAny:
        """Reweighted data"""
        return _get_data(self.base, self._dlnz)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of lnPiArray"""
        return cast("tuple[int, ...]", self.base.data.shape)

    def __len__(self) -> int:
        return len(self.base.data)

    @property
    def ndim(self) -> int:
        return self.base.data.ndim

    @property
    def betamu(self) -> NDArray[np.float64]:
        """Alias to `self.lnz`"""
        return self.lnz

    @property
    def volume(self) -> float | None:
        """Accessor to self.state_kws['volume']."""
        return self.state_kws.get("volume", None)

    @property
    def beta(self) -> float | None:
        """Accessor to self.state_kws['beta']."""
        return self.state_kws.get("beta", None)

    @override
    def __repr__(self) -> str:
        return f"<lnPi(lnz={self.lnz})>"

    @override
    def __str__(self) -> str:
        return repr(self)

    def _index_dict(self, phase: int | str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {f"lnz_{i}": v for i, v in enumerate(self.lnz)}
        if phase is not None:
            out["phase"] = phase
        return out

    # Parameters for xlnPi
    def _lnpi_tot(self, fill_value: float | None = None) -> NDArrayAny:
        return self.filled(fill_value)

    def _pi_params(
        self, fill_value: float | None = None
    ) -> tuple[NDArrayAny, float, float]:
        lnpi = self._lnpi_tot(fill_value)

        lnpi_local_max = lnpi.max()
        pi = np.exp(lnpi - lnpi_local_max)
        pi_sum = pi.sum()
        pi_norm = pi / pi_sum

        lnpi_zero = self.data.ravel()[0] - lnpi_local_max

        return pi_norm, pi_sum, lnpi_zero

    @property
    def _lnz_tot(self) -> NDArrayAny:
        return self.lnz

    # @cached.meth
    def local_argmax(self, *args: Any, **kwargs: Any) -> tuple[int, ...]:
        """
        Calculate index of maximum of masked data.

        Parameters
        ----------
        *args
            Positional arguments to argmax
        **kwargs
            Keyword arguments to argmax

        See Also
        --------
        numpy.ma.MaskedArray.argmax
        numpy.unravel_index
        """
        return tuple(
            int(x)
            for x in np.unravel_index(self.ma.argmax(*args, **kwargs), self.shape)
        )

    # @cached.meth
    def local_max(
        self, *args: Any, **kwargs: Any
    ) -> np.ma.MaskedArray[Any, np.dtype[Any]]:
        """
        Calculate index of maximum of masked data.

        Parameters
        ----------
        *args
            Positional arguments to argmax
        **kwargs
            Keyword arguments to argmax

        See Also
        --------
        numpy.ma.MaskedArray.max
        """
        return validate.maskedarray(
            self.ma[self.local_argmax(*args, **kwargs)],
        )

    # @cached.meth
    def local_maxmask(
        self, *args: Any, **kwargs: Any
    ) -> np.ma.MaskedArray[Any, np.dtype[Any]]:
        """Calculate mask where ``self.ma == self.local_max()``"""
        return validate.maskedarray(
            self.ma == self.local_max(*args, **kwargs),
        )

    @cached.prop
    def edge_distance_matrix(self) -> NDArray[np.float64]:
        """
        Matrix of distance from each element to a background (i.e., masked) point.

        See Also
        --------
        lnpy.core.utils.distance_matrix
        """
        from .core.utils import distance_matrix

        return distance_matrix(~self.mask)

    def edge_distance(self, ref: Self, *args: Any, **kwargs: Any) -> float:
        """
        Distance of local maximum value to nearest background point.

        If `edge_distance` is too small, the value of properties calculated from this
        lnPi cannot be trusted.   This usually is due to the data being reweighted to
        too high a value of `lnz`, or not sampled to sufficiently high values of `N`.


        See Also
        --------
        edge_distance_matrix
        lnpy.core.utils.distance_matrix
        """
        return float(ref.edge_distance_matrix[self.local_argmax(*args, **kwargs)])

    @overload
    def pipe(
        self,
        func: Callable[Concatenate[NDArrayAny, P], NDArrayAny],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self: ...
    @overload
    def pipe(
        self,
        func: Callable[..., NDArrayAny],
        *args: Any,
        **kwargs: Any,
    ) -> Self: ...

    def pipe(
        self,
        func: Callable[..., NDArrayAny],
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        """
        Apply numpy function to underlying data, leaving other meta data unchanged.

        Parameters
        ----------
        func : callable
            Function to apply to data. First argument must accept the data as a
            numpy array.
        *args, **kwargs
            Extra positional and keyword arguments to ``func``.

        Returns
        -------
        lnPiMasked
            New object with data transformed via ``func``.


        Example
        -------
        Apply a gaussian filter to underlying data

        >>> import numpy as np
        >>> data = np.random.default_rng(seed=0).random((3, 3))
        >>> ref = lnPiMasked.from_data(data=data, lnz=[0.0, 0.0])
        >>> ref
        <lnPi(lnz=[0. 0.])>
        >>> ref.data
        array([[0.637 , 0.2698, 0.041 ],
               [0.0165, 0.8133, 0.9128],
               [0.6066, 0.7295, 0.5436]])

        Smooth data using gaussian filter

        >>> from scipy.ndimage import gaussian_filter
        >>> smoothed = ref.pipe(gaussian_filter, mode="nearest", sigma=2)
        >>> smoothed  # No change to metadata
        <lnPi(lnz=[0. 0.])>
        >>> smoothed.data
        array([[0.4638, 0.4249, 0.3835],
               [0.4928, 0.4792, 0.4608],
               [0.5297, 0.5303, 0.5245]])

        """
        return self.new_like(base=self.base.pipe(func, *args, **kwargs))

    def _normalize_axes(self, axes: int | Iterable[int] | None) -> tuple[int, ...]:
        if axes is None:
            return tuple(range(self.ndim))
        if isinstance(axes, int):
            return (axes,)
        return tuple(axes)

    @docfiller.decorate
    def pad(
        self,
        axes: int | Iterable[int] | None = None,
        ffill: bool = True,
        bfill: bool = False,
        limit: None = None,
    ) -> Self:
        """
        Pad nan values in underlying data to values

        Parameters
        ----------
        {fill_axes}
        {ffill}
        {bfill}
        {fill_limit}

        Returns
        -------
        out : lnPiMasked
            Padded object.  Note that final result is the average over
            all axes with specified back and forward fill.
        """

        def _func(data: NDArrayAny, axes: tuple[int, ...]) -> NDArrayAny:
            from .core import array_utils

            datas: list[NDArrayAny] = []

            if ffill:
                datas += [
                    array_utils.ffill(data, axis=axis, limit=limit) for axis in axes
                ]
            if bfill:
                datas += [
                    array_utils.bfill(data, axis=axis, limit=limit) for axis in axes
                ]

            if not datas:
                return data

            if len(datas) == 1:
                return datas[0]

            import bottleneck

            return cast("NDArrayAny", bottleneck.nanmean(datas, axis=0))

        return self.pipe(_func, axes=self._normalize_axes(axes))

    @docfiller.decorate
    def interpolate_na(
        self,
        axes: int | Iterable[int] | None = None,
        add_coords: bool = False,
        method: InterpolationMethods = "linear",
        use_coordinate: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Interpolate ``np.nan`` values.

        Parameters
        ----------
        {fill_axes}
        add_coords: bool, default=False
            Some of the options require the underlying :class:`~xarray.DataArray` to
            have coordinates. Specify ``add_coords=True`` to enable these.
        method: str, default="linear"
            See :meth:`~xarray.DataArray.interpolate_na`
        use_coordinate: bool, default=False
            If True, use coordinates.  If False, assume evenly spaced.
        **kwargs
            Extra arguments to :meth:`~xarray.DataArray.interpolate_na`

        Returns
        -------
        out: object
            Object with `nan` values filled.

        See Also
        --------
        ~xarray.DataArray.interpolate_na
        """

        def _func(data: NDArrayAny, axes: tuple[int, ...]) -> NDArrayAny:
            da = xr.DataArray(data)
            if add_coords:
                da = da.assign_coords({k: range(int(da[k].max()) + 1) for k in da.dims})

            for axis in axes:
                da = da.interpolate_na(
                    dim=da.dims[axis],
                    method=method,
                    use_coordinate=use_coordinate,
                    **kwargs,
                )

            return da.to_numpy()

        return self.pipe(_func, axes=self._normalize_axes(axes))

    def mask_nan(self) -> Self:
        """Return new object with nan values masked."""
        return self.new_like(base=self.base, mask=np.isnan(self.base.data))

    def zeromax(self) -> Self:
        """Shift so that lnpi.max() == 0 on reference"""

        def _func(data: NDArrayAny) -> NDArrayAny:
            return cast("NDArrayAny", data - np.ma.MaskedArray(data, self.mask).max())

        return self.pipe(_func)

    def reweight(self, lnz: float | ArrayLike) -> Self:
        """Create new object at specified value of `lnz`"""
        return self.new_like(lnz=lnz)

    def or_mask(self, mask: NDArrayAny) -> Self:
        """New object with logical or of self.mask and mask"""
        return self.new_like(mask=(mask | self.mask))

    def and_mask(self, mask: NDArrayAny) -> Self:
        """New object with logical and of self.mask and mask"""
        return self.new_like(mask=(mask & self.mask))

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        lnz: float | ArrayLike,
        state_kws: dict[str, Any] | None = None,
        sep: str = r"\s+",
        names: Sequence[Hashable] | None = None,
        csv_kws: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        r"""
        Create lnPi object from text file table with columns [n_0,...,n_ndim, lnpi]

        Parameters
        ----------
        path : path-like
            file object to be read
        lnz : array-like
            :math:`\beta \mu` for each component
        state_kws : dict, optional
            define state variables, like volume, beta
        sep : string, optional
            separator for file read
        names : sequence of str
        csv_kws : dict, optional
            optional arguments to `pandas.read_csv`
        **kwargs
            Passed to lnPi constructor
        """
        lnz = np.atleast_1d(lnz)
        ndim = len(lnz)

        if names is None:
            names = [f"n_{i}" for i in range(ndim)] + ["lnpi"]

        kws: dict[str, Any] = {"sep": sep, "names": names}
        if csv_kws:
            kws.update(csv_kws)

        da = pd.read_csv(path, **kws).set_index(names[:-1])["lnpi"].to_xarray()
        # reindex n_{i}
        da = da.reindex({k: range(int(da[k].max()) + 1) for k in names[:-1]})

        return cls.from_data(
            data=da.values,
            mask=da.isnull().to_numpy(),  # noqa: PD003
            lnz=lnz,
            lnz_data=lnz,
            state_kws=state_kws,
            **kwargs,
        )

    @classmethod
    def from_dataarray(
        cls, da: xr.DataArray, state_as_attrs: bool | None = None, **kwargs: Any
    ) -> Self:
        """
        Create a lnPi object from :class:`xarray.DataArray`

        Parameters
        ----------
        da : DataArray
            DataArray containing the lnPi data
        state_as_attrs : bool, optional
            If True, get `state_kws` from ``da.attrs``.
        **kwargs
            Extra arguments to :meth:`from_data`

        Returns
        -------
        lnPiMasked


        See Also
        --------
        :meth:`from_data`

        """
        kws: dict[str, Any] = {}
        kws["data"] = da.to_numpy()
        if "mask" in da.coords:
            kws["mask"] = da.mask.to_numpy()
        else:
            kws["mask"] = da.isnull().to_numpy()  # noqa: PD003

        # where are state variables
        if state_as_attrs is None:
            state_as_attrs = bool(da.attrs.get("state_as_attrs", False))

        c = da.attrs if state_as_attrs else da.coords

        lnz = []
        state_kws = {}
        for k in da.attrs["dims_state"]:
            val = np.array(c[k])
            if "lnz" in k:
                lnz.append(val)
            else:
                if not val.ndim:
                    val = val[()]
                state_kws[k] = val
        kws["lnz"] = lnz
        kws["state_kws"] = state_kws

        kws["lnz_data"] = lnz

        # any overrides
        kwargs = dict(kws, **kwargs)
        return cls.from_data(**kwargs)

    @docfiller.decorate
    def list_from_masks(
        self, masks: Sequence[NDArrayAny], convention: MaskConvention = "image"
    ) -> list[Self]:
        """
        Create list of lnpis corresponding to masks[i]

        Parameters
        ----------
        {masks_masked}
        {mask_convention}

        Returns
        -------
        lnpis : list
            list of lnpis corresponding to each mask

        See Also
        --------
        lnpy.core.mask.masks_change_convention
        """
        return [
            self.or_mask(m) for m in masks_change_convention(masks, convention, False)
        ]

    @docfiller.decorate
    def list_from_labels(
        self,
        labels: NDArrayAny,
        features: Sequence[int] | None = None,
        include_boundary: bool = False,
        check_features: bool = True,
        **kwargs: Any,
    ) -> list[Self]:
        """
        Create sequence of lnpis from labels array.

        Parameters
        ----------
        {labels}
        {features}
        {include_boundary}
        {check_features}
        **kwargs
            Extra arguments to to :func:`~lnpy.core.mask.labels_to_masks`

        Returns
        -------
        outputs : list of lnPiMasked

        See Also
        --------
        lnPiMasked.list_from_masks
        lnpy.core.mask.labels_to_masks
        """
        masks, features = labels_to_masks(
            labels=labels,
            features=features,
            include_boundary=include_boundary,
            convention=False,
            check_features=check_features,
            **kwargs,
        )
        return self.list_from_masks(masks, convention=False)

    @cached.prop
    @docfiller.decorate
    def xge(self) -> ensembles.GrandCanonicalEnsemble:
        """{accessor.xge}"""
        from .ensembles import GrandCanonicalEnsemble

        return GrandCanonicalEnsemble(self)

    @cached.prop
    @docfiller.decorate
    def xce(self) -> ensembles.CanonicalEnsemble:
        """{accessor.xce}"""
        from .ensembles import CanonicalEnsemble

        return CanonicalEnsemble(self)
