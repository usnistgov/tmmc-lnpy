"""
lnPi data classes and routines (:mod:`~lnpy.lnpidata`)
======================================================
"""
################################################################################
# Delayed
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Iterable

from module_utilities import cached

from ._lazy_imports import np, pd
from .docstrings import docfiller
from .extensions import AccessorMixin
from .utils import labels_to_masks, masks_change_convention

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Mapping, Sequence

    import xarray as xr
    from numpy.typing import ArrayLike, DTypeLike, NDArray
    from typing_extensions import Self

    from . import ensembles
    from ._typing import MaskConvention, MyNDArray


@lru_cache(maxsize=20)
def _get_n_ranges(shape: tuple[int, ...], dtype: DTypeLike) -> list[MyNDArray]:
    return [np.arange(s, dtype=dtype) for s in shape]


@lru_cache(maxsize=20)
def _get_shift(
    shape: tuple[int, ...], dlnz: tuple[float, ...], dtype: DTypeLike
) -> MyNDArray:
    shift = np.zeros([], dtype=dtype)
    for i, (nr, m) in enumerate(zip(_get_n_ranges(shape=shape, dtype=dtype), dlnz)):  # type: ignore
        shift = np.add.outer(shift, nr * m)  # pyright: ignore
    return shift


@lru_cache(maxsize=20)
def _get_data(base: lnPiArray, dlnz: tuple[float, ...]) -> MyNDArray:
    if all(x == 0 for x in dlnz):
        return base.data  # pyright: ignore
    else:
        return (
            _get_shift(base.shape, dlnz, base.data.dtype) + base.data
        )  # pyright: ignore


@lru_cache(maxsize=20)
def _get_maskedarray(
    base: lnPiArray, self: lnPiMasked, dlnz: tuple[float, ...]
) -> np.ma.core.MaskedArray[Any, np.dtype[Any]]:
    return np.ma.MaskedArray(  # type: ignore
        _get_data(base, dlnz),
        mask=self._mask,
        fill_value=base.fill_value,  # pyright: ignore
    )


@lru_cache(maxsize=20)
def _get_filled(
    base: lnPiArray,
    self: lnPiMasked,
    dlnz: tuple[float, ...],
    fill_value: float | None = None,
) -> np.ma.core.MaskedArray[Any, np.dtype[Any]]:
    return _get_maskedarray(base, self, dlnz).filled(fill_value)  # type: ignore


class lnPiArray:
    """
    Wrapper on lnPi lnPiArray

    Parameters
    ----------
    lnz : float or sequence of float
    """

    @docfiller.decorate
    def __init__(
        self,
        lnz: float | ArrayLike,
        data: MyNDArray,
        state_kws: dict[str, Any] | None = None,
        extra_kws: dict[str, Any] | None = None,
        fill_value: float | None = None,
        copy: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        {lnz}
        {data}
        {state_kws}
        {extra_kws}
        {fill_value}
        {copy}
        """

        lnz = np.atleast_1d(lnz)
        data = np.array(data, copy=copy)
        assert data.ndim == len(lnz)

        fill_value = fill_value or np.nan

        if state_kws is None:
            state_kws = {}
        if extra_kws is None:
            extra_kws = {}

        self.data = data
        # make data read only
        self.data.flags.writeable = False

        self.state_kws = state_kws
        self.extra_kws = extra_kws

        self.lnz = lnz
        self.fill_value = fill_value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def new_like(
        self,
        lnz: float | ArrayLike | None = None,
        data: MyNDArray | None = None,
        copy: bool = False,
    ) -> Self:
        """
        Create new object with optional replacements.

        All parameters are optional.  If not passed, use values in `self`

        Parameters
        ----------
        {lnz}
        {data}
        {copy}

        Returns
        -------
        out : lnPiArray
            New object with optionally updated parameters.

        """
        if lnz is None:
            lnz = self.lnz
        if data is None:
            data = self.data

        return type(self)(
            lnz=lnz,
            data=data,
            copy=copy,
            state_kws=self.state_kws,
            extra_kws=self.extra_kws,
            fill_value=self.fill_value,
        )

    @docfiller.decorate
    def pad(
        self,
        axes: int | Iterable[int] | None = None,
        ffill: bool = True,
        bfill: bool = False,
        limit: int | None = None,
    ) -> Self:
        """
        Pad nan values in underlying data to values

        Parameters
        ----------
        {ffill}
        {bfill}
        {fill_limit}

        Returns
        -------
        out : lnPiArray
            object with padded data
        """

        import bottleneck  # pyright: ignore

        from . import utils

        if axes is None:
            axes = range(self.data.ndim)
        elif isinstance(axes, int):
            axes = (axes,)

        data = self.data
        datas: list[MyNDArray] = []

        if ffill:
            datas += [utils.ffill(data, axis=axis, limit=limit) for axis in axes]
        if bfill:
            datas += [utils.bfill(data, axis=axis, limit=limit) for axis in axes]

        if len(datas) > 0:
            data = bottleneck.nanmean(datas, axis=0)

        new = self.new_like(data=data)
        return new

    def zeromax(self, mask: MyNDArray | bool = False) -> Self:
        """
        Shift values such that lnpi.max() == 0

        Parameters
        ----------
        mask : bool or array-like of bool
            Optional mask to apply to data.  Where `mask` is True,
            data is excluded from calculating maximum.
        """

        data = self.data - np.ma.MaskedArray(self.data, mask).max()  # type: ignore
        return self.new_like(data=data)


@docfiller.decorate
class lnPiMasked(AccessorMixin):
    """
    Masked array like wrapper for lnPi data.

    This is the basic data structure for storing the output from a single TMMC simulation.

    Parameters
    ----------
    {lnz}
    {base}
    {mask_masked}
    {copy}

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

    See Also
    --------
    numpy.ma.MaskedArray
    """

    _DataClass = lnPiArray

    def __init__(
        self,
        lnz: float | ArrayLike,
        base: lnPiArray,
        mask: MyNDArray | None = None,
        copy: bool = False,
    ) -> None:
        lnz = np.atleast_1d(lnz)
        assert lnz.shape == base.lnz.shape

        if mask is None:
            mask = np.full(base.data.shape, fill_value=False, dtype=bool)
        else:
            mask = np.array(mask, copy=copy, dtype=bool)
        assert mask.shape == base.data.shape

        self._mask = mask
        # make mask read-only
        self._mask.flags.writeable = False

        self._base = base
        self._lnz = lnz
        self._dlnz = tuple(self._lnz - self._base.lnz)

        self._cache: dict[str, Any] = {}

    @classmethod
    def from_data(
        cls,
        lnz: float | ArrayLike,
        lnz_data: float | ArrayLike,
        data: MyNDArray,
        mask: MyNDArray | None = None,
        state_kws: dict[str, Any] | None = None,
        extra_kws: dict[str, Any] | None = None,
        fill_value: float | None = None,
        copy: bool = False,
    ) -> Self:
        """
        Create :class:`lnPiMasked` object from raw data.

        Parameters
        ----------
        lnz : float or sequence of float
            Value of `lnz` to reweight data to.
        lnz_data : float or sequence of float
            Value of `lnz` at which `data` was collected
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

        base = cls._DataClass(
            lnz=lnz_data,
            data=data,
            state_kws=state_kws,
            extra_kws=extra_kws,
            fill_value=fill_value,
            copy=copy,
        )
        return cls(lnz=lnz, base=base, mask=mask, copy=copy)

    @property
    def _data(self) -> MyNDArray:
        return self._base.data  # pyright: ignore

    @property
    def dtype(self) -> np.dtype[Any]:
        """Type (dtype) of underling data"""
        return self._data.dtype

    def _clear_cache(self) -> None:
        self._cache = {}

    @property
    def state_kws(self) -> dict[str, Any]:
        """State variables."""
        return self._base.state_kws

    @property
    def extra_kws(self) -> dict[str, Any]:
        """Extra parameters."""
        return self._base.extra_kws

    @property
    def ma(self) -> np.ma.core.MaskedArray[Any, np.dtype[Any]]:
        """Masked array view of data reweighted data"""
        return _get_maskedarray(self._base, self, self._dlnz)

    def filled(self, fill_value: float | None = None) -> MyNDArray:
        """Filled view or reweighted data"""
        return _get_filled(self._base, self, self._dlnz, fill_value)

    @property
    def data(self) -> MyNDArray:
        """Reweighted data"""
        return _get_data(self._base, self._dlnz)

    @property
    def mask(self) -> MyNDArray:
        """Where `True`, values are masked out."""
        return self._mask

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of lnPiArray"""
        return self._data.shape

    def __len__(self) -> int:
        return len(self._data)

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def lnz(self) -> MyNDArray:
        """Value of log(activity) evaluated at"""
        return self._lnz

    @property
    def betamu(self) -> MyNDArray:
        """Alias to `self.lnz`"""
        return self._lnz

    @property
    def volume(self) -> float | None:
        """Accessor to self.state_kws['volume']."""
        return self.state_kws.get("volume", None)

    @property
    def beta(self) -> float | None:
        """Accessor to self.state_kws['beta']."""
        return self.state_kws.get("beta", None)

    def __repr__(self) -> str:
        return f"<lnPi(lnz={self._lnz})>"

    def __str__(self) -> str:
        return repr(self)

    def _index_dict(self, phase: int | str | None = None) -> dict[str, Any]:
        out = {f"lnz_{i}": v for i, v in enumerate(self.lnz)}
        if phase is not None:
            out["phase"] = phase
        # out.update(**self.state_kws)
        return out

    # Parameters for xlnPi
    def _lnpi_tot(self, fill_value: float | None = None) -> MyNDArray:
        return self.filled(fill_value)

    def _pi_params(
        self, fill_value: float | None = None
    ) -> tuple[MyNDArray, float, float]:
        lnpi = self._lnpi_tot(fill_value)

        lnpi_local_max = lnpi.max()
        pi = np.exp(lnpi - lnpi_local_max)
        pi_sum = pi.sum()
        pi_norm = pi / pi_sum

        lnpi_zero = self.data.ravel()[0] - lnpi_local_max

        return pi_norm, pi_sum, lnpi_zero

    @property
    def _lnz_tot(self) -> MyNDArray:
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
        return np.unravel_index(self.ma.argmax(*args, **kwargs), self.shape)  # type: ignore

    # @cached.meth
    def local_max(
        self, *args: Any, **kwargs: Any
    ) -> np.ma.core.MaskedArray[Any, np.dtype[Any]]:
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
        return self.ma[self.local_argmax(*args, **kwargs)]  # type: ignore

    # @cached.meth
    def local_maxmask(
        self, *args: Any, **kwargs: Any
    ) -> np.ma.core.MaskedArray[Any, np.dtype[Any]]:
        """Calculate mask where ``self.ma == self.local_max()``"""
        return self.ma == self.local_max(*args, **kwargs)  # type: ignore

    @cached.prop
    def edge_distance_matrix(self) -> NDArray[np.float_]:
        """
        Matrix of distance from each element to a background (i.e., masked) point.

        See Also
        --------
        lnpy.utils.distance_matrix
        """
        from .utils import distance_matrix

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
        lnpy.utils.distance_matrix
        """
        return ref.edge_distance_matrix[self.local_argmax(*args, **kwargs)]  # type: ignore

    @docfiller.decorate
    def new_like(
        self,
        lnz: float | ArrayLike | None = None,
        base: lnPiArray | None = None,
        mask: MyNDArray | None = None,
        copy: bool = False,
    ) -> Self:
        """
        Create new object with optional parameters

        Parameters
        ----------
        {lnz}
        {base}
        {mask_masked}
        {copy}
        """

        if lnz is None:
            lnz = self._lnz
        if base is None:
            base = self._base
        if mask is None:
            mask = self._mask

        return type(self)(lnz=lnz, base=base, mask=mask, copy=copy)

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
        {ffill}
        {bfill}
        {fill_limit}
        inplace : bool, default=False

        Returns
        -------
        out : lnPiMasked
            padded object


        See Also
        --------
        lnPiArray.pad
        """

        base = self._base.pad(axes=axes, ffill=ffill, bfill=bfill, limit=limit)
        return self.new_like(base=base)

    def zeromax(self) -> Self:
        """
        Shift so that lnpi.max() == 0 on reference

        See Also
        --------
        lnPiArray.zeromax
        """

        base = self._base.zeromax(mask=self._mask)
        return self.new_like(base=base)

    def reweight(self, lnz: float | ArrayLike) -> Self:
        """Create new object at specified value of `lnz`"""
        return self.new_like(lnz=lnz)

    def or_mask(self, mask: MyNDArray, **kwargs: Any) -> Self:
        """New object with logical or of self.mask and mask"""
        return self.new_like(mask=(mask | self.mask))

    def and_mask(self, mask: MyNDArray, **kwargs: Any) -> Self:
        """New object with logical and of self.mask and mask"""
        return self.new_like(mask=(mask & self.mask))

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        lnz: float | ArrayLike,
        state_kws: dict[str, Any] | None = None,
        sep: str = r"\s+",
        names: Sequence[str] | None = None,
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

        if csv_kws is None:
            csv_kws = {}

        da = (
            pd.read_csv(path, sep=sep, names=names, **csv_kws)  # type: ignore
            .set_index(names[:-1])["lnpi"]
            .to_xarray()
        )
        return cls.from_data(
            data=da.values,
            mask=da.isnull().values,
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
        kws["data"] = da.values
        if "mask" in da.coords:
            kws["mask"] = da.mask.values
        else:
            kws["mask"] = da.isnull().values

        # where are state variables
        if state_as_attrs is None:
            state_as_attrs = bool(da.attrs.get("state_as_attrs", False))
        if state_as_attrs:
            # state variables from attrs
            c = da.attrs
        else:
            c = da.coords  # type: ignore [assignment]

        lnz = []
        state_kws = {}
        for k in da.attrs["dims_state"]:
            val = np.array(c[k])
            if "lnz" in k:
                lnz.append(val)
            else:
                if val.ndim == 0:
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
        self, masks: Sequence[MyNDArray], convention: MaskConvention = "image"
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
        lnpy.utils.masks_change_convention
        """

        return [
            self.or_mask(m) for m in masks_change_convention(masks, convention, False)
        ]

    @docfiller.decorate
    def list_from_labels(
        self,
        labels: MyNDArray,
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
            Extra arguments to to :func:`~lnpy.utils.labels_to_masks`

        Returns
        -------
        outputs : list of lnPiMasked

        See Also
        --------
        lnPiMasked.list_from_masks
        lnpy.utils.labels_to_masks
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
    def xge(self) -> ensembles.xGrandCanonical:
        """{accessor.xge}"""
        from .ensembles import xGrandCanonical

        return xGrandCanonical(self)

    @cached.prop
    @docfiller.decorate
    def xce(self) -> ensembles.xCanonical:
        """{accessor.xce}"""
        from .ensembles import xCanonical

        return xCanonical(self)


# --- Register accessors ---------------------------------------------------------------
# lnPiMasked.register_accessor("xge", xge_accessor)
# lnPiMasked.register_accessor("xce", xce_accessor)


# reveal_type(lnPiMasked.list_from_labels)
# reveal_type(lnPiMasked.xge)
