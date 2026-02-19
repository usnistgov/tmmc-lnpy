# pyright: reportPrivateUsage=false
"""
Ensemble averages (:mod:`~lnpy.ensembles`)
==========================================
"""

from __future__ import annotations

from functools import lru_cache, wraps
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from module_utilities import cached

from .core.compat import xr_dot
from .core.xr_utils import dim_to_suffix_dataset
from .lnpidata import lnPiMasked
from .lnpiseries import lnPiCollection

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence
    from typing import Any

    import pandas as pd
    from numpy.typing import ArrayLike, NDArray

    from .core.typing import C_Ensemble, EnsembleT, NDArrayAny, P, R, XArrayLike
    from .core.typing_compat import IndexAny

# always check_use_cache here.
cached_prop = cached.prop(check_use_cache=True)
cached_meth = cached.meth(check_use_cache=True)


# * Utilities -----------------------------------------------------------------
@lru_cache(maxsize=10)
def _get_indices(shape: tuple[int, ...]) -> NDArray[np.int64]:
    return np.indices(shape)


@lru_cache(maxsize=10)
def _get_range(n: int) -> NDArray[np.int64]:
    return np.arange(n)


@lru_cache(maxsize=10)
def _get_xrlnpiwrapper(
    shape: tuple[int, ...],
    rec_name: str | None = "sample",
    n_name: str = "n",
    lnz_name: str = "lnz",
    comp_name: str = "component",
    phase_name: str = "phase",
) -> LogPiWrapper:
    return LogPiWrapper(
        shape=shape,
        rec_name=rec_name,
        n_name=n_name,
        lnz_name=lnz_name,
        comp_name=comp_name,
        phase_name=phase_name,
    )


def xr_name(
    long_name: str | None = None,
    name: str | None = None,
    unstack: bool = True,
    **kws: Any,
) -> Callable[
    [C_Ensemble[EnsembleT, P, xr.DataArray]], C_Ensemble[EnsembleT, P, xr.DataArray]
]:
    """Decorator to add name, longname to xarray output"""

    def decorator(
        func: C_Ensemble[EnsembleT, P, xr.DataArray],
    ) -> C_Ensemble[EnsembleT, P, xr.DataArray]:
        name_ = func.__name__.lstrip("_") if name is None else name

        @wraps(func)
        def wrapper(
            self: EnsembleT, /, *args: P.args, **kwargs: P.kwargs
        ) -> xr.DataArray:
            out = func(self, *args, **kwargs).rename(name_)
            attrs = dict(getattr(self, "_standard_attrs", {}), **kws)
            if long_name is not None:
                attrs["long_name"] = long_name
            out = out.assign_attrs(**attrs)
            if unstack and self._xarray_unstack:
                out = out.unstack()  # noqa: PD010
            return out

        return wrapper

    return decorator


# * lnPi wrapper --------------------------------------------------------------
class LogPiWrapper:
    """
    Wraps lnPi objects with xarray functionality

    Most likely, this shouldn't be accessed by the user.
    """

    _use_cache = True

    def __init__(
        self,
        shape: tuple[int, ...],
        rec_name: str | None = "sample",
        n_name: str = "n",
        lnz_name: str = "lnz",
        comp_name: str = "component",
        phase_name: str = "phase",  # noqa: ARG002
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)

        # dimension names
        self.dims_rec: list[str]
        if rec_name is None:
            self.dims_rec = []
        else:
            self.dims_rec = [rec_name]

        self.n_name = n_name
        self.dims_n = [f"{n_name}_{i}" for i in range(self.ndim)]
        self.dims_lnz = [f"{lnz_name}_{i}" for i in range(self.ndim)]
        self.dims_comp = [comp_name]
        self._cache: dict[str, Any] = {}

    @cached_prop
    def coords_n(self) -> dict[str, xr.DataArray]:
        return {
            k: xr.DataArray(_get_range(n), dims=k, attrs={"long_name": rf"${k}$"})
            for k, n in zip(self.dims_n, self.shape, strict=True)
        }

    @cached_meth
    def attrs(self, *args: str) -> dict[str, list[str]]:
        d = {
            "dims_n": self.dims_n,
            "dims_lnz": self.dims_lnz,
            "dims_comp": self.dims_comp,
            "dims_state": self.dims_lnz + list(args),
        }

        if self.dims_rec:
            d["dims_rec"] = self.dims_rec
        return d

    @cached_meth
    def ncoords(self, coords_n: bool = False) -> xr.DataArray:
        coords = self.coords_n if coords_n else None
        return xr.DataArray(
            _get_indices(self.shape),
            dims=self.dims_comp + self.dims_n,
            coords=coords,
            name=self.n_name,
        )

    @cached_meth
    def ncoords_tot(self, coords_n: bool = False) -> xr.DataArray:
        return self.ncoords(coords_n).sum(self.dims_comp)

    def wrap_data(
        self,
        data: NDArrayAny,
        coords_n: bool = False,
        coords: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        if coords is None:
            coords = {}

        if coords_n:
            coords = dict(coords, **self.coords_n)  # pylint: disable=not-a-mapping

        return xr.DataArray(
            data,
            dims=self.dims_rec + self.dims_n,
            coords=coords,
            attrs=self.attrs(*kwargs.keys()),
        )

    def wrap_lnpi(
        self, lnpi: NDArrayAny, coords_n: bool = False, **kwargs: Any
    ) -> xr.DataArray:
        return self.wrap_data(lnpi, coords_n=coords_n, **kwargs)

    def wrap_lnz(self, lnz: NDArrayAny, **kwargs: Any) -> xr.DataArray:
        return xr.DataArray(lnz, dims=self.dims_rec + self.dims_comp, **kwargs)

    def wrap_lnpi_0(self, lnpi_0: float | NDArrayAny, **kwargs: Any) -> xr.DataArray:
        return xr.DataArray(lnpi_0, dims=self.dims_rec, **kwargs)


# * Grandcanonical ensemble (xge) ---------------------------------------------
class GrandCanonicalEnsemble:  # noqa: PLR0904
    """
    :class:`~xarray.DataArray` accessor to Grand Canonical properties from lnPi


    This class is primarily interacted with through the attributes ``xge`` attached
    to :class:`~lnpy.lnpidata.lnPiMasked` and :class:`~lnpy.lnpiseries.lnPiCollection`.
    """

    def __init__(self, parent: lnPiCollection | lnPiMasked) -> None:
        self._parent = parent

        self._rec_name: str | None
        first: lnPiMasked
        if isinstance(parent, lnPiCollection):
            self._rec_name = parent._concat_dim
            first = parent._series.iloc[0]
        else:
            self._rec_name = None
            first = parent

        self._wrapper = _get_xrlnpiwrapper(shape=first.shape, rec_name=self._rec_name)
        self._cache: dict[str, Any] = {}

    @property
    def _use_cache(self) -> bool:
        return getattr(self._parent, "_use_cache", False)

    @property
    def _xarray_unstack(self) -> bool:
        return getattr(self._parent, "_xarray_unstack", True)

    @cached_prop
    def _standard_attrs(self) -> dict[str, Any]:
        return self._wrapper.attrs(*self._parent.state_kws.keys())

    @property
    def first(self) -> lnPiMasked:
        if isinstance(self._parent, lnPiCollection):
            return self._parent.iloc[0]
        return self._parent

    # @cached_prop to much memory
    @property
    def _rec_coords(self) -> dict[str, IndexAny | pd.MultiIndex | float | int]:
        if isinstance(self._parent, lnPiMasked):
            return dict(self._parent._index_dict(), **self._parent.state_kws)

        return {
            self._parent._concat_dim: self._parent.index,
            **self._parent.state_kws,
        }

    @property
    def _xarray_dot_kws(self) -> dict[str, str]:
        return getattr(self._parent, "_xarray_dot_kws", {})

    @property
    def ncoords(self) -> xr.DataArray:
        """Particle number coordinates."""
        return self._wrapper.ncoords()

    @property
    def ncoords_tot(self) -> xr.DataArray:
        """Total number of particles coordinates"""
        return self._wrapper.ncoords_tot()

    @property
    def dims_n(self) -> list[str]:
        """Dimension(s) corresponding to  number of particles"""
        return self._wrapper.dims_n

    @property
    def dims_lnz(self) -> list[str]:
        r"""Dimension corresponding to values of :math:`\ln z`"""
        return self._wrapper.dims_lnz

    @property
    def dims_comp(self) -> list[str]:
        """Dimension of component number"""
        return self._wrapper.dims_comp

    @property
    def dims_state(self) -> list[str]:
        """Dimensions corresponding to 'state variables'"""
        return self.pi_norm.attrs["dims_state"]  # type: ignore[no-any-return]

    @property
    def dims_rec(self) -> list[str]:
        """Dimensions for replicates"""
        return self._wrapper.dims_rec

    @property
    def beta(self) -> float:
        r"""Inverse temperature :math:`\beta = 1 / (k_{\rm B} T)`"""
        return self._parent.state_kws["beta"]  # type: ignore[no-any-return]

    @property
    def volume(self) -> float:
        """System volume :math:`V`."""
        return self._parent.state_kws["volume"]  # type: ignore[no-any-return]

    @cached_prop
    def coords_state(self) -> dict[str, xr.DataArray]:
        return {k: self.pi_norm.coords[k] for k in self.dims_state}

    @cached_prop
    @xr_name(r"$\beta {\bf \mu}$")
    def betamu(self) -> xr.DataArray:
        r"""Scaled chemical potential :math:`\beta \mu`"""
        return self._wrapper.wrap_lnz(self._parent._lnz_tot, coords=self._rec_coords)

    @property
    @xr_name(r"$\ln\beta{\bf\mu}$")
    def lnz(self) -> xr.DataArray:
        r"""Log of activity :math:`\ln z``"""
        return self.betamu

    @xr_name(r"$\ln \Pi(n,\mu,V,T)$", unstack=False)
    def lnpi(self, fill_value: float | None = None) -> xr.DataArray:
        r"""
        :class:`xarray.DataArray` view of :math:`\ln \Pi(N)`

        Notes
        -----
        This value is always in 'stacked' form.
        You must manually unstack it.
        """
        return self._wrapper.wrap_lnpi(
            self._parent._lnpi_tot(fill_value),
            coords=self._rec_coords,
            **self._parent.state_kws,
        )  # .assign_coords(**self._rec_coords)

    @cached_prop
    def _pi_params(self) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        pi_norm, pi_sum, lnpi_zero = self._parent._pi_params(-np.inf)

        return (
            self._wrapper.wrap_lnpi(
                pi_norm, coords=self._rec_coords, **self._parent.state_kws
            ),
            self._wrapper.wrap_lnpi_0(pi_sum, name="pi_sum", coords=self._rec_coords),
            self._wrapper.wrap_lnpi_0(
                lnpi_zero, name="lnpi_zero", coords=self._rec_coords
            ),
        )

    @property
    def pi_norm(self) -> xr.DataArray:
        r"""Normalized value of :math:`\Pi_{\rm norm}(N) = \Pi(N) / \sum_N \Pi(N)`"""
        return self._pi_params[0]

    @property
    def pi_sum(self) -> xr.DataArray:
        r"""Sum of unnormalized :math:`\Pi(N)`"""
        return self._pi_params[1]

    @property
    def _lnpi_zero(self) -> xr.DataArray:
        return self._pi_params[2]

    @property
    def lnpi_norm(self) -> xr.DataArray:
        r""":math:`\ln \Pi_{\rm norm}(N)`."""
        pi = self.pi_norm
        return np.log(xr.where(pi > 0, pi, np.nan))  # type: ignore[no-untyped-call,no-any-return]

    def _get_prop_from_extra_kws(
        self, prop: str | ArrayLike | xr.DataArray
    ) -> ArrayLike | xr.DataArray:
        if isinstance(prop, str):
            p = self.first

            if prop not in p.extra_kws:
                msg = f"{prop} not found in extra_kws"
                raise ValueError(msg)

            prop = p.extra_kws[prop]

        return prop

    def _array_or_callable_to_xarray(
        self,
        func_or_array: str | Callable[..., XArrayLike] | ArrayLike | xr.DataArray,
        allow_extra_kws: bool = True,
        **kwargs: Any,
    ) -> xr.DataArray:
        if callable(func_or_array):
            x = func_or_array(self, **kwargs)
        elif allow_extra_kws:
            x = self._get_prop_from_extra_kws(func_or_array)
        else:
            if isinstance(func_or_array, str):
                raise TypeError
            x = func_or_array

        if not isinstance(x, xr.DataArray):
            x = xr.DataArray(x, dims=self.dims_n)
        return x

    def _mean_pi(self, x: xr.DataArray, **kwargs: Any) -> xr.DataArray:  # noqa: ARG002
        return xr_dot(self.pi_norm, x, dim=self.dims_n, **self._xarray_dot_kws)  # type: ignore[no-any-return]

    @xr_name()
    def mean_pi(
        self,
        /,
        x: str | XArrayLike | Callable[..., XArrayLike],
        allow_extra_kws: bool = True,
        **kwargs: Any,
    ) -> xr.DataArray:
        r"""
        Calculates :math:`\overline{x} = \sum_N \Pi_{\rm norm}(N) x(N)`

        Parameters
        ----------
        x : array, DataArray, callable, or str
            If callable, should have form ``x = x(self, **kwargs)``.
            If string, then set `x = self.parent.extra_kws[x]`.
            Otherwise, should be an array of same shape as single lnPi.
            If x (or result of callable) is not a :class:`~xarray.DataArray`, try to
            convert it to one.
        allow_extra_kws : bool
            If `True`, allow getting property from `extra_kws`.
        *args, **kwargs
            Extra arguments to `x` if passing callable

        Returns
        -------
        ave : DataArray
            x can be an array or a callable of the form
            ``f(self, *args, **kwargs)``

        """
        x = self._array_or_callable_to_xarray(
            x, allow_extra_kws=allow_extra_kws, **kwargs
        )
        return self._mean_pi(x, **kwargs)

    # TODO(wpk): finish this
    # def _central_moment_bar(
    #     self,
    #     x,
    #     y=None,  # noqa: ERA001
    #     xmom=2,  # noqa: ERA001
    #     ymom=None,  # noqa: ERA001
    #     xmom_dim="xmom",  # noqa: ERA001
    #     ymom_dim="ymom",  # noqa: ERA001
    #     *args,
    #     **kwargs,
    # ):
    #     r"""
    #     Calculate central moments of the form

    #     .. math::

    #         \sum_N \Pi(N)  (\bar{x}(N) - \langle \bar{x}(N) \rangle))^n

    #     """
    #     def _get_dx(xx, mom, mom_dim):
    #         xx = self._array_or_callable_to_xarray(xx, *args, **kwargs)  # noqa: ERA001
    #         if isinstance(mom, int):
    #             mom = [mom]  # noqa: ERA001
    #         mom = xr.DataArray(mom, dims=mom_dim, coords={mom_dim: mom})  # noqa: ERA001
    #         return (xx - self._mean_pi(xx)) ** mom  # noqa: ERA001

    #     dx = _get_dx(x, xmom, xmom_dim)  # noqa: ERA001
    #     if y is not None:
    #         dy = _get_dx(y, ymom, ymom_dim)  # noqa: ERA001
    #         dx = dx * dy  # noqa: ERA001
    #     return self._mean_pi(dx)  # noqa: ERA001

    def var_pi(
        self,
        x: XArrayLike | Callable[..., XArrayLike],
        y: ArrayLike | xr.DataArray | Callable[..., XArrayLike] | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        r"""
        Calculate Grand Canonical variance from canonical properties.

        Given x(N) and y(N), calculate

        .. math::

            {\rm var}(x, y) = \overline{(x - \overline{x}) (y - \overline{y})}

        ``x`` and ``y`` can be arrays, or callables, in which case:
        ``x = x(self, **kwargs)``

        See Also
        --------
        :math:`mean_pi`

        """
        x = self._array_or_callable_to_xarray(x, **kwargs)

        # this cuts down on memory usage
        xx = x - self._mean_pi(x)
        if y is None:
            yy = xx
        else:
            y = self._array_or_callable_to_xarray(x, **kwargs)
            yy = y - self._mean_pi(y)

        return xr_dot(self.pi_norm, xx, yy, dim=self.dims_n, **self._xarray_dot_kws)  # type: ignore[no-any-return]

    def pipe(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Apply function to `self`"""
        return func(self, *args, **kwargs)

    @cached_prop
    @xr_name(r"${\bf n}(\mu,V,T)$")
    def nvec(self) -> xr.DataArray:
        r"""Average number of particles of each component :math:`\overline{{\bf N}}`"""
        return self._mean_pi(self.ncoords)

    @cached_prop
    @xr_name(r"$n(\mu,V,T)$")
    def ntot(self) -> xr.DataArray:
        r"""Average total number of particles :math:`\overline{N}`"""
        return self._mean_pi(self.ncoords_tot)

    @property
    @xr_name(r"${\bf x}(\mu,V,T)$")
    def molfrac(self) -> xr.DataArray:
        r"""Average molfrac for each components :math:`{\bf x} = \overline{{\bf N}} / N`"""
        x = self.nvec
        return x / x.sum(self.dims_comp)

    @cached_prop
    @xr_name(r"$var[{\bf n}(\mu,V,T)]$")
    def nvec_var(self) -> xr.DataArray:
        r"""Variance in particle number :math:`{\rm var}\, \bf{N}`"""
        return self.var_pi(self.ncoords)

    @cached_prop
    @xr_name(r"$var[n(\mu,V,T)]$")
    def ntot_var(self) -> xr.DataArray:
        r"""Variance in total number of particles :math:`{\rm var}\, N`"""
        return self.var_pi(self.ncoords_tot)

    @property
    @xr_name(r"${\bf \rho}(\mu,V,T)$")
    def dens(self) -> xr.DataArray:
        r"""Density of each component :math:`{\bf \rho} = \overline{{\bf N}} / V`"""
        # NOTE: keep this here because of some internal calculations
        return self.nvec / self.volume

    @property
    @xr_name(r"$\rho(\mu,V,T)$", standard_name="total_density")
    def dens_tot(self) -> xr.DataArray:
        r"""Total density :math:`\overline{N} / V`"""
        return self.ntot / self.volume

    def max(self) -> xr.DataArray:
        r""":math:`\max_N \ln \Pi(N)`"""
        return self.lnpi().max(self.dims_n)

    @cached_meth
    def _argmax_indexer(self) -> tuple[NDArrayAny, ...]:
        x: NDArrayAny = self.pi_norm.to_numpy()
        xx = x.reshape(x.shape[0], -1)
        idx_flat = xx.argmax(-1)
        return np.unravel_index(idx_flat, x.shape[1:])

    @cached_prop
    def _argmax_indexer_dict(self) -> dict[str, xr.DataArray]:
        if not isinstance(self._parent, lnPiCollection):
            msg = "only implemented for lnPiCollection"
            raise TypeError(msg)

        return {
            k: xr.DataArray(v, dims=self._parent._concat_dim)
            for k, v in zip(self.dims_n, self._argmax_indexer(), strict=True)
        }

    @cached_prop
    def _sample_indexer_dict(self) -> dict[str, xr.DataArray]:
        if not isinstance(self._parent, lnPiCollection):
            msg = "only implemented for lnPiCollection"
            raise TypeError(msg)

        return {
            self._parent._concat_dim: xr.DataArray(
                range(len(self._parent)), dims=self._parent._concat_dim
            )
        }

    @property
    def _sample_argmax_indexer_dict(self) -> dict[str, xr.DataArray]:
        return dict(self._argmax_indexer_dict, **self._sample_indexer_dict)  # pylint: disable=not-a-mapping

    def lnpi_max(
        self, fill_value: float | None = None, add_n_coords: bool = True
    ) -> xr.DataArray:
        r"""Maximum value of :math:`\max_N \ln \Pi(N, ...)`"""
        out = self.lnpi(fill_value).isel(self._sample_argmax_indexer_dict)

        # NOTE : This assumes each n value corresponds to index
        # alternatively, could put through filter like
        # coords = {k : out[k].isel(**{k : v}) for k, v in self._argmax_index_dict.items()}  # noqa: ERA001

        if add_n_coords:
            out = out.assign_coords(self._argmax_indexer_dict)

        return out

    def pi_norm_max(self, add_n_coords: bool = True) -> xr.DataArray:
        r"""Maximum value :math:`\max_N \Pi_{\rm norm}(N, meta)`"""
        out = self.pi_norm.isel(self._sample_argmax_indexer_dict)
        if add_n_coords:
            out = out.assign_coords(self._argmax_indexer_dict)
        return out

    @cached_meth
    def argmax(self) -> NDArrayAny:
        return np.array(self._argmax_indexer()).T

    @xr_name("distance from upper edge")
    def edge_distance(self, ref: lnPiMasked) -> xr.DataArray:
        """Distance from argmax(lnPi) to endpoint"""
        out = ref.edge_distance_matrix[self._argmax_indexer()]
        return xr.DataArray(out, dims=self.dims_rec, coords=self._rec_coords)

    @xr_name("distance from edge of cut value")
    def edge_distance_val(
        self,
        ref: lnPiMasked,
        val: float | xr.DataArray | None = None,
        max_frac: float | None = None,
    ) -> xr.DataArray:
        """
        Calculate min distance from where self.pi_norm > val to edge

        Parameters
        ----------
        ref : lnPiMasked
            reference object to consider.
        val : float

        max_frac : bool, optional
            if not None, val = max_frac * self.pi_norm.max(self.dims_n)
        """
        if max_frac is not None:
            if not (0.0 < max_frac < 1.0):
                msg = f"{max_frac=} outside range (0.0, 1.0)."
                raise ValueError(msg)
            val = self.pi_norm_max(add_n_coords=False) * max_frac
        elif val is None:
            msg = "If not passing max_frac, must set `val`"
            raise TypeError(msg)

        e = xr.DataArray(ref.edge_distance_matrix, dims=self.dims_n)
        mask = self.pi_norm > val
        return e.where(mask).min(self.dims_n)

    @cached_meth
    @xr_name(r"$\beta \Omega(\mu,V,T)$", standard_name="grand_potential")
    def _betaOmega(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        if lnpi_zero is None:
            lnpi_zero = self._lnpi_zero
        return lnpi_zero - np.log(self.pi_sum)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]

    def betaOmega(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""
        Scaled grand potential :math:`\beta \Omega`.

        Parameters
        ----------
        lnpi_zero : float or None
            if None, lnpi_zero = self.data.ravel()[0]
        """
        # Note.  Put calculation in _betaOmega
        # because so many other things
        # call it with lnpi_zero set to None
        # so end up with copies of the same
        # thing in cache
        return self._betaOmega(lnpi_zero)

    @cached_prop
    @xr_name(r"${\rm PE}(\mu,V,T)$", standard_name="potential_energy")
    def PE(self) -> xr.DataArray:
        r"""Potential energy :math:`\overline{PE}`"""
        return self.mean_pi("PE", allow_extra_kws=True)

    @xr_name(r"$\beta F(\mu,V,T)$", standard_name="helmholtz_free_energy")
    def betaF_alt(
        self, betaF_can: xr.DataArray, correction: bool = True
    ) -> xr.DataArray:
        r"""
        Alternate scaled Helmholtz free energy :math:`\beta \overline{F}`.

        Calculated using

        .. math::

            \beta \overline{F} = \sum_N [\Pi(N) \beta F(N) + C(N)]

        Parameters
        ----------
        betaF_can : array-like
            Value of :math:`F(N)`
        correction : bool, default=True
            If True, :math:`C(N) = \ln \Pi(N) `.  Otherwise, :math:`C=0`.
        """
        if correction:
            betaF_can = betaF_can + self.lnpi_norm  # noqa: PLR6104
        return self._mean_pi(betaF_can)

    ################################################################################
    # Other properties
    @xr_name(r"$\beta\omega(\mu,V,T)$", standard_name="grand_potential_per_particle")
    def betaOmega_n(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Grand potential per particle :math:`\beta \Omega / \overline{N}`"""
        return self.betaOmega(lnpi_zero) / self.ntot

    # @cached_meth
    @xr_name(r"$\beta p(\mu,V,T)V$")
    def betapV(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r""":math:`\beta p V = - \beta \Omega`"""
        return -self.betaOmega(lnpi_zero)

    @cached_prop
    @xr_name("mask_stable", description="True where state is most stable")
    def mask_stable(self) -> xr.DataArray:
        """Masks are True where values are stable. Only works for unstacked data."""
        if not isinstance(self._parent, lnPiCollection):
            msg = "only implemented for lnPiCollection"
            raise TypeError(msg)

        if not self._xarray_unstack:
            # raise Value'only mask with unstack')
            pv = self.betapV()
            sample = self._parent._concat_dim
            return (  # type: ignore[no-any-return]
                pv.unstack(sample)  # noqa: PD010, PD013
                .pipe(lambda x: x.max("phase") == x)
                .stack(sample=pv.indexes[sample].names)
                .loc[pv.indexes["sample"]]
            )

        return self.betapV().pipe(lambda x: x.max("phase") == x)  # type: ignore[no-any-return,unused-ignore]

    # @cached_meth
    @xr_name(r"$\beta p(\mu,V,T)/\rho$", standard_name="compressibility_factor")
    def Z(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Compressibility factor :math:`\beta p / \rho`"""
        return -self.betaOmega_n(lnpi_zero)

    @xr_name(r"$p(\mu,V,T)$")
    def pressure(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Pressure :math:`p = -\Omega / V`"""
        return self.betapV(lnpi_zero) / (self.beta * self.volume)

    @property
    @xr_name(r"${\rm PE}(\mu,V,T)/n$", standard_name="potential_energy_per_particle")
    def PE_n(self) -> xr.DataArray:
        r"""Potential energy per particle :math:`\overline{PE}/\overline{N}`"""
        return self.PE / self.ntot

    def table(
        self,
        keys: str | Sequence[str] | None = None,
        default_keys: str | Sequence[str] | None = ("nvec", "betapV", "PE_n"),
        ref: lnPiMasked | None = None,
        mask_stable: bool = False,
        dim_to_suffix: Sequence[Hashable] | None = None,
    ) -> xr.Dataset:
        """
        Create :class:`xarray.Dataset` of calculated properties.

        Parameters
        ----------
        keys : sequence of str, optional
            keys of attributes or methods of `self` to include in output
        default_keys : sequence of str, optional
            Default keys to consider.
        ref : lnPiMasked, optional
            If calculating `edge_distastance`, need a reference :class:`~lnpy.lnpidata.lnPiMasked` object.
        mask_stable : bool, default=False
            If True, remove any unstable values
        dim_to_suffix : sequence of hashable, optional
            dimensions to remove from output.  These are instead added as suffix to variable names

        Returns
        -------
        ds : Dataset
            Containing all the calculated properties in a single object

        Notes
        -----
        The results can be easily convert to a :class:`pandas.DataFrame` using ``ds.to_frame()``
        """
        out: list[xr.DataArray] = []
        if ref is not None:
            out.append(self.edge_distance(ref))

        def _process_keys(x: str | Sequence[str] | None) -> list[str]:
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            return list(x)

        if not (keys := _process_keys(keys) + _process_keys(default_keys)):
            msg = "must specify some keys or default_keys to use"
            raise ValueError(msg)

        # only unique keys
        # this preserves order
        for key in dict.fromkeys(keys):
            try:
                if (v := getattr(self, key, None)) is not None:
                    if callable(v):
                        v = v()
                    out.append(v)
            except Exception:  # noqa: PERF203, BLE001, S110  # pylint: disable=broad-exception-caught
                pass

        ds: xr.Dataset = xr.merge(out, compat="no_conflicts")
        if "lnz" in keys:
            # if including property lnz, then drop lnz_0, lnz_1,...
            ds = ds.drop(ds["lnz"]["dims_lnz"])

        if mask_stable:
            # mask_stable inserts nan in non-stable
            mask = self.mask_stable
            if not self._xarray_unstack:
                ds = ds.where(mask, drop=True)
            else:
                phase = ds.phase
                ds = (
                    ds.where(mask)
                    .max("phase")
                    .assign_coords(phase=lambda x: phase[mask.argmax("phase")])  # noqa: ARG005
                )

        if dim_to_suffix is not None:
            if isinstance(dim_to_suffix, str):
                dim_to_suffix = [dim_to_suffix]
            for dim in dim_to_suffix:
                ds = ds.pipe(dim_to_suffix_dataset, dim=dim)
        return ds

    @property
    @xr_name(r"$\beta G(\mu,V,T)$", standard_name="Gibbs_free_energy")
    def betaG(self) -> xr.DataArray:
        r"""Scaled Gibbs free energy :math:`\beta G = \sum_i \beta \mu_i \overline{N}_i`."""
        return xr_dot(  # type: ignore[no-any-return]
            self.betamu, self.nvec, dim=self.dims_comp, **self._xarray_dot_kws
        )

    @property
    @xr_name(r"$\beta G(\mu,V,T)/n$", standard_name="Gibbs_free_energy_per_particle")
    def betaG_n(self) -> xr.DataArray:
        r"""Scaled Gibbs free energy per particle :math:`\beta G / \overline{N}`."""
        return self.betaG / self.ntot

    @xr_name(r"$\beta F(\mu,V,T)$", standard_name="helmholtz_free_energy")
    def betaF(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Scaled Helmholtz free energy :math:`\beta F = \beta \Omega + \beta G`."""
        return self.betaOmega(lnpi_zero) + self.betaG

    @xr_name(
        r"$\beta F(\mu,V,T)/n$", standard_name="helmholtz_free_energy_per_particle"
    )
    def betaF_n(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Scaled Helmholtz free energy per particle :math:`\beta F / \overline{N}`."""
        return self.betaF(lnpi_zero) / self.ntot

    @xr_name(r"$\beta E(\mu,V,T)$", standard_name="total_energy")
    def betaE(self, ndim: int = 3) -> xr.DataArray:
        r"""Scaled total energy :math:`\beta E = \beta \overline{PE} + \beta \overline{KE}`."""
        return ndim / 2.0 * self.ntot + self.PE * self.beta

    @xr_name(r"$\beta E(\mu,V,T)/ n$", standard_name="total_energy_per_particle")
    def betaE_n(self, ndim: int = 3) -> xr.DataArray:
        r"""Scaled total energy per particle :math:`\beta E / \overline{N}`."""
        return self.betaE(ndim) / self.ntot

    @xr_name(r"$S(\mu,V,T) / k_{\rm B}$", standard_name="entropy")
    def S(self, lnpi_zero: XArrayLike | None = None, ndim: int = 3) -> xr.DataArray:
        r"""Scaled entropy :math:`S / k_{\rm B} = \beta E - \beta F`."""
        return self.betaE(ndim) - self.betaF(lnpi_zero)

    @xr_name(r"$S(\mu,V,T)/(n kB)$", standard_name="entropy_per_particle")
    def S_n(self, lnpi_zero: XArrayLike | None = None, ndim: int = 3) -> xr.DataArray:
        r"""Scaled entropy per particle :math:`S / (N k_{\rm B})`."""
        return self.S(lnpi_zero, ndim) / self.ntot


# * Canonical ensembles (xce) -------------------------------------------------
class CanonicalEnsemble:
    """
    Canonical ensemble properties

    Parameters
    ----------
    parent : lnPiMasked
    """

    def __init__(self, parent: lnPiMasked) -> None:
        self._parent = parent
        self._xge = parent.xge
        self._cache: dict[str, Any] = {}

    def lnpi(self, fill_value: float | None = None) -> xr.DataArray:
        r""":class:`~xarray.DataArray` view of :math:`\ln Pi(N)`"""
        return self._xge.lnpi(fill_value=fill_value)

    @property
    def _xarray_unstack(self) -> bool:
        return getattr(self._parent, "_xarray_unstack", True)

    @cached_prop
    def ncoords(self) -> xr.DataArray:
        """Coordinate vector `dims_n`"""
        return (
            self._xge.ncoords
            # NOTE: if don't use '.values', then get extra coords don't want
            .where(~self._xge.lnpi(np.nan).isnull().to_numpy())  # noqa: PD003
        )

    @property
    def beta(self) -> float:
        return self._xge.beta

    @property
    def volume(self) -> float:
        return self._xge.volume

    @property
    def nvec(self) -> xr.DataArray:
        r"""Number of particles for each components :math:`{\bf N}`"""
        return self.ncoords.rename("nvec")

    @cached_prop
    def ntot(self) -> xr.DataArray:
        """Total number of particles :math:`N`"""
        return self.ncoords.sum(self._xge.dims_comp)

    @cached_meth
    @xr_name(r"$\beta F({\bf n},V,T)$", standard_name="helmholtz_free_energy")
    def _betaF(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        """Helmholtz free energy"""
        x = self._parent.xge

        if lnpi_zero is None:
            # TODO(wpk): Take another look at this...
            lnpi_zero = self._parent.data.ravel()[0]

        return (
            (
                -(x.lnpi(np.nan) - lnpi_zero)  # pyright: ignore[reportOperatorIssue]
                + (x.ncoords * x.betamu).sum(x.dims_comp)
            )
            .assign_coords(x._wrapper.coords_n)
            .drop_vars(x.dims_lnz)
            .assign_attrs(x._standard_attrs)
        )

    def betaF(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Scaled Helmholtz free energy :math:`\beta F`"""
        return self._betaF(lnpi_zero)

    @xr_name(
        r"$\beta F({\bf n},V,T)/n$", standard_name="helmholtz_free_energy_per_particle"
    )
    def betaF_n(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Scaled Helmholtz free energy per particle :math:`\beta F / N`"""
        return self.betaF(lnpi_zero) / self.ntot

    @cached_prop
    @xr_name(r"${\rm PE}({\bf n},V,T)/n$", standard_name="potential_energy")
    def PE(self) -> xr.DataArray:
        """Internal Energy :math:`PE`"""
        # if betaPE available, use that:
        if (PE := self._parent.extra_kws.get("PE", None)) is None:
            msg = 'must set "PE" in "extra_kws" of lnPiMasked'
            raise AttributeError(msg)
        x = self._xge
        coords = dict(x._wrapper.coords_n, **self._parent.state_kws)
        return xr.DataArray(PE, dims=x.dims_n, coords=coords, attrs=x._standard_attrs)

    @property
    @xr_name(
        r"${\rm PE}({\bf n},V,T)/n$", standard_name="potential_energy_per_particle"
    )
    def PE_n(self) -> xr.DataArray:
        r"""Internal energy per particle :math:`PE / N`"""
        return self.PE / self.ntot

    @xr_name(r"$\beta E({\bf n},V,T)$")
    def betaE(self, ndim: int = 3) -> xr.DataArray:
        r"""Scaled total energy :math:`\beta E = \beta PE + \beta KE`"""
        return ndim / 2 * self.ntot + self._xge.beta * self.PE

    @xr_name(r"$\beta E({\bf n},V,T)/n$")
    def betaE_n(self, ndim: int = 3) -> xr.DataArray:
        """Scaled total energy per particle"""
        return self.betaE(ndim) / self.ntot

    @xr_name(r"$S({\bf n},V,T)/k_{\rm B}$")
    def S(self, ndim: int = 3, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Entropy :math:`S / k_{\rm B}`"""
        return self.betaE(ndim) - self.betaF(lnpi_zero)

    @xr_name(r"$S({\bf n},V,T)/(n k_{rm B})$")
    def S_n(self, ndim: int = 3, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Entropy per particle :math:`S / (N k_{\rm B})`"""
        return self.S(ndim, lnpi_zero) / self.ntot

    @cached_meth
    @xr_name(r"$\beta {\bf\mu}({bf n},V,T)$", standard_name="absolute_activity")
    def _betamu(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        return xr.concat(
            [self.betaF(lnpi_zero).differentiate(n) for n in self._xge.dims_n],
            dim=self._xge.dims_comp[0],
        ).assign_attrs(self._xge._standard_attrs)

    def betamu(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Scaled chemical potential :math:`\beta \mu`"""
        return self._betamu(lnpi_zero)

    @property
    @xr_name(r"${\bf \rho}({\bf n},V,T)$")
    def dens(self) -> xr.DataArray:
        r"""Density :math:`\rho = N / V`"""
        return self.ncoords / self._xge.volume

    @cached_meth
    @xr_name(r"$\beta\Omega({\bf n},V,T)$")
    def _betaOmega(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        """Calculate beta * Omega = betaF - lnz .dot. N"""
        return self.betaF(lnpi_zero) - (self.betamu(lnpi_zero) * self.ncoords).sum(
            self._xge.dims_comp
        )

    def betaOmega(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Scaled Grand Potential :math:`\beta \Omega`"""
        return self._betaOmega(lnpi_zero)

    @xr_name(r"$\beta\Omega({\bf n},V,T)/n$")
    def betaOmega_n(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Scaled Grand potential per particle, :math:`\beta\Omega / N`"""
        return self.betaOmega(lnpi_zero) / self.ntot

    @xr_name(r"$\beta p({\bf n},V,T)V$")
    def betapV(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r""":math:`\beta p V = -\beta \Omega`"""
        return -self.betaOmega(lnpi_zero)

    @xr_name(r"$\beta p({\bf n},V,T)/\rho$")
    def Z(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Compressibility :math:`Z = \beta p / \rho = pV / (k_{\rm B} T)`"""
        return -self.betaOmega_n(lnpi_zero)

    @xr_name(r"$p({\bf n},V,T)$")
    def pressure(self, lnpi_zero: XArrayLike | None = None) -> xr.DataArray:
        r"""Pressure :math:`p = -\Omega / V`"""
        return self.betapV(lnpi_zero) / (self.beta * self.volume)

    def table(
        self,
        keys: str | Sequence[str] | None = None,
        default_keys: str | Sequence[str] | None = (
            "betamu",
            "betapV",
            "PE_n",
            "betaF_n",
        ),
        dim_to_suffix: str | Sequence[str] | None = None,
    ) -> xr.Dataset:
        """
        Create Dataset from attributes/methods of `self`

        Parameters
        ----------
        keys : sequence of str
            Names of attributes/methods to access
        default_keys : sequence of str
            Default keys to access.
        dim_to_suffix : str, sequence of str, optional
            If passed, convert dimensions in `dim_to_suffix` from dimensions in output to suffixes to variable names


        Returns
        -------
        table : Dataset
        """
        out: list[xr.DataArray] = []

        def _process_keys(x: str | Sequence[str] | None) -> list[str]:
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            return list(x)

        keys_total = _process_keys(keys) + _process_keys(default_keys)

        # loop over unique keys only, in order
        for key in dict.fromkeys(keys_total):
            try:
                v = getattr(self, key)
                if callable(v):
                    v = v()
                out.append(v)  # pyright: ignore[reportArgumentType]
            except Exception:  # noqa: PERF203, BLE001, S110  # pylint: disable=broad-exception-caught
                pass

        ds = xr.merge(out, compat="no_conflicts")

        if dim_to_suffix is not None:
            if isinstance(dim_to_suffix, str):
                dim_to_suffix = [dim_to_suffix]
            for dim in dim_to_suffix:
                ds = ds.pipe(dim_to_suffix_dataset, dim=dim)  # pylint: disable=redefined-variable-type
        return ds
