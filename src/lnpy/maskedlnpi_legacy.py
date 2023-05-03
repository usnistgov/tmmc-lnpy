"""
Legacy lnPi array routines (:mod:`~lnpy.maskedlnpi_legacy`)
===========================================================
"""
from warnings import warn

import numpy as np
import pandas as pd
from module_utilities import cached
from scipy import ndimage

from .extensions import AccessorMixin
from .utils import labels_to_masks, masks_change_convention

# NOTE : This is a rework of core.
# [ ] : split xarray functionality into wrapper(s)
# [ ] : split splitting into separate classes


class MaskedlnPiLegacy(np.ma.MaskedArray, AccessorMixin):
    r"""
    Class to store masked version of :math:`\ln\Pi(N)`.
    shape is (N0,N1,...) where Ni is the span of each dimension)

    Constructor

    Parameters
    ----------
    data : array-like
        data for lnPi
    lnz : array-like, optional
        if None, set lnz=np.zeros(data.ndim)
    state_kws : dict, optional
        dictionary of state values, such as ``volume`` and ``beta``.
        These parameters will be pushed to ``self.xge`` coordinates.
    extra_kws : dict, optional
        this defines extra parameters to pass along.
        Note that for potential energy calculations, extra_kws should contain
        `PE` (total potential energy for each N vector).
    zeromax : bool, default=False
        if True, shift lnPi = lnPi - lnPi.max()
    pad : bool, default=False
        if True, pad masked region by interpolation
    **kwargs
        Extra arguments to :class:`numpy.ma.MaskedArray`
        e.g., mask=...
    """

    def __new__(cls, data=None, lnz=None, state_kws=None, extra_kws=None, **kwargs):
        warn("MaskedlnPiLegacy is deprecated.  Please use lnPiMasked instead")

        if data is not None and issubclass(data.dtype.type, np.floating):
            kwargs.setdefault("fill_value", np.nan)

        obj = np.ma.array(data, **kwargs).view(cls)
        # fv = kwargs.get('fill_value', None) or getattr(data, 'fill_value', None)
        # if fv is None:
        #     fv = np.nan
        # obj.set_fill_value(fv)

        # make sure to broadcast mask if it is just False
        if obj.mask is False:
            obj.mask = False

        # set mu value:
        if lnz is None:
            lnz = np.zeros(obj.ndim)
        lnz = np.atleast_1d(lnz).astype(obj.dtype)
        if len(lnz) != obj.ndim:
            raise ValueError("bad len on lnz %s" % lnz)

        if state_kws is None:
            state_kws = {}
        if extra_kws is None:
            extra_kws = {}

        obj._optinfo.update(
            lnz=lnz,
            state_kws=state_kws,
            extra_kws=extra_kws,
        )
        return obj

    ##################################################
    # caching
    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._clear_cache()

    def _clear_cache(self):
        self._cache = {}

    ##################################################
    # properties
    @property
    def optinfo(self):
        """All extra properties"""
        return self._optinfo

    @property
    def state_kws(self):
        """State specific parameters"""
        return self._optinfo["state_kws"]

    @property
    def extra_kws(self):
        """All extra parameters"""
        return self._optinfo["extra_kws"]

    def _index_dict(self, phase=None):
        out = {f"lnz_{i}": v for i, v in enumerate(self.lnz)}
        if phase is not None:
            out["phase"] = phase
        # out.update(**self.state_kws)
        return out

    def _lnpi_tot(self, fill_value=None):
        return self.filled(fill_value)

    def _pi_params(self, fill_value=None):
        lnpi = self._lnpi_tot(fill_value)

        lnpi_local_max = lnpi.max()
        pi = np.exp(lnpi - lnpi_local_max)
        pi_sum = pi.sum()
        pi_norm = pi / pi_sum

        lnpi_zero = self.data.ravel()[0] - lnpi_local_max

        return pi_norm, pi_sum, lnpi_zero

    @property
    def _lnz_tot(self):
        return self.lnz

    # @property
    # def _lnpi_0_tot(self):
    #     return self.data.ravel()[0]

    @property
    def lnz(self):
        return self._optinfo.get("lnz", None)

    @property
    def betamu(self):
        return self.lnz

    # @property
    # def mu(self):
    #     return self._optinfo.get('mu', None)
    @property
    def volume(self):
        return self.state_kws.get("volume", None)

    @property
    def beta(self):
        return self.state_kws.get("beta", None)

    def __repr__(self):
        L = []
        L.append(f"lnz={repr(self.lnz)}")
        L.append(f"state_kws={repr(self.state_kws)}")

        L.append(f"data={super().__repr__()}")
        if len(self.extra_kws) > 0:
            L.append(f"extra_kws={repr(self.extra_kws)}")

        indent = " " * 5
        p = "MaskedlnPi(\n" + "\n".join([indent + x for x in L]) + "\n)"

        return p

    def __str__(self):
        return f"MaskedlnPi(lnz={str(self.lnz)})"

    # @cached.meth
    def local_argmax(self, *args, **kwargs):
        return np.unravel_index(self.argmax(*args, **kwargs), self.shape)

    # @cached.meth
    def local_max(self, *args, **kwargs):
        return self[self.local_argmax(*args, **kwargs)]

    # @cached.meth
    def local_maxmask(self, *args, **kwargs):
        return self == self.local_max(*args, **kwargs)

    @cached.prop
    def edge_distance_matrix(self):
        """Matrix of distance from upper bound"""
        from .utils import distance_matrix

        return distance_matrix(~self.mask)

    def edge_distance(self, ref, *args, **kwargs):
        return ref.edge_distance_matrix[self.local_argmax(*args, **kwargs)]

    # make these top level
    # @cached.prop
    # @property
    # def pi(self):
    #     """
    #     basic pi = exp(lnpi)
    #     """
    #     pi = np.exp(self - self.local_max())
    #     return pi

    # @cached.prop
    # def pi_sum(self):
    #     return self.pi.sum()

    # @cached.meth
    # def betaOmega(self, lnpi_zero=None):
    #     if lnpi_zero is None:
    #         lnpi_zero = self.data.ravel()[0]
    #     zval = lnpi_zero - self.local_max()
    #     return  (zval - np.log(self.pi_sum))

    def __setitem__(self, index, value):
        self._clear_cache()
        super().__setitem__(index, value)

    def pad(self, axes=None, ffill=True, bfill=False, limit=None, inplace=False):
        """
        Pad nan values in underlying data to values

        Parameters
        ----------
        ffill : bool, default=True
            do forward filling
        bfill : bool, default=False
            do back filling
        limit : int, default None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.
        inplace : bool, default=False

        Returns
        -------
        out : object
            padded object
        """
        import bottleneck

        from .utils import bfill, ffill

        if axes is None:
            axes = range(self.ndim)

        data = self.data
        datas = []

        if ffill:
            datas += [ffill(data, axis=axis, limit=limit) for axis in axes]
        if bfill:
            datas += [bfill(data, axis=axis, limit=limit) for axis in axes]

        if len(datas) > 0:
            data = bottleneck.nanmean(datas, axis=0)

        if inplace:
            new = self
            new._clear_cache()
        else:
            new = self.copy()

        new.data[...] = data
        return new

    def zeromax(self, inplace=False):
        """Shift so that lnpi.max() == 0"""

        if inplace:
            new = self
            self._clear_cache()
        else:
            new = self.copy()

        new.data[...] = new.data - new.max()
        return new

    def adjust(self, zeromax=False, pad=False, inplace=False):
        """Do multiple adjustments in one go"""

        if inplace:
            new = self
        else:
            new = self.copy()

        if zeromax:
            new.zeromax(inplace=True)
        if pad:
            new.pad(inplace=True)
        return new

    def reweight(self, lnz, zeromax=False, pad=False):
        """
        Get lnpi at new lnz

        Parameters
        ----------
        lnz : array-like
            chem. pot. for new state point

        zeromax : bool, default=False

        pad : bool, default=False

        phases : dict


        Returns
        -------
        object
        """

        lnz = np.atleast_1d(lnz)

        assert len(lnz) == len(self.lnz)

        new = self.copy()
        new._optinfo["lnz"] = lnz

        dlnz = new.lnz - self.lnz

        # s = _get_shift(self.shape,dmu)*self.beta
        # get shift
        # i.e., N * (mu_1 - mu_0)
        # note that this is (for some reason)
        # faster than doing the (more natural) options:
        # N = self.ncoords.values
        # shift = 0
        # for i, m in enumerate(dmu):
        #     shift += N[i,...] * m
        # or
        # shift = (self.ncoords.values.T * dmu).sum(-1).T

        shift = np.zeros([], dtype=float)
        for i, (s, m) in enumerate(zip(self.shape, dlnz)):
            shift = np.add.outer(shift, np.arange(s) * m)

        # scale by beta
        # shift *= self.beta

        new.data[...] += shift
        new.adjust(zeromax=zeromax, pad=pad, inplace=True)

        return new

    def smooth(self, sigma=4, mode="nearest", truncate=4, inplace=False, **kwargs):
        """
        Apply gaussian filter smoothing to data

        Parameters
        ----------
        inplace : bool, default=False
         if True, do inplace modification.
        mode : str, default='nearest'
            Arguments to ``gaussian_filter``
        truncate : int, default=4
            Argument to ``gaussian_filter``

        **kwargs
            Extra arguments to ``gaussian_filter``.

        See Also
        --------
        ~scipy.ndimage.gaussian_filter
        """

        if inplace:
            new = self
            new._clear_cache()
        else:
            new = self.copy()

        ndimage.gaussian_filter(
            new.data,
            output=new.data,
            mode=mode,
            truncate=truncate,
            sigma=sigma,
            **kwargs,
        )
        return new

    def copy_shallow(self, mask=None, **kwargs):
        """
        Create shallow copy

        Parameters
        ----------
        mask : optional
            if specified, new object has this mask
            otherwise, at least copy old mask
        """
        if mask is None:
            mask = self.mask.copy()

        return self.__class__(
            self.data,
            mask=mask,
            fill_value=self.fill_value,
            **dict(self._optinfo, **kwargs),
        )

    def or_mask(self, mask, **kwargs):
        """New object with logical or of self.mask and mask"""
        return self.copy_shallow(mask=mask + self.mask, **kwargs)

    def and_mask(self, mask, **kwargs):
        """New object with logical and of self.mask and mask"""
        return self.copy_shallow(mask=mask * self.mask, **kwargs)

    def __getstate__(self):
        ma = self.view(np.ma.MaskedArray).__getstate__()
        opt = self._optinfo
        return ma, opt

    def __setstate__(self, state):
        ma, opt = state
        super().__setstate__(ma)
        self._optinfo.update(opt)

    #        opt = self._optinfo
    #        return ma, opt
    #         # ma = self.view(np.ma.MaskedArray)
    #         # info = self._optinfo
    #         # return ma, info
    # self._optinfo.update(opt)
    # ma, info = state
    #         # super(MaskedlnPi, self).__setstate__(ma)
    #         # self._optinfo.update(info)

    @classmethod
    def from_table(
        cls, path, lnz, state_kws=None, sep=r"\s+", names=None, csv_kws=None, **kwargs
    ):
        """
        Create lnPi object from text file table with columns [n_0,...,n_ndim, lnpi]

        Parameters
        ----------
        path : path-like
            file object to be read
        lnz : array-like
            beta*(chemical potential) for each component
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
            pd.read_csv(path, sep=sep, names=names, **csv_kws)
            .set_index(names[:-1])["lnpi"]
            .to_xarray()
        )
        return cls(
            data=da.values,
            mask=da.isnull().values,
            lnz=lnz,
            state_kws=state_kws,
            **kwargs,
        )

    @classmethod
    def from_dataarray(cls, da, state_as_attrs=None, **kwargs):
        """Create a lnPi object from xarray.DataArray"""

        kws = {}
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
            c = da.coords

        lnz = []
        state_kws = {}
        for k in da.attrs["dims_state"]:
            val = np.array(c[k])
            if "lnz" in k:
                lnz.append(val)
            else:
                state_kws[k] = val * 1
        kws["lnz"] = lnz
        kws["state_kws"] = state_kws

        # any overrides
        kwargs = dict(kws, **kwargs)
        return cls(**kwargs)

    def list_from_masks(self, masks, convention="image"):
        """
        Create list of lnpis corresponding to masks[i]

        Parameters
        ----------
        masks : list
            masks[i] is the mask for lnpi index `i`.
        convention : str or bool
            convention of input masks
        Returns
        -------
        lnpis : list
            list of lnpis corresponding to each mask
        """

        return [
            self.or_mask(m) for m in masks_change_convention(masks, convention, False)
        ]

    def list_from_labels(
        self,
        labels,
        features=None,
        include_boundary=False,
        check_features=True,
        **kwargs,
    ):
        """Create list of lnpis corresponding to labels"""

        masks, features = labels_to_masks(
            labels=labels,
            features=features,
            include_boundary=include_boundary,
            convention=False,
            check_features=check_features,
            **kwargs,
        )
        return self.list_from_masks(masks, convention=False)
