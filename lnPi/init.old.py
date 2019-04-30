"""
utilities to work with lnPi(N)
"""

import itertools
from collections import Iterable

import numpy as np
# import pandas as pd
from scipy.ndimage import filters
from scipy.spatial.distance import cdist, pdist, squareform

from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries
from skimage import draw
from skimage.graph import route_through_array

import h5py
import xarray as xr

from lnPi.cached_decorators import cached_clear, cached, cached_func
from lnPi._utils import _interp_matrix, get_mu_iter
from lnPi._segment import (_indices_to_markers, _labels_watershed,
                           labels_to_masks, masks_to_labels)
from lnPi.spinodal import *
from lnPi.binodal import *
from lnPi.molfrac import *

from functools import wraps, partial

def gcached(key=None, prop=True):
    def wrapper(func):
        if prop:
            wrapped = property(cached(key)(func))
        else:
            wrapped = cached_func(key)(func)
        return wrapped
    return wrapper


def xrify(key=None, prop=True, cache=False, **kws):
    def wrapper(func):
        if key is None:
            _key = func.__name__
        else:
            _key = key

        @wraps(func)
        def wrapped(self, *args, **kwargs):
            return xr.DataArray(func(self, *args, **kwargs), coords=self._xr_coords, name=_key, **kws)

        if cache:
            if prop:
                wrapped = cached(_key)(wrapped)
            else:
                wrapped = cached_func(_key)(wrapped)
        if prop:
            wrapped = property(wrapped)

        return wrapped
    return wrapper
xrify_comp = partial(xrify, dims='component')


def concatify(key=None, prop=True, cache=False, **kws):
    def wrapper(func):
        if key is None:
            _key = func.__name__
        else:
            _key = key

        @wraps(func)
        def wrapped(self, *args, **kwargs):
            return xr.concat(func(self, *args, **kwargs), **kws)

        if cache:
            if prop:
                wrapped = cached(_key)(wrapped)
            else:
                wrapped = cached_func(_key)(wrapped)
        if prop:
            wrapped = property(wrapped)

        return wrapped
    return wrapper

concatify_phase = partial(concatify, dim='phase', coords='different')
concatify_rec = partial(concatify, dim='rec', coords='all')




class lnPi(np.ma.MaskedArray):
    """
    class to store masked ln[Pi(n0,n1,...)].
    shape is (N0,N1,...) where Ni is the span of each dimension)

    Attributes
    ----------
    self : masked array containing lnPi

    mu : chemical potential for each component

    coords : coordinate array (ndim,N0,N1,...)

    pi : exp(lnPi)

    pi_norm : pi/pi.sum()

    Nave : average number of particles of each component    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",
                              size="2%",
                              pad=cbar_pad)


    molfrac : mol fraction of each component

    Omega : Omega system, relative to lnPi[0,0,...,0]


    set_data : set data array

    set_mask : set mask array


    argmax_local : local argmax (in np.where output form)

    get_phases : return phases


    ZeroMax : set lnPi = lnPi - lnPi.max()

    Pad : fill in masked points by interpolation

    adjust : ZeroMax and/or Pad


    reweight : create new lnPi at new mu

    new_mask : new object (default share data) with new mask

    add_mask : create object with mask = self.mask + mask

    smooth : create smoothed object



    from_file : create lnPi object from file

    from_data : create lnPi object from data array
    """

    def __new__(cls,
                data,
                mu=None,
                ZeroMax=True,
                Pad=False,
                num_phases_max=None,
                volume=None,
                beta=None,
                **kwargs):
        """
        constructor

        Parameters
        ----------
        data : array-like
         data for lnPi

        mu : array-like (Default None)
         if None, set mu=np.zeros(data.ndim)

        ZeroMax : bool (Default False)
         if True, shift lnPi = lnPi - lnPi.max()


        Pad : bool (Default False)
         if True, pad masked region by interpolation


        **kwargs : arguments to np.ma.array
         e.g., mask=...

        """

        obj = np.ma.array(data, **kwargs).view(cls)


        #copy fill value
        obj.set_fill_value(getattr(data, 'fill_value', None))

        #overide?
        fv = kwargs.get('fill_value', None)
        if fv is not None:
            obj.set_fill_value(fv)

        obj.mu = mu
        obj.num_phases_max = num_phases_max
        obj.volume = volume
        obj.beta = beta

        obj.adjust(ZeroMax=ZeroMax, Pad=Pad, inplace=True)

        return obj

    ##################################################
    #caching
    def __array_finalize__(self, obj):
        super(lnPi, self).__array_finalize__(obj)
        self._clear_cache()

    def _clear_cache(self):
        self._cache = {}

    ##################################################
    #properties
    @property
    def mu(self):
        mu = self._optinfo.get('mu', None)
        if mu is None:
            mu = np.zeros(self.ndim)
        mu = np.atleast_1d(mu).astype(self.dtype)

        if len(mu) != self.ndim:
            raise ValueError('bad len on mu %s' % mu)
        return mu

    @mu.setter
    def mu(self, val):
        if val is not None:
            self._optinfo['mu'] = val

    @property
    def num_phases_max(self):
        return self._optinfo.get('num_phases_max', None)

    @num_phases_max.setter
    def num_phases_max(self, val):
        if val is not None:
            self._optinfo['num_phases_max'] = val

    @property
    def volume(self):
        return self._optinfo.get('volume', None)

    @volume.setter
    def volume(self, val):
        if val is not None:
            self._optinfo['volume'] = val

    @property
    def beta(self):
        return self._optinfo.get('beta', None)

    @beta.setter
    def beta(self, val):
        if val is not None:
            self._optinfo['beta'] = val

    @property
    def coords(self):
        return np.indices(self.shape)


    @gcached()
    def _xr_coords(self):
        coords = {'mu_{}'.format(i) : v for i,v in enumerate(self.mu)}
        coords['beta'] = self.beta
        return coords

    @xrify_comp(cache=True)
    def chempot(self):
        return self.mu


    #calculated properties
    @gcached()
    def pi(self):
        """
        basic pi = exp(lnpi)
        """
        shift = self.max()
        pi = np.exp(self - shift)
        return pi

    @property
    def pi_norm(self):
        p = self.pi
        return p / p.sum()

    @xrify_comp(cache=True)
    def Nave(self):
        #<N_i>=sum(N_i*exp(lnPi))/sum(exp(lnPi))
        N = (self.pi_norm * self.coords).reshape(self.ndim,
                                                 -1).sum(axis=-1).data
        return N#self._as_series_ncomp(N, name='Nave')

    @property
    def density(self):
        return self.Nave / self.volume

    @xrify_comp(cache=True)
    def Nvar(self):
        x = (self.coords - self.Nave.reshape((-1, ) + (1, ) * self.ndim))**2
        return (self.pi_norm * x).reshape(self.ndim, -1).sum(axis=-1).data

    @property
    def molfrac(self):
        n = self.Nave
        return n / n.sum()

    @xrify(cache=True, prop=False)
    def Omega(self, *args):
        """
        get omega = zval - ln(sum(pi))

        Parameters
        ----------
        zval : float or None
         if None, zval = self.data.ravel()[0]
        """

        if len(args) == 0:
            zval = None
        else:
            zval = args[0]

        if zval is None:
            zval = self.data.ravel()[0] - self.max()

        omega = (zval - np.log(self.pi.sum())) / self.beta
        return omega

    @property
    def ncomp(self):
        return self.ndim

    ##################################################
    #setters
    @cached_clear()
    def set_data(self, val):
        self.data[:] = val

    @cached_clear()
    def set_mask(self, val):
        self.mask[:] = val

    ##################################################
    #maxima
    def _peak_local_max(self,
                        min_distance=1,
                        threshold_rel=0.00,
                        threshold_abs=0.2,
                        **kwargs):
        """
        find local maxima using skimage.feature.peak_local_max

        Parameters
        ----------
        min_distance : int (Default 1)
          min_distance parameter to peak_local_max

        **kwargs : arguments to peak_local_max (see docs)
         Defaults to min_distance=1,exclude_border=False

        Returns
        -------
        out : tupe of ndarrays
         indices of self where local max

        n : int
         number of maxima found
        """

        kwargs = dict(dict(exclude_border=False, labels=~self.mask), **kwargs)

        data = self.data - np.nanmin(self.data)

        x = peak_local_max(
            data,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            threshold_abs=threshold_abs,
            **kwargs)

        if kwargs.get('indices', True):
            #return indices
            x = tuple(x.T)
            n = len(x[0])
        else:
            n = x.sum()

        return x, n

    def argmax_local(self,
                     min_distance=[5, 10, 15, 20, 25],
                     threshold_rel=0.00,
                     threshold_abs=0.2,
                     smooth='fail',
                     smooth_kwargs={},
                     num_phases_max=None,
                     info=False,
                     **kwargs):
        """
        find local max with fall backs min_distance and filter

        Parameters
        ----------
        min_distance : int or iterable (Default 15)
            min_distance parameter to self.peak_local_max.
            if min_distance is iterable, if num_phase>num_phase_max, try next


        smooth : bool or str (Default 'fail')
            if True, use smooth only
            if False, no smooth
            if 'fail', do smoothing only if non-smooth fails

        smooth_kwargs : dict
            extra arguments to self.smooth

        num_phases_max : int (Default None)
            max number of maxima to find. if None, use self.num_phases_max


        info : bool (Default False)
            if True, return out_info

        **kwargs : extra arguments to peak_local_max

        Returns
        -------
        out : tuple of ndarrays
            indices of self where local max

        out_info : tuple (min_distance,smooth)
            min_distance used and bool indicating if smoothing was used


        """
        if num_phases_max is None:
            num_phases_max = self.num_phases_max

        if not isinstance(min_distance, Iterable):
            min_distance = [min_distance]

        if smooth is True:
            filtA = [True]
        elif smooth is False:
            filtA = [False]
        elif smooth.lower() == 'fail':
            filtA = [False, True]
        else:
            raise ValueError('bad parameter %s' % (smooth))

        for filt in filtA:
            if filt:
                xx = self.Pad().smooth(**smooth_kwargs)
            else:
                xx = self

            for md in min_distance:
                x, n = xx._peak_local_max(
                    min_distance=md,
                    threshold_rel=0.00,
                    threshold_abs=0.2,
                    **kwargs)

                if n <= num_phases_max:
                    if info:
                        return x, (md, filt)
                    else:
                        return x

        #if got here than error
        raise RuntimeError('%i maxima found greater than %i at mu %s' %
                           (n, num_phases_max, repr(self.mu)))

    ##################################################
    #segmentation
    def get_labels_watershed(self,
                             indices,
                             structure='set',
                             size=3,
                             footprint=None,
                             smooth=False,
                             smooth_kwargs={},
                             **kwargs):
        """
        get labels from watershed segmentation

        Parameters
        ----------
        indices : tuple of ndarrays
         indices of location of points to segment to (e.g., maxima).
         len==ndim, seg_indices[i].shape == (nsegs)
         e.g., from output of self.peak_local_max

        structure : array or str or None
         structure passed to ndimage.label.
         if None, use default.
         if 'set', use np.one((3,)*data.ndim)
         else, use structure

        size : size parameter to _labels_watershed

        footprint : footprint parameter to _labels_watershed

        smooth : bool (Default False)
            if True, use smoothed data for label creation

        smooth_kwargs : dict
            arguments to self.smooth()

        **kwargs : dict
            arguments to _labels_watershed

        Returns
        -------
        labels
        """
        markers, n = _indices_to_markers(indices, self.shape, structure)
        if smooth:
            xx = self.Pad().smooth(**smooth_kwargs)
        else:
            xx = self

        labels = _labels_watershed(
            -xx.data,
            markers,
            mask=(~self.mask),
            size=size,
            footprint=footprint,
            **kwargs)

        return labels

    def get_list_regmask(self, regmask, SegLenOne=False):
        """
        create list of lnPi objects with mask = self.mask + regmask[i]
        """
        if not SegLenOne and len(regmask) == 1:
            return None  #[self]
        else:
            return [self.add_mask(r) for r in regmask]

    def get_list_labels(self, labels, SegLenOne=False, **kwargs):
        """
        create list of lnpi's from labels
        """
        regmask = labels_to_masks(labels, **kwargs)
        return self.get_list_regmask(regmask, SegLenOne)

    def get_list_indices(self,
                         indices,
                         SegLenOne=False,
                         smooth=False,
                         smooth_kwargs={},
                         labels_kwargs={},
                         masks_kwargs={}):
        """
        create list of lnpi's from indices of features (argmax's)
        """
        if not SegLenOne and len(indices[0]) == 1:
            return None  #[self]
        else:
            labels = self.get_labels_watershed(
                indices,
                smooth=smooth,
                smooth_kwargs=smooth_kwargs,
                **labels_kwargs)
            return self.get_list_labels(labels, SegLenOne, **masks_kwargs)

    def to_phases(self,
                  argmax_kwargs=None,
                  phases_kwargs=None,
                  build_kwargs=None,
                  ftag_phases=None,
                  ftag_phases_kwargs=None):
        """
        return lnPi_phases object with placeholders for phases/argmax
        """

        return lnPi_phases(
            self,
            argmax_kwargs=argmax_kwargs,
            phases_kwargs=phases_kwargs,
            build_kwargs=build_kwargs,
            ftag_phases=ftag_phases,
            ftag_phases_kwargs=ftag_phases_kwargs)

    ##################################################
    #adjusters
    def ZeroMax(self, inplace=False):
        """
        shift so that lnpi.max() == 0
        """

        if inplace:
            y = self
        else:
            y = self.copy()

        shift = self.max()

        y.set_data(y.data - shift)

        if not inplace:
            return y

    def Pad(self, inplace=False):
        """
        pad self.data rows and columns by last value
        """
        if self.ndim == 1:

            msk = self.mask
            last = self[~msk][-1]

            data = self.data.copy()
            data[msk] = last

        elif self.ndim == 2:
            data = _interp_matrix(self.data, self.mask)

        else:
            raise ValueError('padding only implemented for ndim<=2')

        if inplace:
            y = self
        else:
            y = self.copy()

        y.set_data(data)

        if not inplace:
            return y

    def adjust(self, ZeroMax=False, Pad=False, inplace=False):
        """
        do multiple adjustments in one go
        """

        if inplace:
            Z = self
        else:
            Z = self.copy()

        if ZeroMax:
            Z.ZeroMax(inplace=True)

        if Pad:
            Z.Pad(inplace=True)

        if not inplace:
            return Z

    ##################################################
    #new object/modification
    def reweight(self, mu, ZeroMax=False, Pad=False):
        """
        get lnpi at new mu

        Parameters
        ----------
        mu : array-like
            chem. pot. for new state point

        beta : float
            inverse temperature

        ZeroMax : bool (Default False)

        Pad : bool (Default False)

        phases : dict


        Returns
        -------
        lnPi(mu)
        """

        Z = self.copy()
        Z.mu = mu
        dmu = Z.mu - self.mu

        #s = _get_shift(self.shape,dmu)*self.beta
        #get shift
        #i.e., N * (mu_1 - mu_0)
        shift = np.zeros([], dtype=float)
        for i, (s, m) in enumerate(zip(self.shape, dmu)):
            shift = np.add.outer(shift, np.arange(s) * m)

        #scale by beta
        shift *= self.beta

        Z.set_data(Z.data + shift)

        Z.adjust(ZeroMax=ZeroMax, Pad=Pad, inplace=True)

        return Z

    def new_mask(self, mask=None, **kwargs):
        """
        create copy with new mask
        """
        return lnPi(self.data, mask=mask, **dict(self._optinfo, **kwargs))

    def add_mask(self, mask, **kwargs):
        """
        logical or of self.mask and mask

        Note, if want a copy, pass copy=True
        """
        return self.new_mask(mask=mask + self.mask, **kwargs)

    def smooth(self,
               sigma=4,
               mode='nearest',
               truncate=4,
               inplace=False,
               ZeroMax=False,
               Pad=False,
               **kwargs):
        """
        apply gaussian filter smoothing to data

        Parameters
        ----------
        inplace : bool (Default False)
         if True, do inplace modification.


        **kwargs : (Default sigma=4, mode='nearest',truncate=4)
         arguments to filters.gaussian_filter
        """

        if inplace:
            Z = self
        else:
            Z = self.copy()

        Z._clear_cache()
        Z.adjust(ZeroMax=ZeroMax, Pad=Pad, inplace=True)
        filters.gaussian_filter(
            Z.data,
            output=Z.data,
            mode=mode,
            truncate=truncate,
            sigma=sigma,
            **kwargs)

        if not inplace:
            return Z

    ##################################################
    #create from file/etc
    @classmethod
    def from_file(cls, filename, mu=None, loadtxt_kwargs=None, **kwargs):
        """
        load filename into lnPi object

        Parameters
        ----------
        filename : str
            name of file to read

        mu : array-like or None
            chemical potential.  If None, default to Zero for each
            component

        loadtxt_kwargs : dict
            keyword arguments for numpy.loadtxt

        **kwargs : kwargs to lnPi constructor

        Returns
        -------
        lnPi object
        """

        if loadtxt_kwargs is None:
            loadtxt_kwargs = {}

        data = np.loadtxt(filename, **loadtxt_kwargs)

        return cls.from_data(data, mu, **kwargs)

    @classmethod
    def from_data(cls, data, mu=None, num_phases_max=None, **kwargs):
        """
        parse data into lnpi masked array

        Parameters
        ----------
        data : array of form (n0,n1,...,nd,lnPi)

        mu : array-like (d,)
            chem. pot. for each component

        num_phases_max : int or None (default None)
            max number of phases

        **kwargs : arguments to lnPi constructor
        """

        ndim = data.shape[-1]
        dataT = data.T

        z = dataT[-1, :]

        xx = dataT[:-1, :].astype(int)
        shape = tuple(xx.max(axis=-1) + 1)

        xx = tuple(xx)

        Z = np.zeros(shape, dtype=data.dtype)
        Z[xx] = z

        empty = np.ones(shape, dtype=bool)
        empty[xx] = False

        return cls(
            Z, mask=empty, mu=mu, num_phases_max=num_phases_max, **kwargs)

    @classmethod
    def from_matrix(cls, Z, mask=False, mu=None, num_phases_max=None,
                    **kwargs):
        """
        parse data into lnpi masked array

        Parameters
        ----------
        matrix : lnPi data

        mask : bool or bool array
            mask for matrix (True where data is masked)

        mu : array-like (d,)
            chem. pot. for each component

        num_phases_max : int or None (default None)
            max number of phases

        **kwargs : arguments to lnPi constructor
        """

        return cls(
            Z, mask=mask, mu=mu, num_phases_max=num_phases_max, **kwargs)

    def to_DataArray(self, rec_dim='rec', rec=None, **kwargs):
        """
        create a xarray.DataArray from self.

        Parameters
        ----------
        kwargs : extra arguments to xarray.DataArray

        if rec is not None, insert a record dimension in front
        """

        dims = tuple(['N_{}'.format(i) for i in range(self.ndim)])
        data = self.data
        mask = self.mask
        coords = {}

        if rec is None:
            attrs = self._optinfo
        else:
            dims = (rec_dim, ) + dims
            data = data[None, ...]
            mask = mask[None, ...]

            # other coords
            coords['rec'] = [rec]
            for k, v in self._optinfo.items():
                coords[k] = (rec_dim, np.atleast_1d(v))
            attrs = {}

        coords['mask'] = (dims, mask)

        d = xr.DataArray(data, dims=dims, coords=coords, **kwargs)
        d.attrs.update(**self._optinfo)

        return d

    @classmethod
    def from_DataArray(cls, da, rec_dim='rec', rec=None, **kwargs):
        """
        create a lnPi object from xarray.DataArray
        """

        if rec is not None:
            da = da.sel(rec=rec)
            attrs = {
                k: v.values
                for k, v in da.coords.items() if k not in [rec_dim, 'mask']
            }
        else:
            attrs = da.attrs

        data = da.values
        mask = da['mask'].values
        kwargs = dict(attrs, **kwargs)

        return cls(data, mask=mask, **kwargs)

    def to_hdf(self, path_or_buff, key, overwrite=False):
        """
        push self to h5 file

        Parameters
        ----------
        path_or_buff : string, h5py.File, or h5py.Group
            if string, use as path. otherwise, write to file or group

        key : string
            group to write to (relative to wherever path_or_buff points to)

        overwrite : bool (Default False)
            if key path_or_buff[key] exists, if overwrite is True, overwrite existing
            group, otherwise, create group
        """

        if isinstance(path_or_buff, (bytes, str)):
            p = h5py.File(path_or_buff)

        elif isinstance(path_or_buff, (h5py.File, h5py.Group)):
            p = path_or_buff
        else:
            raise ValueError('bad path_or_buff type %s' % (type(path_or_buff)))

        if key in p:
            if overwrite:
                del p[key]
            else:
                raise RuntimeError('key %s already exists' % key)

        group = p.create_group(key)

        data = group.create_dataset('data', data=self.data)
        mask = group.create_dataset('mask', data=self.mask)

        #add attributes to base
        for k, v in self._optinfo.items():
            group.attrs[k] = v

    @classmethod
    def from_hdf(cls, path_or_buff, key=None):
        """
        read in lnPi from file

        Parameters
        ----------
        path_or_buff : string, h5py.File, or h5py.Group
            if string, use as path. otherwise, write to file or group

        key : string
            group to write to (relative to wherever path_or_buff points to)

        Notes
        -----
        if key is None, assume path_or_buff is the group
        containing data/maks/attributes
        """

        if isinstance(path_or_buff, (bytes, str)):
            group = h5py.File(path_or_buff)[key]

        elif type(path_or_buff) is h5py.File:
            group = path_or_buff[key]

        elif type(path_or_buff) is h5py.Group:
            if key is None:
                group = path_or_buff
            else:
                group = path_or_buff[key]

        data = group['data'].value
        mask = group['mask'].value
        kwargs = dict(group.attrs)
        return lnPi(data, mask=mask, **kwargs)


################################################################################
#phases
################################################################################


def tag_phases_binary(x):
    if x.base.num_phases_max != 2:
        raise ValueError('bad tag function')

    L = []
    for p in x.phases:
        if p.molfrac[0] < 0.5:
            val = 0
        else:
            val = 1
        L.append(val)

    return np.array(L)


#function to tag 'LD' and 'HD' phases
def tag_phases_single(x, density_cut=0.5):
    if x.base.num_phases_max != 2:
        raise ValueError('bad tag function')

    if x.nphase == 1:
        if x[0].density < density_cut:
            return np.array([0])
        else:
            return np.array([1])

    elif x.nphase == 2:
        return np.argsort(x.argmax[0])
    else:
        raise ValueError('bad nphase')


class lnPi_phases(object):
    """
    object containing lnpi base and phases
    """

    def __init__(self,
                 base,
                 phases='get',
                 argmax='get',
                 argmax_kwargs=None,
                 phases_kwargs=None,
                 build_kwargs=None,
                 ftag_phases=None,
                 ftag_phases_kwargs=None):
        """
        object to store base and phases

        Parameters
        ----------
        base : lnPi object
            base object.  Not parsed to phases

        phases : list of lnPi objects, None, or str
            if list of lnPi objects, each corresponds to an independent phase.
            if None, implies single phase (phases=[base])
            if str and 'get', get phases on demand

        argmax : tuple of arrays (indicies) or str
            indices of local max locations.
            if str and 'get', get argmax on demand

        argmax_kwargs : dict
            if argmax=='get', use self.build_phases

        phases_kwargs : dict
            if phases=='get', use self.build_phases


        build_kwargs : dict
            arguments to self.buiild_phases

        ftag_phases : function (default tag_phases_binary)
            function which returns integer phase_id for each phase
            ...
            ftag_phases(self):
                returns [phase_id(i) for i in len(phases)]
        """

        self.base = base
        self.phases = phases
        self.argmax = argmax

        if ftag_phases is None:
            raise ValueError('must specify ftag_phases')

        if isinstance(ftag_phases, (bytes, str)):
            if ftag_phases == 'tag_phases_binary':
                ftag_phases = tag_phases_binary
            elif ftag_phases == 'tag_phases_single':
                ftag_phases = tag_phases_single
            else:
                raise ValueError('ftag_phases not recognized string')

        self._ftag_phases = ftag_phases

        if argmax_kwargs is None:
            argmax_kwargs = {}
        if phases_kwargs is None:
            phases_kwargs = {}
        if build_kwargs is None:
            build_kwargs = {}
        if ftag_phases_kwargs is None:
            ftag_phases_kwargs = {}

        self._argmax_kwargs = argmax_kwargs
        self._phases_kwargs = phases_kwargs
        self._build_kwargs = build_kwargs
        self._ftag_phases_kwargs = ftag_phases_kwargs


    ##################################################
    #copy
    def copy(self, **kwargs):
        """
        create shallow copy of self

        **kwargs : named arguments to lnPi_collection.__init__
        if argument is given, it will overide that in self
        """

        d = {}
        for k in [
                'base', 'phases', 'argmax', 'argmax_kwargs', 'phases_kwargs',
                'build_kwargs', 'ftag_phases', 'ftag_phases_kwargs'
        ]:
            if k in kwargs:
                d[k] = kwargs[k]
            else:
                _k = '_' + k
                d[k] = getattr(self, _k)

        return self.__class__(**d)

    ##################################################
    #reweight
    def reweight(self, mu, ZeroMax=True, Pad=False, **kwargs):
        """
        create a new lnpi_phases reweighted to new mu
        """

        return self.copy(
            base=self.base.reweight(mu, ZeroMax=ZeroMax, Pad=Pad, **kwargs),
            phases='get',
            argmax='get')

    ##################################################
    #properties
    @property
    def base(self):
        return self._base

    @base.setter
    def base(self, val):
        if type(val) is not lnPi:
            raise ValueError('base must be type lnPi %s' % (type(val)))
        self._base = val

    @property
    def argmax(self):
        #set on access
        if self._argmax == 'get':
            #self.argmax = self.base.argmax_local(**self._argmax_kwargs)
            self.build_phases(inplace=True, **self._build_kwargs)
        return self._argmax

    @argmax.setter
    def argmax(self, val):
        if val == 'get':
            pass
        else:
            assert self.base.ndim == len(val)
            assert len(val[0]) <= self.base.num_phases_max

        self._argmax = val

    def argmax_from_phases(self, inplace=False):
        """
        set argmax from max in each phase
        """
        L = [np.array(np.where(p.filled() == p.max())).T for p in self.phases]
        argmax = tuple(np.concatenate(L).T)

        if inplace:
            self.argmax = argmax
        else:
            return argmax

    @property
    def phases(self):
        if self._phases == 'get':
            #self.phases = self.base.get_list_indices(self.argmax,**self._phases_kwargs)
            if self._argmax == 'get':
                self.build_phases(inplace=True, **self._build_kwargs)
            else:
                self._phases = self.base.get_list_indices(
                    self.argmax, **self._phases_kwargs)

        if self._phases is None:
            return [self.base]
        else:
            return self._phases

    @phases.setter
    def phases(self, phases):

        if isinstance(phases, (tuple, list)):
            #assume that have list/tuple of lnPi phases
            for i, p in enumerate(phases):
                if type(p) is not lnPi:
                    raise ValueError('element %i of phases must be lnPi' % (i))

                if np.any(self.base.mu != p.mu):
                    raise ValueError('bad mu between base and comp %i: %s, %s'% \
                                     (i,base.mu,p.mu))

        elif isinstance(phases, np.ndarray):
            #assume labels:
            phases = self.base.get_list_labels(phases)

        elif phases is None or phases == 'get':
            #passing get to later
            pass

        else:
            raise ValueError(
                'phases must be a list of lnPi, label array, None, or "get"')

        self._phases = phases

    def __len__(self):
        return len(self.argmax[0])

    def __getitem__(self, i):
        return self.phases[i]


    @property
    def nphase(self):
        return len(self)

    @property
    def phaseIDs(self):
        return self._ftag_phases(self, **self._ftag_phases_kwargs)


    def phaseIDs_to_indicies(self, IDs):
        """
        convert IDs to indicies in self.phases
        """
        l = list(self.phaseIDs)
        return np.array([l.index(i) for i in IDs])

    @property
    def has_phaseIDs(self):
        """
        return array of bools which are True if index=phaseID is present
        """
        b = np.zeros(self.base.num_phases_max, dtype=bool)
        b[self.phaseIDs] = True
        return b

    @property
    def masks(self):
        return np.array([p.mask for p in self.phases])

    @property
    def labels(self):
        return masks_to_labels(
            self.masks, feature_value=False, values=self.phaseIDs)

    def _get_prop(self, prop):
        return [getattr(x, prop) for x in self]

    def _get_fprop(self, prop, *args, **kwargs):
        return [getattr(x, prop)(*args, **kwargs) for x in self]

    def _reindex(self, x):
        return x.assign_coords(phase=self.phaseIDs).reindex(phase=range(self.base.num_phases_max))

    @concatify_phase(cache=False)
    def molfrac(self):
        return self._get_prop('molfrac')
    @property
    def molfrac_phase(self):
        return self._reindex(self.molfrac)


    @concatify_phase(cache=False)
    def Nave(self):
        return self._get_prop('Nave')
    @property
    def Nave_phase(self):
        return self._reindex(self.Nave)

    @concatify_phase(cache=False)
    def Nvar(self):
        return self._get_prop('Nvar')
    @property
    def Nvar_phase(self):
        return self._reindex(self.Nvar)


    @concatify_phase(cache=False)
    def density(self):
        return self._get_prop('density')
    @property
    def density_phase(self):
        return self._reindex(self.density)

    @concatify_phase(cache=False, prop=False)
    def Omega(self, zval=None):
        return self._get_fprop('Omega',zval)
    def Omega_phase(self, zval=None):
        return self._reindex(self.Omega(zval))

    @property
    def pi(self):
        return [x.pi for x in self]

    @property
    def pi_norm(self):
        return [x.pi_norm for x in self]

    @property
    def mu(self):
        return self.base.mu

    @property
    def chempot(self):
        return self.base.chempot

    @property
    def beta(self):
        return self.base.beta

    @property
    def volume(self):
        return self.base.volume

    @property
    def coords(self):
        return self.base.coords

    ##################################################
    #query
    def _get_boundaries(self, IDs, mode='thick', connectivity=None, **kwargs):
        """
        get the boundary between phase pair

        Parameters
        ----------
        IDs : iterable of phases indices to get boundaries about

        mode : string (Default 'thick')
         mode passed to find_boundaries

        connectivity : int (Default None)
         if None, use self.ndim

        **kwargs : extra arguments to find_boundaries

        Returns
        -------
        output : array of shape self.base.shape of bools
          output==True at boundary locations
        """

        if connectivity is None:
            connectivity = self.base.ndim
        b = []
        for i in IDs:
            p = self.phases[i]
            msk = np.atleast_2d((~p.mask).astype(int))
            b.append(
                find_boundaries(
                    msk, mode=mode, connectivity=connectivity, **kwargs))
        return b

        # b = np.prod(b,axis=0).astype(bool).reshape(self.base.shape)
        # b *= ~self.base.mask

        # return b

    def _get_boundaries_overlap(self,
                                IDs,
                                mode='thick',
                                connectivity=None,
                                **kwargs):
        """
        get overlap between phases in IDs
        """

        boundaries = self._get_boundaries(
            IDs, mode=mode, connectivity=connectivity, **kwargs)

        #loop over all combinations
        ret = {}
        for i, j in itertools.combinations(range(len(IDs)), 2):

            b = np.prod([boundaries[i],boundaries[j]],axis=0)\
                  .astype(bool).reshape(self.base.shape) * \
                  (~self.base.mask)

            if b.sum() == 0:
                b = None

            ret[i, j] = b

        return ret

    @property
    def betaEmin(self):
        """
        betaE_min = -max{lnPi}
        """
        return -np.array([p.max() for p in self.phases])

    def betaEtransition(self, IDs, **kwargs):
        """
        minimum value of energy at boundary between phases
        beta E_boundary = - max{lnPi}_{along boundary}

        if no boundary found between phases (i.e., they are not connected),
        then return vmax
        """

        boundaries = self._get_boundaries_overlap(IDs, **kwargs)

        ret = {}
        for k in boundaries:
            b = boundaries[k]

            if b is None:
                ret[k] = np.nan
            else:
                ret[k] = -(self.base[b].max())

        return ret

    def betaEtransition_matrix(self, **kwargs):
        """
        Transition point energy for all pairs

        out[i,j] = transition energy between phase[i] and phase[j]
        """

        ET = self.betaEtransition(range(self.nphase), **kwargs)

        out = np.empty((self.nphase, ) * 2, dtype=float) * np.nan
        for i, j in ET:
            out[i, j] = out[j, i] = ET[i, j]

        return out

    def DeltabetaE_matrix(self, vmax=1e20, **kwargs):
        """
        out[i,j]=DeltaE between phase[i] and phase[j]

        if no transition between (i,j) , out[i,j] = vmax
        """
        out = self.betaEtransition_matrix(**kwargs) - \
             self.betaEmin[:,None]

        #where nan and not on diagonal, set to vmax
        out[np.isnan(out)] = vmax
        np.fill_diagonal(out, np.nan)

        return out

    def DeltabetaE_matrix_phaseIDs(self, vmin=0.0, vmax=1e20, **kwargs):
        """
        out[i,j] =Delta E between phaseID==i and phaseID==j

        if i does not exist, then out[i,j] = vmin
        if not transition to j, then out[i,j] = vmax
        """

        dE = self.DeltabetaE_matrix(vmax, **kwargs)
        out = np.empty((self.base.num_phases_max, ) * 2, dtype=float) * np.nan
        phaseIDs = self.phaseIDs

        for i, ID in enumerate(phaseIDs):
            out[ID, phaseIDs] = dE[i, :]

        #where nan fill in with vmax
        out[np.isnan(out)] = vmax
        #where phase does not exist, fill with vmin
        has_phaseIDs = self.has_phaseIDs
        out[~has_phaseIDs, :] = vmin
        np.fill_diagonal(out, np.nan)

        return out

    def DeltabetaE_phaseIDs(self, vmin=0.0, vmax=1e20, **kwargs):
        """
        minimum transition energy from phaseID to any other phase
        """
        return np.nanmin(
            self.DeltabetaE_matrix_phaseIDs(vmin, vmax, **kwargs), axis=-1)

    def _betaEtransition_line(self, pair=(0, 1), connectivity=None):
        """
        find transition energy from line connecting location of maxima
k        """

        args = tuple(np.array(self.argmax).T[pair, :].flatten())

        img = np.zeros(self.base.shape, dtype=int)
        img[draw.line(*args)] = 1

        if connectivity is None:
            msk = img.astype(bool)
        else:
            msk = find_boundaries(img, connectivity=connectivity)

        return -(self.base[msk].min())

    def _betaEtransition_path(self, pair=(0, 1), **kwargs):
        """
        construct a path connecting maxima
        """

        idx = np.array(self.argmax).T

        d = -self.base.data.copy()
        d[self.base.mask] = d.max()

        i, w = route_through_array(d, idx[pair[0]], idx[pair[1]], **kwargs)

        i = tuple(np.array(i).T)

        return -(self.base[i].min())

    def sort_by_phaseIDs(self, inplace=False):
        """
        sort self so that self.phaseIDs are increasing
        """

        idx = np.argsort(self.phaseIDs)
        argmax = tuple(x[idx] for x in self.argmax)
        argmax = tuple(np.array(self.argmax).T[idx, :].T)
        L = [self._phases[i] for i in idx]

        if inplace:
            self.argmax = argmax
            self.phases = L
        else:
            return self.copy(phases=L)

    ##################################################
    #repr
    def _repr_html_(self):
        if self._phases == 'get':
            x = self._phases
        else:
            x = self.nphase
        return 'lnPi_phases: nphase=%s, mu=%s' % (x, self.base.mu)

    @staticmethod
    def _get_DE(Etrans, Emin, vmax=1e20):
        DE = Etrans - Emin[:, None]
        DE[np.isnan(DE)] = vmax
        np.fill_diagonal(DE, np.nan)
        return DE

    def merge_phases(self,
                     efac=1.0,
                     vmax=1e20,
                     inplace=False,
                     force=True,
                     **kwargs):
        """
        merge phases such that DeltabetaE[i,j]>efac-tol for all phases

        Parameters
        ----------
        efac : float (Default 1.0)
            cutoff

        vmax : float (1e20)
            vmax parameter in DeltabetaE calculation

        inplace : bool (Default False)
            if True, do inplace modificaiton, else return

        force : bool (Default False)
            if True, iteratively remove minimum energy difference
            (even if DeltabetaE>efac) until have
            exactly self.base.num_phases_max


        **kwargs : arguments to DeltabetE_matrix


        Return
        ------
        out : lnPi_phases object (if inplace is False)
        """

        if self._phases is None:
            if inplace:
                return
            else:
                return self.copy()

        #first get Etrans,Emin
        Etrans = self.betaEtransition_matrix()
        Emin = self.betaEmin

        #will iteratively reduce dE=Etrans - Emin
        #keep placeholder of absolute indices still in dE (number)
        #as well as list of phases merged together (L)
        #where L==None, phase has been removed.  removed phases
        #are appended to L[i] where i is the phase merged into.
        number = np.arange(Emin.shape[0])
        L = [[i] for i in range(len(Emin))]

        while True:
            nphase = len(Emin)
            if nphase == 1:
                break
            DE = self._get_DE(Etrans, Emin, vmax=vmax)

            min_val = np.nanmin(DE)

            if min_val > efac:
                if not force:
                    break
                elif nphase <= self.base.num_phases_max:
                    break

            idx = np.where(DE == min_val)
            idx_kill, idx_keep = idx[0][0], idx[1][0]

            #add in new row taking min of each transition energy
            new_row = np.nanmin(Etrans[[idx_kill, idx_keep], :], axis=0)
            new_row[idx_keep] = np.nan

            #add in new row/col
            Etrans[idx_keep, :] = new_row
            Etrans[:, idx_keep] = new_row

            #delete idx_kill
            Etrans = np.delete(np.delete(Etrans, idx_kill, 0), idx_kill, 1)
            Emin = np.delete(Emin, idx_kill)

            #update L
            L[number[idx_keep]] += L[number[idx_kill]]
            L[number[idx_kill]] = None
            #update number
            number = np.delete(number, idx_kill)

        msk = np.array([x is not None for x in L])

        argmax_new = tuple(x[msk] for x in self._argmax)
        phases_new = []

        for x in L:
            if x is None: continue
            mask = np.all([self._phases[i].mask for i in x], axis=0)
            phases_new.append(self.base.new_mask(mask))

        if inplace:
            self.argmax = argmax_new
            self.phases = phases_new
        else:
            return self.copy(phases=phases_new, argmax=argmax_new)

    def merge_phaseIDs(self, inplace=False, **kwargs):
        """
        if phaseIDs are equal, merge them to a single phase
        """

        if inplace:
            new = self
        else:
            new = self.copy()

        phaseIDs = self._ftag_phases(self, **self._ftag_phases_kwargs)
        if len(phaseIDs) > 0:
            #find distance between phaseIDs
            dist = pdist(phaseIDs.reshape(-1, 1)).astype(int)
            if np.any(dist == 0):
                #have some phases with same phaseIDs.

                #create list of where
                L = []
                for i in range(self.base.num_phases_max):
                    w = np.where(phaseIDs == i)[0]
                    if len(w) > 0:
                        L.append(w)

                #merge phases
                phases_new = []
                for x in L:
                    if len(x) == 1:
                        #only one phase has this phase
                        phases_new.append(self.phases[i])
                    else:
                        #multiple phases have this phaseID
                        mask = np.all([self.phases[i].mask for i in x], axis=0)
                        phases_new.append(self.base.new_mask(mask))

                new.phases = phases_new
                new.argmax = new.argmax_from_phases()
                if hasattr(self, '_phaseIDs'):
                    del self._phaseIDs

        if not inplace:
            return new

    def build_phases(self,
                     merge='full',
                     nmax_start=10,
                     efac=0.1,
                     vmax=1e20,
                     inplace=False,
                     force=True,
                     merge_phaseIDs=True,
                     **kwargs):
        """
        iteratively build phases by finding argmax merging phases

        Parameters
        ----------
        merge : str or None (Default 'partial')
            if None, don't perform merge.
            if 'full', perform normal merge with passed `force` and `efac`
            if 'partial' -> only merge if necessary and with force=True, efac=0.0
            (i.e., will leave phases with dE<`efac`)


        nmax_start : int (Default 10)
            max number of phases argmax_local to start with


        efac : float (Default 0.2)
            merge all phases where DeltabetaE_matrix()[i,j] < efac-tol

        vmax : float (Default 1e20)
            value of DeltabetaE if phase transition between i and j does not exist


        inplace : bool (Default False)
            if True, do inplace modification

        force : bool (Default True)
            if True, keep removing the minimum transition phases regardless of dE until
            nphase==num_phases_max

        merge_phaseIDs : bool (Default True)
            if True, merge phases with same phaseIDs

        **kwargs : extra arguments to self.merge_phases

        Returns
        -------
        out : lnPi_phases (optional, if inplace is False)
        """

        #print 'building phase'

        if inplace:
            t = self
        else:
            t = self.copy()

        if t._argmax == 'get':
            #use _argmax to avoid num_phases_max check
            t._argmax = t._base.argmax_local(
                num_phases_max=nmax_start, **t._argmax_kwargs)

        if t._phases == 'get':
            #use _phases to avoid checks
            t._phases = self._base.get_list_indices(t._argmax,
                                                    **t._phases_kwargs)

        if t.nphase == 1:
            #nothing to do here
            pass

        elif merge is None:
            pass

        elif merge == 'full':
            t.merge_phases(
                efac=efac, vmax=vmax, inplace=True, force=force, **kwargs)

        elif merge == 'partial':
            t.merge_phases(
                efac=0.0, vmax=vmax, inplace=True, force=True, **kwargs)

        #do a sanity check
        t.argmax = t._argmax
        t.phases = t._phases

        if merge_phaseIDs:
            t.merge_phaseIDs(inplace=True)

        if not inplace:
            return t

    @classmethod
    def from_file(cls,
                  filename,
                  mu=None,
                  num_phases_max=None,
                  volume=None,
                  beta=None,
                  ZeroMax=True,
                  Pad=False,
                  loadtxt_kwargs=None,
                  argmax_kwargs=None,
                  phases_kwargs=None,
                  build_kwargs=None,
                  ftag_phases=None,
                  ftag_phases_kwargs=None,
                  **kwargs):
        """
        1. create lnPi object from file.  see documentation of lnPi.from_file
        2. return lnPi_phases object from lnPi object
        """

        base = lnPi.from_file(
            filename=filename,
            mu=mu,
            num_phases_max=num_phases_max,
            volume=volume,
            beta=beta,
            loadtxt_kwargs=loadtxt_kwargs,
            ZeroMax=ZeroMax,
            Pad=Pad,
            **kwargs)

        return cls(
            base,
            phases='get',
            argmax='get',
            argmax_kwargs=argmax_kwargs,
            phases_kwargs=phases_kwargs,
            build_kwargs=build_kwargs,
            ftag_phases=ftag_phases,
            ftag_phases_kwargs=ftag_phases_kwargs)

    @classmethod
    def from_data(cls,
                  data,
                  mu=None,
                  num_phases_max=None,
                  volume=None,
                  beta=None,
                  ZeroMax=True,
                  Pad=False,
                  argmax_kwargs=None,
                  phases_kwargs=None,
                  build_kwargs=None,
                  ftag_phases=None,
                  ftag_phases_kwargs=None,
                  **kwargs):
        """
        1. create lnPi object from data.  see documentation of lnPi.from_data
        2. return lnPi_phases object from lnPi object
        """

        base = lnPi.from_data(
            data,
            mu=mu,
            num_phases_max=num_phases_max,
            volume=volume,
            beta=beta,
            ZeroMax=ZeroMax,
            Pad=Pad,
            **kwargs)
        return cls(
            base,
            phases='get',
            argmax='get',
            argmax_kwargs=argmax_kwargs,
            phases_kwargs=phases_kwargs,
            build_kwargs=build_kwargs,
            ftag_phases=ftag_phases,
            ftag_phases_kwargs=ftag_phases_kwargs)

    @classmethod
    def from_matrix(cls,
                    Z,
                    mask=False,
                    mu=None,
                    num_phases_max=None,
                    volume=None,
                    beta=None,
                    ZeroMax=True,
                    Pad=False,
                    argmax_kwargs=None,
                    phases_kwargs=None,
                    build_kwargs=None,
                    ftag_phases=None,
                    ftag_phases_kwargs=None,
                    **kwargs):
        """
        1. create lnPi object from matrix.  see documentation of lnPi.from_matrix
        2. create lnPi_phases object from lnPi object
        """

        base = lnPi.from_matrix(
            Z,
            mask=mask,
            mu=mu,
            num_phases_max=num_phases_max,
            volume=volume,
            beta=beta,
            ZeroMax=ZeroMax,
            Pad=Pad,
            **kwargs)

        return cls(
            base,
            phases='get',
            argmax='get',
            argmax_kwargs=argmax_kwargs,
            phases_kwargs=phases_kwargs,
            build_kwargs=build_kwargs,
            ftag_phases=ftag_phases,
            ftag_phases_kwargs=ftag_phases_kwargs)

    @classmethod
    def from_labels(cls,
                    base,
                    labels,
                    argmax='get',
                    SegLenOne=False,
                    masks_kwargs={},
                    **kwargs):
        """
        create lnPi_phases from labels
        """
        # TODO

        phases = base.get_list_labels(
            labels, SegLenOne=SegLenOne, **mask_kwargs)
        new = cls(base=base, phases=phases, argmax=argmax)

        if argmax == 'get':
            new.argmax = new.argmax_from_phases()

        # assert type(ref) is lnPi_phases
        # new = ref.reweight(mu, **kwargs)
        # new.phases = new.base.get_list_labels(
        #     labels, SegLenOne=SegLenOne, **masks_kwargs)
        # new.argmax = new.argmax_from_phases()

        return new

#    def to_DataSet()

    def to_hdf(self,
               path_or_buff,
               key,
               overwrite=False,
               base=True,
               phases=True,
               dicts=True):
        """
        push to h5 file

        Parameters
        ----------
        path_or_buff : string, h5py.File, or h5py.Group
            if string, use as path. otherwise, write to file or group

        key : string
            group to write to (relative to wherever path_or_buff points to)

        overwrite : bool (Default False)
            if key path_or_buff[key] exists, if overwrite is True, overwrite existing
            group, otherwise, create group

        base,phases,dicts : bool (default True)
            if True write self.base, self.phases, self._argmax_kwargs,...
        """
        if isinstance(path_or_buff, (bytes, str)):
            p = h5py.File(path_or_buff)

        elif isinstance(path_or_buff, (h5py.File, h5py.Group)):
            p = path_or_buff
        else:
            raise ValueError('bad path_or_buff type %s' % (type(path_or_buff)))

        if key in p:
            if overwrite:
                del p[key]
            else:
                raise RuntimeError('key %s already exists' % key)
        group = p.create_group(key)

        if base:
            self.base.to_hdf(group, 'base', overwrite)

        if phases:
            if 'phases' in group:
                if overwrite:
                    del group['phases']
            group.create_dataset('phases', data=self.labels.astype(np.uint8))

        if dicts:
            for x in ['argmax', 'phases', 'build', 'ftag_phases']:
                kL = x + '_kwargs'
                kR = '_' + kL
                group.attrs[kL] = repr(getattr(self, kR))

            group.attrs['ftag_phases'] = getattr(self._ftag_phases, '__name__',
                                                 None)

    @staticmethod
    def from_hdf(path_or_buff, key=None, base=None, phases=None, **kwargs):
        """
        read in lnpi_phases from h5py file

        Paremters
        ---------
        path_or_buff : string, h5py.File, or h5py.Group
            if string, use as path. otherwise, write to file or group

        key : string
            group to write to (relative to wherever path_or_buff points to)

        base : Default None
            if not None, use this as base

        phases : Default NOne
            if not None, use this as phases

        **kwargs : extra argunets to lnPi_phases
            overides parameters read from file

        Returns
        -------
        out : lnPi_phases

        Notes
        -----
        if key is None, assume path_or_buff is the group
        containing data
        """

        if isinstance(path_or_buff, (bytes, str)):
            group = h5py.File(path_or_buff)[key]

        elif type(path_or_buff) is h5py.File:
            group = path_or_buff[key]

        elif type(path_or_buff) is h5py.Group:
            if key is None:
                group = path_or_buff
            else:
                group = path_or_buff[key]

        if base is None:
            base = lnPi.from_hdf(group, 'base')

        if phases is None:
            phases = group['phases'].value

        ev = lambda x: eval(x) if '{' in x else x
        file_kwargs = {k: ev(v) for k, v in group.attrs.items()}

        #passed kwargs overide read kwargs
        kwargs = dict(file_kwargs, **kwargs)

        if kwargs['ftag_phases'] is None:
            raise ValueError('must specify ftag_phases. None found in file')

        new = lnPi_phases(base=base, phases=phases, **kwargs)
        if phases is not 'get':
            new.argmax_from_phases(inplace=True)

        return new


################################################################################
#collection
################################################################################
class lnPi_collection(object):
    """
    class containing several lnPis
    """

    def __init__(self, lnpis):
        """
        Parameters
        ----------
        lnpis : iterable of lnpi_phases objects
        """
        self.lnpis = lnpis

    ##################################################
    #copy
    def copy(self, lnpis=None):
        """
        create shallow copy of self

        **kwargs : named arguments to lnPi_collection.__init__
        if argument is given, it will overide that in self
        """
        if lnpis is None:
            lnpis = self.lnpis[:]

        return lnPi_collection(lnpis=lnpis)

    ##################################################
    #setters
    def _parse_lnpi(self, x):
        if type(x) is not lnPi_phases:
            raise ValueError('bad value while parsing element %s' % (type(x)))
        else:
            return x

    def _parse_lnpis(self, lnpis):
        """from a list of lnpis, return list of lnpi_phases"""
        if not isinstance(lnpis, (list, tuple)):
            raise ValueError('lnpis must be list or tuple')

        return [self._parse_lnpi(x) for x in lnpis]

    ##################################################
    #properties
    @property
    def lnpis(self):
        return self._lnpis

    @lnpis.setter
    def lnpis(self, val):
        self._lnpis = self._parse_lnpis(val)

    ##################################################
    #list props
    def __len__(self):
        return len(self.lnpis)

    @property
    def shape(self):
        return (len(self), )

    def append(self, val, unique=True, decimals=5):
        """append a value to self.lnpis"""
        if unique:
            if len(self._unique_mus(val.mu, decimals)) > 0:
                self._lnpis.append(self._parse_lnpi(val))
        else:
            self._lnpis.append(self._parse_lnpi(val))

    def extend(self, x, unique=True, decimals=5):
        """extend lnpis"""
        if isinstance(x, lnPi_collection):
            x = x._lnpis

        if isinstance(x, list):
            if unique:
                x = self._unique_list(x, decimals)
            self._lnpis.extend(self._parse_lnpis(x))
        else:
            raise ValueError('only lists or lnPi_collections can be added')

    def extend_by_mu_iter(self, ref, mus, unique=True, decimals=5, **kwargs):
        """extend by mus"""
        if unique:
            mus = self._unique_mus(mus, decimals=decimals)
        new = lnPi_collection.from_mu_iter(ref, mus, **kwargs)
        self.extend(new, unique=False)

    def extend_by_mu(self, ref, mu, x, unique=True, decimals=5, **kwargs):
        """extend self my mu values"""
        mus = get_mu_iter(mu, x)
        self.extend_by_mu_iter(ref, mus, unique, decimals, **kwargs)

    def __add__(self, x):
        if isinstance(x, list):
            L = self._lnpis + self._parse_lnpis(x)
        elif isinstance(x, lnPi_collection):
            L = self._lnpis + x._lnpis
        else:
            raise ValueError('only lists or lnPi_collections can be added')

        return self.copy(lnpis=L)

    def __iadd__(self, x):
        if isinstance(x, list):
            L = self._parse_lnpis(x)
        elif isinstance(x, lnPi_collection):
            L = x._lnpis
        else:
            raise ValueError('only list or lnPi_collections can be added')

        self._lnpis += L
        return self

    def sort_by_mu(self, comp=0, inplace=False):
        """
        sort self.lnpis by mu[:,comp]
        """
        order = np.argsort(self.mus[:, comp])
        L = [self._lnpis[i] for i in order]
        if inplace:
            self._lnpis = L
        else:
            return self.copy(lnpis=L)

    def _unique_list(self, L, decimals=5):
        """
        limit list such that output[i].mu not in self.mus
        """

        tol = 0.5 * 10**(-decimals)
        mus = np.array([x.mu for x in L])
        new = np.atleast_2d(mus)

        msk = np.all(cdist(self.mus, new) > tol, axis=0)

        return [x for x, m in zip(L, msk) if m]

    def _unique_mus(self, mus, decimals=5):
        """
        return only those mus not already in self

        Parameters
        ----------
        mus : arrray of new mus
            shape is (ncomp,) or (m,ncomp). make 2d if not already

        decimals : int (Default 5)
            consider mu replicated if dist between any mu already in
            self and mus[i] <0.5*10**(-decimals)

        Returns
        -------
        output : bool or array of bools
        """

        tol = 0.5 * 10**(-decimals)
        mus = np.asarray(mus)
        new = np.atleast_2d(mus)
        msk = np.all(cdist(self.mus, new) > tol, axis=0)
        return new[msk, :]

    def drop_duplicates(self, decimals=5):
        """
        drop doubles of given mu
        """
        tol = 0.5 * 10**(-decimals)

        mus = self.mus
        msk = squareform(pdist(mus)) < tol
        np.fill_diagonal(msk, False)

        a, b = np.where(msk)
        b = b[b > a]

        keep = np.ones(mus.shape[0], dtype=bool)
        keep[b] = False

        self._lnpis = [x for x, m in zip(self._lnpis, keep) if m]

    def __getitem__(self, i):
        if isinstance(i, (np.int, np.integer)):
            return self.lnpis[i]

        elif type(i) is slice:
            L = self.lnpis[i]

        elif isinstance(i, (list, np.ndarray)):
            idx = np.array(i)

            if np.issubdtype(idx.dtype, np.integer):
                L = [self.lnpis[j] for j in idx]

            elif np.issubdtype(idx.dtype, np.bool):
                assert idx.shape == self.shape
                L = [xx for xx, mm in zip(self.lnpis, idx) if mm]

            else:
                raise KeyError('bad key')

        return self.copy(lnpis=L)

    def merge_phases(self,
                     efac=1.0,
                     vmax=1e20,
                     inplace=False,
                     force=True,
                     **kwargs):
        L = [
            x.merge_phases(
                efac=efac, vmax=vmax, inplace=inplace, force=force, **kwargs)
            for x in self._lnpis
        ]
        if not inplace:
            return self.copy(lnpis=L)

    ##################################################
    #calculations/props
    @property
    def mus(self):
        return np.array([x.mu for x in self.lnpis])

    def _get_prop(self, prop):
        return [getattr(x, prop) for x in self.lnpis]

    def _get_fprop(self, prop, *args, **kwargs):
        return [getattr(x, prop)(*args, **kwargs) for x in self.lnpis]

    @concatify_rec(cache=False)
    def chempot(self):
        return self._get_prop('chempot')
    @concatify_rec(cache=False)
    def molfrac_phase(self):
        return self._get_prop('molfrac_phase')
    @concatify_rec(cache=False)
    def Nave_phase(self):
        return self._get_prop('Nave_phase')
    @concatify_rec(cache=False)
    def Nvar_phase(self):
        return self._get_prop('Nvar_phase')
    @concatify_rec(cache=False)
    def density_phase(self):
        return self._get_prop('density_phase')
    @concatify_rec(cache=False, prop=False)
    def Omega_phase(self, zval=None):
        return self._get_fprop('Omega_phase',zval)

    @property
    def nphases(self):
        return np.array([x.nphase for x in self.lnpis])
    @property
    def has_phaseIDs(self):
        return np.array([x.has_phaseIDs for x in self.lnpis])

    # @property
    # def molfracs(self):
    #     return pd.concat([x.molfracs for x in self.lnpis])

    # @property
    # def molfracs_phaseIDs(self):
    #     return pd.concat([x.molfracs_phaseIDs for x in self.lnpis])

    # @property
    # def Naves(self):
    #     return pd.concat([x.Naves for x in self.lnpis])

    # @property
    # def Naves_phaseIDs(self):
    #     return pd.concat([x.Naves_phaseIDs for x in self.lnpis])

    # @property
    # def densities_phaseIDs(self):
    #     return pd.concat([x.densities_phaseIDs for x in self.lnpis])

    # def Omegas(self, zval=None):
    #     return pd.concat([x.Omegas(zval) for x in self.lnpis])

    # def Omegas_phaseIDs(self, zval=None):
    #     return pd.concat([x.Omegas_phaseIDs(zval) for x in self.lnpis])

    def DeltabetaE_phaseIDs(self, vmin=0.0, vmax=1e20, **kwargs):
        return np.array(
            [x.DeltabetaE_phaseIDs(vmin, vmax, **kwargs) for x in self])

    ##################################################
    #spinodal
    def get_spinodal_phaseID(self,
                             ID,
                             efac=1.0,
                             dmu=0.5,
                             vmin=0.0,
                             vmax=1e20,
                             ntry=20,
                             step=None,
                             nmax=20,
                             reweight_kwargs={},
                             DeltabetaE_kwargs={},
                             close_kwargs={},
                             solve_kwargs={},
                             full_output=False):
        """
        locate spinodal for phaseID ID
        """

        s, r = spinodal.get_spinodal(
            self,
            ID,
            efac=efac,
            dmu=dmu,
            vmin=vmin,
            vmax=vmax,
            ntry=ntry,
            step=step,
            nmax=nmax,
            reweight_kwargs=reweight_kwargs,
            DeltabetaE_kwargs=DeltabetaE_kwargs,
            close_kwargs=close_kwargs,
            solve_kwargs=solve_kwargs,
            full_output=True)

        if full_output:
            return s, r
        else:
            return s

    def get_spinodals(self,
                      efac=1.0,
                      dmu=0.5,
                      vmin=0.0,
                      vmax=1e20,
                      ntry=20,
                      step=None,
                      nmax=20,
                      reweight_kwargs={},
                      DeltabetE_kwargs={},
                      close_kwargs={},
                      solve_kwargs={},
                      inplace=True,
                      append=True):

        L = []
        info = []
        for ID in range(self[0].base.num_phases_max):
            s, r = self.get_spinodal_phaseID(
                ID,
                efac=efac,
                dmu=dmu,
                vmin=vmin,
                vmax=vmax,
                ntry=ntry,
                step=step,
                nmax=nmax,
                reweight_kwargs=reweight_kwargs,
                DeltabetaE_kwargs=DeltabetE_kwargs,
                close_kwargs=close_kwargs,
                solve_kwargs=solve_kwargs,
                full_output=True)

            L.append(s)
            info.append(r)

        if append:
            for x in L:
                if x is not None:
                    self.append(x)

        if inplace:
            self._spinodals = L
            self._spinodals_info = info
        else:
            return L, info

    @property
    def spinodals(self):
        if not hasattr(self, '_spinodals'):
            raise AttributeError('spinodal not set')

        return self._spinodals

    ##################################################
    #binodal
    def get_binodal_pair(self,
                         IDs,
                         spinodals=None,
                         reweight_kwargs={},
                         full_output=False,
                         **kwargs):

        if spinodals is None:
            spinodals = self.spinodals
        spin = [self.spinodals[i] for i in IDs]

        if None in spin:
            #raise ValueError('one of spinodals is Zero')
            b, r = None, None
        else:
            b, r = binodal.get_binodal_point(
                self[0],
                IDs,
                spin[0].mu,
                spin[1].mu,
                reweight_kwargs=reweight_kwargs,
                full_output=True,
                **kwargs)

        if full_output:
            return b, r
        else:
            return b

    def get_binodals(self,
                     spinodals=None,
                     reweight_kwargs={},
                     inplace=True,
                     append=True,
                     **kwargs):

        if spinodals is None:
            spinodals = self.spinodals

        L = []
        info = []
        for IDs in itertools.combinations(
                range(self[0].base.num_phases_max), 2):

            b, r = self.get_binodal_pair(
                IDs,
                spinodals,
                reweight_kwargs=reweight_kwargs,
                full_output=True,
                **kwargs)

            L.append(b)
            info.append(r)

        if append:
            for x in L:
                if x is not None:
                    self.append(x)

        if inplace:
            self._binodals = L
            self._binodals_info = info

        else:
            return L, info

    @property
    def binodals(self):
        if not hasattr(self, '_binodals'):
            raise AttributeError('binodals not set')

        return self._binodals

    def get_binodal_interp(self, mu_axis, IDs=(0, 1)):
        """
        get position of Omega[i]==Omega[j] by interpolation
        """

        idx = np.asarray(IDs)

        assert (len(idx) == 2)

        x = self.Omegas_phaseIDs()[:, idx]
        msk = np.prod(~np.isnan(x), axis=1).astype(bool)
        assert (msk.sum() > 0)

        diff = x[msk, 0] - x[msk, 1]

        mu = self.mus[msk, mu_axis]
        i = np.argsort(diff)

        return np.interp(0.0, diff[i], mu[i])

    ##################################################
    def _repr_html_(self):
        return 'lnPi_collection: %s' % len(self)

    ##################################################
    #builders
    ##################################################
    @classmethod
    def from_mu_iter(cls, ref, mus, **kwargs):
        """
        build lnPi_collection from mus

        Parameters
        ----------
        ref : lnpi_phases object
            lnpi_phases to reweight to get list of lnpi's

        mus : iterable
            chem. pots. to get lnpi

        **kwargs : arguments to ref.reweight

        Returns
        -------
        out : lnPi_collection object
        """

        assert isinstance(ref, lnPi_phases)

        kwargs = dict(dict(ZeroMax=True), **kwargs)

        L = [ref.reweight(mu, **kwargs) for mu in mus]

        return cls(L)

    @classmethod
    def from_mu(cls, ref, mu, x, **kwargs):
        """
        build lnPi_collection from mu builder

        Parameters
        ----------
        ref : lnpi object
            lnpi to reweight to get list of lnpi's

        mu : list
            list with one element equal to None.
            This is the component which will be varied
            For example, mu=[mu0,None,mu2] implies use values
            of mu0,mu2 for components 0 and 2, and vary component 1

        x : array
            values to insert for variable component

        **kwargs : arguments to ref.reweight

        Returns
        -------
        out : lnPi_collection object
        """

        mus = get_mu_iter(mu, x)
        return cls.from_mu_iter(ref, mus, **kwargs)

    def to_hdf(self, path_or_buff, key, ref=None, overwrite=False):
        """
        push self to h5py file

        Parameters
        ----------
        path_or_buff : string, h5py.File, or h5py.Group
            if string, use as path. otherwise, write to file or group

        key : string
            group to write to (relative to wherever path_or_buff points to)

        overwrite : bool (Default False)
            if key path_or_buff[key] exists, if overwrite is True, overwrite existing
            group, otherwise, create group

        ref : lnpi_phases object or None
            if not None, push ref to lnpi_ref
        """
        if isinstance(path_or_buff, (bytes, str)):
            p = h5py.File(path_or_buff)

        elif isinstance(path_or_buff, (h5py.File, h5py.Group)):
            p = path_or_buff
        else:
            raise ValueError('bad path_or_buff type %s' % (type(path_or_buff)))

        if key in p:
            if overwrite:
                del p[key]
            else:
                raise RuntimeError('key %s already exists' % key)
        group = p.create_group(key)

        if ref is not None:
            ref.to_hdf(group, 'lnpi_ref', overwrite=overwrite)

        #push mus
        group.create_dataset('mus', data=self.mus)

        #push labels
        group.create_dataset(
            'phases', data=np.array([x.labels for x in self]).astype(np.uint8))

        #push indices of binodal

        if hasattr(self, '_spinodals'):
            L = []
            for s in self._spinodals:
                if s is None:
                    idx = -1
                else:
                    idx = np.where(np.all(self.mus == s.mu, axis=-1))[0][0]
                L.append(idx)
            group.create_dataset('spinodals', data=np.array(L))

        if hasattr(self, '_binodals'):
            L = []
            for b in self._binodals:
                if b is None:
                    idx = -1
                else:
                    idx = np.where(np.all(self.mus == b.mu, axis=-1))[0][0]
                L.append(idx)
            group.create_dataset('binodals', data=np.array(L))

    @staticmethod
    def from_hdf(path_or_buff, key=None, ref=None):
        """
        read in lnpi_phases from h5py file

        Paremters
        ---------
        path_or_buff : string, h5py.File, or h5py.Group
            if string, use as path. otherwise, write to file or group

        key : string
            group to write to (relative to wherever path_or_buff points to)

        ref : lnpi_phases or None
            if None, read reference from file, else use lnpi_phases as reference.

        Returns
        -------
        out : lnPi_collection

        Notes
        -----
        if key is None, assume path_or_buff is the group
        containing data
        """

        if isinstance(path_or_buff, (bytes, str)):
            group = h5py.File(path_or_buff)[key]

        elif type(path_or_buff) is h5py.File:
            group = path_or_buff[key]

        elif type(path_or_buff) is h5py.Group:
            if key is None:
                group = path_or_buff
            else:
                group = path_or_buff[key]

        if ref is None:
            ref = lnPi_phases.from_hdf(group, 'lnpi_ref')

        mus = group['mus'].value
        phases = group['phases'].value

        new = lnPi_collection.from_mu_iter(ref, mus)

        for i, p in enumerate(new):
            p.phases = phases[i]
            p.argmax_from_phases(inplace=True)

        #spinodals?
        if 'spinodals' in group:
            new._spinodals = [
                new[i] if i >= 0 else None for i in group['spinodals']
            ]

        if 'binodals' in group:
            new._binodals = [
                new[i] if i >= 0 else None for i in group['binodals']
            ]

        return new
