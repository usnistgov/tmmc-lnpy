"""
routines to segment lnPi

 1. find max/peaks in lnPi
 2. segment lnPi about these peaks
 3. determine free energy difference between segments
    a. Merge based on low free energy difference
 4. combination of 1-3.
"""

import warnings
from collections.abc import Iterable

import bottleneck
import numpy as np
import pandas as pd
import xarray as xr
from skimage import feature, morphology, segmentation

from .cached_decorators import gcached
from .collectionlnpi import CollectionlnPi
from .utils import get_tqdm_calc as get_tqdm
from .utils import parallel_map_func_starargs
from .wlnPi import FreeEnergylnPi


def peak_local_max_adaptive(
    data,
    mask=None,
    min_distance=[5, 10, 15, 20, 25],
    threshold_rel=0.00,
    threshold_abs=0.2,
    num_peaks_max=None,
    indices=True,
    errors="warn",
    **kwargs
):
    """
    find local max with fall backs min_distance and filter

    Parameters
    ----------
    data : image to analyze
    mask : mask array of same shape as data, optional
        True means include, False=exclude.  Note that this parameter is called
        `lables` in `peaks_local_max`
    min_distance : int or iterable (Default 15)
        min_distance parameter to self.peak_local_max.
        if min_distance is iterable, if num_phase>num_phase_max, try next
    threshold_rel, threshold_abs : float
        thresholds to use in peak_local_max
    num_peaks_max : int (Default None)
        max number of maxima to find.
    indeces : bool, default=True
        if True, return indicies of peaks.
        if False, return array of ints of shape `data.shape` with peaks
        marked by value > 0.
    errors : {'ignore','raise','warn'}, default='warn'
        - If raise, raise exception if npeaks > num_peaks_max
        - If ignore, return all found maxima
        - If warn, raise warning if npeaks > num_peaks_max


    **kwargs : extra arguments to peak_local_max

    Returns
    -------
    out :
    - if indices is True, tuple of ndarrays
        indices of where local max
    """

    if num_peaks_max is None:
        num_peaks_max = np.inf

    if not isinstance(min_distance, Iterable):
        min_distance = [min_distance]

    data = data - bottleneck.nanmin(data)
    kwargs = dict(dict(exclude_border=False), **kwargs)

    for md in min_distance:
        idx = feature.peak_local_max(
            data,
            min_distance=md,
            labels=mask,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            # this option removed in future
            # indices=True,
            **kwargs
        )

        n = len(idx)
        if n <= num_peaks_max:
            break

    if n > num_peaks_max:
        if errors == "ignore":
            pass
        elif errors in ("raise", "ignore"):
            message = "{} maxima found greater than {}".format(n, num_peaks_max)
            if errors == "raise":
                raise RuntimeError(message)
            else:
                warnings.warn(message)

    idx = tuple(idx.T)
    if indices:
        return idx
    else:
        out = np.zeros_like(data, dtype=bool)
        out[idx] = True
        return out


class Segmenter(object):
    """
    Data segmenter:

    Methods
    -------
    peaks : find peaks of data
    watershep : watershed segementation
    segment_lnpi : helper funciton to segment lnPi
    """

    def __init__(
        self, min_distance=[1, 5, 10, 15, 20], peak_kws=None, watershed_kws=None
    ):
        """
        Parameters
        ----------
        peak_kws : dictionary
            kwargs to `peak_local_max_adaptive`
        watershed_kws : dictionary
            kwargs to `skimage.morphology.watershed`
        """

        if peak_kws is None:
            peak_kws = {}
        peak_kws.update(indices=False)
        self.peak_kws = peak_kws

        if watershed_kws is None:
            watershed_kws = {}
        self.watershed_kws = watershed_kws

    def peaks(
        self,
        data,
        mask=None,
        num_peaks_max=None,
        as_marker=True,
        connectivity=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        data : array
            image to be analyzed
        mask : array
            consider only regions where `mask == True`
        as_marker : bool, default=True
            if True, convert peaks location to labels array
        num_peaks_max : int, optional
        connectivity : int
            connetivity metric, used only if `as_marker==True`
        kwargs : dict
            extra arguments to `peak_local_max_adaptive`.  These overide self.peaks_kws
        Returns
        -------
        out :
            - if `as_marker`, then return label ar
            - else, return indicies of peaks
        Notes
        -----
        All of thes argmuents are in addition to self.peak_kws
        """

        kwargs = dict(self.peak_kws, **kwargs)
        if mask is not None:
            kwargs["mask"] = mask
        if num_peaks_max is not None:
            kwargs["num_peaks_max"] = num_peaks_max
        out = peak_local_max_adaptive(data, **kwargs)
        # combine markers
        if as_marker:
            out = morphology.label(out, connectivity=connectivity)
        return out

    def watershed(self, data, markers, mask, connectivity=None, **kwargs):
        """
        Parameters
        ----------
        data : image array
        markers : int or array of its with shape data.shape
        mask : array of bools of shape data.shape, optional
            if passed, mask==True indicates values to include
        connectivity : int
            connectivity to use in watershed
        kwargs : extra arguments to skimage.morphology.watershed
        Returns
        -------
        labels : array of ints
            Values > 0 correspond to found regions

        """

        if connectivity is None:
            connectivity = data.ndim
        kwargs = dict(self.watershed_kws, connectivity=connectivity, *kwargs)
        return segmentation.watershed(data, markers=markers, mask=mask, **kwargs)

    def segment_lnpi(
        self,
        lnpi,
        find_peaks=True,
        num_peaks_max=None,
        connectivity=None,
        peaks_kws=None,
        watershed_kws=None,
    ):
        """
        Perform segmentations of lnPi object


        Parameters
        ----------
        lnpi : MaskedlnPi
        find_peaks : bool, default=True
        if `True`, then first find peaks using
        """

        if find_peaks:
            if peaks_kws is None:
                peaks_kws = {}
            markers = self.peaks(
                lnpi.data,
                mask=~lnpi.mask,
                num_peaks_max=num_peaks_max,
                connectivity=connectivity,
                **peaks_kws
            )
        else:
            markers = num_peaks_max

        if watershed_kws is None:
            watershed_kws = {}
        labels = self.watershed(
            -lnpi.data, markers=markers, mask=~lnpi.mask, connectivity=connectivity
        )
        return labels


# def _get_delta_w(index, w):
#     return pd.DataFrame(
#         w.delta_w,
#         index=index,
#         columns=index.get_level_values("phase").rename("phase_nebr"),
#     ).stack()


def _get_w_data(index, w):
    w_min = pd.Series(w.w_min[:, 0], index=index, name="w_min")
    w_argmin = pd.Series(w.w_argmin, index=w_min.index, name="w_argmin")

    w_tran = (
        pd.DataFrame(
            w.w_tran,
            index=index,
            columns=index.get_level_values("phase").rename("phase_nebr"),
        )
        .stack()
        .rename("w_tran")
    )

    # get argtrans values for each index
    index_map = {idx: i for i, idx in enumerate(index.get_level_values("phase"))}
    v = w.w_argtran

    argtran = []
    for idxs in zip(
        *[w_tran.index.get_level_values(_) for _ in ["phase", "phase_nebr"]]
    ):
        i, j = [index_map[_] for _ in idxs]

        if (i, j) in v:
            val = v[i, j]
        elif (j, i) in v:
            val = v[j, i]
        else:
            val = None
        argtran.append(val)

    w_argtran = pd.Series(argtran, index=w_tran.index, name="w_argtran")

    return {
        "w_min": w_min,
        "w_tran": w_tran,
        "w_argmin": w_argmin,
        "w_argtran": w_argtran,
    }  # [index_map, w.w_argtran]}


@CollectionlnPi.decorate_accessor("wlnPi")
class wlnPivec(object):
    def __init__(self, parent):
        self._parent = parent
        self._use_joblib = getattr(self._parent, "_use_joblib", False)

    def _get_items_ws(self):
        indexes = []
        ws = []
        for meta, phases in self._parent.groupby_allbut("phase"):
            indexes.append(phases.index)
            masks = [x.mask for x in phases.values]
            ws.append(
                FreeEnergylnPi(data=phases.iloc[0].data, masks=masks, convention=False)
            )
        return indexes, ws

    @gcached()
    def _data(self):
        indexes, ws = self._get_items_ws()
        seq = get_tqdm(zip(indexes, ws), total=len(ws), desc="wlnPi")
        out = parallel_map_func_starargs(
            _get_w_data, items=seq, use_joblib=self._use_joblib, total=len(ws)
        )

        result = {key: pd.concat([x[key] for x in out]) for key in out[0].keys()}

        return result

    @property
    def w_min(self):
        return self._data["w_min"]

    @property
    def w_tran(self):
        return self._data["w_tran"]

    @property
    def w_argmin(self):
        return self._data["w_argmin"]

    @property
    def w_argtran(self):
        return self._data["w_argtran"]

    @property
    def dw(self):
        """Series representation of delta_w"""
        return (self.w_tran - self.w_min).rename("delta_w")

    @property
    def dwx(self):
        """xarray representation of delta_w"""
        return self.dw.to_xarray()

    def get_dwx(self, idx, idx_nebr=None):
        """
        helper function to get the change in energy from
        phase idx to idx_nebr.

        Parameters
        ----------
        idx : int
            phase index to consider transitions from
        idx_nebr : int or list, optional
            if supplied, consider transition from idx to idx_nebr or minimum of all element in idx_nebr.
            Default behavior is to return minimum transition from idx to all other neighboring regions

        Returns
        -------
        dw : float
            - if only phase idx exists, dw = np.inf
            - if idx does not exists, dw = 0.0 (no barrier between idx and anything else)
            - else min of transition for idx to idx_nebr
        """

        delta_w = self.dwx

        # reindex so that has idx in phase
        reindex = delta_w.indexes["phase"].union(pd.Index([idx], name="phase"))
        delta_w = delta_w.reindex(phase=reindex, phase_nebr=reindex)

        # much simpler
        if idx_nebr is None:
            delta_w = delta_w.sel(phase=idx)
        else:
            if not isinstance(idx_nebr, list):
                idx_nebr = [idx_nebr]
            if idx not in idx_nebr:
                idx_nebr.append(idx)
            nebrs = delta_w.indexes["phase_nebr"].intersection(idx_nebr)
            delta_w = delta_w.sel(phase=idx, phase_nebr=nebrs)

        out = delta_w.min("phase_nebr").fillna(0.0)
        return out

    def get_dw(self, idx, idx_nebr=None):
        return self.get_dwx(idx, idx_nebr).to_series()


@CollectionlnPi.decorate_accessor("wlnPi_single")
class wlnPi_single(wlnPivec):
    """
    stripped down version for single phase grouping
    """

    @gcached()
    def dwx(self):
        index = list(self._parent.index.get_level_values("phase"))
        masks = [x.mask for x in self._parent]
        w = FreeEnergylnPi(
            data=self._parent.iloc[0].data, masks=masks, convention=False
        )

        dw = w.w_tran - w.w_min
        dims = ["phase", "phase_nebr"]
        coords = dict(zip(dims, [index] * 2))
        return xr.DataArray(dw, dims=dims, coords=coords)

    @gcached()
    def dw(self):
        """Series representation of delta_w"""
        return self.dwx.to_series()

    def get_dw(self, idx, idx_nebr=None):
        dw = self.dwx
        index = dw.indexes["phase"]

        if idx not in index:
            return 0.0
        elif idx_nebr is None:
            nebrs = index.drop(idx)
        else:
            if not isinstance(idx_nebr, list):
                idx_nebr = [idx_nebr]
            nebrs = [x for x in idx_nebr if x in index]

        if len(nebrs) == 0:
            return np.inf
        return dw.sel(phase=idx, phase_nebr=nebrs).min("phase_nebr").values


class PhaseCreator(object):
    """
    Helper class to create phases
    """

    def __init__(
        self,
        nmax,
        nmax_peak=None,
        ref=None,
        segmenter=None,
        segment_kws=None,
        tag_phases=None,
        phases_factory=CollectionlnPi.from_list,
        FreeEnergylnPi_kws=None,
        merge_kws=None,
    ):
        """
        Parameters
        ----------
        nmax : int
            number of phases to construct
        nmax_peak : int, optional
            if specified, the allowable number of peaks to locate.
            This can be useful for some cases.  These phases will be merged out at the end.
        ref : MaskedlnPi object, optional
        segmenter : Segmenter object, optional
            segmenter object to create labels/masks
        Freeenergy_kws : dict, optional
            dictionary of parameters for the creation of a FreeenergylnPi object
        """

        if nmax_peak is None:
            nmax_peak = nmax * 2
        self.nmax = nmax
        self.ref = ref

        if segmenter is None:
            segmenter = Segmenter()
        self.segmenter = segmenter

        self.tag_phases = tag_phases

        self.phases_factory = phases_factory

        if segment_kws is None:
            segment_kws = {}
        self.segment_kws = segment_kws
        self.segment_kws["num_peaks_max"] = nmax_peak

        if FreeEnergylnPi_kws is None:
            FreeEnergylnPi_kws = {}
        self.FreeEnergylnPi_kws = FreeEnergylnPi_kws

        if merge_kws is None:
            merge_kws = {}
        merge_kws = dict(merge_kws, convention=False, nfeature_max=self.nmax)
        self.merge_kws = merge_kws

    def _merge_phase_ids(sel, ref, phase_ids, lnpis):
        """
        perform merge of phase_ids/index
        """
        from scipy.spatial.distance import pdist

        if len(phase_ids) == 1:
            # only single phase_id
            return phase_ids, lnpis

        phase_ids = np.array(phase_ids)
        dist = pdist(phase_ids.reshape(-1, 1)).astype(int)
        if not np.any(dist == 0):
            # all different
            return phase_ids, lnpis

        phase_ids_new = []
        masks_new = []
        for idx in np.unique(phase_ids):
            where = np.where(idx == phase_ids)[0]
            mask = np.all([lnpis[i].mask for i in where], axis=0)

            phase_ids_new.append(idx)
            masks_new.append(mask)
        lnpis_new = ref.list_from_masks(masks_new, convention=False)

        return phase_ids_new, lnpis_new

    def build_phases(
        self,
        lnz=None,
        ref=None,
        efac=None,
        nmax=None,
        nmax_peak=None,
        connectivity=None,
        reweight_kws=None,
        merge_phase_ids=True,
        phases_factory=None,
        phase_kws=None,
        segment_kws=None,
        FreeEnergylnPi_kws=None,
        merge_kws=None,
    ):
        """
        build phase
        """

        def _combine_kws(class_kws, passed_kws, **default_kws):
            if class_kws is None:
                class_kws = {}
            if passed_kws is None:
                passed_kws = {}
            return dict(class_kws, **default_kws, **passed_kws)

        if ref is None:
            if self.ref is None:
                raise ValueError("must specify ref or self.ref")
            ref = self.ref

        # reweight
        if lnz is not None:
            if reweight_kws is None:
                reweight_kws = {}
            ref = ref.reweight(lnz, **reweight_kws)

        if nmax is None:
            nmax = self.nmax

        if nmax_peak is None:
            nmax_peak = nmax * 2

        connectivity_kws = {}
        if connectivity is not None:
            connectivity_kws["connectivity"] = connectivity

        if nmax > 1:
            # labels
            segment_kws = _combine_kws(
                self.segment_kws,
                segment_kws,
                num_peaks_max=nmax_peak,
                **connectivity_kws
            )
            labels = self.segmenter.segment_lnpi(lnpi=ref, **segment_kws)

            # wlnpi
            FreeEnergylnPi_kws = _combine_kws(
                self.FreeEnergylnPi_kws, FreeEnergylnPi_kws, **connectivity_kws
            )
            wlnpi = FreeEnergylnPi.from_labels(
                data=ref.data, labels=labels, **FreeEnergylnPi_kws
            )

            # merge
            other_kws = {} if efac is None else {"efac": efac}
            merge_kws = _combine_kws(
                self.merge_kws, merge_kws, nfeature_max=nmax, **other_kws
            )
            masks, wtran, wmin = wlnpi.merge_regions(**merge_kws)

            # list of lnpi
            lnpis = ref.list_from_masks(masks, convention=False)

            # tag phases?
            if self.tag_phases is not None:
                index = self.tag_phases(lnpis)
                if merge_phase_ids:
                    index, lnpis = self._merge_phase_ids(ref, index, lnpis)

            else:
                index = list(range(len(lnpis)))
        else:
            lnpis = [ref]
            index = [0]

        if phases_factory is None:
            phases_factory = self.phases_factory
        if isinstance(phases_factory, str) and phases_factory.lower() == "none":
            phases_factory = None
        if phases_factory is not None:
            if phase_kws is None:
                phase_kws = {}
            return phases_factory(items=lnpis, index=index, **phase_kws)
        else:
            return lnpis, index

    def build_phases_mu(self, lnz):
        return BuildPhases_mu(lnz, self)

    def build_phases_dmu(self, dlnz):
        return BuildPhases_dmu(dlnz, self)


class _BuildPhases(object):
    """
    class to build phases object from scalar mu's
    """

    def __init__(self, X, phase_creator):

        self._phase_creator = phase_creator
        self.X = X

    @property
    def X(self):
        return self._X

    @property
    def phase_creator(self):
        return self._phase_creator

    @X.setter
    def X(self, X):
        assert sum([x is None for x in X]) == 1
        self._X = X
        self._ncomp = len(self._X)
        self._index = self._X.index(None)
        self._set_params()

    @property
    def index(self):
        return self._index

    def _set_params(self):
        pass

    def _get_lnz(self, lnz_index):
        # to be implemented in child class
        raise NotImplementedError

    def __call__(self, lnz_index, *args, **kwargs):
        lnz = self._get_lnz(lnz_index)
        return self._phase_creator.build_phases(lnz=lnz, *args, **kwargs)


# from .utils import get_lnz_iter
class BuildPhases_mu(_BuildPhases):
    def __init__(self, lnz, phase_creator):
        """
        Parameters
        ----------
        lnz : list
            list with one element equal to None.  This is the component which will be varied
            For example, lnz=[lnz0,None,lnz2] implies use values of lnz0,lnz2 for components 0 and 2, and
            vary component 1
        phase_creator : PhaseCreator object
        """
        super().__init__(X=lnz, phase_creator=phase_creator)

    def _get_lnz(self, lnz_index):
        lnz = self.X.copy()
        lnz[self.index] = lnz_index
        return lnz


class BuildPhases_dmu(_BuildPhases):
    def __init__(self, dlnz, phase_creator):
        """
        Parameters
        ----------
        dlnz : list
            list with one element equal to None.  This is the component which will be varied
            For example, dlnz=[dlnz0,None,dlnz2] implies use values of dlnz0,dlnz2 for components 0 and 2, and
            vary component 1
            dlnz_i = lnz_i - lnz_index, where lnz_index is the value varied.
        phase_creator : PhaseCreator object
        """
        super().__init__(X=dlnz, phase_creator=phase_creator)

    def _set_params(self):
        self._dlnz = np.array([x if x is not None else 0.0 for x in self.X])

    def _get_lnz(self, lnz_index):
        return self._dlnz + lnz_index


from functools import lru_cache


@lru_cache(maxsize=10)
def get_default_PhaseCreator(nmax):
    return PhaseCreator(nmax=nmax)
