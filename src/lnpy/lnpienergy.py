"""
Local free energy of lnPi (:mod:`~lnpy.lnpienergy`)
===================================================
"""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING, cast, overload

import numpy as np
import pandas as pd
import xarray as xr
from module_utilities import cached

from lnpy.core.utils import peek_at

from .core.docstrings import docfiller
from .core.joblib import parallel_map_func_starargs
from .core.mask import (
    labels_to_masks,
    masks_change_convention,
)
from .core.progress import get_tqdm_calc as get_tqdm
from .core.validate import validate_sequence

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any, Literal

    from .core.typing import MaskConvention, NDArrayAny, NDArrayInt
    from .core.typing_compat import IndexAny, Self, TypeAlias
    from .lnpiseries import lnPiCollection

    _FindBoundariesMode = Literal["thick", "inner", "outer", "subpixel"]
    _FindBoundariesMethod = Literal["exact", "approx"]
    _Extrema = Literal["min", "max"]

    _FillArg: TypeAlias = "int | None"
    _FillVal: TypeAlias = "_FillArg | float"
    _ExtremaArg: TypeAlias = "int | tuple[int, ...] | None"


@docfiller.decorate
def find_boundaries(
    masks: Iterable[NDArrayAny],
    mode: _FindBoundariesMode = "thick",
    connectivity: int | None = None,
    **kws: Any,
) -> list[NDArrayAny]:
    """
    Find boundary region for masks

    Parameters
    ----------
    {masks_image}
    mode : str, default="thick"
        mode to use in :func:`skimage.segmentation.find_boundaries`
    {find_boundary_connectivity}


    Returns
    -------
    boundaries : list of array of bool
        Boundaries of each mask.

    See Also
    --------
    skimage.segmentation.find_boundaries

    """
    from skimage import segmentation

    if connectivity is None:
        first, masks = peek_at(masks)
        connectivity = np.asarray(first).ndim

    return [
        segmentation.find_boundaries(m, connectivity=connectivity, mode=mode, **kws)
        for m in masks
    ]


@overload
def find_boundaries_overlap(
    masks: Iterable[NDArrayAny],
    *,
    boundaries: list[NDArrayAny] | None = ...,
    flag_none: bool = ...,
    mode: _FindBoundariesMode = ...,
    connectivity: int | None = ...,
    method: Literal["exact"] = ...,
) -> dict[tuple[int, int, int], NDArrayAny | None]: ...


@overload
def find_boundaries_overlap(
    masks: Iterable[NDArrayAny],
    *,
    boundaries: list[NDArrayAny] | None = ...,
    flag_none: bool = ...,
    mode: _FindBoundariesMode = ...,
    connectivity: int | None = ...,
    method: Literal["approx"],
) -> dict[tuple[int, int], NDArrayAny | None]: ...


@docfiller.decorate
def find_boundaries_overlap(
    masks: Iterable[NDArrayAny],
    *,
    boundaries: list[NDArrayAny] | None = None,
    flag_none: bool = True,
    mode: _FindBoundariesMode = "thick",
    connectivity: int | None = None,
    method: _FindBoundariesMethod = "exact",
) -> (
    dict[tuple[int, int, int], NDArrayAny | None]
    | dict[tuple[int, int], NDArrayAny | None]
):
    """
    Find regions where boundaries overlap

    Parameters
    ----------
    {masks_image}
    boundaries : list of array of bool, optional
        If supplied, use these areas as boundaries.  Otherwise, calculate
        boundaries using `find_boundaries`
    flag_none : bool, default=True
        if True, replace overlap with None if no overlap between regions
    {find_boundary_connectivity}
    method : str, {{'approx', 'exact'}}
        * approx : consider regions where boundaries overlap and in one of the two region
        * exact : consider regions where boundaries overlap

    Returns
    -------
    overlap : dict of array of bool
        overlap[i, j] = region of overlap between boundaries and masks in regions i and j


    See Also
    --------
    find_boundaries
    """
    masks = validate_sequence(masks)
    n = len(masks)
    possible_methods = {"approx", "exact"}

    if method not in possible_methods:
        msg = f"{method=} not in {possible_methods}"
        raise ValueError(msg)

    if boundaries is None:
        boundaries = find_boundaries(masks, mode=mode, connectivity=connectivity)

    if n != len(boundaries):
        msg = f"{boundaries=} must have length {n}."
        raise ValueError(msg)

    def _get_approx() -> dict[tuple[int, int], NDArrayAny | None]:
        result: dict[tuple[int, int], NDArrayAny | None] = {}
        for i, j in itertools.combinations(range(n), 2):
            # overlap region is where boundaries overlap, and in one of the masks
            overlap = (boundaries[i] & boundaries[j]) & (masks[i] | masks[j])
            if flag_none and not np.any(overlap):
                result[i, j] = None
            else:
                result[i, j] = overlap
        return result

    def _get_exact() -> dict[tuple[int, int, int], NDArrayAny | None]:
        result: dict[tuple[int, int, int], NDArrayAny | None] = {}
        for i, j in itertools.combinations(range(n), 2):
            boundaries_overlap = boundaries[i] & boundaries[j]
            for index, m in enumerate([masks[i], masks[j]]):
                overlap = boundaries_overlap & m
                if flag_none and not np.any(overlap):
                    result[i, j, index] = None
                else:
                    result[i, j, index] = overlap
        return result

    if method == "approx":
        return _get_approx()
    return _get_exact()


@docfiller.decorate
def find_masked_extrema(
    data: NDArrayAny,
    masks: Iterable[NDArrayAny | None],
    convention: MaskConvention = "image",
    extrema: _Extrema = "max",
    fill_val: _FillVal = np.nan,
    fill_arg: _FillArg = None,
    unravel: bool = True,
) -> tuple[list[_ExtremaArg], NDArrayAny]:
    """
    Find position and value of extrema of masked data.

    Parameters
    ----------
    data : ndarray
        input data to consider
    {masks_general}
    {mask_convention}
    extrema : {{'max','min}}
        Type of extrema to calculate
    fill_val : scalar, default=nan
        Value to fill for `out_arg` for empty `mask`
    fill_arg : scalar, optional
        Value to fill for `out_val` for empty `mask`
    unravel : bool, default=True
        If True, unravel flat index to multi dimensional index.


    Returns
    -------
    out_arg : list of int
        index of extrema, one for each `mask`
    out_val : ndarray
        value of extrema, one for each `mask`
    """
    fill_val = fill_val or np.nan

    if extrema == "max":
        func = np.argmax
    elif extrema == "min":
        func = np.argmin
    else:
        msg = 'extrema must be on of {"min", "max}'  # type: ignore[unreachable]  # pyright: ignore[reportUnreachable]
        raise ValueError(msg)

    masks = masks_change_convention(masks, convention, "image")

    data_flat = data.reshape(-1)
    positions_flat = np.arange(data.size)

    out_val = []
    out_arg = []

    arg: _ExtremaArg
    val: _FillVal

    for mask in masks:
        if mask is None or not np.any(mask):
            arg = fill_arg
            val = fill_val
        else:
            mask_flat = mask.reshape(-1)
            arg = positions_flat[mask_flat][func(data_flat[mask_flat])]
            val = data_flat[arg]  # pyright: ignore[reportAssignmentType]

            if unravel:
                arg = np.unravel_index(arg, data.shape)  # type: ignore[assignment]  # pyright: ignore[reportCallIssue, reportArgumentType]

        out_arg.append(arg)
        out_val.append(val)

    return out_arg, np.array(out_val)


@docfiller.decorate
def merge_regions(
    w_tran: NDArrayAny,
    w_min: NDArrayAny,
    masks: Iterable[NDArrayAny],
    nfeature_max: int | None = None,
    efac: float = 1.0,
    force: bool = True,
    convention: MaskConvention = "image",
    warn: bool = True,
) -> tuple[list[NDArrayAny], NDArrayAny, NDArrayAny]:
    r"""
    Merge labels where free energy energy barrier < efac.

    The scaled free energy :math:`w = \beta f = - \ln \Pi ` is analyzed.
    Anywhere where :math:`\Delta w = \text{{w_tran}} - \text{{w_min}} < \text{{efac}}` will be merged into a single phase.  Here,
    ``w_trans`` is the transition energy between phases (i.e., the minimum value of `lnPi`` between regions`) and ``w_min`` is the minimum energy (the maximum `lnPi`) for a phase.



    Parameters
    ----------
    w_tran : array
        Shape of array is ``(n, n)``, where ``n`` is the number of unique regions/phases.
    This is the transitional free energy ()
    w_min : array
        Shape of array is ``(n, 1)``.
    {masks_general}
    nfeature_max : int
        maximum number of features/phases to allow
    efac : float, default=0.5
        Energy difference to merge on. When ``w_trans[i, j] - w_min[i] < efac``, phases
        ``i`` and ``j`` will be merged together.
    force : bool, default=True
        if True, then keep going until nfeature <= nfeature_max
        even if min_val > efac.
    {mask_convention}
    warn : bool, default=True
        if True, give warning messages

    Returns
    -------
    masks : list of ndarray of bool
        output masks using `convention`
    w_trans : array
        transition energy for new masks
    w_min : array
        free energy minima of new masks
    """
    masks = list(masks)

    nfeature = len(masks)
    if nfeature_max is None:
        nfeature_max = nfeature

    w_tran = np.array(w_tran, copy=True)
    w_min = np.array(w_min, copy=True)

    # keep track of keep/kill
    mapping = dict(enumerate(masks))
    for _cnt in range(nfeature):
        # number of finite minima
        nfeature = len(mapping)

        de = w_tran - w_min

        min_arg = np.unravel_index(np.nanargmin(de), de.shape)
        if de[min_arg] > efac:
            if not force:
                if warn and nfeature > nfeature_max:
                    warnings.warn(
                        "min_val > efac, but still too many phases",
                        Warning,
                        stacklevel=2,
                    )
                break
            if nfeature <= nfeature_max:
                break

        idx_keep, idx_kill = min_arg  # pylint: disable=unbalanced-tuple-unpacking
        # keep the one with lower energy
        if w_min[idx_keep, 0] > w_min[idx_kill, 0]:
            idx_keep, idx_kill = idx_kill, idx_keep

        # transition from idx_keep to any other phase equals the minimum transition
        # from either idx_keep or idx_kill to that other phase
        new_tran = w_tran[[idx_keep, idx_kill], :].min(axis=0)
        new_tran[idx_keep] = np.inf
        w_tran[idx_keep, :] = w_tran[:, idx_keep] = new_tran

        # get rid of old one
        w_tran[idx_kill, :] = w_tran[:, idx_kill] = np.inf

        # new mask
        mapping[idx_keep] |= mapping[idx_kill]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
        del mapping[idx_kill]  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    # from mapping create some new stuff
    # new w/de
    idx_min = list(mapping.keys())
    w_min = w_min[idx_min]

    idx_tran = np.ix_(*(idx_min,) * 2)
    w_tran = w_tran[idx_tran]

    # get masks
    masks = masks_change_convention((mapping[i] for i in idx_min), True, convention)

    return masks, w_tran, w_min


@docfiller.decorate
class wFreeEnergy:  # noqa: N801
    r"""
    Analysis of local free energy :math:`w = \beta f = - \ln \Pi`.

    Parameters
    ----------
    data : ndarray
        lnPi data
    {masks_general}
    {mask_convention}
    {find_boundary_connectivity}
    index : sequence of int, optional
        Optional index to apply to phases.
        Not yet fully supported.

    Notes
    -----
    this class uses the 'image' convention for masks.  That is
    where `mask == True` indicates locations in the phase/region,
    and `mask == False` indicates locations not in the phase/region.
    This is opposite the convention used in :mod:`numpy.ma`.
    """

    def __init__(
        self,
        data: NDArrayAny,
        masks: Iterable[NDArrayAny],
        convention: MaskConvention = "image",
        connectivity: int | None = None,
        index: Sequence[int] | NDArrayAny | None = None,
    ) -> None:
        self.data = np.asarray(data)

        # make sure masks in image convention
        self.masks = masks_change_convention(masks, convention, "image")

        self.index: NDArrayInt
        if index is None:
            self.index = np.arange(self.nfeature).astype(np.int64)
        else:
            self.index = np.asarray(index, dtype=np.int64)

        if connectivity is None:
            connectivity = self.data.ndim
        self.connectivity = connectivity

        self._cache: dict[str, Any] = {}

    @property
    def nfeature(self) -> int:
        """Number of features/regions/phases"""
        return len(self.masks)

    @classmethod
    @docfiller.decorate
    def from_labels(
        cls,
        data: NDArrayAny,
        labels: NDArrayAny,
        connectivity: int | None = None,
        features: Sequence[int] | None = None,
        include_boundary: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Create wFreeEnergy from labels

        Parameters
        ----------
        data : ndarray
            lnPi data
        {labels}
        {find_boundary_connectivity}
        {features}
        {include_boundary}
        **kwargs
            Extra arguments to :func:`~lnpy.core.mask.labels_to_masks`

        Returns
        -------
        out : :class:`wFreeEnergy`

        See Also
        --------
        lnpy.core.mask.labels_to_masks
        """
        masks, features = labels_to_masks(
            labels,
            features=features,
            convention="image",
            include_boundary=include_boundary,
            **kwargs,
        )
        return cls(data=data, masks=masks, connectivity=connectivity)

    @cached.prop
    def _data_max(self) -> tuple[list[_ExtremaArg], NDArrayAny]:
        """For lnPi data, find absolute argmax and max"""
        return find_masked_extrema(self.data, self.masks)

    @cached.meth
    def _boundary_max(
        self, method: Literal["approx", "exact"] = "exact", fill_value: float = np.nan
    ) -> tuple[dict[tuple[int, int], _ExtremaArg], NDArrayAny]:
        """
        Find argmax along boundaries of regions.
        Corresponds to argmin(w)

        if method == 'exact', then find the boundary of each region
        and find max.  then find min of those maxes.
        """
        overlap = find_boundaries_overlap(
            self.masks,
            mode="thick",
            connectivity=self.connectivity,
            flag_none=True,
            method=method,
        )

        argmax, valmax = find_masked_extrema(
            self.data,
            list(overlap.values()),
            fill_val=np.nan,
            fill_arg=None,
        )

        # unpack output
        out_arg: dict[tuple[int, int], _ExtremaArg] = {}
        out_max = np.full((self.nfeature,) * 2, dtype=float, fill_value=fill_value)
        if method == "approx":
            for (i, j), arg, val in zip(
                cast("Iterable[tuple[int, int]]", overlap),
                argmax,
                valmax,
                strict=True,
            ):
                out_max[i, j] = out_max[j, i] = val
                out_arg[i, j] = arg

        elif method == "exact":
            # attach keys to argmax, valmax
            overlap_keys = list(overlap)
            argmax_dict = dict(zip(overlap_keys, argmax, strict=True))
            valmax_dict = dict(zip(overlap_keys, valmax, strict=True))

            # loop over unique keys
            for i, j in {
                (i, j) for i, j, _ in cast("Iterable[tuple[int, int, int]]", overlap)
            }:
                vals = [valmax_dict[i, j, index] for index in range(2)]
                # take min value of maxes
                if np.all(np.isnan(vals)):
                    out_arg[i, j] = None
                else:
                    idx_min = np.nanargmin(vals)
                    out_arg[i, j] = argmax_dict[i, j, idx_min]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
                    out_max[i, j] = out_max[j, i] = valmax_dict[i, j, idx_min]  # type: ignore[index]  # pyright: ignore[reportArgumentType]
        return out_arg, out_max

    @property
    def w_min(self) -> NDArrayAny:
        """Minimum value of `w` (max `lnPi`) in each phase/region."""
        return -self._data_max[1][:, None]

    @property
    def w_argmin(self) -> list[_ExtremaArg]:
        """Locations of the minimum of `w` in each phase/region"""
        return self._data_max[0]

    @property
    def w_tran(self) -> NDArrayAny:
        """
        Minimum value of `w` (max `lnPi`) in the boundary between phases.

        `w_tran[i, j]` is the transition energy between phases `i` and `j`.
        """
        return np.nan_to_num(-self._boundary_max()[1], nan=np.inf)  # type: ignore[no-any-return]

    @property
    def w_argtran(self) -> dict[tuple[int, int], _ExtremaArg]:
        """Location of `w_tran`"""
        return self._boundary_max()[0]

    @cached.prop
    def delta_w(self) -> NDArrayAny:
        """Transition energy ``delta_w[i, j] = w_tran[i, j] - w_min[i]``."""
        return self.w_tran - self.w_min

    def merge_regions(
        self,
        nfeature_max: int | None = None,
        efac: float = 1.0,
        force: bool = True,
        convention: MaskConvention = "image",
        warn: bool = True,
    ) -> tuple[Sequence[NDArrayAny], NDArrayAny, NDArrayAny]:
        """
        Merge labels where free energy energy barrier < efac.

        Interface to :func:`merge_regions`

        Parameters
        ----------
        nfeature_max : int
            maximum number of features/phases to allow
        efac : float, default=0.5
            Energy difference to merge on. When ``w_trans[i, j] - w_min[i] < efac``, phases
            ``i`` and ``j`` will be merged together.
        force : bool, default=True
            if True, then keep going until nfeature <= nfeature_max
            even if min_val > efac.
        {mask_convention}
        warn : bool, default=True
            if True, give warning messages

        Returns
        -------
        masks : list of ndarray of bool
            output masks using `convention`
        w_trans : array
            transition energy for new masks
        w_min : array
            free energy minima of new masks
        """
        return merge_regions(
            w_tran=self.w_tran,
            w_min=self.w_min,
            masks=self.masks,
            nfeature_max=nfeature_max,
            efac=efac,
            force=force,
            convention=convention,
            warn=warn,
        )


def _get_w_data(index: pd.MultiIndex, w: wFreeEnergy) -> dict[str, pd.Series[Any]]:
    w_min = pd.Series(w.w_min[:, 0], index=index, name="w_min")
    w_argmin = pd.Series(w.w_argmin, index=w_min.index, name="w_argmin")

    w_tran: pd.Series[Any] = (  # pyright: ignore[reportAssignmentType]
        pd.DataFrame(  # type: ignore[call-overload]  # noqa: PD013
            w.w_tran,
            index=index,
            columns=index.get_level_values("phase").rename("phase_nebr"),
        )
        .stack()
        .rename("w_tran")  # pyright: ignore[reportArgumentType]
    )

    # get argtrans values for each index
    index_map = {idx: i for i, idx in enumerate(index.get_level_values("phase"))}
    v = w.w_argtran

    argtran = []
    for idxs in zip(
        *[w_tran.index.get_level_values(_) for _ in ("phase", "phase_nebr")],
        strict=True,
    ):
        i, j = (index_map[_] for _ in idxs)

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


class wFreeEnergyCollection:  # noqa: N801
    r"""
    Calculate the transition free energies for a :class:`lnpy.lnpiseries.lnPiCollection`.

    :math:`w(N) = \beta f(N) = - \ln \Pi(N)`

    Parameters
    ----------
    parent : lnPiCollection

    Notes
    -----
    An instance of :class:`wFreeEnergyCollection` is normally created from the accessor :meth:`lnpy.lnpiseries.lnPiCollection.wfe`
    """

    def __init__(self, parent: lnPiCollection) -> None:
        self._parent = parent
        self._use_joblib = getattr(self._parent, "_use_joblib", False)

        self._cache: dict[str, Any] = {}

    def _get_items_ws(self) -> tuple[list[IndexAny], list[wFreeEnergy]]:
        indexes = []
        ws = []
        for _, phases in self._parent.groupby_allbut("phase"):
            indexes.append(phases.index)
            masks = [x.mask for x in phases.to_numpy()]

            ws.append(
                wFreeEnergy(data=phases.iloc[0].data, masks=masks, convention=False)
            )
        return indexes, ws

    @cached.prop
    def _data(self) -> dict[str, pd.Series[Any]]:
        indexes, ws = self._get_items_ws()
        seq = get_tqdm(
            zip(indexes, ws, strict=True), total=len(ws), desc="wFreeEnergyCollection"
        )
        out = parallel_map_func_starargs(
            _get_w_data, items=seq, use_joblib=self._use_joblib, total=len(ws)
        )

        return {key: pd.concat([x[key] for x in out]) for key in out[0]}

    @property
    def w_min(self) -> pd.Series[Any]:
        """Minimum energy (maximum `lnPi`) for a given region/phase"""
        return self._data["w_min"]

    @property
    def w_tran(self) -> pd.Series[Any]:
        """Minimum energy (maximum `lnPi`) at boundary between phases"""
        return self._data["w_tran"]

    @property
    def w_argmin(self) -> pd.Series[Any]:
        """Location of :attr:`w_min`"""
        return self._data["w_argmin"]

    @property
    def w_argtran(self) -> pd.Series[Any]:
        """Location of :attr:`w_tran`"""
        return self._data["w_argtran"]

    @property
    def dw(self) -> pd.Series[Any]:
        """Series representation of `dw = w_tran - w_min`"""
        return (self.w_tran - self.w_min).rename("delta_w")

    @property
    def dwx(self) -> xr.DataArray:
        """:mod:`xarray` representation of :attr:`dw`"""
        return self.dw.to_xarray()

    @docfiller.decorate
    def get_dwx(
        self, idx: int, idx_nebr: int | list[int] | None = None
    ) -> xr.DataArray:
        """
        Helper function to get the change in energy from
        phase idx to idx_nebr.

        Parameters
        ----------
        {energy_idx}
        {energy_idx_nebr}

        Returns
        -------
        dw : DataArray
            Transition energy from `idx` to `idx_nebr`
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

        return delta_w.min("phase_nebr").fillna(0.0)

    def get_dw(
        self, idx: int, idx_nebr: int | list[int] | None = None
    ) -> pd.Series[Any]:
        """Series version of :meth:`get_dwx`"""
        return self.get_dwx(idx, idx_nebr).to_series()


# @lnPiCollection.decorate_accessor("wfe_phases")
class wFreeEnergyPhases(wFreeEnergyCollection):  # noqa: N801
    """
    Stripped down version of :class:`wFreeEnergyCollection` for single phase grouping.

    This should be used for a collection of lnPi that is at a single state point, with multiple phases.

    Parameters
    ----------
    parent : lnPiCollection

    Notes
    -----
    This is accessed through :attr:`lnpy.lnpiseries.lnPiCollection.wfe_phases`

    """

    @property
    @cached.meth
    def dwx(self) -> xr.DataArray:
        index = list(self._parent.index.get_level_values("phase"))
        masks = [x.mask for x in self._parent]
        w = wFreeEnergy(data=self._parent.iloc[0].data, masks=masks, convention=False)

        dw = w.w_tran - w.w_min
        dims = ["phase", "phase_nebr"]
        coords = dict(zip(dims, [index] * 2, strict=True))
        return xr.DataArray(dw, dims=dims, coords=coords)

    @property
    @cached.meth
    def dw(self) -> pd.Series[Any]:
        """Series representation of delta_w"""
        return self.dwx.to_series()

    def get_dw(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self, idx: int, idx_nebr: int | Iterable[int] | None = None
    ) -> float | NDArrayAny:
        dw = self.dwx
        index = dw.indexes["phase"]

        if idx not in index:
            return 0.0

        if idx_nebr is None:
            nebrs = index.drop(idx)
        else:
            if isinstance(idx_nebr, int):
                idx_nebr = [idx_nebr]
            nebrs = [x for x in idx_nebr if x in index]

        if len(nebrs) == 0:  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
            return np.inf
        return dw.sel(phase=idx, phase_nebr=nebrs).min("phase_nebr").to_numpy()
