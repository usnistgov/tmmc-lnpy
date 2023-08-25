"""
Segmentation of lnPi (:mod:`~lnpy.segment`)
===========================================
Routines to segment lnPi

 1. find max/peaks in lnPi
 2. segment lnPi about these peaks
 3. determine free energy difference between segments
    a. Merge based on low free energy difference
 4. combination of 1-3.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING, cast, overload  # , TypedDict

from module_utilities.docfiller import DocFiller

from ._lazy_imports import np
from .docstrings import docfiller
from .lnpienergy import wFreeEnergy
from .lnpiseries import lnPiCollection

if TYPE_CHECKING:
    from typing import Any, Literal, Mapping, Sequence

    from numpy.typing import ArrayLike

    from ._typing import (
        MyNDArray,
        PeakError,
        PeakStyle,
        PhasesFactorySignature,
        TagPhasesSignature,
    )
    from .lnpidata import lnPiMasked


# * Common doc strings
_docstrings_local = r"""
Parameters
----------
data : array-like
    Image data to analyze
min_distance : int or sequence of int, optional
    min_distance parameter.  If sequence, then call
    :func:`~skimage.feature.peak_local_max` until number of peaks ``<=num_peaks_max``.
    Default value is ``(5, 10, 15, 20, 25)``.
connectivity_morphology | connectivity : int, optional
    Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
    Accepted values are ranging from 1 to ``input.ndim``. If ``None``, a full
    connectivity of ``input.ndim`` is used.
connectivity_watershed | connectivity : ndarray, optional
    An array with the same number of dimensions as image whose non-zero
    elements indicate neighbors for connection. Following the scipy convention,
    default is a one-connected array of the dimension of the image.
num_peaks_max : int, optional
    Max number of maxima/peaks to find. If not specified, any number of peaks allowed.
peak_style | style : {'indices', 'mask', 'marker'}
    Controls output style

    * indices : indices of peaks
    * mask : array of bool marking peak locations
    * marker : array of int
markers : int, or ndarray of int, optional
    Same shape as image. The desired number of markers, or an array marking the
    basins with the values to be assigned in the label matrix. Zero means not a
    marker. If None (no markers given), the local minima of the image are used
    as markers.

lnz_buildphases_mu | lnz : list of float or None
    list with one element equal to None.  This is the component which will be varied
    For example, lnz=[lnz0, None, lnz2] implies use values of lnz0,lnz2 for components 0 and 2, and
    vary component 1.
dlnz_buildphases_dmu | dlnz : list of float or None
    list with one element equal to None.  This is the component which will be varied
    For example, dlnz=[dlnz0,None,dlnz2] implies use values of dlnz0,dlnz2
    for components 0 and 2, and vary component 1.
    dlnz_i = lnz_i - lnz_index, where lnz_index is the value varied.
phase_creator : :class:`PhaseCreator`
    Factory method to create phases collection object.
    For example, :meth:`.lnPiCollection.from_list`.
phases_factory : callable or bool, default=True
    Function to convert list of phases into Phases object.
    If `phases_factory` ``True``, revert to `self.phases_factory`.
    If `phases_factory` is ``False``, do not apply a factory, and
    return list of :class:`lnpy.lnpidata.lnPiMasked` and array of phase indices.
"""


docfiller_local = docfiller.append(
    DocFiller.from_docstring(_docstrings_local, combine_keys="parameters")
).decorate


@overload
def peak_local_max_adaptive(
    data: MyNDArray,
    *,
    mask: MyNDArray | None = ...,
    min_distance: Sequence[int] | None = ...,
    style: Literal["indices"] = ...,
    threshold_rel: float = ...,
    threshold_abs: float = ...,
    num_peaks_max: int | None = ...,
    connectivity: int | None = ...,
    errors: PeakError = ...,
    **kwargs: Any,
) -> tuple[MyNDArray, ...]:
    ...


@overload
def peak_local_max_adaptive(
    data: MyNDArray,
    *,
    mask: MyNDArray | None = ...,
    min_distance: Sequence[int] | None = ...,
    style: Literal["mask", "marker"],
    threshold_rel: float = ...,
    threshold_abs: float = ...,
    num_peaks_max: int | None = ...,
    connectivity: int | None = ...,
    errors: PeakError = ...,
    **kwargs: Any,
) -> MyNDArray:
    ...


@overload
def peak_local_max_adaptive(
    data: MyNDArray,
    *,
    mask: MyNDArray | None = ...,
    min_distance: Sequence[int] | None = ...,
    style: str,
    threshold_rel: float = ...,
    threshold_abs: float = ...,
    num_peaks_max: int | None = ...,
    connectivity: int | None = ...,
    errors: PeakError = ...,
    **kwargs: Any,
) -> MyNDArray | tuple[MyNDArray, ...]:
    ...


@docfiller_local
def peak_local_max_adaptive(
    data: MyNDArray,
    *,
    mask: MyNDArray | None = None,
    min_distance: Sequence[int] | None = None,
    style: PeakStyle | str = "indices",
    threshold_rel: float = 0.0,
    threshold_abs: float = 0.2,
    num_peaks_max: int | float | None = None,
    connectivity: int | None = None,
    errors: PeakError = "warn",
    **kwargs: Any,
) -> MyNDArray | tuple[MyNDArray, ...]:
    """
    Find local max with fall backs min_distance and filter.

    This is an adaptation of :func:`~skimage.feature.peak_local_max`, which is
    called iteratively until the number of `peaks` is less than `num_peaks_max`.

    Parameters
    ----------
    {data}
    {mask_image}
    {min_distance}
    {peak_style}
    threshold_rel, threshold_abs : float
        thresholds parameters
    {num_peaks_max}
    {connectivity_morphology}
    errors : {{'ignore','raise','warn'}}, default='warn'
        - If raise, raise exception if npeaks > num_peaks_max
        - If ignore, return all found maxima
        - If warn, raise warning if npeaks > num_peaks_max
    **kwargs
        Extra arguments to :func:`~skimage.feature.peak_local_max`

    Returns
    -------
    out : array of int or list of array of bool
        Depending on the value of `indices`.

    Notes
    -----
    The option `mask` is passed as the value `labels` in :func:`~skimage.feature.peak_local_max`

    See Also
    --------
    ~skimage.feature.peak_local_max
    ~skimage.morphology.label
    """
    import bottleneck
    from skimage.feature import peak_local_max
    from skimage.morphology import label as morphology_label

    assert style in ["indices", "mask", "marker"]

    if min_distance is None:
        min_distance = [5, 10, 15, 20, 25]

    if num_peaks_max is None:
        num_peaks_max = np.inf

    if not isinstance(min_distance, Iterable):
        min_distance = [min_distance]

    data = data - bottleneck.nanmin(data)
    kwargs = dict(dict(exclude_border=False), **kwargs)

    n = idx = None
    for md in min_distance:
        idx = peak_local_max(
            data,
            min_distance=md,
            labels=mask,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            # this option removed in future
            **kwargs,
        )

        n = len(idx)
        if n <= num_peaks_max:
            break

    if n is None or idx is None:
        raise ValueError("failed to find peaks")

    if n > num_peaks_max:
        if errors == "ignore":
            pass
        elif errors in ("raise", "ignore"):
            message = f"{n} maxima found greater than {num_peaks_max}"
            if errors == "raise":
                raise RuntimeError(message)
            else:
                warnings.warn(message)

    idx = tuple(idx.T)
    if style == "indices":
        return cast("tuple[MyNDArray, ...]", idx)

    else:
        out = np.zeros_like(data, dtype=bool)
        out[idx] = True

        if style == "marker":
            out = morphology_label(out, connectivity=connectivity)

        return cast("MyNDArray", out)


# TODO: create these for below
# class Peak_Kws(TypedDict, total=False):
#     min_distance: Sequence[int]
#     style: PeakStyle
#     threshold_rel: float
#     threshold_abs: float
#     num_peaks_max: int
#     connectivity: int
#     errors: PeakError
#     # peak_local_max options:
#     exclude_border: int | tuple[int, ...] | bool
#     num_peaks: int
#     footprint: MyNDArray
#     p_norm: float

# class Watershed_Kws(TypedDict, total=False):


@docfiller_local
class Segmenter:
    """
    Data segmenter:


    Parameters
    ----------
    {min_distance}
    {peak_kws}
    {watershed_kws}
    """

    def __init__(
        self,
        peak_kws: Mapping[str, Any] | None = None,
        watershed_kws: Mapping[str, Any] | None = None,
    ) -> None:
        if peak_kws is None:
            peak_kws = {}
        self.peak_kws = peak_kws

        if watershed_kws is None:
            watershed_kws = {}
        self.watershed_kws = watershed_kws

    @overload
    def peaks(
        self,
        data: MyNDArray,
        mask: MyNDArray | None = ...,
        *,
        num_peaks_max: int | None = ...,
        style: Literal["marker"] = ...,
        **kwargs: Any,
    ) -> MyNDArray:
        ...

    @overload
    def peaks(
        self,
        data: MyNDArray,
        mask: MyNDArray | None = ...,
        *,
        num_peaks_max: int | None = ...,
        style: Literal["mask"],
        **kwargs: Any,
    ) -> MyNDArray:
        ...

    @overload
    def peaks(
        self,
        data: MyNDArray,
        mask: MyNDArray | None = ...,
        *,
        num_peaks_max: int | None = ...,
        style: Literal["indices"],
        **kwargs: Any,
    ) -> tuple[MyNDArray, ...]:
        ...

    @overload
    def peaks(
        self,
        data: MyNDArray,
        mask: MyNDArray | None = ...,
        *,
        num_peaks_max: int | None = ...,
        style: str,
        **kwargs: Any,
    ) -> MyNDArray | tuple[MyNDArray, ...]:
        ...

    @docfiller_local
    def peaks(
        self,
        data: MyNDArray,
        mask: MyNDArray | None = None,
        *,
        num_peaks_max: int | None = None,
        style: PeakStyle | str = "marker",
        **kwargs: Any,
    ) -> MyNDArray | tuple[MyNDArray, ...]:
        """
        Interface to :func:`peak_local_max_adaptive` with default values from `self`.


        Parameters
        ----------
        {data}
        {num_peaks_max}
        {peak_style}
        {mask_image}
        **kwargs
            Extra arguments to :func:`peak_local_max_adaptive`.

        Returns
        -------
        ndarray of int or sequence of ndarray
            If ``style=='marker'``, then return label array.  Otherwise,
            return indices of peaks.

        Notes
        -----
        Any value not set will be inherited from `self.peak_kws`


        See Also
        --------
        peak_local_max_adaptive

        """

        if mask is not None:
            kwargs["mask"] = mask
        if num_peaks_max is not None:
            kwargs["num_peaks_max"] = num_peaks_max
        kwargs["style"] = style
        kwargs = dict(self.peak_kws, **kwargs)
        out = peak_local_max_adaptive(data, **kwargs)

        return out  # type: ignore

    @docfiller_local
    def watershed(
        self,
        data: MyNDArray,
        markers: int | MyNDArray,
        mask: MyNDArray,
        connectivity: int | MyNDArray | None = None,
        **kwargs: Any,
    ) -> MyNDArray:
        """
        Interface to :func:`skimage.segmentation.watershed` function

        Parameters
        ----------
        {data}
        {markers}
        {mask_image}
        {connectivity_watershed}
        **kwargs
            Extra arguments to :func:`~skimage.segmentation.watershed`

        Returns
        -------
        {labels}

        See Also
        --------
        ~skimage.segmentation.watershed
        """
        from skimage.segmentation import watershed

        if connectivity is None:
            connectivity = data.ndim

        kwargs = dict(self.watershed_kws, connectivity=connectivity, *kwargs)
        return watershed(data, markers=markers, mask=mask, **kwargs)  # type: ignore

    @docfiller_local
    def segment_lnpi(
        self,
        lnpi: lnPiMasked,
        markers: int | MyNDArray | None = None,
        find_peaks: bool = True,
        num_peaks_max: int | None = None,
        connectivity: MyNDArray | None = None,
        peaks_kws: Mapping[str, Any] | None = None,
        watershed_kws: Mapping[str, Any] | None = None,
    ) -> MyNDArray:
        """
        Perform segmentations of lnPi object using watershed on negative of lnPi data.

        Parameters
        ----------
        lnpi : lnPiMasked
            Object to be segmented
        {markers}

        find_peaks : bool, default=True
            If True, use :func:`peak_local_max_adaptive` to construct `markers`.
        {num_peaks_max}
        {connectivity_watershed}
        {peak_kws}
        {watershed_kws}

        Returns
        -------
        {labels}


        See Also
        --------
        Segmenter.watershed
        ~skimage.segmentation.watershed

        """

        if markers is None:
            if find_peaks:
                if peaks_kws is None:
                    peaks_kws = {}
                else:
                    peaks_kws = dict(peaks_kws)
                    peaks_kws.pop("style", None)

                markers = self.peaks(
                    lnpi.data,
                    mask=~lnpi.mask,
                    num_peaks_max=num_peaks_max,
                    connectivity=connectivity,
                    style="marker",
                    **peaks_kws,
                )
            else:
                markers = num_peaks_max

        assert isinstance(markers, (int, np.ndarray))

        if watershed_kws is None:
            watershed_kws = {}
        labels = self.watershed(
            -lnpi.data, markers=markers, mask=~lnpi.mask, connectivity=connectivity
        )
        return labels


class PhaseCreator:
    """
    Helper class to create phases

    Parameters
    ----------
    nmax : int
        Maximum number of phases to allow
    nmax_peak : int, optional
        if specified, the allowable number of peaks to locate.
        This can be useful for some cases.  These phases will be merged out at the end.
    ref : lnPiMasked, optional
        Reference object.
    segmenter : :class:`Segmenter`, optional
        segmenter object to create labels/masks. Defaults to using base segmenter.
    segment_kws : mapping, optional
        Optional arguments to be passed to :meth:`.Segmenter.segment_lnpi`.
    tag_phases : callable, optional
        Optional function which takes a list of :class:`~.lnPiMasked` objects
        and returns on integer label for each object.
    phases_factory : callable, optional
        Factory function for returning Collection from a list of :class:`~lnpy.lnpidata.lnPiMasked` object.
        Defaults to :meth:`.lnPiCollection.from_list`.
    lnPiFreeEnergy_kws : mapping, optional
        Optional arguments to ...
    merge_kws : mapping, optional
        Optional arguments to :func:`.merge_regions`

    """

    def __init__(
        self,
        nmax: int,
        nmax_peak: int | None = None,
        ref: lnPiMasked | None = None,
        segmenter: Segmenter | None = None,
        segment_kws: Mapping[str, Any] | None = None,
        tag_phases: TagPhasesSignature | None = None,
        phases_factory: PhasesFactorySignature | None = None,
        lnPiFreeEnergy_kws: Mapping[str, Any] | None = None,
        merge_kws: Mapping[str, Any] | None = None,
    ) -> None:
        self.nmax = nmax
        self.ref = ref

        self.segmenter = segmenter or Segmenter()
        self.tag_phases = tag_phases
        self.phases_factory = phases_factory or lnPiCollection.from_list

        self.segment_kws = {} if segment_kws is None else dict(segment_kws)
        self.segment_kws["num_peaks_max"] = nmax_peak or nmax * 2

        self.lnPiFreeEnergy_kws = lnPiFreeEnergy_kws or {}
        self.merge_kws = (
            {}
            if merge_kws is None
            else dict(merge_kws, convention=False, nfeature_max=self.nmax)
        )

    # TODO: make this work with integer or string phase_ids
    def _merge_phase_ids(
        self,
        ref: lnPiMasked,
        phase_ids: Sequence[int] | MyNDArray,
        lnpis: list[lnPiMasked],
    ) -> tuple[MyNDArray, list[lnPiMasked]]:
        """Perform merge of phase_ids/index"""
        from scipy.spatial.distance import pdist

        phase_ids = np.asarray(phase_ids)
        if len(phase_ids) == 1:
            # only single phase_id
            return phase_ids, lnpis

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

        return np.asarray(phase_ids_new), lnpis_new

    @overload
    def build_phases(
        self,
        lnz: float | Sequence[float] | ArrayLike | MyNDArray | None = ...,
        ref: lnPiMasked | None = ...,
        *,
        efac: float | None = ...,
        nmax: int | None = ...,
        nmax_peak: int | None = ...,
        connectivity: int | None = ...,
        reweight_kws: Mapping[str, Any] | None = ...,
        merge_phase_ids: bool = True,
        merge_phases: bool = True,
        phases_factory: PhasesFactorySignature | Literal[True] = ...,
        phase_kws: Mapping[str, Any] | None = ...,
        segment_kws: Mapping[str, Any] | None = ...,
        lnPiFreeEnergy_kws: Mapping[str, Any] | None = ...,
        merge_kws: Mapping[str, Any] | None = ...,
        tag_phases: TagPhasesSignature | None = ...,
    ) -> lnPiCollection:
        ...

    @overload
    def build_phases(
        self,
        lnz: float | Sequence[float] | ArrayLike | MyNDArray | None = ...,
        ref: lnPiMasked | None = ...,
        *,
        efac: float | None = ...,
        nmax: int | None = ...,
        nmax_peak: int | None = ...,
        connectivity: int | None = ...,
        reweight_kws: Mapping[str, Any] | None = ...,
        merge_phase_ids: bool = True,
        merge_phases: bool = True,
        phases_factory: Literal[False],
        phase_kws: Mapping[str, Any] | None = ...,
        segment_kws: Mapping[str, Any] | None = ...,
        lnPiFreeEnergy_kws: Mapping[str, Any] | None = ...,
        merge_kws: Mapping[str, Any] | None = ...,
        tag_phases: TagPhasesSignature | None = ...,
    ) -> tuple[list[lnPiMasked], MyNDArray]:
        ...

    @overload
    def build_phases(
        self,
        lnz: float | Sequence[float] | ArrayLike | MyNDArray | None = ...,
        ref: lnPiMasked | None = ...,
        *,
        efac: float | None = ...,
        nmax: int | None = ...,
        nmax_peak: int | None = ...,
        connectivity: int | None = ...,
        reweight_kws: Mapping[str, Any] | None = ...,
        merge_phase_ids: bool = True,
        merge_phases: bool = True,
        phases_factory: PhasesFactorySignature | bool,
        phase_kws: Mapping[str, Any] | None = ...,
        segment_kws: Mapping[str, Any] | None = ...,
        lnPiFreeEnergy_kws: Mapping[str, Any] | None = ...,
        merge_kws: Mapping[str, Any] | None = ...,
        tag_phases: TagPhasesSignature | None = ...,
    ) -> tuple[list[lnPiMasked], MyNDArray] | lnPiCollection:
        ...

    @docfiller_local
    def build_phases(
        self,
        lnz: float | Sequence[float] | ArrayLike | MyNDArray | None = None,
        ref: lnPiMasked | None = None,
        *,
        efac: float | None = None,
        nmax: int | None = None,
        nmax_peak: int | None = None,
        connectivity: int | None = None,
        reweight_kws: Mapping[str, Any] | None = None,
        merge_phase_ids: bool = True,
        merge_phases: bool = True,
        phases_factory: PhasesFactorySignature | bool = True,
        phase_kws: Mapping[str, Any] | None = None,
        segment_kws: Mapping[str, Any] | None = None,
        lnPiFreeEnergy_kws: Mapping[str, Any] | None = None,
        merge_kws: Mapping[str, Any] | None = None,
        tag_phases: TagPhasesSignature | None = None,
    ) -> tuple[list[lnPiMasked], MyNDArray] | lnPiCollection:
        """
        Construct 'phases' for a lnPi object.

        This is quite an involved process.  The steps are

        * Optionally find the location of the maxima in lnPi.
        * Segment lnpi using watershed
        * Merge phases which are energetically similar
        * Optionally merge phases which have the same `phase_id`

        Parameters
        ----------
        lnz : int or sequence of int, optional
            lnz value to evaluate `ref` at.  If not specified, use
            `ref.lnz`
        ref : lnPiMasked
            Object to be segmented
        efac : float, optional
            Optional value to use in energetic merging of phases.
        nmax : int, optional
            Maximum number of phases.  Defaults to `self.nmax`
        nmax_peak : int, optional
            Maximum number of peaks to allow in :func:`peak_local_max_adaptive`.
            Note that this value can be larger than `nmax`.  Defaults to `self.nmax_peak`.
        {connectivity_morphology}
        reweight_kws : mapping, optional
            Extra arguments to `ref.reweight`
        merge_phase_ids : bool, default=True
            If True and calling `tag_phases` routine, merge phases with same phase_id.
        {phases_factory}
        phase_kws : mapping, optional
            Extra arguments to `phases_factory`
        segment_kws : mapping, optional
            Extra arguments to `self.segmenter.segment_lnpi`
        lnPiFreeEnergy_kws : mapping, optional
            Extra arguments to free energy calculation
        merge_kws : mapping, optional
            Extra arguments to merge
        tag_phases : callable, optional
            Function to tag phases.  Defaults to `self.tag_phases`

        Returns
        -------
        output : list of lnPiMasked and ndarray, or lnPiCollection
            If no phase creator, return list of lnPiMasked objects and array of phase indices.
            Otherwise, lnPiCollection object.
        """

        def _combine_kws(
            class_kws: Mapping[str, Any] | None,
            passed_kws: Mapping[str, Any] | None,
            **default_kws: Any,
        ) -> dict[str, Any]:
            return dict(class_kws or {}, **default_kws, **(passed_kws or {}))

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
            # segment lnpi using watershed
            segment_kws = _combine_kws(
                self.segment_kws,
                segment_kws,
                num_peaks_max=nmax_peak,
                **connectivity_kws,
            )
            labels = self.segmenter.segment_lnpi(lnpi=ref, **segment_kws)

            # analyze w = - lnPi
            lnPiFreeEnergy_kws = _combine_kws(
                self.lnPiFreeEnergy_kws, lnPiFreeEnergy_kws, **connectivity_kws
            )
            wlnpi = wFreeEnergy.from_labels(
                data=ref.data, labels=labels, **lnPiFreeEnergy_kws
            )

            if merge_phases:
                # merge
                other_kws = {} if efac is None else {"efac": efac}
                merge_kws = _combine_kws(
                    self.merge_kws, merge_kws, nfeature_max=nmax, **other_kws
                )
                masks, _, _ = wlnpi.merge_regions(**merge_kws)
            else:
                masks = wlnpi.masks

            # list of lnpi
            lnpis = ref.list_from_masks(masks, convention=False)

            # tag phases?
            if tag_phases is None:
                tag_phases = self.tag_phases
            if tag_phases is not None:
                index = tag_phases(lnpis)
                if merge_phase_ids:
                    index, lnpis = self._merge_phase_ids(ref, index, lnpis)
            else:
                index = list(range(len(lnpis)))
        else:
            lnpis = [ref]
            index = [0]

        if isinstance(phases_factory, bool):
            if phases_factory:
                phases_factory = self.phases_factory
            else:
                return lnpis, np.asarray(index)

        phase_kws = phase_kws or {}
        return phases_factory(items=lnpis, index=index, **phase_kws)

    def build_phases_mu(self, lnz: list[float | None]) -> BuildPhases_mu:
        """
        Factory constructor at fixed values of `mu`

        Parameters
        ----------
        {lnz_buildphases_mu}

        See Also
        --------
        BuildPhases_mu

        Examples
        --------
        >>> import lnpy.examples
        >>> e = lnpy.examples.hsmix_example()

        The default build phases from this multicomponent system requires specifies the
        activity for both species.  For example:

        >>> e.phase_creator.build_phases([0.1, 0.2])
        <class lnPiCollection>
        lnz_0  lnz_1  phase
        0.1    0.2    0        [0.1, 0.2]
                      1        [0.1, 0.2]
        dtype: object


        But if we want to creat phases at a fixed value of either lnz_0 or lnz_1, we can
        do the following:

        >>> b = e.phase_creator.build_phases_mu([None, 0.5])

        Note the syntax [None, 0.5].  This means that calling `b(lnz_0)` will
        create a new object at [lnz_0, 0.5].

        >>> b(0.1)
        <class lnPiCollection>
        lnz_0  lnz_1  phase
        0.1    0.5    0        [0.1, 0.5]
                      1        [0.1, 0.5]
        dtype: object

        Likewise, we can fix lnz_0 with

        >>> b = e.phase_creator.build_phases_mu([0.5, None])

        >>> b(0.1)
        <class lnPiCollection>
        lnz_0  lnz_1  phase
        0.5    0.1    0        [0.5, 0.1]
                      1        [0.5, 0.1]
        dtype: object


        To create an object at fixed value of ``dmu_i = lnz_i - lnz_fixed``, we use the following:

        >>> b = e.phase_creator.build_phases_dmu([None, 0.5])

        Now any phase created will have ``lnz = [lnz_0, 0.5 + lnz_0]``

        >>> b(0.5)
        <class lnPiCollection>
        lnz_0  lnz_1  phase
        0.5    1.0    0        [0.5, 1.0]
                      1        [0.5, 1.0]
        dtype: object

        """
        return BuildPhases_mu(lnz, self)

    def build_phases_dmu(self, dlnz: list[float | None]) -> BuildPhases_dmu:
        """
        Factory constructor at fixed values of `dmu`.

        Parameters
        ----------
        {dlnz_buildphases_dmu}


        See Also
        --------
        BuildPhases_dmu
        build_phases_mu
        """
        return BuildPhases_dmu(dlnz, self)


class BuildPhasesBase:
    """Base class to build Phases objects from scalar values of `lnz`."""

    def __init__(self, x: list[float | None], phase_creator: PhaseCreator) -> None:
        self._phase_creator = phase_creator
        self._set_x(x)

    @property
    def x(self) -> list[float | None]:
        return self._x

    @property
    def phase_creator(self) -> PhaseCreator:
        return self._phase_creator

    def _set_x(self, x: list[float | None]) -> None:
        assert sum([x is None for x in x]) == 1
        self._x = x
        self._ncomp = len(self._x)
        self._index = self._x.index(None)
        self._set_params()

    @x.setter  # type: ignore
    def x(self, x: list[float | None]) -> None:
        self._set_x(x)

    @property
    def index(self) -> int:
        """Index number which varies"""
        return self._index

    def _set_params(self) -> None:
        pass

    def _get_lnz(self, lnz_index: float) -> MyNDArray:  # pyright: ignore
        # to be implemented in child class
        raise NotImplementedError

    @overload
    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | Literal[True] = ...,
        **kwargs: Any,
    ) -> lnPiCollection:
        ...

    @overload
    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: Literal[False],
        **kwargs: Any,
    ) -> tuple[list[lnPiMasked], MyNDArray]:
        ...

    @overload
    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | bool,
        **kwargs: Any,
    ) -> tuple[list[lnPiMasked], MyNDArray] | lnPiCollection:
        ...

    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | bool = True,
        **kwargs: Any,
    ) -> tuple[list[lnPiMasked], MyNDArray] | lnPiCollection:
        """
        Build phases from scalar value of lnz.

        Parameters
        ----------
        lnz_index : float
            Value of lnz for `self.index` index.
        {phases_factory}
        **kwargs
            Extra arguments to :meth:`PhaseCreator.build_phases`

        Returns
        -------
        output : list of lnPiMasked and ndarray, or lnPiCollection
            If no phase creator, return list of lnPiMasked objects and array of phase indices.
            Otherwise, lnPiCollection object.

        See Also
        --------
        PhaseCreator.build_phases
        """
        lnz = self._get_lnz(lnz_index)
        return self._phase_creator.build_phases(
            lnz=lnz, phases_factory=phases_factory, **kwargs
        )


# from .utils import get_lnz_iter
@docfiller_local
class BuildPhases_mu(BuildPhasesBase):
    """
    create phases from scalar value of mu for fixed value of mu for other species

    Parameters
    ----------
    {lnz_buildphases_mu}
    {phase_creator}
    """

    def __init__(self, lnz: list[float | None], phase_creator: PhaseCreator) -> None:
        super().__init__(x=lnz, phase_creator=phase_creator)

    def _get_lnz(self, lnz_index: float) -> MyNDArray:
        lnz = self.x.copy()
        lnz[self.index] = lnz_index
        return np.asarray(lnz)


@docfiller_local
class BuildPhases_dmu(BuildPhasesBase):
    """
    Create phases from scalar value of mu at fixed value of dmu for other species

    Parameters
    ----------
    {dlnz_buildphases_dmu}
    {phase_creator}
    """

    def __init__(self, dlnz: list[float | None], phase_creator: PhaseCreator) -> None:
        super().__init__(x=dlnz, phase_creator=phase_creator)

    def _set_params(self) -> None:
        self._dlnz: MyNDArray = np.array([x if x is not None else 0.0 for x in self.x])

    def _get_lnz(self, lnz_index: float) -> MyNDArray:
        return self._dlnz + lnz_index


@lru_cache(maxsize=10)
def get_default_PhaseCreator(nmax: int) -> PhaseCreator:
    return PhaseCreator(nmax=nmax)
