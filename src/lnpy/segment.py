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
from functools import partial
from typing import TYPE_CHECKING, cast, overload

import attrs
import attrs.validators as av
import numpy as np
from module_utilities.docfiller import DocFiller
from skimage.feature import peak_local_max

from .core import validate
from .core._attrs_utils import MyAttrsMixin
from .core.docstrings import docfiller
from .core.typing_compat import override
from .core.typing_kwargs import (
    MergeKwargs,
    PeakLocalMaxAdaptiveKwargs,
    WatershedKwargs,
    convert_free_energy_kws,
    convert_merge_kws,
    convert_peak_kws,
    convert_watershed_kws,
    wFreeEnergyKwargs,
)
from .lnpidata import lnPiMasked
from .lnpienergy import wFreeEnergy
from .lnpiseries import lnPiCollection, validate_lnpicollection

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, Literal

    from numpy.typing import ArrayLike

    from .core.typing import (
        NDArrayAny,
        OptionalKwsAny,
        PeakError,
        PeakStyle,
        PhasesFactorySignature,
        TagPhasesSignature,
    )
    from .core.typing_compat import Unpack


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
    data: NDArrayAny,
    *,
    style: Literal["indices"] = ...,
    mask: NDArrayAny | None = ...,
    **kwargs: Unpack[PeakLocalMaxAdaptiveKwargs],
) -> tuple[NDArrayAny, ...]: ...


@overload
def peak_local_max_adaptive(
    data: NDArrayAny,
    *,
    style: Literal["mask", "marker"],
    mask: NDArrayAny | None = ...,
    **kwargs: Unpack[PeakLocalMaxAdaptiveKwargs],
) -> NDArrayAny: ...


@overload
def peak_local_max_adaptive(
    data: NDArrayAny,
    *,
    style: str,
    mask: NDArrayAny | None = ...,
    **kwargs: Unpack[PeakLocalMaxAdaptiveKwargs],
) -> NDArrayAny | tuple[NDArrayAny, ...]: ...


@docfiller_local
def peak_local_max_adaptive(
    data: NDArrayAny,
    *,
    style: PeakStyle | str = "indices",
    mask: NDArrayAny | None = None,
    min_distance: int | Sequence[int] | None = None,
    threshold_rel: float = 0.0,
    threshold_abs: float = 0.2,
    num_peaks_max: float | None = None,
    connectivity: int | None = None,
    errors: PeakError = "warn",
    peak_local_max_kws: OptionalKwsAny = None,
) -> NDArrayAny | tuple[NDArrayAny, ...]:
    """
    Find local max with fall backs min_distance and filter.

    This is an adaptation of :func:`~skimage.feature.peak_local_max`, which is
    called iteratively until the number of `peaks` is less than `num_peaks_max`.

    Parameters
    ----------
    {data}
    {peak_style}
    {mask_image}
    {min_distance}
    threshold_rel, threshold_abs : float
        thresholds parameters
    {num_peaks_max}
    {connectivity_morphology}
    errors : {{'ignore','raise','warn'}}, default='warn'
        - If raise, raise exception if npeaks > num_peaks_max
        - If ignore, return all found maxima
        - If warn, raise warning if npeaks > num_peaks_max
    peak_local_max_kws : dict
        Extra arguments to :func:`~skimage.feature.peak_local_max`

    Returns
    -------
    out : array of int or list of array of bool
        Depending on the value of `indices`.

    See Also
    --------
    ~skimage.feature.peak_local_max
    ~skimage.morphology.label

    Notes
    -----
    The option `mask` is passed as the value `labels` in :func:`~skimage.feature.peak_local_max`
    """
    import bottleneck
    from skimage.morphology import label as morphology_label

    possible_styles = {"indices", "mask", "marker"}
    if style not in possible_styles:
        msg = f"{style=} not in {possible_styles}"
        raise ValueError(msg)

    if min_distance is None:
        min_distance = [5, 10, 15, 20, 25]

    if num_peaks_max is None:
        num_peaks_max = np.inf

    if not isinstance(min_distance, Iterable):
        min_distance = [min_distance]

    data = data - bottleneck.nanmin(data)  # noqa: PLR6104

    peak_local_max_kws = {} if peak_local_max_kws is None else dict(peak_local_max_kws)
    peak_local_max_kws.setdefault("exclude_border", False)

    n = idx = None
    for md in min_distance:
        idx = peak_local_max(
            data,
            min_distance=md,
            labels=mask,
            threshold_abs=threshold_abs,
            threshold_rel=threshold_rel,
            # this option removed in future
            **peak_local_max_kws,
        )

        if (n := len(idx)) <= num_peaks_max:
            break

    if n is None or idx is None:
        msg = "failed to find peaks"
        raise ValueError(msg)

    if n > num_peaks_max:
        if errors == "ignore":
            pass
        elif errors in {"raise", "ignore"}:
            message = f"{n} maxima found greater than {num_peaks_max}"
            if errors == "raise":
                raise RuntimeError(message)
            warnings.warn(message, stacklevel=1)

    idx = tuple(idx.T)
    if style == "indices":
        return cast("tuple[NDArrayAny, ...]", idx)

    out = np.zeros_like(data, dtype=bool)
    out[idx] = True

    if style == "marker":
        out = validate.ndarrayany(morphology_label(out, connectivity=connectivity))
    return out


@attrs.define(frozen=True)
@docfiller_local
class Segmenter(MyAttrsMixin):
    """
    Data segmenter:


    Parameters
    ----------
    {min_distance}
    {peak_kws}
    {watershed_kws}
    """

    peak_kws: PeakLocalMaxAdaptiveKwargs = attrs.field(
        factory=PeakLocalMaxAdaptiveKwargs,
        converter=convert_peak_kws,
    )
    watershed_kws: WatershedKwargs = attrs.field(
        factory=WatershedKwargs,
        converter=convert_watershed_kws,
    )

    @overload
    def peaks(
        self,
        data: NDArrayAny,
        *,
        style: Literal["marker"] = ...,
        mask: NDArrayAny | None = ...,
        **kwargs: Unpack[PeakLocalMaxAdaptiveKwargs],
    ) -> NDArrayAny: ...

    @overload
    def peaks(
        self,
        data: NDArrayAny,
        *,
        style: Literal["mask"],
        mask: NDArrayAny | None = ...,
        **kwargs: Unpack[PeakLocalMaxAdaptiveKwargs],
    ) -> NDArrayAny: ...

    @overload
    def peaks(
        self,
        data: NDArrayAny,
        *,
        style: Literal["indices"],
        mask: NDArrayAny | None = ...,
        **kwargs: Unpack[PeakLocalMaxAdaptiveKwargs],
    ) -> tuple[NDArrayAny, ...]: ...

    @overload
    def peaks(
        self,
        data: NDArrayAny,
        *,
        style: str,
        mask: NDArrayAny | None = ...,
        **kwargs: Unpack[PeakLocalMaxAdaptiveKwargs],
    ) -> NDArrayAny | tuple[NDArrayAny, ...]: ...

    @docfiller_local
    def peaks(
        self,
        data: NDArrayAny,
        *,
        style: PeakStyle | str = "marker",
        mask: NDArrayAny | None = None,
        **kwargs: Unpack[PeakLocalMaxAdaptiveKwargs],
    ) -> NDArrayAny | tuple[NDArrayAny, ...]:
        """
        Interface to :func:`peak_local_max_adaptive` with default values from `self`.


        Parameters
        ----------
        {data}
        {peak_style}
        {mask_image}
        **kwargs
            Extra arguments to :func:`peak_local_max_adaptive`.  Note that these override
            ``self.peak_kws``.

        Returns
        -------
        ndarray of int or sequence of ndarray
            If ``style=='marker'``, then return label array.  Otherwise,
            return indices of peaks.

        See Also
        --------
        peak_local_max_adaptive

        Notes
        -----
        Any value not set will be inherited from `self.peak_kws`

        """
        kwargs = {**self.peak_kws, **kwargs}
        return peak_local_max_adaptive(data, style=style, mask=mask, **kwargs)

    @docfiller_local
    def watershed(
        self,
        data: NDArrayAny,
        markers: int | NDArrayAny,
        mask: NDArrayAny,
        *,
        connectivity: int | NDArrayAny | None = None,
        **kwargs: Any,
    ) -> NDArrayAny:
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
        from skimage.segmentation import watershed  # pylint: disable=no-name-in-module

        kwargs = dict(self.watershed_kws, **kwargs)
        if connectivity is not None:
            kwargs["connectivity"] = connectivity
        else:
            _ = kwargs.setdefault("connectivity", data.ndim)

        return validate.ndarrayany(
            watershed(data, markers=markers, mask=mask, **kwargs)
        )

    @docfiller_local
    def segment_lnpi(
        self,
        lnpi: lnPiMasked,
        markers: int | NDArrayAny | None = None,
        peak_kws: OptionalKwsAny = None,
        watershed_kws: OptionalKwsAny = None,
    ) -> NDArrayAny:
        """
        Perform segmentations of lnPi object using watershed on negative of lnPi data.

        Parameters
        ----------
        lnpi : lnPiMasked
            Object to be segmented
        {markers}
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
            peak_kws_: PeakLocalMaxAdaptiveKwargs = (
                self.peak_kws
                if peak_kws is None
                else {**self.peak_kws, **convert_peak_kws(peak_kws)}
            )

            markers = self.peaks(
                lnpi.data,
                mask=~lnpi.mask,
                style="marker",
                **peak_kws_,
            )

        watershed_kws_: WatershedKwargs = (
            self.watershed_kws
            if watershed_kws is None
            else {**self.watershed_kws, **convert_watershed_kws(watershed_kws)}
        )

        return self.watershed(
            -lnpi.data,
            markers=markers,
            mask=~lnpi.mask,
            **watershed_kws_,
        )


if TYPE_CHECKING:

    def _convert_merge_kws(value: OptionalKwsAny) -> MergeKwargs: ...
    def _convert_phase_creator_peak_kws(
        value: Mapping[Any, Any],
    ) -> PeakLocalMaxAdaptiveKwargs: ...

else:

    @partial(attrs.Converter, takes_self=True)
    def _convert_merge_kws(value: OptionalKwsAny, self_: Any) -> MergeKwargs:
        out = MergeKwargs() if value is None else convert_merge_kws(value)
        out.update(convention=False, nfeature_max=self_.nmax)
        return out

    @partial(attrs.Converter, takes_self=True)
    def _convert_phase_creator_peak_kws(
        value: Mapping[Any, Any], self_: Any
    ) -> PeakLocalMaxAdaptiveKwargs:
        value = convert_peak_kws(value)
        value["num_peaks_max"] = self_.nmax_peak or self_.nmax * 2

        return value


@docfiller_local
@attrs.define(frozen=True)
class PhaseCreator(MyAttrsMixin):
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
    {peak_kws}
    {watershed_kws}
    tag_phases : callable, optional
        Optional function which takes a list of :class:`~.lnPiMasked` objects
        and returns on integer label for each object.
    phases_factory : callable, optional
        Factory function for returning Collection from a list of :class:`~lnpy.lnpidata.lnPiMasked` object.
        Defaults to :meth:`.lnPiCollection.from_list`.
    free_energy_kws : mapping, optional
        Optional arguments to :meth:`.wFreeEnergy.from_labels`
    merge_phases: bool, default=True
        If True, merge phases using :meth:`.wFreeEnergy.merge_regions`
    merge_kws : mapping, optional
        Optional arguments to :meth:`.wFreeEnergy.merge_regions`
    merge_phase_ids: bool, default=True
        If ``True``, merge phases with same phase id (from ``tag_phases``).
    """

    nmax: int
    ref: lnPiMasked = attrs.field(validator=av.instance_of(lnPiMasked))
    nmax_peak: int | None = None
    segmenter: Segmenter = attrs.field(
        factory=Segmenter, validator=av.instance_of(Segmenter)
    )
    peak_kws: PeakLocalMaxAdaptiveKwargs = attrs.field(
        factory=PeakLocalMaxAdaptiveKwargs,
        converter=_convert_phase_creator_peak_kws,
    )
    watershed_kws: WatershedKwargs = attrs.field(
        factory=WatershedKwargs,
        converter=convert_watershed_kws,
    )
    tag_phases: TagPhasesSignature | None = None
    phases_factory: PhasesFactorySignature = attrs.field(
        default=lnPiCollection.from_list
    )

    free_energy_kws: wFreeEnergyKwargs = attrs.field(
        factory=wFreeEnergyKwargs,
        converter=convert_free_energy_kws,
    )
    merge_phases: bool = True
    merge_kws: MergeKwargs = attrs.field(
        factory=MergeKwargs,
        converter=_convert_merge_kws,
    )
    merge_phase_ids: bool = True

    # TODO(wpk): make this work with integer or string phase_ids
    @staticmethod
    def _merge_phase_ids(
        ref: lnPiMasked,
        phase_ids: Sequence[int] | NDArrayAny,
        lnpis: list[lnPiMasked],
    ) -> tuple[NDArrayAny, list[lnPiMasked]]:
        """Perform merge of phase_ids/index"""
        from scipy.spatial.distance import pdist

        phase_ids = np.asarray(phase_ids)
        if len(phase_ids) == 1:
            # only single phase_id
            return phase_ids, lnpis

        dist = pdist(phase_ids.reshape(-1, 1)).astype(int)
        if not np.any(dist == 0):  # pylint: disable=use-implicit-booleaness-not-comparison-to-zero
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
        lnz: float | Sequence[float] | ArrayLike | NDArrayAny | None = ...,
        *,
        efac: float | None = ...,
        phases_factory: PhasesFactorySignature | Literal[True] = ...,
        phase_kws: Mapping[str, Any] | None = ...,
    ) -> lnPiCollection: ...

    @overload
    def build_phases(
        self,
        lnz: float | Sequence[float] | ArrayLike | NDArrayAny | None = ...,
        *,
        efac: float | None = ...,
        phases_factory: Literal[False],
        phase_kws: Mapping[str, Any] | None = ...,
    ) -> tuple[list[lnPiMasked], NDArrayAny]: ...

    @overload
    def build_phases(
        self,
        lnz: float | Sequence[float] | ArrayLike | NDArrayAny | None = ...,
        *,
        efac: float | None = ...,
        phases_factory: PhasesFactorySignature | bool,
        phase_kws: Mapping[str, Any] | None = ...,
    ) -> tuple[list[lnPiMasked], NDArrayAny] | lnPiCollection: ...

    @docfiller_local
    def build_phases(
        self,
        lnz: float | Sequence[float] | ArrayLike | NDArrayAny | None = None,
        *,
        efac: float | None = None,
        phases_factory: PhasesFactorySignature | bool = True,
        phase_kws: Mapping[str, Any] | None = None,
    ) -> tuple[list[lnPiMasked], NDArrayAny] | lnPiCollection:
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
        efac : float, optional
            Optional value to use in energetic merging of phases.
        {connectivity_morphology}
        {phases_factory}
        phase_kws : mapping, optional
            Extra arguments to `phases_factory`

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

        ref = self.ref
        nmax = self.nmax

        # reweight
        if lnz is not None:
            ref = ref.reweight(lnz)

        if nmax > 1:
            # segment lnpi using watershed
            labels = self.segmenter.segment_lnpi(
                lnpi=ref, peak_kws=self.peak_kws, watershed_kws=self.watershed_kws
            )

            # analyze w = - lnPi
            wlnpi = wFreeEnergy.from_labels(
                data=ref.data, labels=labels, **self.free_energy_kws
            )

            if self.merge_phases:
                # merge
                merge_kws = (
                    self.merge_kws
                    if efac is None
                    else convert_merge_kws(dict(self.merge_kws, efac=efac))
                )
                masks, _, _ = wlnpi.merge_regions(**merge_kws)
            else:
                masks = wlnpi.masks

            # list of lnpi
            lnpis = ref.list_from_masks(masks, convention=False)

            # tag phases?
            tag_phases = self.tag_phases
            if tag_phases is not None:
                index = tag_phases(lnpis)
                if self.merge_phase_ids:
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

        Parameter
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

    def __init__(self, x: Iterable[float | None], phase_creator: PhaseCreator) -> None:
        self.phase_creator = phase_creator
        self.x = list(x)

        if sum(x is None for x in x) != 1:
            msg = f"{x=} must have a single element which is None.  This will be the dimension varied."
            raise ValueError(msg)

    @property
    def ncomp(self) -> int:
        return len(self.x)

    @property
    def index(self) -> int:
        return self.x.index(None)

    def _get_lnz(self, lnz_index: float) -> NDArrayAny:
        raise NotImplementedError

    @overload
    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | Literal[True] = ...,
        **kwargs: Any,
    ) -> lnPiCollection: ...

    @overload
    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: Literal[False],
        **kwargs: Any,
    ) -> tuple[list[lnPiMasked], NDArrayAny]: ...

    @overload
    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | bool,
        **kwargs: Any,
    ) -> tuple[list[lnPiMasked], NDArrayAny] | lnPiCollection: ...

    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | bool = True,
        **kwargs: Any,
    ) -> tuple[list[lnPiMasked], NDArrayAny] | lnPiCollection:
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
        return self.phase_creator.build_phases(
            lnz=lnz, phases_factory=phases_factory, **kwargs
        )

    def _call_and_validate_is_lnpicollection(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | bool = True,
        **kwargs: Any,
    ) -> lnPiCollection:
        return validate_lnpicollection(
            self(lnz_index=lnz_index, phases_factory=phases_factory, **kwargs)
        )


@docfiller_local
class BuildPhases_mu(BuildPhasesBase):  # noqa: N801
    """
    create phases from scalar value of mu for fixed value of mu for other species

    Parameters
    ----------
    {lnz_buildphases_mu}
    {phase_creator}
    """

    def __init__(self, lnz: list[float | None], phase_creator: PhaseCreator) -> None:
        super().__init__(x=lnz, phase_creator=phase_creator)

    @override
    def _get_lnz(self, lnz_index: float) -> NDArrayAny:
        lnz = self.x.copy()
        lnz[self.index] = lnz_index
        return np.asarray(lnz)


@docfiller_local
class BuildPhases_dmu(BuildPhasesBase):  # noqa: N801
    """
    Create phases from scalar value of mu at fixed value of dmu for other species

    Parameters
    ----------
    {dlnz_buildphases_dmu}
    {phase_creator}
    """

    def __init__(self, dlnz: list[float | None], phase_creator: PhaseCreator) -> None:
        super().__init__(x=dlnz, phase_creator=phase_creator)
        self.dlnz = np.array([x if x is not None else 0.0 for x in self.x])

    @override
    def _get_lnz(self, lnz_index: float) -> NDArrayAny:
        return self.dlnz + lnz_index


class BuildPhases_Fixed_betaOmega(BuildPhasesBase):  # noqa: N801
    """
    Here, None is the index we will set.
    Free_index is the one which will be varied to reach specified beta_omega
    """

    def __init__(
        self,
        lnz: list[float | None],
        free_index: int,
        beta_omega: float,
        phase_creator: PhaseCreator,
    ) -> None:
        super().__init__(x=lnz, phase_creator=phase_creator)
        self._beta_omega = beta_omega
        self._free_index = free_index
        self._last_stable: lnPiMasked | None = None

    @override
    def _get_lnz(self, lnz_index: float) -> NDArrayAny:
        raise NotImplementedError

    def _get_lnz_total(self, lnz_index: float, lnz_free_index: float) -> list[float]:
        lnz = self.x.copy()
        lnz[self.index] = lnz_index
        lnz[self._free_index] = lnz_free_index
        return cast("list[float]", lnz)

    def _build_stable(self, lnz: Sequence[float]) -> lnPiMasked:
        lnpis, _ = self.phase_creator.build_phases(lnz, phases_factory=False)
        # return minimum betaOmega
        idx = np.argmin([lnpi.xge.betaOmega() for lnpi in lnpis])
        return lnpis[idx]

    def _get_stable(self, lnz: Sequence[float]) -> lnPiMasked:
        if self._last_stable is not None and np.allclose(lnz, self._last_stable.lnz):
            return self._last_stable
        self._last_stable = self._build_stable(lnz)
        return self._last_stable

    def _objective(self, lnz_free_index: float, lnz_index: float) -> float:
        lnz = self._get_lnz_total(lnz_index, lnz_free_index)
        lnpi = self._get_stable(lnz)
        return float(lnpi.xge.betaOmega()) - self._beta_omega

    def _jacobian(self, lnz_free_index: float, lnz_index: float) -> float:
        lnz = self._get_lnz_total(lnz_index, lnz_free_index)
        lnpi = self._get_stable(lnz)
        return -float(lnpi.xge.nvec.values[self._free_index])

    @overload
    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | Literal[True] = ...,
        lnz_free_index: float | None = None,
        **kwargs: Any,
    ) -> lnPiCollection: ...

    @overload
    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: Literal[False],
        lnz_free_index: float | None = None,
        **kwargs: Any,
    ) -> tuple[list[lnPiMasked], NDArrayAny]: ...

    @overload
    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | bool,
        lnz_free_index: float | None = None,
        **kwargs: Any,
    ) -> tuple[list[lnPiMasked], NDArrayAny] | lnPiCollection: ...

    def __call__(
        self,
        lnz_index: float,
        *,
        phases_factory: PhasesFactorySignature | bool = True,
        lnz_free_index: float | None = None,
        **kwargs: Any,
    ) -> tuple[list[lnPiMasked], NDArrayAny] | lnPiCollection:

        from scipy.optimize import newton

        x0: Any = (
            (self._last_stable.lnz if self._last_stable is not None else self.x)[
                self._free_index
            ]
            if lnz_free_index is None
            else lnz_free_index
        )

        _, result, *_ = newton(
            self._objective,
            args=(lnz_index,),
            fprime=self._jacobian,
            x0=x0,
            full_output=True,
        )
        if not result.converged:
            msg = f"Failed {result=}"
            raise ValueError(msg)

        return self.phase_creator.build_phases(
            lnz=self._get_lnz_total(lnz_index, result.root),
            phases_factory=phases_factory,
            **kwargs,
        )
