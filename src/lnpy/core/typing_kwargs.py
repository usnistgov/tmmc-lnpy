"""TypedDict keyword arguments to various functions/methods"""
# ruff: noqa: TC001, TC003

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cattrs

from .typing import OptionalKwsAny, PeakError
from .typing_compat import TypedDict


# * kwargs
class PeakLocalMaxAdaptiveKwargs(TypedDict, total=False, closed=True):  # type: ignore[call-arg]
    """Keyword arguments to :func:`.peak_local_max_adaptive`"""

    min_distance: Sequence[int] | None
    threshold_rel: float
    threshold_abs: float
    num_peaks_max: int | None
    connectivity: int | None
    errors: PeakError
    peak_local_max_kws: OptionalKwsAny


class WatershedKwargs(TypedDict, total=False):
    """
    Keywords to :meth:`.Segmenter.watershed`

    Extras passed to segmentation.watershed

    """

    connectivity: Any  # TODO(wpk):  get `int | NDArrayAny | None` to work with cattrs


class wFreeEnergyKwargs(TypedDict, total=False):  # noqa: N801
    """
    Keywords to :meth:`.wFreeEnergy.from_labels`

    Extra arguments passed to :func:`skimage.segmentation.find_boundaries`
    """

    connectivity: int | None
    features: Sequence[int] | None
    include_boundary: bool
    check_features: bool


class MergeKwargs(TypedDict, total=False, closed=True):  # type: ignore[call-arg]
    """Keyword arguments to :func:`.wFreeEnergy.merge_regions`"""

    nfeature_max: int | None
    efac: float
    force: bool
    convention: Any  # TODO(wpk): get `Literal["mask", "image"] | bool` to work
    warn: bool


_forbid_extra_keys_converter = cattrs.Converter(forbid_extra_keys=True)


def convert_peak_kws(x: Mapping[Any, Any]) -> PeakLocalMaxAdaptiveKwargs:
    return _forbid_extra_keys_converter.structure(x, PeakLocalMaxAdaptiveKwargs)


def convert_watershed_kws(x: Mapping[Any, Any]) -> WatershedKwargs:
    return cattrs.structure(x, WatershedKwargs)


def convert_free_energy_kws(x: Mapping[Any, Any]) -> wFreeEnergyKwargs:
    return cattrs.structure(x, wFreeEnergyKwargs)


def convert_merge_kws(x: Mapping[Any, Any]) -> MergeKwargs:
    return _forbid_extra_keys_converter.structure(x, MergeKwargs)
