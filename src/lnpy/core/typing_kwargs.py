"""TypedDict keyword arguments to various functions/methods"""
# ruff: noqa: TC001, TC003

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import cattrs
import numpy as np

from .typing import NDArrayAny, PeakError
from .typing_compat import TypedDict


# * kwargs
class PeakLocalMaxAdaptiveKwargs(TypedDict, total=False):
    """
    Keyword arguments to :func:`.peak_local_max_adaptive`

    Extras passed to :func:`skimage.feature.peak_local_max`
    """

    min_distance: Sequence[int] | None
    threshold_rel: float
    threshold_abs: float
    num_peaks_max: int | None
    connectivity: int | None
    errors: PeakError


class WatershedKwargs(TypedDict, total=False):
    """
    Keywords to :meth:`.Segmenter.watershed`

    Extras passed to :func:`skimage.segmentation.watershed`
    """

    connectivity: int | NDArrayAny | None


class wFreeEnergyKwargs(TypedDict, total=False):  # noqa: N801
    """
    Keywords to :meth:`.wFreeEnergy.from_labels`

    Extra arguments passed to :func:`skimage.segmentation.find_boundaries`
    """

    connectivity: int | None
    features: Sequence[int] | None
    include_boundary: bool
    check_features: bool


class MergeKwargs(TypedDict, total=False, closed=True):
    """Keyword arguments to :func:`.wFreeEnergy.merge_regions`"""

    nfeature_max: int | None
    efac: float
    force: bool
    convention: Literal["image", "masked"] | bool
    warn: bool


# converters
_peak_converter = cattrs.Converter(forbid_extra_keys=True)
_watershed_converter = cattrs.Converter()
_merge_converter = cattrs.Converter(forbid_extra_keys=True)


@_watershed_converter.register_structure_hook
def _validate_connectivity(val: Any, _: Any) -> int | NDArrayAny | None:
    if val is None or isinstance(val, int):
        return val
    return np.asarray(val)


@_merge_converter.register_structure_hook
def _validate_convention(val: Any, _: Any) -> Literal["image", "masked"] | bool:
    if isinstance(val, bool):
        return val

    if isinstance(val, str) and val in {"image", "masked"}:
        return cast("Literal['image','masked']", val)

    msg = "Convention must be bool, 'image', or 'masked'"
    raise ValueError(msg)


def convert_peak_kws(x: Mapping[Any, Any]) -> PeakLocalMaxAdaptiveKwargs:
    return _peak_converter.structure(x, PeakLocalMaxAdaptiveKwargs)


def convert_watershed_kws(x: Mapping[Any, Any]) -> WatershedKwargs:
    return _watershed_converter.structure(x, WatershedKwargs)


def convert_free_energy_kws(x: Mapping[Any, Any]) -> wFreeEnergyKwargs:
    return cattrs.structure(x, wFreeEnergyKwargs)


def convert_merge_kws(x: Mapping[Any, Any]) -> MergeKwargs:
    return _merge_converter.structure(x, MergeKwargs)
