"""Masking routines."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload

import numpy as np

from .docstrings import docfiller
from .validate import validate_list, validate_sequence

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from numpy.typing import DTypeLike, NDArray

    from .typing import MaskConvention, NDArrayAny, NDArrayBool


# * Mask conventions ----------------------------------------------------------
def _convention_to_bool(convention: MaskConvention) -> bool:
    if isinstance(convention, bool):
        return convention
    if convention == "image":
        return True
    if convention == "masked":
        return False

    msg = f"Bad value {convention} sent to _convention_to_bool"  # type: ignore[unreachable]  # pyright: ignore[reportUnreachable]
    raise ValueError(msg)


@overload
def mask_change_convention(
    mask: None,
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> None: ...


@overload
def mask_change_convention(
    mask: NDArrayAny,
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> NDArrayBool: ...


@docfiller.decorate
def mask_change_convention(
    mask: NDArrayAny | None,
    convention_in: MaskConvention = "image",
    convention_out: MaskConvention = "masked",
) -> NDArrayBool | None:
    """
    Convert an array from one 'mask' convention to another.

    Parameters
    ----------
    {mask_general}
    convention_in, convention_out : string or bool
        Convention for input and output.
        Convention for mask.  Allowable values are:

        * ``'image' ``or ``True`` : `True` values included, `False` values excluded.
          This is the normal convention in :mod:`scipy.ndimage`.
        * ``'masked'`` or ``False``:  `False` values are included, `True` values are excluded.
          This is the convention in :mod:`numpy.ma`

    Returns
    -------
    ndarray
        New 'mask' array with specified convention.
    """
    if mask is None:
        return mask

    mask = mask.astype(np.bool_, copy=False)
    if _convention_to_bool(convention_in) != _convention_to_bool(convention_out):
        mask = ~mask
    return mask


@overload
def masks_change_convention(  # pyright: ignore[reportOverlappingOverload]
    masks: Iterable[NDArrayAny],
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> list[NDArrayBool]: ...
@overload
def masks_change_convention(
    masks: Iterable[None],
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> list[None]: ...
@overload
def masks_change_convention(
    masks: Iterable[NDArrayAny | None],
    convention_in: MaskConvention = ...,
    convention_out: MaskConvention = ...,
) -> list[NDArrayBool | None]: ...


def masks_change_convention(
    masks: Iterable[NDArrayAny] | Iterable[None] | Iterable[NDArrayAny | None],
    convention_in: MaskConvention = "image",
    convention_out: MaskConvention = "masked",
) -> list[NDArrayBool] | list[None] | list[NDArrayBool | None]:
    """
    Perform convention change of sequence of masks

    Parameters
    ----------
    masks : sequence of array-like
        masks[i] is the 'ith' mask
    convention_in, convention_out : string or bool or None
        Convention for input and output.
        Convention for mask.  Allowable values are:

        * 'image' or True : `True` values included, `False` values excluded.
          This is the normal convention in :mod:`scipy.ndimage`.
        * 'masked' or False:  `False` values are included, `True` values are excluded.
          This is the convention in :mod:`numpy.ma`

        If `None`, then pass return input `mask`

    Returns
    -------
    new_masks : list of array
        New 'masks' array with specified convention.
    """
    same_convention = _convention_to_bool(convention_in) == _convention_to_bool(
        convention_out
    )

    out: list[NDArrayBool | None] = []
    for mask in masks:
        if mask is None:
            out.append(mask)
        else:
            m = mask.astype(np.bool_, copy=False)
            out.append(m if same_convention else ~m)

    return out


# ** labels
@docfiller.decorate
def labels_to_masks(
    labels: NDArrayAny,
    features: Iterable[int] | None = None,
    include_boundary: bool = False,
    convention: MaskConvention = "image",
    check_features: bool = True,
    **kwargs: Any,
) -> tuple[list[NDArrayBool], list[int]]:
    """
    Convert labels array to list of masks

    Parameters
    ----------
    labels : ndarray of int
        Each unique value `i` in `labels` indicates a mask.
        That is ``labels == i``.
    {features}
    {include_boundary}
    {mask_convention}
    {check_features}

    **kwargs
        arguments to find_boundary if include_boundary is True
        default to mode='outer', connectivity=labels.ndim

    Returns
    -------
    output : list of array of bool
        list of mask arrays, each with same shape as ``labels``, with
        mask convention `convention`.
    features : list
        features

    See Also
    --------
    masks_to_labels
    :func:`skimage.segmentation.find_boundaries`

    """
    from skimage import segmentation

    if include_boundary:
        kwargs = dict({"mode": "outer", "connectivity": labels.ndim}, **kwargs)
    if features is None:
        features = [i for i in np.unique(labels) if i > 0]
    else:
        features = validate_list(features)
        if check_features:
            vals = np.unique(labels)
            if not np.all([x in vals for x in features]):
                raise ValueError

    convention = _convention_to_bool(convention)
    output: list[NDArrayBool] = []

    for feature in features:
        m: NDArray[np.bool_] = labels == feature
        if include_boundary:
            # fmt: off
            b = cast(
                "NDArray[np.bool_]",
                segmentation.find_boundaries(m.astype(int), **kwargs),
            )
            # fmt: on
            m |= b
        if not convention:
            m = ~m
        output.append(m)

    return output, features


@docfiller.decorate
def masks_to_labels(
    masks: Iterable[NDArrayAny],
    features: Iterable[int] | None = None,
    convention: MaskConvention = "image",
    dtype: DTypeLike = np.int64,
) -> NDArrayAny:
    """
    Convert list of masks to labels

    Parameters
    ----------
    masks : list of array-like of bool
        list of mask arrays.
    {features}
    {mask_convention}

    Returns
    -------
    ndarray
        Label array.


    See Also
    --------
    labels_to_masks
    """
    masks = validate_sequence(masks)
    if features is None:
        features = range(1, len(masks) + 1)
    else:
        features = validate_sequence(features)
        if len(features) != len(masks):
            raise ValueError

    labels = np.full(masks[0].shape, fill_value=0, dtype=dtype)
    masks = masks_change_convention(masks, convention, True)

    for i, m in zip(features, masks, strict=True):
        labels[m] = i
    return labels
