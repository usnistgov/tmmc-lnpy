"""routines to find constant molfracs"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .core.array_utils import array_to_scalar
from .core.rootresults import RootResultDict, rootresults_to_rootresultdict
from .lnpiseries import lnPiCollection

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import xarray as xr

    from .lnpidata import lnPiMasked
    from .segment import BuildPhasesBase


def _initial_bracket_molfrac(
    target: float,
    collection: lnPiCollection,
    build_phases: BuildPhasesBase,
    phase_id: int | str | None = 0,
    component: int | None = None,
    dlnz: float = 0.5,
    dfac: float = 1.0,
    ref: lnPiMasked | None = None,
    build_kws: Mapping[str, Any] | None = None,
    ntry: int = 20,
) -> tuple[lnPiCollection, lnPiCollection, dict[str, Any]]:
    """Find bracket for molfrac"""
    lnz_idx = build_phases.index
    if component is None:
        component = lnz_idx
    if build_kws is None:
        build_kws = {}

    if isinstance(phase_id, str) and phase_id.lower() == "none":
        # select stable phase
        skip_phase_id = True

        def getter(x: lnPiCollection) -> xr.DataArray:
            v = x.xge.molfrac.sel(component=component)
            if len(x) == 1:
                return v.isel(phase=0)
            return v.where(x.xge.mask_stable).max("phase")

    else:
        skip_phase_id = False

        def getter(x: lnPiCollection) -> xr.DataArray:
            return x.xge.molfrac.sel(component=component, phase=phase_id)

    s = getter(collection).to_series().dropna()

    # get left bound
    left = None
    ntry_left = 0
    ss = s[s < target]
    if len(ss) > 0:
        left = collection.mloc[ss.index[[-1]]]
    else:
        index = s.index[[0]] if len(s) > 0 else collection.index[[0]].droplevel("phase")
        new_lnz = collection.mloc[index]._get_lnz(lnz_idx)
        dlnz_ = dlnz
        for i in range(ntry):
            new_lnz -= dlnz_
            dlnz_ *= dfac
            p = build_phases(new_lnz, ref=ref, **build_kws)
            if (skip_phase_id or phase_id in p._get_level("phase")) and getter(
                p
            ).to_numpy() < target:
                left = p
                ntry_left = i
                break
        else:
            ntry_left = ntry

    if left is None:
        msg = "could not find left bounds"
        raise RuntimeError(msg)

    # right bracket
    right = None
    ntry_right = 0
    ss = s[s > target]
    if len(ss) > 0:
        right = collection.mloc[ss.index[[0]]]
    else:
        index = (
            s.index[[-1]] if len(s) > 0 else collection.index[[-1]].droplevel("phase")
        )
        new_lnz = collection.mloc[index]._get_lnz(lnz_idx)
        dlnz_ = dlnz

        for i in range(ntry):
            new_lnz += dlnz_
            p = build_phases(new_lnz, ref=ref, **build_kws)
            if (not skip_phase_id) and (phase_id not in p._get_level("phase")):
                # went to far
                new_lnz -= dlnz_
                # reset to half dlnz
                dlnz_ *= 0.5
            elif getter(p).to_numpy() > target:
                right = p
                ntry_right = i
                break
            else:
                dlnz_ *= dfac
        else:
            ntry_right = ntry

    if right is None:
        msg = "could not find right bounds"
        raise RuntimeError(msg)

    info = {"ntry_left": ntry_left, "ntry_right": ntry_right}
    return left, right, info


def _solve_lnz_molfrac(
    target: float,
    left: lnPiCollection | float,
    right: lnPiCollection | float,
    build_phases: BuildPhasesBase,
    phase_id: int | str | None = 0,
    component: int | None = None,
    build_kws: Mapping[str, Any] | None = None,
    ref: lnPiMasked | None = None,
    tol: float = 1e-4,
    **kwargs: Any,
) -> tuple[lnPiCollection, RootResultDict]:
    """
    Calculate `lnz` which provides ``molfrac == target``.

    Parameters
    ----------
    target : float
        target molfraction
    left, right : float
        lnz values bracketing solution
    build_phases : BuildPhasesBase object
    phase_id : int
        target phase
    ref : lnPiMasked
        object to reweight
    component : int, optional
        if not specified, use build_phases.index
    tol : float, default=1e-4
        solver tolerance
    **kwargs : extra arguments to scipy.optimize.brentq

    Returns
    -------
    output : lnPi_phases object
        object with desired molfraction
    info : solver info (optional, returned if full_output is `True`)
    """
    from scipy.optimize import brentq

    if build_kws is None:
        build_kws = {}

    if component is None:
        component = build_phases.index

    if isinstance(left, lnPiCollection):
        left = left._get_lnz(build_phases.index)
    if isinstance(right, lnPiCollection):
        right = right._get_lnz(build_phases.index)

    a, b = sorted((left, right))

    if isinstance(phase_id, str) and phase_id.lower() == "none":
        # select stable phase
        skip_phase_id = True

        def getter(x: lnPiCollection) -> xr.DataArray:
            v = x.xge.molfrac.sel(component=component)
            if len(x) == 1:
                return v.isel(phase=0)
            return v.where(x.xge.mask_stable).max("phase")

    else:
        skip_phase_id = False

        def getter(x: lnPiCollection) -> xr.DataArray:
            return x.xge.molfrac.sel(component=component, phase=phase_id)

    shared: dict[str, lnPiCollection] = {}

    def f(x: float) -> float:
        p = build_phases(x, ref=ref, **build_kws)
        shared["lnpi"] = p

        # by not using the ListAccessor,
        # can parallelize
        mf: float = (
            array_to_scalar(getter(p).values)
            if skip_phase_id or phase_id in p._get_level("phase")
            else np.inf
        )

        return mf - target

    xx, r = brentq(f, a, b, full_output=True, **kwargs)

    residual = f(xx)
    if np.abs(residual) > tol:
        msg = "something went wrong with solve"
        raise RuntimeError(msg)

    return shared["lnpi"], rootresults_to_rootresultdict(r, residual=residual)


def find_lnz_molfrac(
    target: float,
    collection: lnPiCollection,
    build_phases: BuildPhasesBase,
    phase_id: int | str | None = 0,
    component: int | None = None,
    build_kws: Mapping[str, Any] | None = None,
    ref: lnPiMasked | None = None,
    dlnz: float = 0.5,
    dfac: float = 1.0,
    ntry: int = 20,
    tol: float = 1e-4,
    **kwargs: Any,
) -> tuple[lnPiCollection, RootResultDict]:
    left, right, _info = _initial_bracket_molfrac(
        target=target,
        collection=collection,
        build_phases=build_phases,
        phase_id=phase_id,
        component=component,
        dlnz=dlnz,
        dfac=dfac,
        ref=ref,
        build_kws=build_kws,
        ntry=ntry,
    )

    lnpi, r = _solve_lnz_molfrac(
        target=target,
        left=left,
        right=right,
        build_phases=build_phases,
        phase_id=phase_id,
        component=component,
        build_kws=build_kws,
        ref=ref,
        tol=tol,
        **kwargs,
    )
    return lnpi, r
