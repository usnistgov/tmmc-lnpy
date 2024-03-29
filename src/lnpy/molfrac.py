"""routines to find constant molfracs"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._lazy_imports import np
from .utils import RootResultDict, rootresults_to_rootresultdict

if TYPE_CHECKING:
    from typing import Any, Mapping

    import xarray as xr

    from ._typing import MyNDArray
    from .lnpidata import lnPiMasked
    from .lnpiseries import lnPiCollection
    from .segment import BuildPhasesBase


def _initial_bracket_molfrac(
    target: float,
    C: lnPiCollection,
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
            else:
                return v.where(x.xge.mask_stable).max("phase")

    else:
        skip_phase_id = False

        def getter(x: lnPiCollection) -> xr.DataArray:
            return x.xge.molfrac.sel(component=component, phase=phase_id)

    s = getter(C).to_series().dropna()

    # get left bound
    left = None
    ntry_left = 0
    ss = s[s < target]
    if len(ss) > 0:
        left = C.mloc[ss.index[[-1]]]
    else:
        if len(s) > 0:
            index = s.index[[0]]
        else:
            index = C.index[[0]].droplevel("phase")
        new_lnz = C.mloc[index]._get_lnz(lnz_idx)
        dlnz_ = dlnz
        for i in range(ntry):
            new_lnz -= dlnz_
            dlnz_ *= dfac
            p = build_phases(new_lnz, ref=ref, **build_kws)
            if (skip_phase_id or phase_id in p._get_level("phase")) and getter(
                p
            ).values < target:
                left = p
                break
        ntry_left = i

    if left is None:
        raise RuntimeError("could not find left bounds")

    # right bracket
    right = None
    ntry_right = 0
    ss = s[s > target]
    if len(ss) > 0:
        right = C.mloc[ss.index[[0]]]
    else:
        if len(s) > 0:
            index = s.index[[-1]]
        else:
            index = C.index[[-1]].droplevel("phase")
        new_lnz = C.mloc[index]._get_lnz(lnz_idx)
        dlnz_ = dlnz

        for i in range(ntry):
            new_lnz += dlnz_
            p = build_phases(new_lnz, ref=ref, **build_kws)
            if (not skip_phase_id) and (phase_id not in p._get_level("phase")):
                # went to far
                new_lnz -= dlnz_
                # reset to half dlnz
                dlnz_ = dlnz_ * 0.5
            elif getter(p).values > target:
                right = p
                break
            else:
                dlnz_ *= dfac
        ntry_right = i

    if right is None:
        raise RuntimeError("could not find right bounds")

    info = dict(ntry_left=ntry_left, ntry_right=ntry_right)
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
    a, b : float
        lnz values bracketing solution
    phase_id : int
        target phase
    ref : lnPiMasked
        object to reweight
    component : int, optional
        if not specified, use build_phases.index
    full_output : bool, default=False
        if True, return solve stats
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

    if not isinstance(left, float):
        left = left._get_lnz(build_phases.index)
    if not isinstance(right, float):
        right = right._get_lnz(build_phases.index)

    a, b = sorted([x for x in (left, right)])

    if isinstance(phase_id, str) and phase_id.lower() == "none":
        # select stable phase
        skip_phase_id = True

        def getter(x: lnPiCollection) -> xr.DataArray:
            v = x.xge.molfrac.sel(component=component)
            if len(x) == 1:
                return v.isel(phase=0)
            else:
                return v.where(x.xge.mask_stable).max("phase")

    else:
        skip_phase_id = False

        def getter(x: lnPiCollection) -> xr.DataArray:
            return x.xge.molfrac.sel(component=component, phase=phase_id)

    def f(x: float) -> float | MyNDArray:
        p = build_phases(x, ref=ref, **build_kws)
        f.lnpi = p  # type: ignore

        # by not using the ListAccessor,
        # can parralelize
        mf: MyNDArray | float
        if skip_phase_id or phase_id in p._get_level("phase"):
            mf = getter(p).values
            # mf = (
            #     p.s.xs(phase_id, level='phase').iloc[0]
            #     .xge
            #     .molfrac
            #     .sel(component=component)
            #     .values
            # )
        else:
            mf = np.inf

        return mf - target

    xx, r = brentq(f, a, b, full_output=True, **kwargs)

    residual = f(xx)
    if np.abs(residual) > tol:
        raise RuntimeError("something went wrong with solve")

    return f.lnpi, rootresults_to_rootresultdict(r, residual=residual)  # type: ignore


def find_lnz_molfrac(
    target: float,
    C: lnPiCollection,
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
    left, right, info = _initial_bracket_molfrac(
        target=target,
        C=C,
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
