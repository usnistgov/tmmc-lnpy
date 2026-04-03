"""Set of helper utilities to work with single component system"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from .core.array_utils import array_to_scalar
from .core.rootresults import RootResultDict, rootresults_to_rootresultdict
from .lnpiseries import lnPiCollection

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import xarray as xr
    from numpy.typing import ArrayLike

    from .core.typing import NDArrayAny
    from .lnpidata import lnPiMasked
    from .segment import BuildPhasesBase


def tag_phases_singlecomp(x: Sequence[lnPiMasked]) -> Sequence[int] | NDArrayAny:
    """
    Function to tag phases with a unique id.

    This is for analyzing a single component system.
    If multiple phases passed (length of input > 1), then
    sort by ``argmax = x.local_argmax()``, i.e., the location of the maxima.
    Otherwise, assign `phase_id = 0` if ``argmax < Nmax // 2`` where ``Nmax`` is
    the maximum number of particles in lnPi.

    Parameters
    ----------
    x : sequence of lnPiMasked
        lnPi objects to be tagged.

    Returns
    -------
    phase_id : sequence of int
        Phase code for each element in ``x``.
    """
    if len(x) > 2:
        msg = "bad tag function"
        raise ValueError(msg)

    argmax0 = np.array([xx.local_argmax()[0] for xx in x])
    if len(x) == 2:
        return np.argsort(argmax0)

    return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)


def get_lnz_min(
    target: float,
    collection: lnPiCollection,
    build_phases: BuildPhasesBase,
    phase_id: int | None = 0,
    component: int | str | None = None,
    dlnz: float = 0.5,
    dfac: float = 1.0,
    ref: lnPiMasked | None = None,
    build_kws: Mapping[str, Any] | None = None,
    ntry: int = 20,
    solve_kws: Mapping[str, Any] | None = None,
) -> tuple[lnPiCollection, RootResultDict]:
    """
    Solve for target density

    Parameters
    ----------
    target : float
        target density of specified component
    collection : CollectionPhases
        initial guess to work from.  This is assumed to be sorted in ascending `lnz` order
    build_phases : callable
        function to build new phases objects
    phase_id : int, default=0
        phase id tag to consider.  If None, consider min over 'phases'
    component : int, optional
        component to consider.
        If component is None, use build_phases.index.
        If component=='None', then consider total density
    dlnz : float, default=0.5
        how to change the chemical potential if `collection` doesn't already
        bracket solution
    ref : lnPiMasked, optional
        optional lnPiMasked object to pass to build_phases
    build_kws : dictionary, optional
        optional arguments to `build_phases`
    ntry : int, default=20
        number of times to attempted finding left/right bracketing objects
    solve_kws : dictionary, optional
        optional arguments to `scipy.optimize.brentq`

    Returns
    -------
    phases : Phases object as solution lnz
        solution_parameters : optional
    info : :class:`RootResultDict`
    """
    from scipy.optimize import brentq

    if phase_id is not None and phase_id not in collection.index.get_level_values(
        "phase"
    ):
        msg = f"no phase {phase_id}"
        raise ValueError(msg)

    lnz_idx = build_phases.index
    if isinstance(component, str) and component.lower() == "none":

        def getter_comp(x: lnPiCollection) -> xr.DataArray:
            return x.xge.dens_tot

    else:
        component = component or lnz_idx

        def getter_comp(x: lnPiCollection) -> xr.DataArray:
            return x.xge.dens.sel(component=component)

    if phase_id is None:

        def getter(x: lnPiCollection) -> xr.DataArray:
            return getter_comp(x).min("phase")

    else:

        def getter(x: lnPiCollection) -> xr.DataArray:
            return getter_comp(x).sel(phase=phase_id)

    s = getter(collection).to_series().dropna()
    # builder
    if build_kws is None:
        build_kws = {}

    # left bracket
    left = None
    ss = s[s < target]
    if len(ss) > 0:
        left = collection.mloc[ss.index[[-1]]]
    else:
        # TODO(wpk): Need to make collection work better.  Might need to just have a single class for everything...
        new_lnz = collection.mloc[s.index[[0]]]._get_lnz(lnz_idx)
        dlnz_left = dlnz
        for _i in range(ntry):
            new_lnz -= dlnz_left
            p = build_phases(new_lnz, ref=ref, **build_kws)
            if phase_id in p._get_level("phase") and getter(p).to_numpy() < target:
                left = p
                break
            dlnz_left *= dfac

    if left is None:
        msg = "could not find left bounds"
        raise RuntimeError(msg)

    # right bracket
    right = None
    ss = s[s > target]
    if len(ss) > 0:
        right = collection.mloc[ss.index[[0]]]
    else:
        new_lnz = collection.mloc[s.index[[-1]]]._get_lnz(lnz_idx)
        dlnz_right = dlnz

        for _i in range(ntry):
            new_lnz += dlnz_right
            p = build_phases(new_lnz, ref=ref, **build_kws)

            if phase_id not in p._get_level("phase"):
                # went to far
                new_lnz -= dlnz_right
                # reset to half dlnz
                dlnz_right *= 0.5
            elif getter(p).to_numpy() > target:
                right = p
                break
            else:
                dlnz_right *= dfac

    if right is None:
        msg = "could not find right bounds"
        raise RuntimeError(msg)

    shared: dict[str, Any] = {}

    def f(x: float) -> float:
        lnz_new = x
        p = build_phases(lnz_new, ref=ref, **build_kws)
        shared["lnpi"] = p
        return array_to_scalar(getter(p).values) - target

    a, b = sorted([x._get_lnz(lnz_idx) for x in (left, right)])

    xx, r = brentq(f, a, b, full_output=True, **(solve_kws or {}))

    return shared["lnpi"], rootresults_to_rootresultdict(r, residual=f(xx))


def get_lnz_max(
    edge_distance_min: int,
    build_phases: BuildPhasesBase,
    ref: lnPiMasked | None = None,
    lnz_start: float | None = None,
    collection: lnPiCollection | None = None,
    dlnz: float = 0.5,
    dfac: float = 1.0,
    threshold_abs: float = 1e-4,
    ntry: int = 50,
    build_kws: Mapping[str, Any] | None = None,
) -> tuple[lnPiCollection, dict[str, int | float]]:
    """Find max lnz by bisection"""
    build_kws = build_kws or {}
    if (ref := ref or build_phases.phase_creator.ref) is None:
        msg = "must specify `ref` or build_phases must have access to reference lnPiMasked object"
        raise ValueError(msg)

    lnz_idx = build_phases.index
    lnz_start = cast("float", lnz_start or ref.lnz[lnz_idx])

    # need left/right bounds
    # left is greatest lnz point with edge_distance > edge_distance_min
    # right is least lnz point with edge_distance < edge_distance_min
    left: lnPiCollection | None = None
    right: lnPiCollection | None = None
    lnz_left: float | None = None
    lnz_right: float | None = None
    n_left: int = -1
    n_right: int = -1

    def getter(p: lnPiCollection) -> xr.DataArray:
        v = p.xge.edge_distance(ref)
        if not p._xarray_unstack:
            v = v.unstack(p._concat_dim)
        return v.min("phase")

    # if have collection, try to use it
    if collection is not None:
        # see if bound contained in collection

        s = getter(collection).to_series()

        # left
        ss = s[s > edge_distance_min]
        if len(ss) > 0:
            left = collection.mloc[ss.index[[-1]]]
        else:
            lnz_left = collection.zloc[[0]]._get_lnz(lnz_idx)

        # right
        ss = s[s < edge_distance_min]
        if len(ss) > 0:
            right = collection.mloc[ss.index[[0]]]
        else:
            lnz_right = collection.zloc[[-1]]._get_lnz(lnz_idx)

    # if left not set, try to find it
    if left is None:
        if lnz_left is None:
            lnz_left = lnz_start
        dlnz_loc = dlnz
        for i in range(ntry):
            lnz_left -= dlnz_loc
            p = build_phases(lnz_left, ref=ref, **build_kws)
            if getter(p).to_numpy() >= edge_distance_min:
                left = p
                n_left = i
                break
            # grow dlnz
            dlnz_loc *= dfac

    if left is None:
        msg = "could not find left bound"
        raise RuntimeError(msg)

    # if right not set, try to find it
    if right is None:
        if lnz_right is None:
            lnz_right = lnz_start

        dlnz_loc = dlnz
        for i in range(ntry):
            lnz_right += dlnz_loc
            p = build_phases(lnz_right, ref=ref, **build_kws)

            if getter(p).to_numpy() < edge_distance_min:
                right = p
                n_right = i
                break
            # grow dlnz
            dlnz_loc *= dfac

    if right is None:
        msg = "could not find right bound"
        raise RuntimeError(msg)

    # not do bisection
    bracket: list[lnPiCollection] = [left, right]
    values = [getter(x).to_numpy() for x in bracket]

    for i in range(ntry):
        lnz = [x._get_lnz(lnz_idx) for x in bracket]
        if (delta := np.abs(lnz[1] - lnz[0])) < threshold_abs:
            tried = i
            break
        lnz_mid = 0.5 * (lnz[0] + lnz[1])
        mid = build_phases(lnz_mid, ref=ref, **build_kws)
        y_mid = getter(mid).to_numpy()

        index = 0 if y_mid >= edge_distance_min else 1

        bracket[index] = mid
        values[index] = y_mid
    else:
        tried = ntry
        delta = 0.0

    left, right = bracket

    info = {
        "ntry": tried,
        "ntry_left": n_left,
        "ntry_right": n_right,
        "precision": delta,
    }
    return left, info


def build_grid(
    x: ArrayLike | NDArrayAny | None = None,
    dx: float | None = None,
    x_range: Sequence[float] | NDArrayAny | None = None,
    x0: float | None = None,
    offsets: Sequence[float] | NDArrayAny | None = None,
    even_grid: bool = False,
    digits: int | None = None,
    unique: bool = True,
    outlier: bool = False,
) -> NDArrayAny:
    """
    Build a grid of values


    Parameters
    ----------
    x : array, optional
        if passes, use this as the base array
    dx : float, optional
        delta x. See below for uses
    x_range : array of length 2, optional
        if x not specified, make an array with
        `x=np.arange(x_range[0], x_range[1] + dx/2, dx)`
    x0 : float, optional
        see offsets
    offsets : array of length 2, optional
        if x or x_range not specified, set x_range = x0 + offsets
    even_grid : bool, default=True
        if True, make sure all values are multiples of dx
    digits : int, optional
        if passes, round output to this many digits
    outlier : bool, default=False
        if outlier is True, allow values outsize x

    Returns
    -------
    out : array of values
    """
    if x is None:
        if dx is None:
            msg = "Must specify dx"
            raise ValueError(msg)

        if x_range is not None:
            x_range = np.asarray(x_range)
        elif offsets is not None and x0 is not None:
            x_range = x0 + np.asarray(offsets)
        else:
            msg = "must specify x_range or x0 and offsets"
            raise ValueError(msg)

        if len(x_range) != 2:
            raise ValueError

        # for consistency with previous code.
        if even_grid:
            new_range = np.round(x_range / dx) * dx
            if new_range[0] < x_range[0]:
                new_range[0] += dx
            if new_range[1] > x_range[1]:
                new_range[1] -= dx
            x_range = new_range
        x = np.arange(x_range[0], x_range[1] + dx * 0.5, dx)
    else:
        x = np.asarray(x)

    bounds = None if outlier else (x.min(), x.max())

    if even_grid:
        if dx is None:
            raise TypeError
        x = np.round(x / dx) * dx

    if digits is not None:
        x = np.round(x, digits)

    if unique:
        x = np.unique(x)

    if bounds is not None:
        x = x[(bounds[0] <= x) & (x <= bounds[1])]

    return x  # type: ignore[return-value]


def limited_collection(
    build_phases: BuildPhasesBase,
    dlnz: float,
    lnz_range: Sequence[float] | NDArrayAny | None = None,
    offsets: list[int] | None = None,
    digits: int | None = None,
    even_grid: bool = True,
    course_step: int | None = None,
    edge_distance_min: int | None = None,
    dens_min: float | None = None,
    lnz_min_kws: Mapping[str, Any] | None = None,
    lnz_max_kws: Mapping[str, Any] | None = None,
    ref: lnPiMasked | None = None,
    build_kws: Mapping[str, Any] | None = None,
    nmax: int | None = None,
    xarray_output: bool = True,
    collection_kws: Mapping[str, Any] | None = None,
    limit_course: bool = False,
) -> tuple[lnPiCollection, lnPiCollection]:
    """
    Build a lnPiCollection over a range of lnz values

    Parameters
    ----------
    build_phases :
    """
    if lnz_range is None:
        if (ref := ref or build_phases.phase_creator.ref) is None:
            msg = "Must pass in ref or build_phases must have access to reference lnPiMasked object"
            raise ValueError(msg)
        x0 = ref.lnz[build_phases.index]
    else:
        x0 = None

    # TODO: update tests to get rid of outlier=True  # noqa: TD002
    lnzs = build_grid(
        dx=dlnz,
        x_range=lnz_range,
        x0=x0,
        offsets=offsets,
        even_grid=even_grid,
        digits=digits,
        outlier=True,
    )

    if course_step is None:
        course_step = max(len(lnzs) // 40, 1)
    if collection_kws is None:
        collection_kws = {}

    # limit lnz
    c_course = None
    lnz_min, lnz_max = lnzs[0], lnzs[-1]
    if edge_distance_min is not None or dens_min is not None:
        c_course = lnPiCollection.from_builder(
            lnzs[::course_step],
            build_phases=build_phases,
            build_kws=build_kws,
            nmax=nmax,
            xarray_output=xarray_output,
            **collection_kws,
        )

        if dens_min is not None:
            try:
                if lnz_min_kws is None:
                    lnz_min_kws = {}
                p_min, o = get_lnz_min(
                    dens_min, c_course, build_phases, build_kws=build_kws, **lnz_min_kws
                )
                if o["converged"]:
                    lnz_min = p_min.iloc[0].lnz[build_phases.index] - dlnz
            except Exception:  # noqa: BLE001, S110
                pass

        if edge_distance_min is not None:
            try:
                if lnz_max_kws is None:
                    lnz_max_kws = {}
                p_max, _ = get_lnz_max(
                    edge_distance_min,
                    build_phases,
                    build_kws=build_kws,
                    collection=c_course,
                    **lnz_max_kws,
                )
                lnz_max = p_max.iloc[0].lnz[build_phases.index]
            except Exception:  # noqa: BLE001,S110
                pass

    lnzs = lnzs[(lnzs >= lnz_min) & (lnzs <= lnz_max)]
    c = lnPiCollection.from_builder(
        lnzs,
        build_phases=build_phases,
        build_kws=build_kws,
        nmax=nmax,
        xarray_output=xarray_output,
        **collection_kws,
    )

    if c_course is None:
        return c, c

    if limit_course:
        slnz = f"lnz_{build_phases.index}"
        q = f"{slnz} >= {lnz_min} and {slnz} <= {lnz_max}"
        return c_course.query(q), c  # pylint: disable=too-many-function-args

    return c_course, c
