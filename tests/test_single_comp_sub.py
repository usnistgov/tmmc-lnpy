from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lnpy
import lnpy.examples


def tag_phases2(x):
    if len(x) > 2:
        msg = "bad tag function"
        raise ValueError(msg)
    argmax0 = np.array([xx.local_argmax()[0] for xx in x])
    return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)


@pytest.fixture(scope="session")
def path_data(path_data_single_comp_sub):
    return path_data_single_comp_sub


@pytest.fixture(scope="session")
def ref(ref_single_comp_sub):
    return ref_single_comp_sub


@pytest.fixture(scope="session")
def phase_creator(ref):
    return lnpy.segment.PhaseCreator(
        nmax=2,
        nmax_peak=4,
        ref=ref,
        merge_kws={"efac": 0.8},
        # if want to id phases based on some
        # callable, then set this.
        # the callable is passes a list of
        # MaskedlnPi objects, and should return
        # a list of phaseid
        tag_phases=tag_phases2,
    )


@pytest.fixture(scope="session")
def build_phases(phase_creator):
    # choose a particular factory method for creating phases
    return phase_creator.build_phases_mu([None])


@pytest.fixture(params=[1])
def obj(request, ref, phase_creator, build_phases):
    if not request.param:
        return lnpy.examples.Example(
            ref=ref, phase_creator=phase_creator, build_phases=build_phases
        )
    return lnpy.examples.lj_sub_example()


@pytest.fixture(scope="session")
def test_table(path_data):
    return pd.read_csv(path_data / "data_1.csv")


def get_test_table(o, ref):
    return (
        o.xge
        .table(
            keys=[
                "betaOmega",
                "nvec",
                "PE",
                "dens",
                "betaF",
                "S",
                "betaG",
                "edge_distance",
            ],
            ref=ref,
        )
        .to_dataframe()
        .reset_index()
    )  # .to_csv('data_0.csv', index=False)


def test_collection_properties(build_phases, test_table, obj) -> None:
    ref = obj.ref

    # for big builds, take advantage of progress bar, and parallel builds
    lnzs = np.linspace(-10, 3, 2000)

    # by default, progress bar hides itself after completion.  use context manager to keep it
    # note that for this example (where only have a single phase), doesn't really make a difference
    with lnpy.set_options(tqdm_leave=True, joblib_use=False, tqdm_bar="text"):
        o = lnpy.lnPiCollection.from_builder(lnzs, build_phases)

    other = get_test_table(o, ref)

    pd.testing.assert_frame_equal(test_table, other)


@pytest.fixture(scope="session")
def test_table_can(path_data):
    return pd.read_csv(path_data / "data_1_can.csv")


def get_test_table_can(ref):
    return (
        ref.xce
        .table(keys=["S", "Z", "betaE", "betaF", "betaOmega", "betamu", "dens", "ntot"])
        .to_dataframe()
        .reset_index()
    )


def test_canonical_properties(obj, test_table_can) -> None:
    ref = obj.ref
    other = get_test_table_can(ref)
    pd.testing.assert_frame_equal(other, test_table_can)


def test_nice_grid(obj, path_data) -> None:
    ref = obj.ref
    build_phases = obj.build_phases

    import lnpy.lnpicollectionutils

    with lnpy.set_options(joblib_use=True):
        o_course, o = lnpy.lnpicollectionutils.limited_collection(
            build_phases,
            dlnz=0.01,
            offsets=[-10, +10],
            even_grid=True,  # but lnzs on same grid as dlnz
            digits=2,  # round lnzs to this number of digits
            edge_distance_min=10,
            dens_min=0.001,
        )

    other_course = get_test_table(o_course, ref)
    test_course = pd.read_csv(path_data / "data_1_course.csv")
    pd.testing.assert_frame_equal(other_course, test_course)

    other_fine = get_test_table(o, ref)
    test = pd.read_csv(path_data / "data_1_fine.csv")
    pd.testing.assert_frame_equal(other_fine, test)

    # spinodal/binodal
    o_course.spinodal(2, build_phases)  # pylint: disable=too-many-function-args
    o_course.binodal(2, build_phases)  # pylint: disable=too-many-function-args

    other = get_test_table(o_course.spinodal.access, ref)
    test = pd.read_csv(path_data / "data_1_spin.csv")
    pd.testing.assert_frame_equal(other, test)

    other = get_test_table(o_course.binodal.access, ref)
    test = pd.read_csv(path_data / "data_1_bino.csv")
    pd.testing.assert_frame_equal(other, test)

    # check dw
    other = o_course.spinodal.access.wfe.dw.to_frame().reset_index()
    test = pd.read_csv(path_data / "data_1_spin_dw.csv")
    pd.testing.assert_frame_equal(other, test)
