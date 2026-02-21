from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import lnPi
import lnpy.examples


# get meta data
@pytest.fixture(scope="session")
def ref(ref_lnPi):  # noqa: N803
    return ref_lnPi


@pytest.fixture(scope="session")
def path_data(path_data_lnPi):  # noqa: N803
    return path_data_lnPi


@pytest.fixture(scope="session")
def phase_creator(ref):
    return lnPi.segment.PhaseCreator(
        nmax=1,
        nmax_peak=4,
        ref=ref,
        merge_kws={"efac": 0.8},
        # if want to id phases based on some
        # callable, then set this.
        # the callable is passes a list of
        # MaskedlnPi objects, and should return
        # a list of phaseid
        tag_phases=None,
    )


@pytest.fixture(scope="session")
def build_phases(phase_creator):
    # choose a particular factory method for creating phases
    return phase_creator.build_phases_mu([None])


# can drop param = 0 after make sure all good
@pytest.fixture(params=[1])
def obj(request, ref, phase_creator, build_phases):
    if not request.param:
        return lnpy.examples.Example(
            ref=ref, phase_creator=phase_creator, build_phases=build_phases
        )

    return lnpy.examples.lj_sup_example()


@pytest.fixture(scope="session")
def test_table(path_data):
    return pd.read_csv(path_data / "data_0.csv")


def get_test_table(o, ref):
    return (
        o.xge.table(
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


def test_collection_properties(obj, test_table) -> None:
    ref, build_phases = obj.unpack(["ref", "build_phases"])

    # for big builds, take advantage of progress bar, and parallel builds
    lnzs = np.linspace(-10, 3.5, 2000)

    # by default, progress bar hides itself after completion.  use context manager to keep it
    # note that for this example (where only have a single phase), doesn't really make a difference
    with lnPi.set_options(tqdm_leave=True, joblib_use=False, tqdm_bar="text"):
        o = lnPi.CollectionlnPi.from_builder(lnzs, build_phases)

    other = get_test_table(o, ref)

    pd.testing.assert_frame_equal(test_table, other)


@pytest.fixture(scope="session")
def test_table_can(path_data):
    return pd.read_csv(path_data / "data_0_can.csv")


def get_test_table_can(ref):
    return (
        ref.xce.table(
            keys=["S", "Z", "betaE", "betaF", "betaOmega", "betamu", "dens", "ntot"]
        )
        .to_dataframe()
        .reset_index()
    )


def test_canonical_properties(obj, test_table_can) -> None:
    ref = obj.ref
    other = get_test_table_can(ref)
    pd.testing.assert_frame_equal(other, test_table_can)


def test_nice_grid(obj, path_data) -> None:
    ref, build_phases = obj.unpack(["ref", "build_phases"])
    import lnPi.collectionlnpiutils

    with lnPi.set_options(joblib_use=True):
        o_course, o = lnPi.collectionlnpiutils.limited_collection(
            build_phases,
            dlnz=0.01,
            offsets=[-10, +10],
            even_grid=True,  # but lnzs on same grid as dlnz
            digits=2,  # round lnzs to this number of digits
            edge_distance_min=10,
            dens_min=0.001,
        )

    other_course = get_test_table(o_course, ref)
    test_course = pd.read_csv(path_data / "data_0_course.csv")

    other_fine = get_test_table(o, ref)
    test_fine = pd.read_csv(path_data / "data_0_fine.csv")

    v0 = other_fine.lnz_0.to_numpy()
    v1 = test_fine.lnz_0.to_numpy()

    np.testing.assert_allclose(v0, v1)

    pd.testing.assert_frame_equal(other_course, test_course)

    pd.testing.assert_frame_equal(other_fine, test_fine)
