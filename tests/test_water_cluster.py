import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lnpy
import lnpy.stability

path_data = Path(__file__).parent / "water_cluster"


def tag_phases2(x):
    if len(x) > 2:
        raise ValueError("bad tag function")
    argmax0 = np.array([xx.local_argmax()[0] for xx in x])
    return np.where(argmax0 <= x[0].shape[0] / 2, 0, 1)


def load_ref():
    ensemble = json.load(open(path_data / "water_MOF_ensemble.json"))
    lnz = ensemble["betamu0"]
    box = ensemble["box"]
    state_kws = {
        "beta": ensemble["beta"],
        "volume": box[0] * box[1] * box[2],  # being lazy here because the box is cubic
    }

    data = pd.read_csv(path_data / "water_MOF_example.csv")

    # potential energy
    pe = data["U"].values

    # Build the lnPi object
    # DWS Note to self: I'm doing something sloppy here, in that the N values are not specified
    ref = lnpy.lnPiMasked.from_data(
        data=data["lnPi"].values,
        fill_value=np.nan,
        lnz=lnz,
        lnz_data=lnz,
        # state_kws needs to be defined if want to calculate some properties down the road
        state_kws=state_kws,
        # extra_kws is where you pass things which will be passed along to new lnPis, like potential energy
        extra_kws={"PE": pe, "other": pe},
    )
    return ref


@pytest.fixture(params=[0, 1])
def ref(request):
    if request.param == 0:
        return load_ref()
    else:
        import lnpy.examples

        return lnpy.examples.load_example_maskddata("watermof")


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


def do_test(phase_creator, ref, fname):
    p = phase_creator.build_phases()

    other = get_test_table(p, ref)
    test = pd.read_csv(path_data / fname)

    pd.testing.assert_frame_equal(other, test)


def test_0(ref):
    phase_creator = lnpy.segment.PhaseCreator(
        nmax=2,  # number of phases
        nmax_peak=10,  # max number of peaks in lnPi
        ref=ref,
        tag_phases=tag_phases2,
        merge_kws={"efac": 0.8},
    )
    do_test(phase_creator, ref, "data_0.csv")


def test_1(ref):
    phase_creator = lnpy.segment.PhaseCreator(
        nmax=20,  # number of phases
        nmax_peak=50,  # max number of peaks in lnPi
        ref=ref,
        tag_phases=None,
        merge_kws={"efac": 0.1},
        segment_kws={"peaks_kws": {"min_distance": [5]}},
    )
    do_test(phase_creator, ref, "data_1.csv")
