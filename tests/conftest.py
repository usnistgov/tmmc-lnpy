from __future__ import annotations

import json
import locale
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lnpy import lnPiMasked  # pylint: disable=no-name-in-module


def get_lnz(path):
    k_b = 1.3806503e-23  # J/K
    kg = 1.66054e-27  # 1/amu
    h_planck = 6.62606876e-34  # Js

    with Path(path).open(encoding=locale.getpreferredencoding(False)) as f:
        metadata = json.load(f)

    debroglie = (
        h_planck
        / np.sqrt(
            2.0
            * np.pi
            * (metadata["mass"] * kg)
            * k_b
            * (metadata["T*"] * metadata["eps_kB"])
        )
        * 1.0e10
    )  # ang

    mu = metadata["mu*"] + 3.0 * metadata["T*"] * np.log(metadata["sigma"] / debroglie)
    temp = metadata["T*"]

    lnz = mu / temp

    return lnz, {"beta": 1.0 / temp, "volume": metadata["V*"]}


def get_ref(
    path_data: str | Path,
    metadata_name: str,
    lnpi_name: str,
    energy_name: str,
) -> lnPiMasked:
    path_data = Path(path_data)
    lnz, state_kws = get_lnz(path_data / metadata_name)

    # read in potential energy
    pe = pd.read_csv(
        path_data / energy_name,
        header=None,
        sep="\\s+",
        names=["n", "e"],
    )["e"].to_numpy()

    return (
        lnPiMasked
        .from_table(
            path_data / lnpi_name,
            fill_value=np.nan,
            lnz=lnz,
            # state_kws needs to be defined if want to calculate some properties
            # down the road
            state_kws=state_kws,
            # extra_kws is where you pass things which will be passed along to
            # new lnPis, like potential energy
            extra_kws={"PE": pe},
        )
        .zeromax()
        .pad()
    )


@pytest.fixture(scope="session")
def path_data_lnPi():  # noqa: N802
    return Path(__file__).parent / "../examples/archived/LJ_cfs_2.5sig"


@pytest.fixture(scope="session")
def ref_lnPi(path_data_lnPi):  # noqa: N802,N803
    import lnPi

    path_data = path_data_lnPi

    lnz, state_kws = get_lnz(path_data / "t150.metadata.json")

    # read in potential energy
    pe = pd.read_csv(
        path_data / "ljsf.t150.bulk.v512.r1.energy.dat",
        header=None,
        sep="\\s+",
        names=["n", "e"],
    )["e"].to_numpy()

    return (
        lnPi.MaskedlnPiDelayed
        .from_table(
            path_data / "ljsf.t150.bulk.v512.r1.lnpi.dat",
            fill_value=np.nan,
            lnz=lnz,
            # state_kws needs to be defined if want to calculate some properties
            # down the road
            state_kws=state_kws,
            # extra_kws is where you pass things which will be passed along to
            # new lnPis, like potential energy
            extra_kws={"PE": pe},
        )
        .zeromax()
        .pad()
    )


@pytest.fixture(scope="session")
def path_data_single_comp_super():
    return Path(__file__).parent / "../examples/archived/LJ_cfs_2.5sig"


@pytest.fixture(scope="session")
def ref_single_comp_super(path_data_single_comp_super):
    return get_ref(
        path_data=path_data_single_comp_super,
        metadata_name="t150.metadata.json",
        energy_name="ljsf.t150.bulk.v512.r1.energy.dat",
        lnpi_name="ljsf.t150.bulk.v512.r1.lnpi.dat",
    )


@pytest.fixture(scope="session")
def path_data_single_comp_sub():
    return Path(__file__).parent / "../examples/archived/LJ_cfs_2.5sig"


@pytest.fixture(scope="session")
def ref_single_comp_sub(path_data_single_comp_super):
    return get_ref(
        path_data=path_data_single_comp_super,
        metadata_name="t072871.metadata.json",
        energy_name="ljsf.t072871.bulk.v512.r1.energy.dat",
        lnpi_name="ljsf.t072871.bulk.v512.r1.lnpi.dat",
    )
