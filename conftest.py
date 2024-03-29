# flake8: noqa
import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace):
    import numpy as np
    import pandas as pd
    import xarray as xr

    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["xr"] = xr

    # always seed numpy.random to make the examples deterministic
    np.random.seed(0)

    np.set_printoptions(precision=4)
