"""Config for doctests and test collection"""

from functools import lru_cache

import pytest


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace) -> None:  # noqa: ARG001
    import numpy as np

    np.set_printoptions(precision=4)


@lru_cache
def _is_default_version() -> bool:
    import sys
    from pathlib import Path

    return sys.version_info[:2] == tuple(
        map(int, Path(".python-version").read_text(encoding="utf-8").strip().split("."))
    )


def pytest_ignore_collect(collection_path) -> bool:
    if not _is_default_version():
        return "tmmc-lnpy/tests" not in str(collection_path)
    return False
