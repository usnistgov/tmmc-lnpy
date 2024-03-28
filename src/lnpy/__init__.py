"""Publicly accessible classes/routines."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ensembles, lnpienergy, segment  # noqa: TCH004
    from .combine import combine_lnpi  # noqa: TCH004
    from .lnpidata import lnPiMasked  # noqa: TCH004
    from .lnpiseries import lnPiCollection  # noqa: TCH004
    from .options import OPTIONS, set_options  # noqa: TCH004
    from .segment import PhaseCreator  # noqa: TCH004
else:
    import lazy_loader as lazy

    __getattr__, __dir__, _ = lazy.attach(
        __name__,
        submodules=["ensembles", "lnpienergy", "segment"],
        submod_attrs={
            "lnpidata": ["lnPiMasked"],
            "lnpiseries": ["lnPiCollection"],
            "options": ["OPTIONS", "set_options"],
            "segment": ["PhaseCreator"],
        },
    )


# updated versioning scheme
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("tmmc-lnpy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"


__all__ = [
    "OPTIONS",
    "PhaseCreator",
    "__author__",
    "__email__",
    "__version__",
    "combine_lnpi",
    "ensembles",
    "lnPiCollection",
    "lnPiMasked",
    "lnpienergy",
    "segment",
    "set_options",
]
