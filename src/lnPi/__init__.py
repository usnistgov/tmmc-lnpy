"""
Legacy interface to :mod:`lnpy` (:mod:`lnPi`)
=============================================
"""
# ruff: noqa: RUF067

from warnings import warn

from lnpy import __version__, options, segment, set_options
from lnpy.lnpidata import lnPiMasked as MaskedlnPiDelayed
from lnpy.lnpiseries import lnPiCollection as CollectionlnPi

from . import stability

msg = """\
lnPi is deprecated.  Please transition to using lnpy instead.  Name changes are:

Modules:
lnPi -> lnpy
lnPi.collectionlnpiutils -> lnpy.lnpicollectionutils

Classes:
MaskedlnPiDelayed -> lnPiMasked
CollectionlnPi    -> lnPiCollection
"""


warn(msg, DeprecationWarning)


__all__ = [
    "CollectionlnPi",
    "MaskedlnPiDelayed",
    "__version__",
    "options",
    "segment",
    "set_options",
    "stability",
]
