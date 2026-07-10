"""Handle typing compatibility issues."""
# pyright: reportUnreachable=false

import sys
from typing import (  # pylint: disable=unused-import
    TYPE_CHECKING,
    Any,
    Concatenate,
    ParamSpec,
    TypeAlias,
)

if sys.version_info >= (3, 11):
    from typing import Self, TypedDict, Unpack, assert_never
else:
    from typing_extensions import Self, TypedDict, Unpack, assert_never


if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if sys.version_info >= (3, 13):  # pragma: no cover
    from typing import TypeIs, TypeVar
else:
    from typing_extensions import TypeIs, TypeVar

if sys.version_info >= (3, 15):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

if TYPE_CHECKING:
    import pandas as pd

IndexAny: TypeAlias = "pd.Index[Any]"


__all__ = [
    "Concatenate",
    "IndexAny",
    "ParamSpec",
    "Self",
    "TypeAlias",
    "TypeIs",
    "TypeVar",
    "TypedDict",
    "Unpack",
    "assert_never",
    "override",
]
