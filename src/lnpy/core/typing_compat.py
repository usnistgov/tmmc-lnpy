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
    from typing import Self
else:
    from typing_extensions import Self

if sys.version_info >= (3, 13):  # pragma: no cover
    from typing import TypeIs, TypeVar
else:
    from typing_extensions import TypeIs, TypeVar

if TYPE_CHECKING:
    import pandas as pd

IndexAny: TypeAlias = "pd.Index[Any]"  # type: ignore[misc, unused-ignore]  # get pd.Index working for python 3.8


__all__ = [
    "Concatenate",
    "IndexAny",
    "ParamSpec",
    "Self",
    "TypeAlias",
    "TypeIs",
    "TypeVar",
]
