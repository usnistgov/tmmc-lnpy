from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

    from .typing_compat import Self


def _get_smart_filter(
    self_: Any,
    include: str | Sequence[str] | None = None,
    exclude: str | Sequence[str] | None = None,
    exclude_private: bool = True,
    exclude_no_init: bool = True,
) -> Callable[..., Any]:
    """
    Create a filter to include exclude names.

    Parameters
    ----------
    include : sequence of str
        Names to include
    exclude : sequence of str
        names to exclude
    exclude_private : bool, default=True
        If True, exclude any names starting with '_'
    exclude_no_init : bool, default=True
        If True, exclude any fields defined with ``field(init=False)``.

    Notes
    -----
    Precedence is in order
    `include, exclude, exclude_private exclude_no_init`.
    That is, if a name is in include and exclude and is private/no_init,
    it will be included
    """
    fields = attrs.fields(type(self_))

    if include is None:
        include = []
    elif isinstance(include, str):
        include = [include]

    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    includes: list[Any] = []
    for f in fields:
        if f.name in include:
            includes.append(f)

        elif (
            f.name in exclude
            or (exclude_private and f.name.startswith("_"))
            or (exclude_no_init and not f.init)
        ):
            pass

        else:
            includes.append(f)
    return attrs.filters.include(*includes)


class MyAttrsMixin:
    """Baseclass for adding some sugar to attrs.derived classes."""

    def asdict(self) -> dict[str, Any]:
        """Convert object to dictionary."""
        return attrs.asdict(self, filter=_get_smart_filter(self))  # type: ignore[arg-type]

    def new_like(self, **kws: Any) -> Self:
        """
        Create a new object with optional parameters.

        Parameters
        ----------
        **kws
            `attribute`, `value` pairs.
        """
        return attrs.evolve(self, **kws)  # type: ignore[misc]

    def assign(self, **kws: Any) -> Self:
        """Alias to :meth:`new_like`."""
        return self.new_like(**kws)
