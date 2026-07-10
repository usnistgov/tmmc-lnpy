from __future__ import annotations

from typing import TYPE_CHECKING

from lnpy.core.typing import NDArrayAny

if TYPE_CHECKING:
    from lnpy.core.typing_kwargs import PeakLocalMaxAdaptiveKwargs


def type_peak_local_max_adaptive(
    data: NDArrayAny, kws: PeakLocalMaxAdaptiveKwargs
) -> None:

    from typing_extensions import assert_type

    from lnpy.segment import peak_local_max_adaptive

    _ = assert_type(
        peak_local_max_adaptive(data, style="indices", **kws), tuple[NDArrayAny, ...]
    )
    _ = assert_type(peak_local_max_adaptive(data, style="mask", **kws), NDArrayAny)
    _ = assert_type(peak_local_max_adaptive(data, style="marker", **kws), NDArrayAny)
    _ = assert_type(
        peak_local_max_adaptive(data, style="other", **kws),
        tuple[NDArrayAny, ...] | NDArrayAny,
    )
