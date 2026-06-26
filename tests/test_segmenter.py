from __future__ import annotations

import numpy as np

import lnpy


def test_new_like() -> None:
    rng = np.random.default_rng()
    refs = [
        lnpy.lnPiMasked.from_data(data=rng.random(n), lnz=n, lnz_data=n)
        for n in [5, 10]
    ]

    creators = [lnpy.PhaseCreator(ref=ref, nmax=len(ref)) for ref in refs]

    new = creators[0].new_like(ref=refs[1], nmax=len(refs[1]))

    assert new.ref is creators[1].ref
    assert new.nmax == creators[1].nmax
