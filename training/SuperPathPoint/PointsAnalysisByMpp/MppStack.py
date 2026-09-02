"""Read one co-registered stack: the same level-0 centre at every rung.

    chains = MppStack.chains(tiles_root, wsi_stem, tile=256)
    f = MppStack.f_stack(chains[7])          # {ds: [tile, tile, 3] uint8}
    r = MppStack.r_stack(chains[7], rungs)   # derived, no second read

THE 'R' STACK IS DERIVED, NOT EXTRACTED
=========================================
An 'R' rung is, by its own definition (`TileSampler.resolution_plan`), `tile`
level-0 pixels read at level 0, shrunk by `ds` and grown back. A chain's ds 1
tile IS `tile` level-0 pixels read at level 0. So the whole 'R' stack is
`degrade_resolution` applied to that one image, and it needs no extraction, no
second store and no WSI handle.

THAT IS NOT A SHORTCUT, IT IS THE ONLY WAY THE TWO AXES STAY COMPARABLE.
新生歸因 asks whether a point born late on 'F' is also born late on 'R', which
requires the two axes to be about the SAME PHYSICAL POINT. Two independent
extractions would choose centres by their own admissibility -- an 'F' centre has
to fit a `tile * ds_max` footprint and an 'R' centre only `tile`, so the coarse
rungs would drop centres the 'R' run keeps -- and the axes would have to be
joined spatially afterwards, with a tolerance, on exactly the quantity the
analysis is about. Deriving 'R' from the chain makes the centres identical by
construction.

`degrade_resolution` lives in `TileSampler` and is called from both places for
the same reason: two spellings of the shrink-and-grow would make a survival
number a statement about which resampling filter each half used.

WHAT A CHAIN IS AND WHY INCOMPLETE ONES ARE DROPPED
=====================================================
A chain is one level-0 centre with a tile at every rung -- `inherit_id` in the
pre-tile index groups them. A chain missing a rung is DROPPED rather than
carried with a gap, `TileSampler.stacks`'s reason: a four-rung chain handed over
as if it were six reads as "the keypoint died at the two missing rungs" when it
means "those rungs never cut a tile there", and telling those two apart is the
whole of a survival measurement.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import PreTileStore                                          # noqa: E402
from PreTileStore import centre_crop                         # noqa: E402
from TileSampler import degrade_resolution                   # noqa: E402


@dataclass
class Chain:
    """One level-0 centre, and the pre-tile of every rung at it."""
    wsi_stem: str
    inherit_id: int
    #: level-0 CENTRE, not the top-left. The centre is what the rungs share --
    #: their top-left corners differ because their footprints differ -- so
    #: carrying the corner would make "the same centre" a derived quantity that
    #: each caller re-derives with its own rounding.
    cx: float
    cy: float
    #: ds -> (store folder, index record). Sorted finest first by `rungs`.
    members: Dict[float, tuple]

    @property
    def rungs(self) -> List[float]:
        return sorted(self.members)


def chains(tiles_root, wsi_stem: str, *, tile: int,
           rungs: Optional[Sequence[float]] = None) -> Dict[int, Chain]:
    """Every COMPLETE chain of one slide, keyed by `inherit_id`.

    Complete against `rungs` when given, and against whatever rungs the store
    holds when not. The explicit form is the one to use: "complete" measured
    against what happened to be extracted cannot notice a rung that failed to
    extract at all.
    """
    want = sorted(float(r) for r in rungs) if rungs else None
    found: Dict[int, Dict[float, tuple]] = {}
    centres: Dict[int, tuple] = {}
    seen_rungs = set()
    # ONE STORE PER (slide, rung), REFUSED RATHER THAN UNIONED, and the failure
    # here is worse than the one the same guard prevents in `Datasets`. Two
    # corpora in a root both number their chains from 0, so a union does not
    # merely double the data -- it MERGES CHAINS THAT ARE NOT THE SAME CHAIN,
    # and a "complete" one can be four rungs of one corpus and two of another
    # at a different level-0 centre. Every survival number computed on it would
    # be about points that were never co-registered, and nothing would say so.
    # It happened: two smoke runs into one root, 2026-09-01.
    seen: Dict[float, object] = {}

    for folder in sorted(PreTileStore.find(tiles_root, tile=int(tile))):
        meta = PreTileStore.load_meta(folder)
        if meta.wsi_stem != wsi_stem:
            continue
        if want is not None and not any(abs(meta.ds - r) < 1e-6 for r in want):
            continue
        if float(meta.ds) in seen:
            raise ValueError(
                f'two pre-tile stores for {wsi_stem} at ds {meta.ds:g} under '
                f'{tiles_root}:\n  {seen[float(meta.ds)].name}\n  '
                f'{folder.name}\nBoth number their chains from 0, so reading '
                f'them together merges chains that are not the same chain -- a '
                f'stack whose rungs are at different level-0 centres, which is '
                f'the one thing this whole analysis assumes cannot happen. '
                f'Point --tiles-root at one of them')
        seen[float(meta.ds)] = folder
        seen_rungs.add(float(meta.ds))
        for record in PreTileStore.load_index(folder):
            cid = int(getattr(record, 'inherit_id', -1))
            if cid < 0:
                continue
            found.setdefault(cid, {})[float(meta.ds)] = (folder, record, meta)
            if float(meta.ds) == min(seen_rungs):
                half = 0.5 * float(meta.tile) * float(meta.ds)
                centres[cid] = (record.x + half, record.y + half)

    target = want if want is not None else sorted(seen_rungs)
    out: Dict[int, Chain] = {}
    for cid, members in found.items():
        if len(members) < len(target):
            continue
        cx, cy = centres.get(cid, (float('nan'), float('nan')))
        out[cid] = Chain(wsi_stem=wsi_stem, inherit_id=cid,
                         cx=cx, cy=cy, members=members)
    return out


def f_stack(chain: Chain, *, tile: int) -> Dict[float, np.ndarray]:
    """`{ds: [tile, tile, 3] uint8}` read from the store, one per rung.

    The stored image is the PRE-TILE (`pre_tile_factor` times the tile, so a
    warp has somewhere to come from); the tile is its centre crop, which is what
    every other consumer uses (`Datasets.__getitem__`). Cropping here rather
    than storing both is what keeps the two from disagreeing about where the
    centre is.
    """
    out = {}
    for ds, (folder, record, _meta) in sorted(chain.members.items()):
        pre = PreTileStore.read_tile(folder, record)
        out[float(ds)] = centre_crop(pre, int(tile))
    return out


def r_stack(chain: Chain, rungs: Sequence[float], *, tile: int
            ) -> Dict[float, np.ndarray]:
    """`{ds: [tile, tile, 3] uint8}` DERIVED from the chain's ds 1 tile.

    No store read beyond that one tile and no WSI handle. See the module
    docstring: an 'R' rung is `tile` level-0 px shrunk by `ds` and grown back,
    and the ds 1 tile is `tile` level-0 px.

    Raises when the chain has no ds 1 member rather than falling back to the
    finest available: an 'R' stack built from a ds 2 tile would be degrading
    something already degraded, so every `ds` label on it would be wrong by a
    factor of 2 -- and it would look completely normal.
    """
    if not any(abs(ds - 1.0) < 1e-6 for ds in chain.members):
        raise ValueError(
            f'chain {chain.inherit_id} of {chain.wsi_stem} has no ds 1 tile '
            f'(rungs {chain.rungs}), so an R stack cannot be derived from it. '
            f'Degrading a ds 2 tile instead would mislabel every rung by 2x '
            f'and nothing downstream would notice')
    base = f_stack(chain, tile=tile)[1.0]
    return {float(ds): degrade_resolution(base, float(ds), int(tile))
            for ds in sorted(float(r) for r in rungs)}


def rung_scale(ds: float, stack_kind: str) -> float:
    """How many level-0 px one output pixel SPANS. For `to_level0`.

    THIS IS NOT `rung_shrink` AND CONFUSING THEM IS SILENT. On the 'F' axis a
    tile of `tile` pixels covers `tile * ds` level-0 px, so one pixel spans
    `ds`. On the 'R' axis the footprint is `tile` level-0 px at EVERY rung, so
    one pixel spans 1.0 no matter how degraded the image is -- the shrink and
    grow happen inside a frame that never moves.

    Using `ds` here for an 'R' rung would scatter every coarse-rung point `ds`
    times too far from the centre. The points would still be inside a plausible
    range, the table would fill, and the survival numbers would be a picture of
    the bug.
    """
    if stack_kind == 'F':
        return float(ds)
    if stack_kind == 'R':
        return 1.0
    raise ValueError(f"stack_kind must be 'F' or 'R', got {stack_kind!r}")


def rung_shrink(ds: float, stack_kind: str) -> float:
    """How many level-0 px one output pixel is worth, per axis.

    'F': `ds`, because the tile covers `tile * ds` level-0 px in `tile` pixels.
    'R': also `ds`, but for a different reason -- the footprint is `tile` level-0
    px at every rung and `ds` is the DEGRADATION, so a position is only knowable
    to `ds` level-0 px even though the image is `tile` px wide.

    The two agree numerically and disagree in meaning, which is exactly the case
    a function exists for: `tau` needs "how far off can a position be", and a
    caller that reached for `footprint / tile` would get 1.0 on the 'R' axis and
    a tau that no coarse rung could ever satisfy.

    Its twin is `rung_scale`, which answers the OTHER question -- how far apart
    two pixels are in level-0 -- and answers it differently on the 'R' axis.
    They are two functions because they are two quantities that happen to agree
    on one of the two axes.
    """
    if stack_kind not in ('F', 'R'):
        raise ValueError(f"stack_kind must be 'F' or 'R', got {stack_kind!r}")
    return float(ds)
