"""The three axes of Stage B. spec.md 3.2 "三個軸，一張表".

    chains = chains(corpus, wsi_stem, tile=256)      # corpus: Corpora.Corpus
    f = FStack.read(chains[7], tile=256)             # {ds: [tile,tile,3] uint8}
    r = RStack.derive(chains[7], rungs, tile=256)     # derived, no second read
    groups = CStack.pyramid(chains[7].cx, chains[7].cy, rungs, tile=256)

Each class answers a DIFFERENT SHAPE of question and is split the same way
internally: a `footprint`/`pyramid` staticmethod that is PURE GEOMETRY (no
pixel read, no store, no WSI handle -- can be called and tested with no
corpus at all), and a `read`/`derive` staticmethod that is the IO built on
top of it (`Corpus.read`, `TileSampler`'s `degrade_resolution`). Keeping the
split inside each class rather than only across files is what lets a test
call `FStack.footprint(...)` without a store existing anywhere.

    FStack   ONE tile per rung, footprint grows with ds, independently read
    RStack   ONE tile per rung, footprint fixed at `tile` px, DERIVED
    CStack   MANY tiles per rung -- a recursive tree, real reads throughout

WHAT A CHAIN IS AND WHY INCOMPLETE ONES ARE DROPPED
=====================================================
A chain is one level-0 centre with a tile at every rung -- `inherit_id` in the
corpus's draw groups them. A chain missing a rung is DROPPED rather than
carried with a gap, `TileSampler.stacks`'s reason: a four-rung chain handed over
as if it were six reads as "the keypoint died at the two missing rungs" when it
means "those rungs never cut a tile there", and telling those two apart is the
whole of a survival measurement.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from TileSampler import centre_crop                          # noqa: E402
from PatchingLib import PatchGrid, PatchInfo                  # noqa: E402
from ReadGeometry import ReadSpec                             # noqa: E402
from SlideReader import SlideReader, degrade_resolution       # noqa: E402

#: Local on-disk cache for tiles that do NOT come from a `TileSampler` draw --
#: `CStack`'s descendant reads. NOT a corpus: a corpus is one draw's identity
#: (`sampler_id`, the mask, the plan) and a computed tile's position was never
#: sampled -- filing it there would invent a `sampler_id` for something that
#: has none.
#:
#: ONE DIRECTORY PER PYRAMID, NOT ONE FLAT POOL: in a flat pool nothing on disk
#: says which files belong together, so "how big is THIS pyramid's cache" is a question
#: nobody could answer without re-deriving every position. `_pyramid_dir`
#: gives every (axis, wsi_stem, root tile, finest rung) its own subdirectory --
#: `slide=<s>/chainstack/<Axis>Stack/<key>/` -- so a directory listing is how
#: anyone finds these.
#:
#: WHOSE CACHE IS THE CALLER'S. Every `cache_root` here is a job name, the
#: tree the tiles go to (`_cache_dir`), and defaults to None (no cache); an
#: entry point passes its own (`training/SuperPathPoint/cli.chainstack_root`).


def _cache_dir(job: str, wsi_stem: str) -> str:
    """`slide=<wsi_stem>/chainstack/` in `job`'s cache tree."""
    from Cache import Address                                    # noqa: PLC0415
    return str(Address(job, slide=wsi_stem).dir / 'chainstack')


def _pyramid_dir(cache_root: str, axis: str, wsi_stem: str, root: PatchInfo,
                 tile: int, finest: float) -> str:
    """`<_cache_dir>/<axis>Stack/<wsi_stem>__ds<root.ds>_x<x>_y<y>_t<tile>_to<finest>/`

    `root` is the pyramid's own root tile -- `CStack`'s mother -- not a centre
    point, so every axis can key off the same kind of object it already has
    in hand.

    `finest` DISAMBIGUATES two pyramids at the same root position and `tile`
    but a DIFFERENT `--rungs` ladder (e.g. `1 2 4 8 16` vs `1 2 4 8 16 32`):
    one directory would mix files from two different requests -- harmless
    per FILE
    (`_read_wsi_tile`'s cache is content-addressed by `(wsi_stem, x, y, ds,
    tile)`, so a shared file is always the same real pixels), but wrong for
    the one thing this per-pyramid layout exists to answer: "how big is THIS
    pyramid's cache" (module docstring). No cryptographic hash needed --
    `pyramid()`/`_children_of` only accept a CONTIGUOUS run of 2x-adjacent
    rungs, so `(root.ds, finest)` already names the rung set exactly.
    """
    name = f'{wsi_stem}__ds{root.ds:g}_x{root.x}_y{root.y}_t{tile}_to{finest:g}'
    return os.path.join(_cache_dir(cache_root, wsi_stem), f'{axis}Stack', name)


def _cache_path(cache_root: str, wsi_stem: str, x: int, y: int, ds: float,
                tile: int) -> Path:
    return Path(cache_root) / f'{wsi_stem}__x{int(x)}_y{int(y)}_ds{float(ds):g}_t{int(tile)}.png'


def _cache_get(cache_root: Optional[str], wsi_stem: str, x: int, y: int,
               ds: float, tile: int) -> Optional[np.ndarray]:
    if not cache_root:
        return None
    path = _cache_path(cache_root, wsi_stem, x, y, ds, tile)
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if image is None else image[:, :, ::-1]      # BGR -> RGB


def _cache_put(cache_root: Optional[str], wsi_stem: str, x: int, y: int,
               ds: float, tile: int, image: np.ndarray) -> None:
    if not cache_root:
        return
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_root, wsi_stem, x, y, ds, tile)
    ok, buf = cv2.imencode('.png', np.asarray(image)[:, :, ::-1])   # RGB -> BGR
    if not ok:
        return
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'wb') as handle:
        handle.write(buf.tobytes())
    os.replace(tmp, path)


def _read_wsi_tile(wsi, x: int, y: int, ds: float, tile: int, *,
                    wsi_stem: str, cache_root: Optional[str] = None
                    ) -> np.ndarray:
    """One real tile, `tile` x `tile` RGB uint8, top-left `(x, y)` level-0.

    Cache-checked first (`_cache_get`), and written back on a miss
    (`_cache_put`) -- the only IO this project has for a tile whose position
    was not chosen by a `TileSampler` run.

    THE SAME READ EVERY PRE-TILE GOT, NOT A SECOND SPELLING
    =========================================================
    `SlideReader(wsi, resize='area').read` of a plain tile at `ds`: the level
    rule, the level px and the filter `Corpora.Corpus.read` reads the pre-tiles
    with -- which is what keeps a 'C' descendant tile and an 'F' tile at the
    same `ds` reading the pyramid the same way.
    """
    cached = _cache_get(cache_root, wsi_stem, x, y, ds, tile)
    if cached is not None:
        return cached
    image = SlideReader(wsi, resize='area').read(int(x), int(y),
                                                 ReadSpec(int(tile), int(tile)),
                                                 float(ds))
    if image is None:
        raise ValueError(f'{wsi_stem} ds {ds:g} ({x}, {y}): a {tile} px tile '
                         f'there runs off the slide')
    _cache_put(cache_root, wsi_stem, x, y, ds, tile, image)
    return image


def _read_store_tile(corpus, tile_of, tile: int) -> np.ndarray:
    """One real tile of a corpus: the centre crop of `corpus.read(tile_of)`.

    Shared by `FStack.read`, `RStack`'s own tiles and `CStack`'s own mother
    tile (`OwnTiles`/`OwnForest.__getitem__`). The read is the pre-tile,
    `factor` times the tile (warp context, spec.md 6.6); the tile is its centre
    crop, which is what every other consumer uses (`Datasets.__getitem__`).
    One definition keeps the three from silently cropping differently -- the
    same reason `_read_wsi_tile` exists for the WSI side.
    """
    return centre_crop(corpus.read(tile_of), int(tile))


@dataclass
class Chain:
    """One level-0 centre, and the corpus tile of every rung at it."""
    wsi_stem: str
    inherit_id: int
    #: level-0 CENTRE, not the top-left. The centre is what the rungs share --
    #: their top-left corners differ because their footprints differ -- so
    #: carrying the corner would make "the same centre" a derived quantity that
    #: each caller re-derives with its own rounding.
    cx: float
    cy: float
    #: ds -> `Corpora.Tile`. Sorted finest first by `rungs`.
    members: Dict[float, object]
    #: The corpus the members are read from (`FStack.read`).
    corpus: object = None

    @property
    def rungs(self) -> List[float]:
        return sorted(self.members)


def chains(corpus, wsi_stem: str, *, tile: int,
           rungs: Optional[Sequence[float]] = None) -> Dict[int, Chain]:
    """Every COMPLETE chain of one slide, keyed by `inherit_id`.

    Complete against `rungs` when given, and against whatever rungs the draw
    holds when not. The explicit form is the one to use: "complete" measured
    against what happened to be extracted cannot notice a rung that failed to
    extract at all.

    SHARED ACROSS ALL THREE AXES -- a chain's identity is its centre and its
    'F' members; 'R' derives from the same object and 'C' roots its pyramid
    at the same `(cx, cy)`. There is one `chains()`, not one per class.

    `corpus` (a `Corpora.Corpus`) is ONE draw: stage A and stage B share a
    cache on purpose, and their different sampler_ids put them at different
    addresses, so a chain can never be assembled from two.
    """
    want = sorted(float(r) for r in rungs) if rungs else None
    found: Dict[int, Dict[float, object]] = {}
    centres: Dict[int, tuple] = {}
    seen_rungs = set()
    # ONE CORPUS, BY ADDRESS. Two corpora both number their chains from 0,
    # so a union would MERGE CHAINS THAT ARE NOT THE SAME CHAIN -- a
    # "complete" one could be four rungs of one corpus and two of another at a
    # different level-0 centre. One call reads one draw.
    for ds, tiles in corpus.tiles(wsi_stem, want).items():
        seen_rungs.add(ds)
        for t in tiles:
            cid = int(t.meta.inherit_id)
            if cid < 0:
                continue
            found.setdefault(cid, {})[ds] = t
            centres.setdefault(cid, t.centre_l0)

    target = want if want is not None else sorted(seen_rungs)
    out: Dict[int, Chain] = {}
    for cid, members in found.items():
        if len(members) < len(target):
            continue
        cx, cy = centres.get(cid, (float('nan'), float('nan')))
        out[cid] = Chain(wsi_stem=wsi_stem, inherit_id=cid,
                         cx=cx, cy=cy, members=members, corpus=corpus)
    return out


def rung_scale(ds: float, stack_kind: str) -> float:
    """How many level-0 px one output pixel SPANS. For `to_level0`.

    THIS IS NOT `rung_shrink` AND CONFUSING THEM IS SILENT. On the 'F' and 'C'
    axes a tile of `tile` pixels covers `tile * ds` level-0 px, so one pixel
    spans `ds` -- 'C' is real reads exactly like 'F', just tiled recursively
    instead of independently, so the same formula applies to a descendant tile
    at rung `ds`. On the 'R' axis the footprint is `tile` level-0 px at EVERY
    rung, so one pixel spans 1.0 no matter how degraded the image is -- the
    shrink and grow happen inside a frame that never moves.

    Using `ds` here for an 'R' rung would scatter every coarse-rung point `ds`
    times too far from the centre. The points would still be inside a plausible
    range, the table would fill, and the survival numbers would be a picture of
    the bug.
    """
    if stack_kind in ('F', 'C'):
        return float(ds)
    if stack_kind == 'R':
        return 1.0
    raise ValueError(f"stack_kind must be 'F', 'R' or 'C', got {stack_kind!r}")


def rung_shrink(ds: float, stack_kind: str) -> float:
    """How many level-0 px one output pixel is worth, per axis.

    'F'/'C': `ds`, because the tile covers `tile * ds` level-0 px in `tile`
    pixels. 'R': also `ds`, but for a different reason -- the footprint is
    `tile` level-0 px at every rung and `ds` is the DEGRADATION, so a position
    is only knowable to `ds` level-0 px even though the image is `tile` px wide.

    The two agree numerically and disagree in meaning, which is exactly the case
    a function exists for: `tau` needs "how far off can a position be", and a
    caller that reached for `footprint / tile` would get 1.0 on the 'R' axis and
    a tau that no coarse rung could ever satisfy.

    Its twin is `rung_scale`, which answers the OTHER question -- how far apart
    two pixels are in level-0 -- and answers it differently on the 'R' axis.
    They are two functions because they are two quantities that happen to agree
    on two of the three axes.
    """
    if stack_kind not in ('F', 'R', 'C'):
        raise ValueError(f"stack_kind must be 'F', 'R' or 'C', got {stack_kind!r}")
    return float(ds)


@dataclass
class OwnChains:
    """`FStack.from_own`'s result. LAZY -- same idiom as `TileSampler.Sample`
    ("metadata always, pixels only if asked for"): every chain's identity
    (`inherit_id`, centre, which corpus tile backs each rung) is already
    known from `chains()`, which never reads a pixel. `__getitem__` is where
    `FStack.read` actually happens, once per access -- so building this over
    a draw with thousands of chains costs nothing until something is
    actually indexed.
    """
    chains: Dict[int, Chain]
    tile: int

    def __len__(self) -> int:
        return len(self.chains)

    def __iter__(self):
        return iter(sorted(self.chains))

    def __getitem__(self, inherit_id: int) -> Dict[float, np.ndarray]:
        return FStack.read(self.chains[inherit_id], tile=self.tile)


class FStack:
    """ONE tile per rung, independently read, footprint grows with ds.

        info = FStack.footprint(cx, cy, ds, tile=256)   # pure geometry
        stack = FStack.read(chain, tile=256)             # IO
    """

    @staticmethod
    def footprint(cx: float, cy: float, ds: float, *, tile: int) -> PatchInfo:
        """Level-0 position+size of the rung `ds` tile. PURE, no IO.

            rung d 的 tile：level-0 上一塊 (tile_size * d) x (tile_size * d)
            的方形，以 (cx, cy) 為中心，讀成 tile_size x tile_size 像素

        Also the geometric definition `CStack`'s mother tile reuses -- a 'C'
        pyramid is rooted at exactly this rectangle, not a second definition
        of "the coarsest tile".
        """
        half = 0.5 * float(tile) * float(ds)
        return PatchInfo(row=0, col=0,
                          x=int(round(cx - half)), y=int(round(cy - half)),
                          size_px=int(round(float(tile) * float(ds))),
                          kind='main', ds=float(ds), level=None)

    @staticmethod
    def read(chain: Chain, *, tile: int) -> Dict[float, np.ndarray]:
        """`{ds: [tile, tile, 3] uint8}` read from the corpus, one per rung
        (`_read_store_tile`: the centre crop of the pre-tile).
        """
        out = {}
        for ds, member in sorted(chain.members.items()):
            out[float(ds)] = _read_store_tile(chain.corpus, member, int(tile))
        return out

    @staticmethod
    def from_own(corpus, wsi_stem: str, *, tile: int,
                rungs: Optional[Sequence[float]] = None) -> OwnChains:
        """Every complete chain in `stageB-fOwn`'s draw -- LAZY, see
        `OwnChains`. `chains()` already does the whole enumeration (groups by
        `inherit_id`, drops incomplete chains); this just wraps its result so
        `__getitem__` can call `read()` on demand instead of eagerly reading
        every chain's pixels up front.
        """
        found = chains(corpus, wsi_stem, tile=tile, rungs=rungs)
        return OwnChains(found, int(tile))


class RStack:
    """ONE tile per rung, footprint fixed at `tile` px, DERIVED not extracted.

        info = RStack.footprint(cx, cy, ds, tile=256)         # pure geometry
        stack = RStack.derive(chain, rungs, tile=256)          # IO, source='F'

    THE DEFAULT SOURCE IS DERIVED, NOT EXTRACTED, AND THAT IS NOT A SHORTCUT
    ============================================================================
    An 'R' rung is, by its own definition (`TileSampler.resolution_plan`),
    `tile` level-0 pixels read at level 0, shrunk by `ds` and grown back. A
    chain's ds 1 tile IS `tile` level-0 pixels read at level 0. So the whole
    'R' stack is `degrade_resolution` applied to that one image, and it needs
    no extraction, no second store and no WSI handle.

    THAT IS THE ONLY WAY THE TWO AXES STAY COMPARABLE. 新生歸因 asks whether a
    point born late on 'F' is also born late on 'R', which requires the two
    axes to be about the SAME PHYSICAL POINT. Two independent extractions
    would choose centres by their own admissibility -- an 'F' centre has to
    fit a `tile * ds_max` footprint and an 'R' centre only `tile`, so the
    coarse rungs would drop centres the 'R' run keeps -- and the axes would
    have to be joined spatially afterwards, with a tolerance, on exactly the
    quantity the analysis is about. Deriving 'R' from the chain makes the
    centres identical by construction.

    `degrade_resolution` lives in `SlideReader` and is called from both places
    for the same reason: two spellings of the shrink-and-grow would make a
    survival number a statement about which resampling filter each half used.

    `source='C'`/`source='own'` are the two OTHER ways to get an 'R' tile.
    `source='C'` is a WSI bypass, same as `CStack` itself -- it
    reads via `CStack.read_one`/`_read_wsi_tile`, not a corpus, because the
    descendant it reads was never a `TileSampler` draw either. `own` is a
    third corpus (`stageB-rOwn`), independently sampled
    -- not a `derive` branch (see below), reached instead through
    `RStack.from_own`.

    TWO LAYERS, NOT ONE
    ===================
    `from_tile` is the PRIMITIVE: one real tile plus the rung it was read at
    -> one 'R' stack. Pure -- no `Chain`, no `wsi`, no store. `derive` is a
    convenience wrapper around it for the ONE-CHAIN-AT-A-TIME case
    (`source='F'`/`'C'`, one `base_rung`, one stack out) -- what
    `build_survival.py` already loops over per chain.

    `own` DOES NOT FIT `derive`'S SIGNATURE. `derive(chain, ...)` requires a `Chain` --
    `cx`/`cy`/`inherit_id`-grouped `members` -- and a `stageB-rOwn` tile has
    none of that: it is a standalone corpus tile, independently
    sampled, never grouped by `inherit_id`. Forcing it through `derive` would
    mean inventing a fake one-member `Chain` per tile just to satisfy a
    parameter the tile does not actually have.

    THE ACTUAL CARDINALITY IS "ONE STACK PER REAL TILE USED AS A BASE", AND
    IT IS DELIBERATELY A CROSS PRODUCT: `own` with 100 tiles at ds 1 and 100 at ds 4 is 200 stacks;
    100 F chains x 2 chosen `base_rung`s is 200 stacks; 200 C pyramids x 256
    descendants at one rung is 51,200 stacks. `RStack.from_own` is the
    enumerator for own's whole store -- LAZY, same idiom as
    `TileSampler.Sample`, see `OwnTiles`. Enumerating a chain's chosen
    `base_rung`s or a pyramid's chosen rung's tiles is a plain loop calling
    `derive`/`from_tile` once per one -- there is no separate class for it,
    unlike own's store scan.

    `base_rung` (`footprint`, `from_tile` and `derive`, default 1.0) is WHICH
    real rung supplies the tile being degraded -- default is F's own ds 1
    tile, genuinely the sharpest read there is. See `from_tile`'s docstring
    for what a larger `base_rung` means and why it is not a bug.
    """

    @staticmethod
    def footprint(cx: float, cy: float, ds: float, *, tile: int,
                 base_rung: float = 1.0) -> PatchInfo:
        """Level-0 position+size of an 'R' rung. PURE, no IO.

        Always `FStack.footprint(cx, cy, base_rung, ...)` -- `ds` is accepted
        (every rung asks for it) but IGNORED, because the whole point of the
        'R' axis is that the footprint does not grow: 'R' 沒有這個代價
        (`spec.md` 3.2) -- every rung's real coverage is the SAME `tile`
        level-0 px (at `base_rung`, default the ds-1 window), degraded by
        different amounts, not a bigger window.

        `base_rung` MUST MATCH whatever `derive(..., base_rung=...)` built the
        array with, or this reports a window size the array does not actually
        have -- there is no way to check that from here, `footprint` never
        sees the array.
        """
        return FStack.footprint(cx, cy, base_rung, tile=tile)

    @staticmethod
    def from_tile(image: np.ndarray, base_rung: float,
                 rungs: Sequence[float], *, tile: int
                 ) -> Dict[float, np.ndarray]:
        """ONE real tile -> one 'R' stack. PURE -- no `Chain`, no `wsi`, no
        store. The primitive every source (`own`, `F`, `C`) reduces to once
        it has found its one real tile; they differ only in HOW they find
        one (see the class docstring), never in what happens after.

        `base_rung` is WHICH real `ds` `image` was actually read at.
        `base_rung=1.0` IS THE SHARPEST READ THERE IS, NOT A SPECIAL CASE --
        and `degrade_resolution` ONLY SKIPS DEGRADING FOR `ds<=1.0`, NEVER
        for `ds<=base_rung`. So every requested rung above 1.0 gets a FULL
        shrink-and-grow pass at that ABSOLUTE `ds`, compounding on top of
        whatever blur `image` already carries -- intentional ("當作 ds=1 去二次
        降解", a second full degradation stacked on the first, not a relative
        one). A LARGER
        `base_rung` -- an F tile at ds 2/4/..., or a `source='C'` tile above
        ds 1, mother included -- means `R[ds=X]` for any `X` in
        `(1, base_rung]` is MORE degraded than a `base_rung=1` `R[ds=X]`
        would be at the SAME `X` (that one degrades once; this one degrades
        an already-blurred tile again) -- not merely "as blurry as
        ds=base_rung", blurrier than that. Only `ds<=1.0` is ever spared.
        This is NOT an error to guard against -- it is choosing to start 'R'
        from an already-coarser real read on purpose, and
        `footprint(..., base_rung=...)` reports the true window size that
        goes with it. Every `SurvivalMeta` this feeds has to carry
        `base_rung`: two 'R' stacks are only comparable rung-for-rung when
        they share one.
        """
        return {float(ds): degrade_resolution(image, float(ds), int(tile))
                for ds in sorted(float(r) for r in rungs)}

    @staticmethod
    def derive(chain: Chain, rungs: Sequence[float], *, tile: int,
               source: str = 'F', base_rung: float = 1.0, wsi=None,
               cache_root: Optional[str] = None
               ) -> Dict[float, np.ndarray]:
        """ONE chain, ONE `base_rung` -> one 'R' stack. Finds the real tile,
        then hands it to `from_tile`. The convenience for the ONE-AT-A-TIME
        case (`build_survival.py`'s per-chain loop); enumerating MANY chains,
        MANY rungs or MANY descendants and calling this (or `from_tile`
        directly) once per tile is a separate, not-yet-written layer
        (`prepare_chain_stack.py`, plan.md 2.1) -- see the class docstring's
        cardinality note.

        `source='F'`: no store read beyond the chain's own `base_rung` tile
        and no WSI handle -- see the class docstring. Raises when the chain
        has no `base_rung` member rather than falling back to the nearest
        available: a caller that asked for a specific rung and silently got
        a different one would not notice its `R` stack was built from the
        wrong floor.

        `source='C'`: blur a real `CStack` tile at `base_rung` instead of
        'F''s own read -- to test whether the DERIVATION itself
        (shrink-and-grow) is the artefact rather than the tissue.
        `base_rung` equal to the pyramid's own mother `ds` means the mother
        tile itself; any finer rung means the descendant tile NEAREST the
        chain centre at that rung (`CStack.nearest` -- `main` only, not
        `overlap`: overlap tiles exist for the SAME-RUNG 相依型 comparison,
        spec.md 3.2, not as a second candidate set for "nearest the centre",
        which would make this base tile's identity depend on a coin flip).
        Needs `wsi` (the tile is read fresh, cached under `cache_root` --
        see `CStack.read_one`). Builds a full `CStack.pyramid` rooted at this
        chain's own coarsest rung to get there; the geometry is cheap (no
        IO), only the one tile that gets read costs anything.

        `source='own'` IS NOT A VALID VALUE HERE, ON PURPOSE -- see the class
        docstring's "own does not fit derive's signature". An own tile is a
        standalone corpus tile with no `Chain` to pass in; calling
        `from_tile` directly, once per tile `stageB-rOwn` yields, is the
        correct shape once that enumeration is written -- there is nothing
        for `derive` to do with a `chain` it would never receive.
        """
        if source == 'F':
            if not any(abs(ds - base_rung) < 1e-6 for ds in chain.members):
                raise ValueError(
                    f'chain {chain.inherit_id} of {chain.wsi_stem} has no ds '
                    f'{base_rung:g} tile (rungs {chain.rungs}), so an R stack '
                    f'cannot be derived from it at base_rung={base_rung:g}. '
                    f'Falling back to a different rung would mislabel every '
                    f'output rung and nothing downstream would notice')
            base = FStack.read(chain, tile=tile)[base_rung]
            return RStack.from_tile(base, base_rung, rungs, tile=tile)
        if source == 'C':
            if wsi is None:
                raise ValueError("source='C' needs wsi= -- the base tile is "
                                 "a real read")
            c_rungs = sorted({1.0, float(base_rung), *chain.members, *rungs})
            finest = min(c_rungs)
            groups_by_ds = CStack.pyramid(chain.cx, chain.cy, c_rungs, tile=tile)
            mother = CStack.mother(chain.cx, chain.cy, max(c_rungs), tile=tile)
            if abs(float(base_rung) - mother.ds) < 1e-6:
                base_tile = mother
            else:
                if base_rung not in groups_by_ds:
                    raise ValueError(
                        f'base_rung={base_rung:g} is not the mother\'s own ds '
                        f'({mother.ds:g}) or one of the descendant rungs '
                        f'{sorted(groups_by_ds)} built from c_rungs {c_rungs}')
                candidates = [m for g in groups_by_ds[base_rung] for m in g.main]
                base_tile = CStack.nearest(candidates, chain.cx, chain.cy)
            base = CStack.read_one(base_tile, wsi, wsi_stem=chain.wsi_stem,
                                   root=mother, finest=finest, tile=tile,
                                   cache_root=cache_root)
            return RStack.from_tile(base, base_rung, rungs, tile=tile)
        if source == 'own':
            raise ValueError(
                "source='own' does not fit derive()'s signature -- an own "
                "tile is a standalone corpus tile with no Chain to "
                "pass in here. Use RStack.from_own(...)[i] instead, which "
                "reads the tile and calls from_tile for you -- see the "
                "class docstring")
        raise ValueError(f"source must be 'F', 'C' or 'own', got {source!r}")

    @staticmethod
    def from_own(corpus, wsi_stem: str, rungs: Sequence[float], *,
                tile: int, cache_root: Optional[str] = None) -> 'OwnTiles':
        """Every tile of `stageB-rOwn`'s draw -- LAZY, see `OwnTiles`.

        Takes EVERY rung the draw holds, not one -- own's batch can span
        more than one `base_rung` in a single run (e.g. 100 tiles at ds 1
        AND 100 at ds 4), each tile an independent draw with no chain to
        keep them together, unlike `chains()` which groups by `inherit_id`.

        `cache_root` DEFAULTS OFF (unlike every other `cache_root` in
        this module). `degrade_resolution` is a resize on an
        array already in memory -- cheap, nothing like the WSI reads
        `_read_wsi_tile` exists to avoid repeating -- so caching every 'R'
        rung of a large `from_own` run would be disk IO spent to save
        something that was never expensive. Pass a `cache_root` only for a
        SMALL run (a demo, a handful of tiles) where re-generating the same
        few figures repeatedly is worth not recomputing at all; leave it
        `None` for anything that scans a real corpus.
        """
        items = [t for ts in corpus.tiles(wsi_stem).values() for t in ts]
        return OwnTiles(corpus, items, list(rungs), int(tile), wsi_stem,
                        cache_root)


@dataclass
class OwnTiles:
    """`RStack.from_own`'s result. LAZY, same idiom as `OwnChains` -- every
    tile's identity (a `Corpora.Tile`) is already known from the draw, no
    pixel read; `__getitem__` is where `_read_store_tile` +
    `RStack.from_tile` actually happen. Indexed by plain position
    (0..len-1) over every rung.
    """
    corpus: object
    items: List[object]                      # Corpora.Tile
    rungs: Sequence[float]
    tile: int
    wsi_stem: str
    cache_root: Optional[str] = None

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(range(len(self.items)))

    def __getitem__(self, i: int) -> Dict[float, np.ndarray]:
        """See `RStack.from_own` for why `cache_root` defaults off -- degrade
        is cheap, this is a convenience for small/repeated runs, not a
        performance fix.
        """
        t = self.items[i]
        x, y = int(t.meta.x), int(t.meta.y)
        out: Dict[float, np.ndarray] = {}
        missing = []
        for ds in sorted(float(r) for r in self.rungs):
            cached = (_cache_get(_cache_dir(self.cache_root, self.wsi_stem),
                                 self.wsi_stem, x, y, ds, self.tile)
                      if self.cache_root else None)
            if cached is not None:
                out[ds] = cached
            else:
                missing.append(ds)
        if missing:
            image = _read_store_tile(self.corpus, t, self.tile)
            computed = RStack.from_tile(image, float(t.meta.ds), missing,
                                        tile=self.tile)
            out.update(computed)
            if self.cache_root:
                for ds, img in computed.items():
                    _cache_put(_cache_dir(self.cache_root, self.wsi_stem),
                               self.wsi_stem, x, y, ds,
                              self.tile, img)
        return {ds: out[ds] for ds in sorted(float(r) for r in self.rungs)}


@dataclass
class TileGroup:
    """One parent tile's real children at the next-finer rung. All level-0.

    `overlap` is the ONE tile that shares exactly 1/4 of its area with EACH
    tile in `main` (`PatchingLib.py:250-263`) -- that shared quarter is where
    相依型 keypoint comparison happens (spec.md 3.2). `main` is also the set
    of parents `CStack.pyramid` recurses on for the next rung down.
    """
    ds: float
    parent: PatchInfo
    main: List[PatchInfo]
    overlap: PatchInfo


class CStack:
    """MANY real tiles per rung -- a recursive descendant tree. spec.md 3.2
    "第三個軸：C（子嗣／組合 stack）".

        groups = CStack.pyramid(cx, cy, rungs, tile=256)   # pure geometry
        groups[8][0].main                                   # 4 real tiles
        groups[8][0].overlap                                # shares 1/4 with each
        mother = CStack.mother(cx, cy, max(rungs), tile=256)
        img = CStack.read_one(groups[8][0].main[0], wsi, wsi_stem=...,
                              root=mother, tile=256)

    PURE GEOMETRY IN `pyramid`/`nearest`, NO PIXEL READ, NO DETECTOR, NO WSI
    HANDLE -- same contract as `utilities/PatchingLib.PatchGrid`, which it is
    built on rather than a new tiling scheme (spec.md 3.2 "子嗣格子的切法：
    沿用 PatchGrid，不是新排列"). `read_one`/`read` are the IO half, built on
    `_read_wsi_tile`'s local disk cache (module docstring) rather than a
    corpus -- a descendant's position was computed, not sampled, so a draw's
    identity does not describe it.

    ONLY `main` RECURSES
    ======================
    A rung's tile is tiled by the next-finer rung's tiles, recursively down to
    the finest rung in `rungs` (spec.md 3.2's "金字塔"). Doing that recursion
    on the overlap tiles too would make every overlap tile spawn its own
    subtree, and the count across a six-rung chain would be 5**5 instead of
    4**5 -- 3125 instead of 1024. That is not what an overlap tile is for: it
    exists to be compared against ITS OWN four main tiles AT THE SAME RUNG
    (相依型 keypoint, spec.md 3.2), which is a same-level comparison, not a
    request for finer coverage under it. So only `.main` becomes the next
    rung's parents; `.overlap` is a leaf, produced once per group and never
    recursed into.

    WHY EACH GROUP IS EXACTLY 4 MAIN + 1 OVERLAP, AND WHY THAT IS NOT GENERAL
    ============================================================================
    `DsLadder.DEFAULT_RUNGS` is a fixed 2x ladder, so every parent -> child
    transition this project actually runs tiles a `2*tile x 2*tile`
    (child-native) footprint with `tile`-sized children: a 2x2 main grid and
    exactly one overlap tile, sitting at the one interior corner shared by all
    4 mains (`PatchingLib.py:250-263`). That is a property of a STEP-2
    transition specifically -- a step-4 transition (skipping rungs) would give
    a 4x4 main grid and 9 overlap tiles, each shared by a DIFFERENT set of 4
    neighbouring mains, and "the group's mains" would stop meaning "this
    overlap's four partners". Nothing here has that geometry, so
    `_children_of` raises on a non-2x step rather than silently building
    group boundaries that would be wrong for it.

    THE GRID IS BUILT DIRECTLY IN LEVEL-0 UNITS, NOT CHILD-NATIVE
    =================================================================
    `PatchGrid.from_size` never rescales -- whatever unit `width`/`tile_size`/
    `x_offset` are given in is the unit `PatchInfo.x/y` comes back in, so
    `_children_of` passes `parent.x`/`parent.size_px` straight through
    (level-0) rather than dividing by `d_child` first: `parent.x` has no
    reason to be a multiple of `d_child`, so `round(parent.x/d_child)*d_child` silently drops up to `d_child/2` level-0
    px. Every `PatchInfo` this class hands out is level-0 -- the unit
    `Chain.cx/cy` and `SurvivalTable` use -- by construction, not by a
    conversion step that can drop a remainder.
    """

    @staticmethod
    def mother(cx: float, cy: float, ds: float, *, tile: int) -> PatchInfo:
        """The root of the pyramid: the 'F' tile at the coarsest rung.

        Literally `FStack.footprint` -- not a second definition. A 'C'
        pyramid's mother tile IS an 'F' tile; the only thing 'C' adds is what
        tiles it below.
        """
        return FStack.footprint(cx, cy, ds, tile=tile)

    @staticmethod
    def _children_of(parent: PatchInfo, d_child: float, *, tile: int
                      ) -> TileGroup:
        """One parent tile's 4-main-plus-1-overlap children. Level-0 `PatchInfo`.

        Requires `parent.ds / d_child == 2` -- see the class docstring for why
        a step other than 2 would make "this group's mains" ambiguous.

        BUILT DIRECTLY IN LEVEL-0 UNITS. This function never reads a pixel --
        it only computes positions -- and `parent.x` is an arbitrary level-0
        integer with no reason to be a multiple of `d_child`, so building in
        CHILD-NATIVE units (`round(parent.x / d_child) * d_child`) would drop
        the remainder: 1 level-0 px in all four corners at ds 4.0 for a chain
        rooted at (12345.0, 6789.0) (`test_chain_stack.py`'s reconstruction
        test). Passing `parent.x`/`parent.size_px`
        straight through as `x_offset`/`width` avoids the division entirely --
        `tile * d_child` is always an exact integer (a power-of-2 `ds` times an
        integer `tile`), so nothing here needs `round()` at all, and children
        reconstruct their parent's rectangle EXACTLY by construction, not by
        coincidence.
        """
        d_parent = float(parent.ds)
        step = d_parent / float(d_child)
        if abs(step - 2.0) > 1e-6:
            raise ValueError(
                f'_children_of assumes a 2x step (DsLadder.DEFAULT_RUNGS), got '
                f'd_parent={d_parent} -> d_child={d_child} (step={step:g}). A '
                f'step of 4 would need per-corner grouping this function does '
                f'not do -- see the class docstring')

        child_tile_l0 = int(tile) * int(d_child)
        grid = PatchGrid.from_size(
            width=parent.size_px, height=parent.size_px,
            tile_size=child_tile_l0, overlap=True,
            x_offset=parent.x, y_offset=parent.y, ds=float(d_child))

        def _tag(info: PatchInfo) -> PatchInfo:
            return PatchInfo(row=info.row, col=info.col, x=info.x, y=info.y,
                              size_px=info.size_px, kind=info.kind,
                              ds=float(d_child), level=info.level)

        main = [_tag(i) for i in grid.main_patch_infos]
        overlap = [_tag(i) for i in grid.overlap_patch_infos]
        if len(main) != 4 or len(overlap) != 1:
            raise ValueError(
                f'expected 4 main + 1 overlap from a 2x step, got {len(main)} '
                f'main + {len(overlap)} overlap -- parent tile does not tile '
                f'evenly at tile={tile}')
        return TileGroup(ds=float(d_child), parent=parent, main=main,
                          overlap=overlap[0])

    @staticmethod
    def pyramid(cx: float, cy: float, rungs: Sequence[float], *, tile: int
                ) -> Dict[float, List[TileGroup]]:
        """The full C pyramid, `{ds: [TileGroup, ...]}`, one entry per rung
        except the coarsest (the mother tile has no parent to be a child of).

        `rungs` must be sorted (either order) and every adjacent pair a 2x
        step -- `DsLadder.DEFAULT_RUNGS` (1, 2, 4, 8, 16, 32) is exactly that.
        At rung `d` there are `4 ** k` groups, where `k` is how many
        finer-than-the-mother transitions have happened -- the count is
        exponential in depth BY DESIGN (class docstring): a six-rung chain's
        finest rung has 4**5 = 1024 real, independently-detected tiles under
        one mother.
        """
        order = sorted((float(r) for r in rungs), reverse=True)
        if len(order) < 2:
            return {}

        out: Dict[float, List[TileGroup]] = {}
        parents = [CStack.mother(cx, cy, order[0], tile=tile)]
        for d_parent, d_child in zip(order, order[1:]):
            groups = [CStack._children_of(p, d_child, tile=tile)
                      for p in parents]
            out[d_child] = groups
            parents = [m for g in groups for m in g.main]
        return out

    @staticmethod
    def nearest(tiles: Sequence[PatchInfo], cx: float, cy: float) -> PatchInfo:
        """Whichever tile's CENTRE is closest to `(cx, cy)`. PURE, no IO.

        Not "which tile CONTAINS `(cx, cy)`" -- at a shared corner more than
        one main tile's footprint contains the same point, and nearest-centre
        is the unambiguous tie-break `RStack.derive(source='C')` needs to pick
        exactly one.
        """
        if not tiles:
            raise ValueError('nearest() needs at least one tile')
        def _d2(t):
            tx, ty = t.x + 0.5 * t.size_px, t.y + 0.5 * t.size_px
            return (tx - cx) ** 2 + (ty - cy) ** 2
        return min(tiles, key=_d2)

    @staticmethod
    def read_one(info: PatchInfo, wsi, *, wsi_stem: str, root: PatchInfo,
                 finest: float, tile: int,
                 cache_root: Optional[str] = None
                 ) -> np.ndarray:
        """ONE descendant tile's real pixels. Cached -- see `_read_wsi_tile`.

        `root` is the pyramid's mother (`CStack.mother(...)`) -- it decides
        WHICH pyramid's cache directory `info` is filed under
        (`_pyramid_dir`), not what gets read; `info` alone still says where.
        `finest` is the pyramid's own finest rung -- NOT `info.ds` (`info` is
        one tile, which may sit above the pyramid's actual bottom) -- it is
        what makes the directory name unique to THIS `--rungs` request, see
        `_pyramid_dir`. Required rather than defaulted: a caller that does not
        know its own pyramid's finest rung should not be filing anything under
        it.

        The unit everything else (`read`, `RStack.derive(source='C')`,
        eventual coverage-confirmation / 相依型 comparison) is built from: a
        pyramid has up to 1024 tiles under one mother, and almost nothing
        needs all of them at once, so the primitive is "read one, cheaply
        repeatable" rather than "read the whole tree".
        """
        pyramid_dir = _pyramid_dir(cache_root, 'C', wsi_stem, root, tile,
                                   finest) if cache_root else None
        return _read_wsi_tile(wsi, info.x, info.y, info.ds, tile,
                              wsi_stem=wsi_stem, cache_root=pyramid_dir)

    @staticmethod
    def read_tree(groups_by_ds: Dict[float, List[TileGroup]], wsi, *,
                 wsi_stem: str, root: PatchInfo, tile: int,
                 cache_root: Optional[str] = None
                 ) -> Dict[float, List[np.ndarray]]:
        """Real pixels for EVERY descendant tile in a pyramid, rung by rung.

        NAMED `read_tree`, NOT `read` -- `FStack.read`/
        `RStack.derive` both return ONE tile per rung (`Dict[ds, image]`);
        this returns MANY per rung (`Dict[ds, List[image]]`), and sharing the
        name `read` across an incompatible shape is exactly the kind of thing
        that reads fine until a caller assumes the wrong one.

        Built on `read_one`, so a repeated call after the first is cheap (disk
        cache, no second WSI hit) -- but the FIRST call at six rungs is still
        1024 real reads under one mother, a per-chain cost nobody has measured
        (plan.md 2.3). Call `read_one` directly instead of this when only a
        few specific tiles are actually needed (`RStack.derive(source='C')`
        does).
        """
        finest = min(groups_by_ds)
        out: Dict[float, List[np.ndarray]] = {}
        for ds, groups in groups_by_ds.items():
            tiles = [m for g in groups for m in g.main] + \
                    [g.overlap for g in groups]
            out[ds] = [CStack.read_one(t, wsi, wsi_stem=wsi_stem, root=root,
                                       finest=finest, tile=tile,
                                       cache_root=cache_root)
                      for t in tiles]
        return out

    @staticmethod
    def from_own(corpus, wsi_stem: str, rungs: Sequence[float], wsi, *,
                tile: int, cache_root: Optional[str] = None
                ) -> 'OwnForest':
        """Every tile of `stageB-cOwn`'s draw -> one tree each, as a
        mother -- LAZY, see `OwnForest`, but UNLIKE `OwnChains`/`OwnTiles`:
        the GEOMETRY for every tree in the forest is built here, up front
        (`pyramid()` is pure -- cheap even for hundreds of trees). Only the
        PIXELS wait for `__getitem__`: the mother's (corpus-backed,
        `_read_store_tile`, this tile IS the mother) and every
        descendant's (`wsi`-backed, `CStack.read_tree`, never store-backed --
        see the class docstring).

        Takes every rung the draw holds, same reason as `RStack.from_own`.
        Each tile roots its OWN tree at its OWN `ds` -- `rungs` must not ask
        for anything coarser than that tile's `ds`, or `pyramid()` would
        silently build from a DIFFERENT, coarser mother than the one this
        tile actually is.
        """
        items = []
        for ds, tiles in corpus.tiles(wsi_stem).items():
            for t in tiles:
                if any(float(r) > ds + 1e-6 for r in rungs):
                    raise ValueError(
                        f'rungs {sorted(rungs)} asks for something coarser '
                        f'than this tile\'s own ds {ds:g} -- that tile '
                        f'cannot be the mother of a pyramid rooted above '
                        f'itself')
                cx, cy = t.centre_l0
                c_rungs = sorted({ds, *(float(r) for r in rungs)})
                mother = CStack.mother(cx, cy, max(c_rungs), tile=tile)
                groups_by_ds = CStack.pyramid(cx, cy, c_rungs, tile=tile)
                items.append((t, mother, groups_by_ds))
        return OwnForest(corpus, items, wsi, wsi_stem, int(tile), cache_root)

    @staticmethod
    def from_mother(mother_image: np.ndarray, cx: float, cy: float,
                    rungs: Sequence[float], wsi, *, wsi_stem: str, tile: int,
                    cache_root: Optional[str] = None):
        """Build and read a pyramid reusing an ALREADY-READ mother tile
        (e.g. `FStack.read(chain)[ds]`) instead of reading it again -- source
        #1 of the mother's two documented sources (plan.md 2.1); `from_own`
        is source #2.

        `mother_image` is taken on faith -- this does not re-derive it from
        `wsi`, only from `cx, cy, max(rungs)` geometrically (`CStack.mother`)
        to know WHERE it is. If the image does not actually match that
        rectangle (wrong rung, wrong centre), nothing here catches that; the
        caller is the one that knows it read `FStack.footprint(cx, cy,
        max(rungs), tile=tile)`'s own rectangle.

        Descendants still need `wsi` -- they never have a store-backed
        source, `own` included (see the class docstring). Returns the same
        shape `OwnForest.__getitem__` does: `(mother, mother_image,
        groups_by_ds, images_by_ds)`.
        """
        mother = CStack.mother(cx, cy, max(rungs), tile=tile)
        groups_by_ds = CStack.pyramid(cx, cy, rungs, tile=tile)
        images_by_ds = CStack.read_tree(groups_by_ds, wsi, wsi_stem=wsi_stem,
                                        root=mother, tile=tile,
                                        cache_root=cache_root)
        return mother, mother_image, groups_by_ds, images_by_ds


@dataclass
class OwnForest:
    """`CStack.from_own`'s result. LAZY like `OwnChains`/`OwnTiles`, but the
    GEOMETRY for the whole forest is already built (`from_own` did it) --
    `__getitem__` is where every PIXEL read happens: the mother's own
    corpus-backed pixels (this tile always IS the mother) and every
    descendant's (`wsi`-backed, never store-backed -- descendant positions
    are computed, not sampled, see the class docstring).
    """
    corpus: object
    items: List[tuple]       # (Corpora.Tile, mother, groups_by_ds)
    wsi: object
    wsi_stem: str
    tile: int
    cache_root: Optional[str]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(range(len(self.items)))

    def __getitem__(self, i: int):
        """`(mother, mother_image, groups_by_ds, images_by_ds)` -- geometry
        AND pixels for both the mother and every descendant, all at once.
        """
        t, mother, groups_by_ds = self.items[i]
        mother_image = _read_store_tile(self.corpus, t, self.tile)
        images_by_ds = CStack.read_tree(groups_by_ds, self.wsi,
                                        wsi_stem=self.wsi_stem, root=mother,
                                        tile=self.tile,
                                        cache_root=self.cache_root)
        return mother, mother_image, groups_by_ds, images_by_ds
