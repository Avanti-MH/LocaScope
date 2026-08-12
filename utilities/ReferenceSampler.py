"""Choose which tiles become a slide's reference set, and say why for each one.

The reference bank behind stage 1 is currently 40 tiles per level drawn at random
from inside the tissue mask. That is enough to be saturated for a KNN
(result/MppEstimate: 40 -> 640 buys 2.7 points) but it controls nothing about
what those tiles contain, and the pooling stores showed what slips through --
up to 13% of the tiles in one slide's deep banks were bit-identical twins, all of
them holes or unscanned canvas that SafeSlide fills with a flat colour and that
therefore encode to the same vector at every level.

This module decides the tiles. It does not read pixels (with one deliberate
exception, below) and it does not encode. It answers: which coordinates, from
which bucket, by which route, and -- when it cannot deliver -- exactly what it
fell short of.

What it controls
----------------
* **Background mix.** Tiles are bucketed by the fraction of their footprint the
  mask calls background, and each bucket carries a quota: mostly tissue-dense,
  a bounded minority of emptier ones, because queries contain those too.
* **Near-duplicates.** A bucket that the grid cannot fill is topped up by
  displacing an existing coordinate. Every offered offset moves a full tile in
  one axis, so a displaced tile shares NO pixels with its parent, and none of
  them is a multiple of half a tile, so none lands back on a position the grid
  already offered. It is still the parent's neighbour, so `origin` and the
  parent's COORDINATE are recorded and the topped-up share is capped.
* **Cross-level correspondence.** A fixed set of level-0 locations is carried to
  every other level, centre-aligned, so the same physical tissue is present at
  every magnification and only the scale differs. Each carries an `inherit_id`,
  which turns cross-level lookup into an index rather than a coordinate search.
* **Holes.** Whether a tile was actually photographed is a property of
  (location, level) -- a corrupt stored tile at level 0 says nothing about level
  3 -- so it cannot be answered from a mask and cannot be shared between levels.
  It is checked per level at read time, and the inheritance set is checked at
  EVERY level before it is fixed, because a correspondence with holes in it is
  not a correspondence.

The one read before the plan
----------------------------
Everything above except the hole check is geometry over the mask, so the whole
plan -- including whether a level will fall short -- is knowable in milliseconds,
before a single pixel is read. `preflight()` reports it. The inheritance set is
the exception: it must be read at every level to be validated, and it must be
fixed before the per-level quotas are filled, since it consumes them.

What it does not do
-------------------
Encoding, storing, and anything about features. The caller reads the coordinates,
encodes them, and hands the arrays to FeatureStore.save -- `to_store_args` builds
that call. Feature-space criteria (a diverse reference rather than merely a
background-balanced one) would need the features to exist first and are
deliberately out of scope: there is no evidence yet that coverage rather than
contamination is the binding constraint, and `white_frac`/`bucket` are recorded
so that question becomes measurable rather than arguable.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PatchingLib import PatchGrid

#: Bucket names, coarsest tissue first. The order is the allocation order.
BUCKETS = ('lt15', 'mid', 'gt70', 'gt80', 'full')

#: Displacements offered when a bucket runs out of grid positions, in units of
#: the tile at THAT level. Two properties, and a table entry must have both:
#:
#:   disjoint from the parent    max(|dx|, |dy|) >= tile
#:       The overlap area is max(0, T-|dx|) * max(0, T-|dy|), so one component
#:       of a full tile is enough to make it zero. Written in level-n units on
#:       purpose; as level-0 constants they would be 87% overlap at ds=8.
#:
#:   NOT on a grid position      dx % (tile//2) != 0  or  dy % (tile//2) != 0
#:       The main grid steps by the tile and the overlap grid sits half a tile
#:       in, so the union of the two IS every multiple of tile//2. A
#:       displacement onto one of those has produced a coordinate the grid
#:       already offered -- and the bucket was short precisely because the grid
#:       had run out there, so it cannot help. Three of the five offsets
#:       originally here were multiples of 128 and did exactly that.
JITTER_OFFSETS = (
    (64, 256), (256, 64), (192, 256), (256, 192), (320, 320),
)

#: Above this many mask pixels the summed-area table is built on a decimated
#: view. Named for what it does rather than for its unit, and deliberately in
#: the same shape as TissuesRegionsMask._CC_DECIMATE_ABOVE_PX: both are the
#: point at which a whole-mask array stops fitting comfortably, and both respond
#: by losing resolution rather than by failing.
#:
#: 1 << 28 is 268 Mpx, so the int32 table stays near 1 GB. Without it this
#: module would put back the last thing from_wsi got rid of -- an array that
#: scales with the slide -- at four bytes per mask pixel: BRACS_1228's level 1
#: is 411 Mpx, and an MRXS level 1 is four times that again.
#:
#: Nothing measurable is lost. A background fraction is an average over a
#: footprint of 256*ds level-0 pixels, which is tens to thousands of mask
#: pixels across, so a stride of 2 or 4 moves it by far less than the
#: segmentation's own error. int32 rather than float64 for the same reason it is
#: exact: the mask is 0/1, so the running sum is an integer and float32 would
#: silently lose precision past ~1.7e7 -- as a drift in every fraction at once,
#: which is the kind of wrong that never raises.
_INTEGRAL_DECIMATE_ABOVE_PX = 1 << 28


@dataclass(frozen=True)
class SamplerConfig:
    """Everything that decides which tiles are chosen. Hashed into `sampler_id`.

    The hash is the point: FeatureStore's cfg_hash covers the encoder and the
    mask but nothing about sampling, so two runs with different quotas or a
    different seed produce the SAME filename today. Feeding sampler_id into the
    store's identity is what keeps them apart.
    """
    tile: int = 256
    n_target: int = 1000
    #: A level may hold at most this multiple of its own grid, so topping up can
    #: never dominate. 1.25 makes the displaced share exactly 25/125 = 20%, the
    #: same number as jitter_cap -- they agree by construction, but jitter_cap is
    #: per bucket and this is per level, so both are checked and the stricter binds.
    over: float = 1.25
    jitter_cap: float = 0.20
    #: white-fraction cuts: [0, e0) [e0, e1] (e1, e2] (e2, 1) [1, 1]
    edges: Tuple[float, float, float] = (0.15, 0.70, 0.80)
    #: floor for lt15, share for mid
    floor_lt15: float = 0.85
    share_mid: float = 0.10
    #: nested caps on the emptier buckets: gt70+gt80+full, gt80+full, full
    cap_gt70: float = 0.10
    cap_gt80: float = 0.075
    cap_full: float = 0.05
    jitter_offsets: Tuple[Tuple[int, int], ...] = JITTER_OFFSETS
    #: carried from level 0 to every other level
    inherit_frac: float = 0.50
    inherit_over: float = 2.0
    #: a tile whose photographed fraction is below this is discarded at read time
    min_valid: float = 0.95
    #: consecutive draws that hit an already-taken coordinate before giving up
    max_miss: int = 20
    #: Below this a level is reported as unusable rather than merely short.
    #: Falling short of n_target is not by itself a failure: a deep level offers
    #: a few hundred grid positions at most, and 40 per level is already
    #: saturated for the KNN that reads this (result/MppEstimate: 40 -> 640 buys
    #: 2.7 points). What matters is whether enough survived to be worth having.
    min_useful: int = 200
    seed: int = 42

    #: Fields that do not change WHICH tiles are chosen, and so must stay out of
    #: the identity. min_useful only decides whether the run complains; two runs
    #: differing only here selected the same tiles and have to share a filename,
    #: or the store would fork on a reporting threshold.
    _NOT_IDENTITY = frozenset({'min_useful'})

    def sampler_id(self) -> str:
        parts = []
        for f in sorted(dataclasses.fields(self), key=lambda f: f.name):
            if f.name in self._NOT_IDENTITY:
                continue
            parts.append(f'{f.name}={getattr(self, f.name)!r}')
        return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:8]


@dataclass
class LevelGeoms:
    """Every candidate position at one level, with no pixel read."""
    level: int
    ds: float
    footprint_l0: int                 # tile * ds
    xy: np.ndarray                    # [N, 2] int64, level-0 top-left
    region: np.ndarray                # [N] int32
    grid_rc: np.ndarray               # [N, 2] int32
    kind: np.ndarray                  # [N] int8, 0 main / 1 overlap
    white: np.ndarray                 # [N] float32
    bucket: np.ndarray                # [N] int8, index into BUCKETS

    def n_in(self, b: int) -> int:
        return int((self.bucket == b).sum())


@dataclass
class LevelSample:
    """The chosen tiles for one level. `valid_frac` is filled at read time."""
    level: int
    ds: float
    x: np.ndarray                     # [N] int64
    y: np.ndarray                     # [N] int64
    region: np.ndarray                # [N] int32
    grid_rc: np.ndarray               # [N, 2] int32, (-1,-1) off-grid
    kind: np.ndarray                  # [N] int8, -1 off-grid
    white_frac: np.ndarray            # [N] float32
    bucket: np.ndarray                # [N] int8
    origin: np.ndarray                # [N] int8, 0 grid / 1 jitter / 2 inherit
    #: Where a displaced tile came from, as a level-0 COORDINATE rather than an
    #: index. An index into the grid cannot be resolved from the store, which
    #: does not hold the grid; an index into the sample is invalidated the
    #: moment a caller drops a row, and both drivers do exactly that when a read
    #: turns out to be a hole. A coordinate survives both, and it is already the
    #: key the answer lookup uses. (-1, -1) unless origin is a displacement.
    parent_x: np.ndarray              # [N] int64
    parent_y: np.ndarray              # [N] int64
    inherit_id: np.ndarray            # [N] int32, -1 unless inherit
    valid_frac: np.ndarray            # [N] float32, NaN until read

    def __len__(self) -> int:
        return int(len(self.x))

    def subset(self, keep) -> 'LevelSample':
        k = np.asarray(keep)
        return LevelSample(
            self.level, self.ds,
            *[getattr(self, f)[k] for f in
              ('x', 'y', 'region', 'grid_rc', 'kind', 'white_frac', 'bucket',
               'origin', 'parent_x', 'parent_y', 'inherit_id', 'valid_frac')])


@dataclass
class BucketPlan:
    bucket: str
    want: int
    available: int
    from_grid: int
    from_jitter: int
    short: int
    reason: str = ''

    @property
    def got(self) -> int:
        return self.from_grid + self.from_jitter


@dataclass
class LevelPlan:
    level: int
    n_grid: int
    n_target: int
    buckets: List[BucketPlan]
    inherited: int = 0
    caps_ok: bool = True
    cap_note: str = ''
    min_useful: int = 200

    @property
    def got(self) -> int:
        return self.inherited + sum(b.got for b in self.buckets)

    @property
    def short(self) -> bool:
        """Fewer than asked for. Informational -- at a deep level the whole grid
        is a few hundred positions and n_target was never reachable."""
        return self.got < self.n_target

    @property
    def unusable(self) -> bool:
        """Few enough to be worth stopping for. This is the one to gate on."""
        return self.got < self.min_useful


# ── white fraction over every candidate, in one pass ──────────────────────────

def _integral(mask: np.ndarray) -> tuple:
    """Summed-area table, padded so a rect is four lookups.

    Returns (table, step). The mask is decimated by `step` first when it is
    larger than _INTEGRAL_DECIMATE_ABOVE_PX, and every lookup afterwards has to
    be in decimated coordinates -- which is why the stride comes back with the
    table instead of being recoverable from its shape alone.

    The stride is derived from the mask, not fixed, so a small mask gets stride
    1 and nothing about the existing behaviour moves.
    """
    step = 1
    while (mask.shape[0] // step) * (mask.shape[1] // step) > _INTEGRAL_DECIMATE_ABOVE_PX:
        step *= 2
    small = mask[::step, ::step]
    s = np.zeros((small.shape[0] + 1, small.shape[1] + 1), dtype=np.int32)
    s[1:, 1:] = small.astype(np.int32).cumsum(0).cumsum(1)
    return s, step


def white_fractions(mask, xy: np.ndarray, level: int, tile: int) -> np.ndarray:
    """Background fraction of each tile's footprint, from the mask alone.

    main_mask holds 1 where the segmenter found tissue, so background is its
    complement. Area outside the mask counts as background rather than being
    dropped -- the same choice has_tissue documents, and for the same reason: a
    tile hanging half off the scanned region must not score as pure tissue on the
    half that happens to land on some.

    Vectorised through a summed-area table because "is this level even fillable"
    has to be answerable before any pixel is read, and a level can offer 200,000
    candidates.
    """
    if not hasattr(mask, '_ref_integral'):
        mask._ref_integral = _integral(mask.main_mask)
    S, step = mask._ref_integral
    H, W = S.shape[0] - 1, S.shape[1] - 1        # in DECIMATED mask pixels

    mw, mh = mask._levelLength_converter(tile, tile, level)
    # Everything below is in decimated units. The step**2 that would scale both
    # the tissue count and the requested area cancels in the ratio, so it never
    # has to appear -- but the footprint does have to be at least one cell, or a
    # tile smaller than the stride would divide by zero.
    mw = max(1, mw // step)
    mh = max(1, mh // step)

    mx0 = np.empty(len(xy), dtype=np.int64)
    my0 = np.empty(len(xy), dtype=np.int64)
    for i, (x, y) in enumerate(xy):
        cx, cy = mask.to_mask_xy(int(x), int(y))
        mx0[i], my0[i] = cx // step, cy // step

    x0 = np.clip(mx0, 0, W)
    y0 = np.clip(my0, 0, H)
    x1 = np.clip(mx0 + mw, 0, W)
    y1 = np.clip(my0 + mh, 0, H)

    tissue = (S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0])
    return (1.0 - tissue / float(mw * mh)).astype(np.float32)


def assign_buckets(white: np.ndarray, cfg: SamplerConfig) -> np.ndarray:
    e0, e1, e2 = cfg.edges
    b = np.full(len(white), BUCKETS.index('mid'), dtype=np.int8)
    b[white < e0] = BUCKETS.index('lt15')
    b[(white > e1) & (white <= e2)] = BUCKETS.index('gt70')
    b[white > e2] = BUCKETS.index('gt80')
    b[white >= 0.999] = BUCKETS.index('full')
    return b


def build_level_geoms(mask, levels: Sequence[int], downsamples: Sequence[float],
                      cfg: SamplerConfig) -> Dict[int, LevelGeoms]:
    """Grid positions and their background fractions, for every level. No reads."""
    out = {}
    for lv in levels:
        ds = float(downsamples[lv])
        xs, ys, regs, rows, cols, kinds = [], [], [], [], [], []
        for ri, region in enumerate(mask.tissue_regions):
            w_n, h_n = int(region.w / ds), int(region.h / ds)
            if w_n < cfg.tile or h_n < cfg.tile:
                continue
            grid = PatchGrid.from_size(
                w_n, h_n, cfg.tile, overlap=True,
                x_offset=int(region.x / ds), y_offset=int(region.y / ds),
                ds=ds, level=lv)
            for info in grid.iter_infos():
                xs.append(int(round(info.x * ds)))
                ys.append(int(round(info.y * ds)))
                regs.append(ri)
                rows.append(info.row)
                cols.append(info.col)
                kinds.append(0 if info.kind == 'main' else 1)
        if not xs:
            continue
        xy = np.array([xs, ys], dtype=np.int64).T
        white = white_fractions(mask, xy, lv, cfg.tile)
        out[lv] = LevelGeoms(
            level=lv, ds=ds, footprint_l0=int(cfg.tile * ds), xy=xy,
            region=np.array(regs, dtype=np.int32),
            grid_rc=np.array([rows, cols], dtype=np.int32).T,
            kind=np.array(kinds, dtype=np.int8),
            white=white, bucket=assign_buckets(white, cfg))
    return out


# ── quotas ────────────────────────────────────────────────────────────────────

def plan_level(g: LevelGeoms, cfg: SamplerConfig, inherited: int = 0) -> LevelPlan:
    """What each bucket wants, what it can have, and what it will fall short of.

    Pure arithmetic on counts -- no drawing yet, so this is the report that can
    be produced before the expensive part starts.
    """
    n_grid = len(g.xy)
    n_target = int(min(cfg.n_target, np.floor(n_grid * cfg.over)))
    free = max(0, n_target - inherited)

    want = {
        'lt15': int(round(cfg.floor_lt15 * free)),
        'mid': int(round(cfg.share_mid * free)),
    }
    rest = free - want['lt15'] - want['mid']
    avail = {b: g.n_in(i) for i, b in enumerate(BUCKETS)}

    # The emptier buckets are filled least-empty first, so the reference stays as
    # tissue-rich as the caps allow while still carrying some background.
    for b in ('gt70', 'gt80', 'full'):
        want[b] = min(rest, avail[b])
        rest -= want[b]

    plans = []
    for b in BUCKETS:
        w = want.get(b, 0)
        from_grid = min(w, avail[b])
        room = int(np.floor(cfg.jitter_cap / (1 - cfg.jitter_cap) * max(from_grid, 0)))
        from_jit = min(w - from_grid, room) if avail[b] else 0
        short = w - from_grid - from_jit
        reason = ''
        if short > 0:
            reason = (f'grid offers {avail[b]}, jitter capped at '
                      f'{cfg.jitter_cap:.0%} of the bucket')
        plans.append(BucketPlan(b, w, avail[b], from_grid, from_jit, short, reason))

    got = {p.bucket: p.got for p in plans}
    caps = [('gt70+gt80+full', got['gt70'] + got['gt80'] + got['full'],
             int(np.floor(cfg.cap_gt70 * n_target))),
            ('gt80+full', got['gt80'] + got['full'],
             int(np.floor(cfg.cap_gt80 * n_target))),
            ('full', got['full'], int(np.floor(cfg.cap_full * n_target)))]
    bad = [f'{name} = {v} > {c}' for name, v, c in caps if v > c]

    return LevelPlan(level=g.level, n_grid=n_grid, n_target=n_target,
                     buckets=plans, inherited=inherited,
                     caps_ok=not bad, cap_note='; '.join(bad),
                     min_useful=cfg.min_useful)


# ── inheritance ───────────────────────────────────────────────────────────────

@dataclass
class InheritPlan:
    """Level-0 locations carried to every level, centre-aligned.

    Centre rather than top-left: the footprint grows with ds, so sharing a
    top-left corner would mean the coarse tile extends away from the fine one
    rather than covering the same tissue.
    """
    n_candidates: int
    fits: Dict[int, int] = field(default_factory=dict)      # geometry only
    valid: Dict[int, int] = field(default_factory=dict)     # after reads
    xy0: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))
    region: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    binding_level: Optional[int] = None
    #: half the level-0 footprint of the level the coordinates were drawn from,
    #: which is what turns their top-left corners back into centres. A declared
    #: field rather than an attribute bolted on by pick_inheritance, so the empty
    #: plan returned when there are no candidates is still a usable object.
    half0: float = 0.0

    def at(self, ds: float, tile: int) -> np.ndarray:
        """Top-left at a level of downsample `ds`, keeping the centre fixed."""
        if len(self.xy0) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        centre = self.xy0.astype(np.float64) + self.half0
        return np.rint(centre - tile * ds / 2.0).astype(np.int64)


def pick_inheritance(geoms: Dict[int, LevelGeoms], mask, cfg: SamplerConfig,
                     rng: np.random.Generator) -> InheritPlan:
    """Candidates from level 0's tissue-dense bucket that fit at EVERY level.

    Geometry only. The intersection is taken before anything is read, so the read
    validation that follows costs a few hundred tiles rather than a few thousand.
    """
    base = geoms[min(geoms)]
    pool = np.flatnonzero(base.bucket == BUCKETS.index('lt15'))
    n_cand = int(min(len(pool),
                     np.ceil(cfg.inherit_over * cfg.inherit_frac * cfg.n_target)))
    if n_cand == 0:
        return InheritPlan(0)
    take = rng.choice(pool, size=n_cand, replace=False)

    xy0 = base.xy[take]
    half0 = base.footprint_l0 / 2.0
    centre = xy0.astype(np.float64) + half0
    reg = base.region[take]

    keep = np.ones(n_cand, dtype=bool)
    fits, binding, worst = {}, None, n_cand + 1
    for lv, g in sorted(geoms.items()):
        half = g.footprint_l0 / 2.0
        tl = centre - half
        ok = np.ones(n_cand, dtype=bool)
        for i, ri in enumerate(reg):
            region = mask.tissue_regions[int(ri)]
            ok[i] = (tl[i, 0] >= region.x and tl[i, 1] >= region.y and
                     tl[i, 0] + g.footprint_l0 <= region.x + region.w and
                     tl[i, 1] + g.footprint_l0 <= region.y + region.h)
        fits[lv] = int(ok.sum())
        if fits[lv] < worst:
            worst, binding = fits[lv], lv
        keep &= ok

    return InheritPlan(n_candidates=n_cand, fits=fits, binding_level=binding,
                       xy0=xy0[keep], region=reg[keep], half0=half0)


def validate_inheritance(plan: InheritPlan, geoms: Dict[int, LevelGeoms],
                         read_valid, cfg: SamplerConfig) -> InheritPlan:
    """Read every surviving candidate at every level and keep only those
    photographed everywhere.

    This is the module's one read before the plan is fixed, and it is not
    optional: a hole is a property of (location, level), so a location can be
    solid at level 0 and missing at level 3. A correspondence set with holes in
    it is not a correspondence, and the whole reason to carry locations across
    levels is that the tissue is then held constant.

    `read_valid(x, y, level, size) -> float` is injected rather than imported so
    this module stays free of slide IO.
    """
    if len(plan.xy0) == 0:
        return plan
    keep = np.ones(len(plan.xy0), dtype=bool)
    for lv, g in sorted(geoms.items()):
        tl = plan.at(g.ds, cfg.tile)
        for i in range(len(tl)):
            if not keep[i]:
                continue
            if read_valid(int(tl[i, 0]), int(tl[i, 1]), lv, cfg.tile) < cfg.min_valid:
                keep[i] = False
        plan.valid[lv] = int(keep.sum())
    plan.xy0 = plan.xy0[keep]
    plan.region = plan.region[keep]
    return plan


# ── the sampler ───────────────────────────────────────────────────────────────

class ReferenceSampler:
    """Draws a level's tiles, and stays alive to replace the ones reads reject.

    It is not a one-shot plan: whether a tile was photographed only surfaces when
    it is read, and a rejected tile has to be replaced from the SAME bucket or the
    background mix quietly drifts.
    """

    def __init__(self, geoms: Dict[int, LevelGeoms], cfg: SamplerConfig,
                 mask, inherit: Optional[InheritPlan] = None):
        self.geoms = geoms
        self.cfg = cfg
        #: needed because an inherited tile sits off the grid, so its background
        #: fraction is not one of the values build_level_geoms already computed
        #: -- and it is a DIFFERENT fraction at every level, since the footprint
        #: grows with ds. Leaving it NaN would quietly exempt exactly the tiles
        #: the buckets exist to control.
        self.mask = mask
        self.inherit = inherit
        self.rng = np.random.default_rng(cfg.seed)
        self._taken: Dict[int, set] = {lv: set() for lv in geoms}
        self._notes: Dict[int, List[str]] = {lv: [] for lv in geoms}

    # -- helpers ----------------------------------------------------------
    def _free_in(self, lv: int, b: int) -> np.ndarray:
        g = self.geoms[lv]
        idx = np.flatnonzero(g.bucket == b)
        taken = self._taken[lv]
        return np.array([i for i in idx
                         if (int(g.xy[i, 0]), int(g.xy[i, 1])) not in taken],
                        dtype=np.int64)

    def _jitter_from(self, level: int, parent_indices: np.ndarray,
                     n_wanted: int, bucket: int):
        """Displace parents until n_wanted new coordinates land in `bucket`.

        Three things are checked, and each of them was a way a displaced tile
        could look new while not being:

        the coordinate is computed in LEVEL-N and converted back exactly the way
            grid_coords does. Adding offset * ds to the level-0 coordinate and
            truncating is a different arithmetic path, and a non-integer ds makes
            the two disagree by a pixel -- a full-tile step produced 25298 where
            the grid holds 25299, so `_taken` could not tell they were the same
            place and the duplicate went into the pool.

        the tile still fits inside its parent's region. Without this a
            displacement could hang off the edge; the white fraction counts the
            outside as background, which happens to bounce it from the
            tissue-dense bucket but lets it straight into the empty ones.

        the background fraction is RECOMPUTED where it landed. Every offset is a
            whole tile in one axis, so parent and child share no pixel and a tile
            drawn for the tissue-dense bucket can easily arrive on glass.
            Recording the parent's value would satisfy the quota on paper while
            filling it with the opposite of what it asked for.

        Gives up after max_miss consecutive refusals -- collision, off-region or
        wrong bucket, all meaning the neighbourhood is used up rather than the
        draw being unlucky.

        Returns (rows, gave_up), each row (x, y, parent_index, white_fraction).
        """
        geometry = self.geoms[level]
        footprint_level0 = self.cfg.tile * geometry.ds
        displaced_rows, consecutive_misses = [], 0

        while (len(displaced_rows) < n_wanted
               and consecutive_misses < self.cfg.max_miss
               and len(parent_indices)):
            parent_index = int(self.rng.choice(parent_indices))
            offset_x, offset_y = self.cfg.jitter_offsets[
                self.rng.integers(len(self.cfg.jitter_offsets))]
            sign_x, sign_y = self.rng.choice([-1, 1], size=2)

            parent_level_n_x = int(round(geometry.xy[parent_index, 0]
                                         / geometry.ds))
            parent_level_n_y = int(round(geometry.xy[parent_index, 1]
                                         / geometry.ds))
            displaced_x = int(round((parent_level_n_x + sign_x * offset_x)
                                    * geometry.ds))
            displaced_y = int(round((parent_level_n_y + sign_y * offset_y)
                                    * geometry.ds))

            if (displaced_x, displaced_y) in self._taken[level]:
                consecutive_misses += 1
                continue

            region = self.mask.tissue_regions[
                int(geometry.region[parent_index])]
            fits_in_region = (
                displaced_x >= region.x and displaced_y >= region.y
                and displaced_x + footprint_level0 <= region.x + region.w
                and displaced_y + footprint_level0 <= region.y + region.h)
            if not fits_in_region:
                consecutive_misses += 1
                continue

            displaced_white = float(white_fractions(
                self.mask, np.array([[displaced_x, displaced_y]]),
                level, self.cfg.tile)[0])
            if int(assign_buckets(np.array([displaced_white]),
                                  self.cfg)[0]) != bucket:
                consecutive_misses += 1
                continue

            consecutive_misses = 0
            self._taken[level].add((displaced_x, displaced_y))
            displaced_rows.append((displaced_x, displaced_y, parent_index,
                                   displaced_white))
        return displaced_rows, consecutive_misses >= self.cfg.max_miss

    # -- the plan ---------------------------------------------------------
    def plan(self, lv: int, plan: LevelPlan) -> LevelSample:
        g = self.geoms[lv]
        cols = {k: [] for k in ('x', 'y', 'region', 'rc', 'kind', 'white',
                                'bucket', 'origin', 'parent_x', 'parent_y',
                                'inherit')}

        if self.inherit is not None and len(self.inherit.xy0):
            tl = self.inherit.at(g.ds, self.cfg.tile)
            w = white_fractions(self.mask, tl, lv, self.cfg.tile)
            # The bucket is recomputed, not carried over. A location chosen for
            # being tissue-dense at level 0 covers 4x or 64x the area here and
            # may well no longer be, and recording where it actually lands is
            # what makes "does inheritance skew the mix" a question the store can
            # answer.
            wb = assign_buckets(w, self.cfg)
            for i in range(len(tl)):
                self._taken[lv].add((int(tl[i, 0]), int(tl[i, 1])))
                cols['x'].append(int(tl[i, 0]))
                cols['y'].append(int(tl[i, 1]))
                cols['region'].append(int(self.inherit.region[i]))
                cols['rc'].append((-1, -1))
                cols['kind'].append(-1)
                cols['white'].append(float(w[i]))
                cols['bucket'].append(int(wb[i]))
                cols['origin'].append(2)
                cols['parent_x'].append(-1)
                cols['parent_y'].append(-1)
                cols['inherit'].append(i)

        for bp in plan.buckets:
            b = BUCKETS.index(bp.bucket)
            free = self._free_in(lv, b)
            take = free if len(free) <= bp.from_grid else \
                self.rng.choice(free, size=bp.from_grid, replace=False)
            for i in np.atleast_1d(take):
                i = int(i)
                self._taken[lv].add((int(g.xy[i, 0]), int(g.xy[i, 1])))
                cols['x'].append(int(g.xy[i, 0]))
                cols['y'].append(int(g.xy[i, 1]))
                cols['region'].append(int(g.region[i]))
                cols['rc'].append((int(g.grid_rc[i, 0]), int(g.grid_rc[i, 1])))
                cols['kind'].append(int(g.kind[i]))
                cols['white'].append(float(g.white[i]))
                cols['bucket'].append(b)
                cols['origin'].append(0)
                cols['parent_x'].append(-1)
                cols['parent_y'].append(-1)
                cols['inherit'].append(-1)

            if bp.from_jitter > 0:
                displaced, gave_up = self._jitter_from(
                    lv, np.atleast_1d(take), bp.from_jitter, b)
                for disp_x, disp_y, parent_index, disp_white in displaced:
                    cols['x'].append(disp_x)
                    cols['y'].append(disp_y)
                    cols['region'].append(int(g.region[parent_index]))
                    cols['rc'].append((-1, -1))
                    cols['kind'].append(-1)
                    cols['white'].append(disp_white)
                    cols['bucket'].append(b)
                    cols['origin'].append(1)
                    cols['parent_x'].append(int(g.xy[parent_index, 0]))
                    cols['parent_y'].append(int(g.xy[parent_index, 1]))
                    cols['inherit'].append(-1)
                if gave_up:
                    self._notes[lv].append(
                        f'{bp.bucket}: gave up after {self.cfg.max_miss} '
                        f'consecutive refusals; got {len(displaced)} of '
                        f'{bp.from_jitter} displaced tiles')

        n = len(cols['x'])
        return LevelSample(
            level=lv, ds=g.ds,
            x=np.array(cols['x'], np.int64), y=np.array(cols['y'], np.int64),
            region=np.array(cols['region'], np.int32),
            grid_rc=np.array(cols['rc'], np.int32).reshape(n, 2),
            kind=np.array(cols['kind'], np.int8),
            white_frac=np.array(cols['white'], np.float32),
            bucket=np.array(cols['bucket'], np.int8),
            origin=np.array(cols['origin'], np.int8),
            parent_x=np.array(cols['parent_x'], np.int64),
            parent_y=np.array(cols['parent_y'], np.int64),
            inherit_id=np.array(cols['inherit'], np.int32),
            valid_frac=np.full(n, np.nan, np.float32))

    def replace(self, lv: int, bucket: int, n: int = 1) -> List[tuple]:
        """Fresh coordinates from one bucket, for tiles a read rejected."""
        free = self._free_in(lv, bucket)
        if len(free) == 0:
            return []
        g = self.geoms[lv]
        take = self.rng.choice(free, size=min(n, len(free)), replace=False)
        out = []
        for i in np.atleast_1d(take):
            i = int(i)
            self._taken[lv].add((int(g.xy[i, 0]), int(g.xy[i, 1])))
            out.append((int(g.xy[i, 0]), int(g.xy[i, 1]), i))
        return out

    def notes(self, lv: int) -> List[str]:
        return self._notes[lv]


# ── handing the result to FeatureStore ────────────────────────────────────────

def to_store_args(s: LevelSample, features) -> dict:
    """The keyword arguments for FeatureStore.save.

    The four core arrays map one to one; everything this module adds rides in
    `extra`, which takes arbitrary named tensors and only checks that the names
    do not collide. So none of the provenance needs a change to FeatureStore --
    the query stores already carry ans_main and fov_id the same way.

    The one thing that DOES need a change lives in the metadata: cfg_hash covers
    the encoder and the mask but nothing about sampling, so two runs with
    different quotas or a different seed collide on one filename today. Put
    `SamplerConfig.sampler_id()` into StoreMeta and into _IDENTITY_FIELDS, or the
    stores silently mix.
    """
    import torch
    return dict(
        features=features,
        x=torch.from_numpy(s.x),
        y=torch.from_numpy(s.y),
        region=torch.from_numpy(s.region),
        grid_rc=torch.from_numpy(s.grid_rc),
        extra={
            'kind': torch.from_numpy(s.kind),
            'white_frac': torch.from_numpy(s.white_frac),
            'bucket': torch.from_numpy(s.bucket),
            'origin': torch.from_numpy(s.origin),
            'parent_x': torch.from_numpy(s.parent_x),
            'parent_y': torch.from_numpy(s.parent_y),
            'inherit_id': torch.from_numpy(s.inherit_id),
            'valid_frac': torch.from_numpy(s.valid_frac),
        })


# ── preflight ─────────────────────────────────────────────────────────────────

def render_preflight(stem: str, cfg: SamplerConfig,
                     geoms: Dict[int, LevelGeoms],
                     plans: Dict[int, LevelPlan],
                     inherit: Optional[InheritPlan]) -> str:
    """The report that decides whether the expensive part is worth starting.

    Returned rather than printed: this module is a library, and the caller owns
    stdout. It carries the distribution and not only the verdict, because a fixed
    background threshold means different things at different levels -- a tile's
    footprint grows with ds, so a half-millimetre square almost always contains
    background -- and the percentile each threshold lands on is what makes that
    visible instead of merely fatal.
    """
    L = [f'{"=" * 78}',
         f' ReferenceSampler pre-flight   {stem}   sampler_id {cfg.sampler_id()}'
         f'   seed {cfg.seed}',
         f' tile {cfg.tile}   target {cfg.n_target}/level   over {cfg.over}'
         f'   jitter cap {cfg.jitter_cap:.0%}',
         f'{"=" * 78}']

    for lv in sorted(geoms):
        g, p = geoms[lv], plans[lv]
        L.append('')
        L.append(f' L{lv}  ds={g.ds:g}  footprint {g.footprint_l0} level-0 px'
                 f'   grid {len(g.xy):,}'
                 f'  (main {int((g.kind == 0).sum()):,} /'
                 f' ovlp {int((g.kind == 1).sum()):,})')
        q = np.percentile(g.white, [10, 25, 50, 75, 90]) if len(g.white) else []
        if len(q):
            L.append('   white fraction  p10 {:.2f}  p25 {:.2f}  p50 {:.2f}'
                     '  p75 {:.2f}  p90 {:.2f}'.format(*q))
            pct = [float((g.white < e).mean() * 100) for e in cfg.edges]
            L.append('   the fixed cuts land at:  {:.0%} -> p{:.0f}'
                     '    {:.0%} -> p{:.0f}    {:.0%} -> p{:.0f}'.format(
                         cfg.edges[0], pct[0], cfg.edges[1], pct[1],
                         cfg.edges[2], pct[2]))
        L.append(f'   {"bucket":<8}{"want":>7}{"avail":>9}{"grid":>7}'
                 f'{"jitter":>8}{"short":>7}   why')
        for bp in p.buckets:
            L.append(f'   {bp.bucket:<8}{bp.want:>7}{bp.available:>9}'
                     f'{bp.from_grid:>7}{bp.from_jitter:>8}{bp.short:>7}'
                     f'   {bp.reason}')
        if p.inherited:
            L.append(f'   {"inherit":<8}{p.inherited:>7}{"":>9}'
                     f'{"":>7}{"":>8}{"":>7}   carried from level 0')
        verdict = ('UNUSABLE' if p.unusable else 'SHORT' if p.short else 'OK')
        L.append(f'   n_target {p.n_target} = min({cfg.n_target}, '
                 f'{len(g.xy):,} x {cfg.over})   achievable {p.got}'
                 f'   >>> {verdict}')
        if p.unusable:
            L.append(f'   !! below min_useful {cfg.min_useful} -- too few to be '
                     f'worth having')
        if not p.caps_ok:
            L.append(f'   !! cap violated: {p.cap_note}')

    if inherit is not None:
        L.append('')
        L.append(' inheritance (all from level 0, centre-aligned)')
        L.append(f'   candidates .................. {inherit.n_candidates}')
        for lv in sorted(inherit.fits):
            mark = '  <-- binding' if lv == inherit.binding_level else ''
            L.append(f'   fits at L{lv} ................. '
                     f'{inherit.fits[lv]}{mark}')
        if inherit.valid:
            for lv in sorted(inherit.valid):
                L.append(f'   photographed at L{lv} ......... {inherit.valid[lv]}')
        L.append(f'   surviving everywhere ........ {len(inherit.xy0)}')

    bad = [lv for lv, p in sorted(plans.items()) if p.unusable]
    if bad:
        L.append('')
        L.append(f' UNUSABLE levels: {bad}   (fewer than {cfg.min_useful} tiles'
                 f' achievable)')

    L.append('')
    L.append(' SHORT is not a failure. A deep level offers a few hundred grid'
             ' positions in total,')
    L.append(' and 40 per level is already saturated for the KNN that reads'
             ' this -- 40 to 640')
    L.append(' bought 2.7 points (result/MppEstimate). Only UNUSABLE is worth'
             ' stopping for.')

    L.append('')
    L.append(' NOT covered above: holes and unscanned canvas outside the'
             ' inheritance set.')
    L.append(' They are a property of (location, level) and only surface on read,'
             ' so every')
    L.append(' tile is re-checked at encode time and rejects are replaced from'
             ' the same bucket.')
    L.append('=' * 78)
    return '\n'.join(L)
