'''Episode construction for metric-based meta-learning -- which positions a
training step, a validation step and a test step compare.

TRAINING AND VALIDATION draw from a MIXED POOL (`pool_by_rung`): every
position of a rung, across every WSI, in one list. Which WSI a position came
from is not part of the task -- a scale question on one slide is the same
question on another with the same base mpp and rungs --
so there is no same-WSI coin and no borrowing. What the WSI still decides is
the OVERLAP RULE: two positions on the same WSI whose footprints overlap
(across rungs too) never land on opposite sides. `draw` enforces it, and
every position in one draw is distinct.

ONE DRAW, K BATCHES A SIDE. `draw` returns `ks` support batches and `kq`
query batches for one rung combination. What is trained on is a list of
(support batch, query batch) PAIRS, one optimizer step each (`reuse_pairs`):

    none     1 x 1     no hold: a fresh draw every step
    hold_s   1 x K     the support batch held, K query batches through it
    hold_q   K x 1     the query batch held, K support batches against it
    val      K x K     every support batch against every query batch

TRAINING COMBOS ARE ENUMERATED, NOT DRAWN (`training_combos`): every 3-, 4-
and 5-of-6 subset except `HELD_OUT_COMBOS` -- 19 + 14 + 6 = 39 -- each once
per epoch, in a fresh order every epoch.

K IS MEASURED, NOT GUESSED (`max_feasible_k`): the largest K for which the
real `draw` succeeds on every combination under the rules above.

TEST keeps its original per-episode draw as well (`sample_val_episode`,
`_draw_episode`, `render_episode` -- the `p_same_wsi` coin and all) so
`cli/evaluate.py` can report the numbers it always reported next to the
new K x K ones.

NOT NAMED `Datasets.py`: `training/MppRoutingHead/` already has one, and this
file imports it fully qualified. Position sampling itself (mask + TileSampler)
is `MppRoutingHead.Datasets.build_manifest`, run once up front; this module
only draws from what it built.
'''
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# MppRoutingHead's -- see module docstring. CameraBank/render_row are public
# there so this import is legitimate reuse.
from training.MppRoutingHead.Datasets import (
    CameraBank, ManifestRow, RUNGS, RenderConfig, render_row)
from WsiSplit import BRACS_RUNGS


#: Rung combinations reserved for validation -- `training_combos` never
#: yields them. Tests "has the metric space generalised to this specific
#: combination", a strictly harder question than "has it seen each rung in it
#: alone" (every rung here still appears in plenty of training combinations).
#:
#: The full 6-way tuple changes nothing about training (`n_choices` is 3, 4,
#: 5, so no 6-tuple is ever enumerated). It is here for validation: spec.md's
#: "Task" names the full 6-way classification as the DEPLOYMENT task, never
#: rehearsed by any training draw, and val is where it gets checked.
#:
#: The other two entries' contents are TBD -- the user's own examples from
#: spec.md, not a validated final list.
HELD_OUT_COMBOS: Tuple[Tuple[float, ...], ...] = (
    (1.0, 4.0, 16.0, 32.0),
    (1.0, 2.0, 4.0),
    (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
)


@dataclass(frozen=True)
class Episode:
    '''One episode of the ORIGINAL test draw (`sample_val_episode`): one
    support and one query set. `support`/`query`:
    `{rung: [ManifestRow, ...]}`, one key per rung in `rungs`, never a key
    for a rung this episode did not choose.'''
    rungs: Tuple[float, ...]
    support: Dict[float, List[ManifestRow]]
    query: Dict[float, List[ManifestRow]]


def group_by_wsi_name(rows: List[ManifestRow]) -> Dict[str, List[ManifestRow]]:
    '''`build_manifest`'s own flat list, grouped by WSI -- the shape the
    original test draw (`sample_val_episode`) takes.'''
    out: Dict[str, List[ManifestRow]] = {}
    for row in rows:
        out.setdefault(row.wsi_name, []).append(row)
    return out


def _rung_subsets(n: int, rungs: Sequence[float] = RUNGS) -> List[Tuple[float, ...]]:
    '''Every n-of-`rungs` combination, sorted ascending within each tuple so
    a combination and its held-out counterpart (`HELD_OUT_COMBOS`) compare
    equal regardless of draw order.'''
    return list(combinations(sorted(rungs), n))


def _draw_episode(support_pool: Dict[str, List[ManifestRow]],
                  query_pool: Dict[str, List[ManifestRow]],
                  rng: random.Random, chosen_rungs: Tuple[float, ...], *,
                  p_same_wsi: float, n_support: int, n_query: int
                  ) -> Optional[Episode]:
    '''The ORIGINAL per-episode draw, kept for `cli/evaluate.py`'s original
    test table so its numbers stay what they always were. Training and val
    use `draw` (mixed pool, overlap rule) instead.

    `support_pool is query_pool` is the ordinary case; two different pools
    turn the "same WSI" machinery below off.

    Draws support/query WSI (same with probability `p_same_wsi` when the
    two pools are the SAME domain, otherwise always an independent draw --
    there is no "same WSI" to coin-flip across two different datasets),
    then slices `n_support` + `n_query` positions per chosen rung.

    SUPPORT borrows across WSIs WITHIN `support_pool` when the chosen
    `support_wsi` alone falls short (`draw_support` below) -- a
    prototype's whole job is to represent "what rung R looks like" from a
    SET, not to represent one particular slide, so which WSI(s) actually
    supplied its members is not part of what it means (spec.md's
    class-agnostic framing: the model never keys on a rung's identity,
    only on support-vs-query similarity). Borrowing never crosses INTO
    `query_pool` even when the two are the same domain -- that would just
    be a second, redundant way of drawing support, not a new capability.
    QUERY stays tied to a single `query_wsi`, unborrowed: each query tile
    is scored as its own independent example, not aggregated, and it is
    the thing render_row provenance / native-vs-resampled bookkeeping
    already treats as "this one photo, from this one slide".

    Returns `None` on a supply shortfall QUERY still cannot recover from,
    or one SUPPORT cannot recover from even pooling every other WSI in
    `support_pool` together -- a genuine floor, not a sampling failure;
    the caller re-draws, same stance `TileSampler`'s own rejection
    sampling takes.

    Same-WSI draws get DISJOINT support/query positions (`exclude` below);
    cross-WSI and cross-domain draws need no such care -- the two pools
    have nothing in common to begin with. This still holds with
    borrowing: `exclude=s` only ever removes rows that could physically be
    in `query_pool`'s own pool (i.e. `support_wsi`-origin ones when the
    pools are the same domain AND `support_wsi == query_wsi`); a borrowed
    row's home WSI is never `query_wsi` in that branch by construction
    (`draw_support` borrows from `support_pool` EXCLUDING `support_wsi`,
    so no borrowed row can quietly already be that WSI's own).
    '''
    same_domain = support_pool is query_pool
    support_wsi = rng.choice(list(support_pool))
    query_wsi = (support_wsi if same_domain and rng.random() < p_same_wsi
                else rng.choice(list(query_pool)))

    def draw_support(rung: float, count: int) -> Optional[List[ManifestRow]]:
        primary = [r for r in support_pool.get(support_wsi, ())
                  if r.rung == rung]
        if len(primary) >= count:
            return rng.sample(primary, count)
        shortfall = count - len(primary)
        borrowed_pool = [r for name, rows in support_pool.items()
                         if name != support_wsi
                         for r in rows if r.rung == rung]
        if len(borrowed_pool) < shortfall:
            return None
        return primary + rng.sample(borrowed_pool, shortfall)

    def draw(wsi_name: str, rung: float, count: int,
            exclude: Sequence[ManifestRow] = ()) -> Optional[List[ManifestRow]]:
        pool = [r for r in query_pool.get(wsi_name, ())
               if r.rung == rung and r not in exclude]
        return rng.sample(pool, count) if len(pool) >= count else None

    support: Dict[float, List[ManifestRow]] = {}
    query: Dict[float, List[ManifestRow]] = {}
    for rung in chosen_rungs:
        s = draw_support(rung, n_support)
        if s is None:
            return None
        q = draw(query_wsi, rung, n_query,
                exclude=s if same_domain and support_wsi == query_wsi else ())
        if q is None:
            return None
        support[rung], query[rung] = s, q
    return Episode(rungs=chosen_rungs, support=support, query=query)


def sample_val_episode(manifest_by_wsi: Dict[str, List[ManifestRow]],
                       rng: random.Random, *, p_same_wsi: float,
                       n_support: int, n_query: int,
                       combos: Sequence[Tuple[float, ...]] = HELD_OUT_COMBOS
                       ) -> Optional[Episode]:
    '''One episode of the ORIGINAL test draw: the rung subset is chosen FROM
    `combos`, then `_draw_episode`. `cli/evaluate.py`'s original table only.

    Deliberately NOT cross-domain -- it measures accuracy on the DEPLOYMENT task's own shape
    (spec.md's "Task"), where support and query come from the SAME WSI's
    domain; cross-domain pairing is a TRAINING-time technique for forcing
    domain-invariant parameters, not a different definition of the task
    being scored. `--eval-datasets` already covers the cross-dataset
    generalisation question on the query/support-together axis (bracs/test
    vs ki67_with_photo, each internally consistent), which is what this
    function keeps measuring.

    `manifest_by_wsi` is expected to already be restricted to HELD-OUT WSIs
    by the caller -- this function has no opinion about which WSIs it is
    handed.
    '''
    chosen_rungs = rng.choice(list(combos))
    return _draw_episode(manifest_by_wsi, manifest_by_wsi, rng, chosen_rungs,
                         p_same_wsi=p_same_wsi, n_support=n_support,
                         n_query=n_query)


# ══════════════════════════════════════════════════════════════════════════
# Mixed-pool draws: training (none / hold_s / hold_q) and validation (K x K)
# ══════════════════════════════════════════════════════════════════════════

#: `--episode-reuse` values and the (support batches, query batches) each one
#: draws for a given K. `none` never holds anything: K only scales how many
#: draws an epoch makes (`episodes_per_epoch`), never the shape of one.
REUSE_MODES = ('none', 'hold_s', 'hold_q')


def batch_shape(mode: str, k: int) -> Tuple[int, int]:
    """`(ks, kq)` for one training draw."""
    if mode == 'none':
        return 1, 1
    if mode == 'hold_s':
        return 1, int(k)
    if mode == 'hold_q':
        return int(k), 1
    raise ValueError(f'unknown episode reuse mode {mode!r}; one of {REUSE_MODES}')


def reuse_pairs(ks: int, kq: int) -> List[Tuple[int, int]]:
    """Every (support index, query index) pair of a draw, support-major --
    one optimizer step each in training, one scored pair each in val."""
    return [(i, j) for i in range(ks) for j in range(kq)]


def training_combos(n_choices: Sequence[int] = (3, 4, 5),
                    held_out: Sequence[Tuple[float, ...]] = HELD_OUT_COMBOS
                    ) -> List[Tuple[float, ...]]:
    """Every n-of-6 rung subset for n in `n_choices`, minus `held_out`,
    fewest rungs first. 39 with the defaults."""
    held = {tuple(float(r) for r in c) for c in held_out}
    return [c for n in sorted(n_choices) for c in _rung_subsets(n) if c not in held]


def episodes_per_epoch(mode: str, k: int, n_combos: int) -> int:
    """The auto value: every combination once per epoch; `none` makes K times
    as many draws so its number of optimizer steps equals the held modes'."""
    return n_combos * (int(k) if mode == 'none' else 1)


def epoch_schedule(combos: Sequence[Tuple[float, ...]], n_draws: int,
                   rng: random.Random) -> List[Tuple[float, ...]]:
    """The combinations one epoch draws, in order: whole rounds of every
    combination, each round freshly shuffled, cut at `n_draws`. With the auto
    count every combination appears exactly once (x K under `none`)."""
    out: List[Tuple[float, ...]] = []
    while len(out) < n_draws:
        round_ = list(combos)
        rng.shuffle(round_)
        out += round_
    return out[:n_draws]


def pool_by_rung(rows: Sequence[ManifestRow]) -> Dict[float, List[ManifestRow]]:
    """The MIXED pool: every position of a rung, across every WSI, in one
    list, in manifest order (the draw shuffles, not this)."""
    out: Dict[float, List[ManifestRow]] = {}
    for row in rows:
        out.setdefault(float(row.rung), []).append(row)
    return out


def overlaps(a: ManifestRow, b: ManifestRow, max_ratio: float = 0.0) -> bool:
    """True when `a` and `b` sit on the same WSI and their level-0
    footprints share more than `max_ratio` of the smaller one. 0 means any
    shared area at all. Different rungs compare too: a ds 16 footprint that
    contains a ds 1 one is the same tissue."""
    if a.wsi_name != b.wsi_name or a.dataset != b.dataset:
        return False
    fa, fb = int(a.footprint_l0), int(b.footprint_l0)
    if fa <= 0 or fb <= 0:
        raise ValueError(f'a row without footprint_l0 cannot be overlap-tested: '
                         f'{a if fa <= 0 else b}')
    w = min(a.x + fa, b.x + fb) - max(a.x, b.x)
    h = min(a.y + fa, b.y + fb) - max(a.y, b.y)
    if w <= 0 or h <= 0:
        return False
    return (w * h) / float(min(fa, fb) ** 2) > max_ratio


@dataclass(frozen=True)
class Draw:
    """One draw for one rung combination: `supports[i]` / `queries[j]` are
    `{rung: [ManifestRow, ...]}`, `n_support` / `n_query` rows per rung."""
    rungs: Tuple[float, ...]
    supports: Tuple[Dict[float, List[ManifestRow]], ...]
    queries: Tuple[Dict[float, List[ManifestRow]], ...]


def draw(support_pool: Dict[float, List[ManifestRow]],
         query_pool: Dict[float, List[ManifestRow]],
         rng: random.Random, rungs: Sequence[float], *,
         n_support: int, n_query: int, ks: int, kq: int,
         max_overlap: float = 0.0) -> Optional[Draw]:
    """`ks` support and `kq` query batches for `rungs`, or None when the pool
    cannot supply them under the rules:

      * every position in the draw is distinct;
      * no support position overlaps a query position on the same WSI
        (`overlaps`, any rung against any rung) -- positions on the SAME side
        may overlap each other, nothing about the task forbids that.

    The side that needs FEWER positions is drawn first, at random; the other
    side then walks its rung's pool in a random order, skipping anything
    already taken or overlapping the first side. Taking the small side first
    removes the fewest candidates from the large one.

    `support_pool is query_pool` is the ordinary case. Two different pools
    (the cross-domain draw) cannot share a WSI, so the overlap rule has
    nothing to test there.
    """
    rungs = tuple(float(r) for r in rungs)
    same = support_pool is query_pool
    need = {'s': int(ks) * int(n_support), 'q': int(kq) * int(n_query)}
    pools = {'s': support_pool, 'q': query_pool}
    first, second = ('s', 'q') if need['s'] <= need['q'] else ('q', 's')

    taken: Dict[str, Dict[float, List[ManifestRow]]] = {'s': {}, 'q': {}}
    for rung in rungs:
        pool = pools[first].get(rung, [])
        if len(pool) < need[first]:
            return None
        taken[first][rung] = rng.sample(pool, need[first])

    by_wsi: Dict[Tuple[str, str], List[ManifestRow]] = {}
    if same:
        for rows in taken[first].values():
            for r in rows:
                by_wsi.setdefault((r.dataset, r.wsi_name), []).append(r)
    used = {id(r) for rows in taken[first].values() for r in rows}

    for rung in rungs:
        pool = list(pools[second].get(rung, []))
        rng.shuffle(pool)
        picked: List[ManifestRow] = []
        for r in pool:
            if len(picked) == need[second]:
                break
            if id(r) in used:
                continue
            if same and any(overlaps(r, o, max_overlap)
                            for o in by_wsi.get((r.dataset, r.wsi_name), ())):
                continue
            picked.append(r)
        if len(picked) < need[second]:
            return None
        taken[second][rung] = picked

    def split(side: str, k: int, n: int):
        return tuple({rung: taken[side][rung][i * n:(i + 1) * n] for rung in rungs}
                     for i in range(k))

    return Draw(rungs=rungs, supports=split('s', ks, n_support),
                queries=split('q', kq, n_query))


def training_pools(train_pool, cross_pool, rng: random.Random,
                   rungs: Sequence[float]):
    """`(support_pool, query_pool)` for one training draw. With a cross-domain
    pool and a combination that fits inside `BRACS_RUNGS`, a coin decides
    which domain supplies which side -- forcing the comparison across a
    staining domain neither side can shortcut through."""
    if cross_pool is not None and set(rungs) <= BRACS_RUNGS:
        return ((cross_pool, train_pool) if rng.random() < 0.5
                else (train_pool, cross_pool))
    return train_pool, train_pool


def _feasible(pool_pairs, combos, *, n_support, n_query, ks, kq, max_overlap,
              tries: int, seed: int) -> bool:
    for combo in combos:
        for sp, qp in pool_pairs(combo):
            rng = random.Random(seed)
            if not any(draw(sp, qp, rng, combo, n_support=n_support,
                            n_query=n_query, ks=ks, kq=kq,
                            max_overlap=max_overlap) is not None
                       for _ in range(tries)):
                return False
    return True


def max_feasible_k(pool_pairs, combos, shapes, *, n_support: int, n_query: int,
                   max_overlap: float = 0.0, tries: int = 5, seed: int = 0,
                   k_cap: int = 1000) -> int:
    """The largest K for which `draw` succeeds on EVERY combination, for
    EVERY shape in `shapes` (`K -> (ks, kq)` callables), within `tries`
    attempts each -- the real draw, not a formula, so it is exactly what a
    run can execute. `pool_pairs(combo)` lists the (support pool, query pool)
    pairs a combination can be drawn under (two for a cross-domain combo:
    each domain on each side). 0 when not even K = 1 fits.

    Binary search: a K that fits means every smaller K fits, since a smaller
    draw asks for a subset of the same positions.
    """
    def ok(k: int) -> bool:
        return all(_feasible(pool_pairs, combos, n_support=n_support,
                             n_query=n_query, ks=shape(k)[0], kq=shape(k)[1],
                             max_overlap=max_overlap, tries=tries, seed=seed)
                   for shape in shapes)

    if not ok(1):
        return 0
    lo, hi = 1, 2
    while hi <= k_cap and ok(hi):
        lo, hi = hi, hi * 2
    hi = min(hi, k_cap + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


@dataclass(frozen=True)
class RenderedDraw:
    """A `Draw` rendered: each batch `{rung: [(patch, native), ...]}`."""
    rungs: Tuple[float, ...]
    supports: Tuple[Dict[float, List[Tuple[np.ndarray, bool]]], ...]
    queries: Tuple[Dict[float, List[Tuple[np.ndarray, bool]]], ...]


def render_draw(d: Draw, bank: CameraBank, cfg: RenderConfig, *,
                deterministic: bool, support_native: bool = False,
                rng: Optional[random.Random] = None) -> Optional[RenderedDraw]:
    """Every batch of `d`, through `render_row`. Training passes `rng` (the
    episode sampler's), and each capture gets its own generator seeded from
    it, so the photos are a function of the saved RNG state; evaluation
    passes `deterministic=True` and each row renders from its own identity.
    None if any position fails to render -- the caller draws again."""
    def one(batch, native):
        out = {}
        for rung, rows in batch.items():
            got = []
            for row in rows:
                cap_rng = (None if deterministic or rng is None
                           else random.Random(rng.getrandbits(64)))
                result = render_row(bank, row, cfg, deterministic=deterministic,
                                    native=native, rng=cap_rng)
                if result is None:
                    return None
                patch, _label, native_pyramid = result
                got.append((patch, native_pyramid))
            out[rung] = got
        return out

    supports, queries = [], []
    for batch in d.supports:
        r = one(batch, support_native)
        if r is None:
            return None
        supports.append(r)
    for batch in d.queries:
        r = one(batch, False)
        if r is None:
            return None
        queries.append(r)
    return RenderedDraw(rungs=d.rungs, supports=tuple(supports), queries=tuple(queries))


# ══════════════════════════════════════════════════════════════════════════
# Rendering an Episode's positions into pixels.
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RenderedEpisode:
    '''One `Episode`, rendered. `support`/`query`: `{rung: [(patch, native),
    ...]}`, `patch` a `[tile, tile, 3]` uint8 array.

    NO label field, on purpose. `ManifestRow.label` is the GLOBAL rung index
    (0..5) -- what a caller needs here is the LOCAL index into THIS
    episode's own `rungs` (0..N-1, `CosineTauHead` returns logits shaped by
    however many prototypes it was handed). Deriving it as `rungs.index
    (rung)` at the point a target tensor gets built is one source of truth;
    carrying a second, separately-computed local-index number here is the
    kind of thing that can quietly disagree with `rungs` after an edit
    somewhere else.
    '''
    rungs: Tuple[float, ...]
    support: Dict[float, List[Tuple[np.ndarray, bool]]]
    query: Dict[float, List[Tuple[np.ndarray, bool]]]


def render_episode(episode: Episode, bank: CameraBank, cfg: RenderConfig, *,
                   deterministic: bool, support_native: bool = False
                   ) -> Optional[RenderedEpisode]:
    '''Renders every position in `episode` through `render_row`
    (`MppRoutingHead.Datasets`'s own render path -- fresh augmentation when
    `deterministic=False`, the training case; `rng` seeded from each row's
    own identity when `True`, the eval case -- see `RenderConfig`'s own
    docstring, "Camera: train vs eval").

    `support_native` (default `False`):
    passed straight through to `render_row`'s own `native` switch for the
    SUPPORT side only -- `query` always renders `native=False` (`CAMERA_
    FULL`, `Datasets.py`'s own module docstring), because a query genuinely
    IS a photograph, simulated or real, at both train and deploy time.
    Support is different: at real Stage 1 inference a reference/support
    tile is read straight off the target WSI, never photographed (see
    `CAMERA_GEOMETRY_ONLY`'s own docstring) -- this switch is what lets
    training match that on the support side, when a caller asks for it.

    `None` if ANY position fails to render (`render_row` returns `None`
    near a region edge, same rare per-position case `MppRoutingHead` already
    lives with) -- the caller draws a fresh episode rather than scoring a
    partial one.
    '''
    def render_group(rows_by_rung: Dict[float, List[ManifestRow]], *,
                     native: bool
                     ) -> Optional[Dict[float, List[Tuple[np.ndarray, bool]]]]:
        out: Dict[float, List[Tuple[np.ndarray, bool]]] = {}
        for rung, rows in rows_by_rung.items():
            rendered = []
            for row in rows:
                result = render_row(bank, row, cfg, deterministic=deterministic,
                                    native=native)
                if result is None:
                    return None
                patch, _global_label, native_pyramid = result
                rendered.append((patch, native_pyramid))
            out[rung] = rendered
        return out

    support = render_group(episode.support, native=support_native)
    if support is None:
        return None
    query = render_group(episode.query, native=False)
    if query is None:
        return None
    return RenderedEpisode(rungs=episode.rungs, support=support, query=query)
