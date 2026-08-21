#!/usr/bin/env python3
"""Does a different token pooling, or a different window score, retrieve the
right window better than production does?

    python utilities/bench_modules/bench_slidewin_pooling.py --n-fov 100
    python utilities/bench_modules/bench_slidewin_pooling.py --report-only \
        /work/u26130998/result/SlidewinPooling/slidewin_pooling.csv

Stage 2 as it actually runs -- mask, regions, sliding window -- with two axes
laid over it. GigaPath emits 197 tokens per tile and production keeps one, the
CLS; `SlidingWindowSimilarity` then turns the query's per-tile cosines into one
window score with an arithmetic mean. Both of those are choices, neither has
been measured against an alternative at the window level, and retrieval's
largest failure bucket is "the truth was never proposed" (32.3% of 1404 shots,
result/BenchLocaScope).

    pooling   cls  cls_avg  cls_std  rings3  grid2x2      (aiNNModel pooling_kinds)
    score     mean  geomean  min                          (how R_q x C_q cosines
                                                           become one number)

Five by three is fifteen arms; `cls` + `mean` IS production and is the
baseline every other arm is measured against.


═══════════════════════════════════════════════════════════════════════════════
 VOCABULARY -- every term used below, and where the number comes from
═══════════════════════════════════════════════════════════════════════════════

  查詢 / FoV        one photograph. Camera at a random position inside a tissue
   (query)          region, rotation fixed at 0. `--n-fov` per (slide, level).

  候選視窗          every sliding-window position of that (slide, level): the
   (candidate)      main grid AND the overlap grid, all regions, in ONE pool --
                    production proposes from both, so ranking must too.

  pool              how many candidate windows there are.          [stored]

  rank              1 + (number of candidates scoring strictly higher).
                    1 is best. Ties take the optimistic value, which matters
                    because `min` produces equal scores more often than `mean`.

  nearest main      round(x/256)*256 , round(y/256)*256
  nearest overlap   round((x-128)/256)*256+128 , same in y
                    x, y = the FoV's top-left in level-n coordinates relative to
                    its region's origin.

  d_main            Euclidean distance from the FoV's top-left to that grid
  d_overlap         point. Rotation is 0 and the two footprints are the same
                    size, so top-left distance == centre distance. [stored]

  truth             whichever of the two is geometrically closer.
                    rank_truth = rank_main if d_main <= d_overlap else
                                 rank_overlap
                    This is the metric of record.

  fine              whichever of the two ranks better.
                    rank_fine = min(rank_main, rank_overlap)
                    Strictly easier than truth. Reported as a diagnostic only.

  arm               one (pooling, score) pair. Written `cls_avg+geomean`.
                    baseline = `cls+mean`.

  rank_main         rank of the window at the nearest main grid point   [stored]
  rank_overlap      rank of the window at the nearest overlap point     [stored]

Everything below is COMPUTED from those five stored numbers, which is why
`--report-only` can redraw every table without a GPU.

  hit@k             rank_truth <= k, as a rate over the queries in the group:
                        truth@k(G,a) = |{q in G : rank_truth(q,a) <= k}| / |G|
                    A rate, hence a percentage.

  gap@k             fine@k(G,a) - truth@k(G,a).  Non-negative by construction,
                    since rank_fine <= rank_truth always. Reads as "retrieval
                    put the OTHER grid point in the top k but not the
                    geometrically closer one" -- location right to within half a
                    tile, wrong member of the overlapping pair. If this is large
                    the strict truth definition is doing a lot of the work and
                    the whole bench should be read differently.

  k@f%              max(1, ceil(f * pool)). Per (slide, level), because pool
                    varies 250x across levels.

  top@f%            rank_truth <= k@f%, as a rate. When a group spans several
                    (slide, level), EACH query uses its own k@f% and the hits
                    are then averaged -- not a single k applied to everything.

                    This is the only level metric that survives aggregation,
                    and the reason is that its null is constant: a random
                    ranking scores f% at top@f% whatever the pool size, while
                    rank@100 scores 100/pool -- 35% at L2, 0.13% at L0.

                    That holds while f * pool >= 1. Below it the max(1, ...)
                    takes over, k is 1 whatever f says, and the null goes back
                    to 1/pool. Only @0.01% reaches that floor on these slides,
                    and only outside L0 -- see K_FRACTIONS for which levels.

  W / L / T         paired against the baseline, per query, on rank_truth:
                        W: rank_truth(arm) <  rank_truth(baseline)
                        L: rank_truth(arm) >  rank_truth(baseline)
                        T: equal
  n_cmp             W + L. The sign test's real sample size -- ties do not
                    inform it, and at k=1 a great many queries tie.
  win%              W / (W + L). '-' when n_cmp is 0.

  ratio             rank_truth(baseline) / rank_truth(arm). Above 1 means the
                    arm ranked the truth better. Reported as Q1 / median / Q3
                    over the group. A ratio rather than a difference because
                    ranks span 1..76,435 and beating the baseline by 10 places
                    means something different at each end.

  med rank          median and 90th percentile of rank_truth. ABSOLUTE, so
  p90 rank          they appear only where pool is a single number.


═══════════════════════════════════════════════════════════════════════════════
 WHAT IS STORED
═══════════════════════════════════════════════════════════════════════════════

Per query:                     slide, level, fov_id, pool, d_main, d_overlap
Per (query, arm):              rank_main, rank_overlap

At --n-fov 100 that is 7 slides x 3 levels x 100 x 15 arms = 31,500 rows, a
few MB. Every table, every k, every aggregation is derived, so changing the k
list or the grouping never costs a GPU hour.


═══════════════════════════════════════════════════════════════════════════════
 THE FOUR TABLES -- what each is for, and why four
═══════════════════════════════════════════════════════════════════════════════

They differ only in the grouping key. The paired block (W L T n_cmp win% Q1 med
Q3, one column per K_FRACTIONS entry) is IDENTICAL in all four.

  單片單層   slide x level    21 tables   n = --n-fov
             The only place absolute numbers are valid, because pool is a
             single value: pool, k@f%, med/p90 rank, fixed rank@k, gap@k.

  同層跨片   level             3 tables   n = 7 x --n-fov       PRIMARY
             The first grouping where pool sizes are comparable -- within a
             level they differ about 6x (Ki67 L0 ~7,200 tiles against
             BRACS_1936 ~43,500), against 250x across levels. Fixed rank@k is
             still meaningful here and is printed.

  單片跨層   slide             7 tables   n = 3 x --n-fov
             Exists because stain type has already bitten this project once:
             bench_feature_axes found the three H&E slides carry mpp on PC1
             while two of the four Ki67 slides carry it on PC2. No fixed
             rank@k -- it mixes a 250x pool range inside one slide.

  全部       none              1 table    n = 21 x --n-fov      CONCLUSION
             One row per arm. No fixed rank@k, no absolute ranks.


═══════════════════════════════════════════════════════════════════════════════
 THE GATES, WHICH RUN BEFORE ANY GPU HOUR IS SPENT
═══════════════════════════════════════════════════════════════════════════════

Nothing here fails loudly. A broken coordinate mapping produces "no pooling
improves retrieval", which reads as a finding. So three checks run first, each
taking seconds, each able to pass only if the machinery means something:

  baseline is production   pooling_kinds(...,'cls') against .features().
                           Without it every arm is compared to a baseline that
                           is not the shipped feature. test_gigapath_pooling
                           already pins this at cos 1.2e-7; it is repeated here
                           because this bench's whole claim is relative to it.

  concat identity          each slot L2-normalised, concatenated, normalised
                           again gives a cosine equal to the MEAN of the
                           per-slot cosines. That identity is what lets five
                           multi-slot poolings run through an unmodified
                           SlidingWindowSimilarity, and it is a derivation --
                           derivations are not evidence.

  grid geometry            a FoV lands somewhere in a 128x128 cell, so the
                           nearest main point is at most 181 px away, the
                           nearest overlap point likewise, and the CLOSER of
                           the two is at most 128 -- NOT 90.51, because the two
                           grids interleave diagonally and their union is a
                           checkerboard, so (128, 0) belongs to neither. A
                           derived bound, not a tolerance. It caught its own
                           first version.

And one gate on the run itself: `decoy`, the percentile of a uniformly random
window, which must sit at 0.50. If the coordinate mapping is wrong the truth
window is effectively random, and this is what says so.

Rotation is fixed at 0 throughout. `SlidingWindowSimilarity` scores both grids
against the MAIN query kernel only (GigaPathSlidingWinSim.py:54); a rotated FoV
would fail for reasons that have nothing to do with pooling, and rotation has
its own test in test_gigapath_slide_win_sim.py step 5.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('utilities', 'aiNNModel', 'query_sim', '2_retrieval'):
    _p = str(_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

from PatchingLib import (FeaturesMap, QueryPatchContainer,      # noqa: E402
                         WsiTissuesContainer)
from SafeSlide import SafeSlide                                  # noqa: E402
from TissuesRegionsMask import TissuesRegionsMask                # noqa: E402
from TissueSegFunc import HestSegConfig                          # noqa: E402
from TileEncoderFunc import (admissible_poolings, encoder_config,  # noqa: E402
                             encoder_names, pooling_kinds)
from RetrievalReport import (K_FIXED, K_FRACTIONS,              # noqa: E402,F401
                             attach_baseline, frac_label, k_at,
                             report, truth_rank)
from GigaPathSlidingWinSim import SlidingWindowSimilarity        # noqa: E402
from camera import Camera                                        # noqa: E402
from config import DomainGapConfig                               # noqa: E402
from _paths import job_result_dir                                # noqa: E402

TILE = 256
HALF_TILE = TILE // 2

#: Rows of the comparison. Every one is a `pooling_kinds` mode; the slot counts
#: are 1, 2, 2, 4, 5, so the concatenated descriptors are 1536 .. 7680 wide.
POOLINGS = ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2')

#: Columns. All three reduce the SAME [R_q, C_q] tensor of per-tile cosines, so
#: they cost no capture and no forward pass -- see `combine`.
SCORES = ('mean', 'geomean', 'min')

#: Production. Every other arm is measured against this one, per query.
BASELINE = 'cls+mean'

#: Cosines of L2-normalised features live in [-1, 1] and log is undefined at or
#: below zero. Raising this floor pulls geomean towards the arithmetic mean,
#: because it stops the worst tile from dominating -- it is a parameter, not a
#: guard. Same value as bench_offgrid_score, so the two benches agree.
GEOMEAN_FLOOR = 1e-3

#: K_FIXED and K_FRACTIONS are imported from RetrievalReport, which owns them
#: along with every statistic computed from them. What the pools look like HERE,
#: which is what decides whether @0.01% means anything on a given row:
#:
#:     L0   pool 18,296..183,833   k = 2..19    a real 0.01% criterion
#:     L1   pool  3,432.. 10,495   k = 1..2     partly floored
#:     L2   pool    311..    499   k = 1        truth@1, null 0.20..0.32%
#:
#: Read @0.01% in 單片單層 and in the L0 row of 同層跨片; in 全部 it averages a
#: genuine 0.01% at L0 with truth@1 at L2, which is not one criterion.

#: Furthest a FoV can be from the NEAREST POINT OF ONE GRID. Each grid steps by
#: TILE on both axes, so the worst position is a cell centre: hypot(128, 128).
MAX_SINGLE_GRID_DISTANCE = math.hypot(HALF_TILE, HALF_TILE)      # 181.02

#: Furthest a FoV can be from the CLOSER of the two grids, and the reason it is
#: not what it first looks like.
#:
#: The obvious derivation says: the two grids interleave, so a FoV sits in a
#: 128x128 cell and the worst case is its centre at 128/sqrt(2) = 90.51. That is
#: wrong, and 10,000 random positions say 128.00.
#:
#: The overlap grid is offset DIAGONALLY, by (128, 128), so the union is
#: {(0,0) mod 256} u {(128,128) mod 256} -- a checkerboard. (128, 0) belongs to
#: neither. The union is therefore a square lattice rotated 45 degrees with
#: nearest-neighbour spacing 128*sqrt(2) = 181.02, whose covering radius is
#: 181.02/sqrt(2) = 128 exactly.
#:
#: CLAUDE.md records the mirror image of this mistake: an earlier DELTA_MAX was
#: set to 128 "by a derivation that applied to the union of both grids, while
#: the value being checked was the distance to one grid, whose bound is 181".
#: Same two numbers, swapped. Both times the derivation lost to the data, which
#: is the argument for keeping the check rather than the reasoning.
MAX_TRUTH_DISTANCE = float(HALF_TILE)                            # 128.00

BRACS = '/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test'
KI67 = '/work/u26130998/datasets/Ki67'
SLIDES = (
    f'{BRACS}/Group_AT/Type_ADH/BRACS_1228.svs',
    f'{BRACS}/Group_MT/Type_DCIS/BRACS_1476.svs',
    f'{BRACS}/Group_AT/Type_FEA/BRACS_1936.svs',
    f'{KI67}/S1104233,G7E,110208.mrxs',
    f'{KI67}/S1104360,G7E,110208.mrxs',
    f'{KI67}/S1137178,G7E,110926.mrxs',
    f'{KI67}/S1151088,G7E,111220.mrxs',
)


# ══════════════════════════════════════════════════════════════════════════════
#  Descriptors: pooling -> one vector per tile
# ══════════════════════════════════════════════════════════════════════════════

def concat_slots(slots: torch.Tensor) -> torch.Tensor:
    """[N, n, D] slots -> [N, n*D] unit vectors whose cosine IS the mean of the
    per-slot cosines.

    Each slot is normalised first, so every slot contributes a cosine in
    [-1, 1]; concatenating n unit vectors gives a vector of norm sqrt(n), and
    normalising divides by exactly that, leaving

        cos(A, B) = (1/n) * sum_k cos(a_k, b_k)

    which is why five multi-slot poolings can be scored by an unmodified
    `SlidingWindowSimilarity`. Weighting the slots differently would be a third
    axis; it is deliberately not opened.
    """
    slots = F.normalize(slots.float(), dim=-1)   # pooling_kinds already does this;
    return F.normalize(slots.flatten(1), dim=-1)  # repeated so the gate can feed
                                                  # raw tensors and still be valid


def pooled_descriptors(patches, encoder, poolings) -> dict:
    """{pooling: [N, n*D]} for one patch list, encoding the tokens ONCE.

    Encoding once per pooling instead would be five forward passes over one set
    of pixels, so all five reductions run off the same tokens. The question is
    only where those tokens live while it happens.

    They stay on the GPU. `reduce` is applied inside the encoder's batch loop,
    so what crosses to the host is the 14 concatenated slots -- 86 KB per tile
    against the 1.21 MB of [197, 1536] fp32 -- and the tokens for a batch are
    freed on the next iteration. 68k tiles is 5.9 GB accumulated rather than
    82 GB, which is why this takes no chunk parameter: `batch_size` bounds the
    GPU side and nothing else needs bounding.

    The earlier version pooled on the host, which needed an outer loop over
    `--token-chunk` to bound the same total. That loop paid to move every token
    across PCIe and only then discarded 93% of them.

    Widths are recorded as the first batch is reduced rather than derived from
    POOLINGS, so the split below cannot disagree with what pooling_kinds actually
    returned. The five results are views into one tensor, which is the whole
    allocation -- slicing does not copy.
    """
    widths = {}
    spec = encoder.model_spec

    def reduce(tokens):                       # [B, 197, 1536] fp32, on device
        parts = []
        for name in poolings:
            slots = pooling_kinds(tokens, name, spec)   # (feats, slots, layout)
            flat = concat_slots(slots)
            widths[name] = flat.shape[1]
            parts.append(flat)
        return torch.cat(parts, dim=1).cpu()  # one transfer, not five

    packed = encoder.tokens(patches, reduce=reduce)
    out, start = {}, 0
    for name in poolings:
        out[name] = packed[:, start:start + widths[name]]
        start += widths[name]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Scoring
# ══════════════════════════════════════════════════════════════════════════════

def combine(per_tile: torch.Tensor) -> dict:
    """[..., R_q, C_q] per-tile cosines -> one score per window, three ways.

    mean     arithmetic, what production ships. Strong tiles carry a window
             even when the rest disagree -- an OR over the query's tiles.
    geomean  exp(mean(log(x))). Multiplicative, so one tile near zero drags the
             window down however good the others are -- an AND. "Use AND
             instead of OR" and "multiply instead of add" are the same change;
             a linear combiner cannot be an AND.
    min      the hardest AND: one tile can veto the window.
    """
    values = per_tile.float()
    floored = values.clamp_min(GEOMEAN_FLOOR)
    return {
        'mean': values.mean(dim=(-2, -1)),
        'geomean': torch.exp(torch.log(floored).mean(dim=(-2, -1))),
        'min': values.amin(dim=(-2, -1)),
    }


def grid_answers(x: int, y: int) -> dict:
    """The two grid points nearest a FoV whose top-left is (x, y), level-n and
    relative to its region's origin, plus which of them is the truth.

    The main grid steps by TILE from the region origin; the overlap grid is the
    same lattice shifted by HALF_TILE on both axes, so `round` on the shifted
    coordinate is the whole calculation. Distances are top-left to top-left,
    which equals centre to centre because rotation is 0 and the footprints are
    the same size.
    """
    main_col = int(round(x / TILE))
    main_row = int(round(y / TILE))
    ovlp_col = int(round((x - HALF_TILE) / TILE))
    ovlp_row = int(round((y - HALF_TILE) / TILE))

    d_main = math.hypot(x - main_col * TILE, y - main_row * TILE)
    d_ovlp = math.hypot(x - (ovlp_col * TILE + HALF_TILE),
                        y - (ovlp_row * TILE + HALF_TILE))
    return {'main_rc': (main_row, main_col), 'ovlp_rc': (ovlp_row, ovlp_col),
            'd_main': d_main, 'd_overlap': d_ovlp,
            'truth': 'main' if d_main <= d_ovlp else 'overlap'}


def rank_of(answer_score: float, pools: list) -> int:
    """1 + how many candidates beat this one, across every region and BOTH grids.

    The pool is deliberately the union: production proposes from the main grid
    and the overlap grid together, so a ranking that saw only one of them would
    be answering a different question.
    """
    higher = 0
    for scores in pools:
        higher += int((scores > answer_score).sum())
    return higher + 1


# ══════════════════════════════════════════════════════════════════════════════
#  Gates
# ══════════════════════════════════════════════════════════════════════════════

def gate_baseline_is_production(patches, encoder) -> tuple:
    """`cls` pooling must be the feature production actually ships.

    BOTH sides run at the same `dtype`. autocast changes the forward pass, so
    comparing an fp16 pooled feature against an fp32 shipped one would measure
    the precision difference and blame it on the pooling -- a gate failing for
    a reason that is not a bug is worse than no gate.
    """
    sample = patches[:min(32, len(patches))]
    tokens = encoder.tokens(sample)
    slots = pooling_kinds(tokens, 'cls', encoder.model_spec)
    pooled = F.normalize(slots[:, 0].float(), dim=-1)
    shipped = F.normalize(encoder.features(sample).float(), dim=-1)
    cos = float((pooled * shipped).sum(-1).min())
    return cos, cos > 1 - 1e-4


def gate_reduce_matches_host(patches, encoder, poolings) -> tuple:
    """Pooling on the GPU must give what pooling on the host gave.

    `pooled_descriptors` hands the pooling to the encoder so the tokens never
    cross to the host -- 86 KB per tile instead of 1.21 MB. That moves the
    arithmetic onto the model's device and packs five descriptors of different
    widths into one tensor, and neither of the two ways it can go wrong raises:

      dtype    under autocast the tokens arrive fp16, and .mean(1) / .std(1)
               over 196 of them in fp16 degrades the averaged slots while
               leaving cls exactly right. That reads as "averaging poolings do
               not help", which is a result, not an error.
      packing  a wrong split offset hands each mode another mode's numbers, in
               the right shape.

    Scored against those two failures rather than a tolerance -- the right
    tolerance is unknown, the wrong answers are not. Not asserted as exact: the
    reference reduces on the CPU and this reduces on the GPU, and two sums of
    1536 floats in different orders need not match bit for bit.
    """
    sample = patches[:min(32, len(patches))]
    dim = int(encoder.model_spec.dim)
    spec = encoder.model_spec

    # `poolings`, not the module-level POOLINGS: that one is what this bench
    # WANTS compared, and admissible_poolings has already narrowed it to what
    # this encoder's patch grid admits. Gating the wider list would raise inside
    # pooling_kinds -- with a shape message, not a "this encoder cannot do
    # grid2x2" one -- before the run that would have dropped it ever started.
    got = pooled_descriptors(sample, encoder, poolings)
    tokens = encoder.tokens(sample)

    worst_ratio, worst_name = math.inf, ''
    for name in poolings:
        want = concat_slots(pooling_kinds(tokens, name, spec))
        same = float((got[name] - want).abs().max())
        decoys = [float((concat_slots(pooling_kinds(tokens.half(), name, spec))
                         - want).abs().max())]
        if want.shape[1] > dim:                  # cls has one slot to roll
            decoys.append(float((got[name] - want.roll(dim, dims=1)).abs().max()))
        ratio = min(decoys) / same if same > 0 else math.inf
        if ratio < worst_ratio:
            worst_ratio, worst_name = ratio, name
    return (worst_ratio, worst_name), worst_ratio > 1000


def gate_concat_identity(seed: int = 0) -> tuple:
    """cos(concat) must equal the mean of the per-slot cosines."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(512, 4, 1536, generator=generator)
    b = torch.randn(512, 4, 1536, generator=generator)
    per_slot = F.cosine_similarity(a, b, dim=-1).mean(-1)
    concat = (concat_slots(a) * concat_slots(b)).sum(-1)
    delta = float((per_slot - concat).abs().max())
    return delta, delta < 1e-5


def gate_grid_geometry(seed: int = 0, n: int = 10000) -> tuple:
    """The closer grid point can never be further than 128/sqrt(2)."""
    rng = np.random.default_rng(seed)
    worst_main = worst_ovlp = worst_truth = 0.0
    for x, y in rng.integers(0, 4096, size=(n, 2)):
        found = grid_answers(int(x), int(y))
        worst_main = max(worst_main, found['d_main'])
        worst_ovlp = max(worst_ovlp, found['d_overlap'])
        worst_truth = max(worst_truth, min(found['d_main'], found['d_overlap']))
    ok = (worst_main <= MAX_SINGLE_GRID_DISTANCE + 1e-9
          and worst_ovlp <= MAX_SINGLE_GRID_DISTANCE + 1e-9
          and worst_truth <= MAX_TRUTH_DISTANCE + 1e-9)
    return (worst_main, worst_ovlp, worst_truth), ok


def gate_tiles(path: str, n: int = 32) -> list:
    """A few real tiles from the middle of a slide, read directly.

    The gates need pixels the encoder will actually see, but they must not pay
    for them: building a WsiTissuesContainer would read a whole tissue region,
    which on a level-0 BRACS slide is the 4083-second case in log/TODO.log. One
    2048x1024 read is enough and costs nothing.
    """
    slide = SafeSlide(path)
    try:
        width, height = slide.dimensions
        cols, rows = 8, (n + 7) // 8
        x0 = max(0, width // 2 - cols * TILE // 2)
        y0 = max(0, height // 2 - rows * TILE // 2)
        image, _ = slide.read_region_valid((x0, y0), 0,
                                           (cols * TILE, rows * TILE))
        return [image[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
                for r in range(rows) for c in range(cols)][:n]
    finally:
        slide.close()


def run_gates(patches, encoder, poolings) -> bool:
    print('[gate] baseline is production', end='  ', flush=True)
    cos, ok_base = gate_baseline_is_production(patches, encoder)
    print(f'cos={cos:.7f}  {"OK" if ok_base else "FAIL"}')

    print('[gate] reduce == pooling on the host', end='  ', flush=True)
    (ratio, worst), ok_reduce = gate_reduce_matches_host(
        patches, encoder, poolings)
    print(f'worst margin over a decoy {ratio:.3g}x ({worst})  '
          f'{"OK" if ok_reduce else "FAIL"}')

    print('[gate] concat identity', end='  ', flush=True)
    delta, ok_concat = gate_concat_identity()
    print(f'max|Δ|={delta:.2e} over 512 pairs  {"OK" if ok_concat else "FAIL"}')

    print('[gate] grid geometry', end='  ', flush=True)
    (wm, wo, wt), ok_grid = gate_grid_geometry()
    print(f'd(main) {wm:.2f}/{MAX_SINGLE_GRID_DISTANCE:.2f}  '
          f'd(overlap) {wo:.2f}/{MAX_SINGLE_GRID_DISTANCE:.2f}  '
          f'd(truth) {wt:.2f}/{MAX_TRUTH_DISTANCE:.2f}  '
          f'{"OK" if ok_grid else "FAIL"}')
    return ok_base and ok_reduce and ok_concat and ok_grid


# ══════════════════════════════════════════════════════════════════════════════
#  One (slide, level)
# ══════════════════════════════════════════════════════════════════════════════

def sample_fovs(mask, containers, camera, ds, n_fov, white_max, rng) -> list:
    """(region_index, level-0 x, y, region-relative level-n x, y) per query.

    Rejection sampling, with three conditions that all have to hold or the
    query is not answerable:

      the FoV rectangle lies inside the region     -- Camera would read past it
      both answer windows exist in the sliding-window output -- a window whose
        index is off the end is not in the pool, and its "rank" would be a
        number nobody could earn
      the footprint is below `white_max` background -- on blank glass every
        window matches every other and the ranking measures nothing

    The second is the one that is easy to forget and impossible to see later:
    an out-of-range answer index does not raise, it just makes that query
    unanswerable for every arm at once, which reads as "retrieval is bad here".
    """
    footprint_w = camera.qfw.rect_w_l0
    footprint_h = camera.qfw.rect_h_l0
    picked = []
    for _ in range(200 * n_fov):
        if len(picked) >= n_fov:
            break
        index = int(rng.integers(0, len(containers)))
        container = containers[index]
        region = container.tissue_region
        if region.w <= footprint_w or region.h <= footprint_h:
            continue
        x0 = int(region.x + rng.integers(0, region.w - footprint_w))
        y0 = int(region.y + rng.integers(0, region.h - footprint_h))

        x_n = (x0 - region.x) / ds
        y_n = (y0 - region.y) / ds
        found = grid_answers(x_n, y_n)
        if not window_exists(container, found, camera):
            continue

        footprint = np.array(
            [[x0 + c * TILE * ds, y0 + r * TILE * ds]
             for r in range(camera.qfw.output_h // TILE)
             for c in range(camera.qfw.output_w // TILE)], dtype=np.int64)
        white = float(mask.white_fractions(footprint, container.at_level,
                                           TILE).mean())
        if white >= white_max:
            continue
        picked.append({'region': index, 'x0': x0, 'y0': y0,
                       'white_frac': round(white, 4), **found})
    return picked


def window_exists(container, found: dict, camera) -> bool:
    """Are both answer windows inside this region's sliding-window output?

    H_out = R_wsi - R_q + 1 on the main grid, and one less on each axis for the
    overlap grid, which is what `SlidingWindowSimilarity` returns.
    """
    rows_q = camera.qfw.output_h // TILE
    cols_q = camera.qfw.output_w // TILE
    main_h = container.grid.grid_rows - rows_q + 1
    main_w = container.grid.grid_cols - cols_q + 1
    ovlp_h, ovlp_w = main_h - 1, main_w - 1
    mr, mc = found['main_rc']
    orow, ocol = found['ovlp_rc']
    return (0 <= mr < main_h and 0 <= mc < main_w
            and 0 <= orow < ovlp_h and 0 <= ocol < ovlp_w)


def score_maps(query_map, wsi_maps: list) -> dict:
    """{score: (list of per-region main maps, list of overlap maps)}.

    `SlidingWindowSimilarity` is called once per region and its [H, W, R_q,
    C_q] output is reduced three ways, because the three combiners read the
    same tensor -- adding them costs no forward pass.
    """
    out = {name: ([], []) for name in SCORES}
    for wsi_map in wsi_maps:
        main_sim, ovlp_sim = SlidingWindowSimilarity(query_map, wsi_map)
        main_by = combine(main_sim) if main_sim.numel() else None
        ovlp_by = combine(ovlp_sim) if ovlp_sim.numel() else None
        for name in SCORES:
            out[name][0].append(main_by[name] if main_by is not None
                                else torch.empty(0))
            out[name][1].append(ovlp_by[name] if ovlp_by is not None
                                else torch.empty(0))
    return out


def ranks_for(maps: tuple, region: int, found: dict) -> tuple:
    """(rank_main, rank_overlap, pool) for one arm on one query."""
    main_maps, ovlp_maps = maps
    mr, mc = found['main_rc']
    orow, ocol = found['ovlp_rc']
    main_answer = float(main_maps[region][mr, mc])
    ovlp_answer = float(ovlp_maps[region][orow, ocol])
    everything = [m.flatten() for m in main_maps + ovlp_maps if m.numel()]
    pool = int(sum(int(m.numel()) for m in everything))
    return (rank_of(main_answer, everything),
            rank_of(ovlp_answer, everything), pool)


def run_slide_level(slide, stem, level, mask, args, encoder,
                    poolings, rng) -> list:
    """Every query of one (slide, level), scored by every arm."""
    ds = float(slide.level_downsamples[level])
    base_mpp = slide.base_mpp  # SafeSlide.base_mpp: mean of mpp-x/y, one definition
    config = DomainGapConfig(
        wh_ratio='45:32', MPixels=1.47456, query_mpp=base_mpp * ds,
        angle_jitter_deg=0.0, scale_range=(1.0, 1.0),
        query_mpp_jitter=0.0, stage_shift_max=0, photometric=True)
    camera = Camera(slide, cfg=config, mask=mask, seed=args.seed + level)

    started = time.time()
    # from_ds, not the constructor: it drops regions that cannot host a tile at
    # this level. Constructing directly left them in, and S1104233 L2 -- 23
    # regions, several of them slivers -- reached the encoder with an empty
    # batch and died in torch.cat naming neither the region nor the level.
    wsi = WsiTissuesContainer.from_ds(slide, ds, tile_size=TILE,
                                      overlap=True, mask=mask)
    containers = wsi.tissue_patches
    read_s = time.time() - started

    started = time.time()
    wsi_maps = {name: [] for name in poolings}
    n_tiles = 0
    for container in containers:
        patches = list(container)
        n_tiles += len(patches)
        pooled = pooled_descriptors(patches, encoder, poolings)
        for name in poolings:
            wsi_maps[name].append(FeaturesMap(container.grid, pooled[name]))
    encode_s = time.time() - started

    queries = sample_fovs(mask, containers, camera, ds, args.n_fov,
                          args.white_max, rng)
    if not queries:
        print(f'  L{level}: no position leaves room for the scan -- skipped',
              flush=True)
        return []

    rows = []
    for fov_id, query in enumerate(queries):
        shot, _ = camera.capture_with_gt(query['x0'], query['y0'], rotation=0)
        if shot is None:
            continue
        container = QueryPatchContainer(shot)
        container.extract_all(TILE, overlap=False)   # only the main kernel is
        patches = list(container)                    # ever used, see module doc
        pooled = pooled_descriptors(patches, encoder, poolings)
        for name in poolings:
            maps = score_maps(FeaturesMap(container.grid, pooled[name]),
                              wsi_maps[name])
            for score in SCORES:
                rank_main, rank_ovlp, pool = ranks_for(
                    maps[score], query['region'], query)
                rows.append({
                    # First column on purpose. Every table below averages over
                    # rows, so a CSV that cannot say which encoder produced it
                    # reads as one comparison when two were merged; --report-only
                    # refuses a mixed set on this key.
                    'encoder': args.encoder,
                    'slide': stem, 'level': level, 'fov_id': fov_id,
                    'pool': pool, 'd_main': round(query['d_main'], 2),
                    'd_overlap': round(query['d_overlap'], 2),
                    'white_frac': query['white_frac'],
                    'arm': f'{name}+{score}',
                    'rank_main': rank_main, 'rank_overlap': rank_ovlp})

    pool = rows[0]['pool'] if rows else 0
    base = [r for r in rows if r['arm'] == BASELINE]
    truths = sorted(truth_rank(r) for r in base)
    # The truth's mean percentile. A uniformly random window would sit at
    # 0.500, so this reads as a decoy comparison: near 0.5 means the coordinate
    # mapping is broken and every arm is ranking noise.
    truth_pct = float(np.mean([truth_rank(r) / max(1, r['pool'])
                               for r in base])) if base else float('nan')
    print(f'  L{level}  {n_tiles:,} tiles  pool {pool:,}  '
          f'{len(base)} FoV   baseline rank_truth med '
          f'{truths[len(truths) // 2] if truths else 0:,}  '
          f'read {read_s:.0f}s  encode {encode_s:.0f}s  '
          f'truth_pctile {truth_pct:.4f} (random=0.5000)',
          flush=True)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Derived metrics -- everything reads only the stored integers
#
#  They live in utilities/RetrievalReport.py because bench_gigapath_pooling asks
#  the same question one scale down and prints the same tables. Two copies of
#  "@1%" would never have raised: both would print a plausible number under the
#  same header, and a comparison between the two benches would be quietly
#  invalid. Moving them was pure relocation -- `--report-only` on the existing
#  CSV redraws the previous log byte for byte, which is the check that it was.
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(rows: list, path: Path) -> None:
    if not rows:
        print(f'  (nothing to write to {path.name})')
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, restval='')
        writer.writeheader()
        writer.writerows(rows)
    print(f'  {path}  ({len(rows):,} rows)')


def read_rows(paths) -> list:
    integers = {'level', 'fov_id', 'pool', 'rank_main', 'rank_overlap'}
    floats = {'d_main', 'd_overlap', 'white_frac'}
    rows = []
    for path in paths:
        with open(path, newline='') as handle:
            for raw in csv.DictReader(handle):
                row = {}
                for key, value in raw.items():
                    if key in integers:
                        row[key] = int(float(value))
                    elif key in floats:
                        row[key] = float(value)
                    else:
                        row[key] = value
                rows.append(row)
        print(f'  read {path}')
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description='pooling x window-score, measured through stage 2')
    parser.add_argument('csv', nargs='*',
                        help='with --report-only, the CSVs to re-tabulate')
    parser.add_argument('--report-only', action='store_true',
                        help='rebuild every table from stored ranks. No GPU, '
                             'no WSI, no model -- every metric is derived from '
                             'rank_main and rank_overlap, so changing the k '
                             'list or the grouping never costs a GPU hour.')
    parser.add_argument('--slides', nargs='*', default=list(SLIDES))
    parser.add_argument('--levels', type=int, nargs='*', default=[0, 1, 2])
    parser.add_argument('--n-fov', type=int, default=100,
                        help='queries per (slide, level). 100 makes every rate '
                             'a whole percent and puts the single-combo win%% '
                             'standard error at 5 pp; 25 would quantise the '
                             'rates to 4%% steps.')
    parser.add_argument('--white-max', type=float, default=0.15)
    parser.add_argument('--mask-ds', type=float, default=4.0)
    parser.add_argument('--seg-chunk-px', type=float, default=4_000_000)
    parser.add_argument('--min-region-ratio', type=float, default=0.01)
    parser.add_argument(
        '--encoder', default='gigapath', choices=encoder_names(),
        help='which tile encoder. Only the module for THIS one is imported: '
             'every implementation sets HF_HOME above its own timm import and '
             'setdefault is first-one-wins, so importing all three would point '
             'two of them at the wrong weight cache -- silently. See '
             'TileEncoderFunc._IMPLEMENTATIONS. Arms this encoder cannot do '
             'are dropped by name and printed; see admissible_poolings.')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='tiles per forward pass. 1024 matches the rest of '
                             'the benches and is affordable because the token '
                             'path runs under fp16 autocast')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='run the forward pass under fp16 autocast. Output '
                             'is fp32 either way (GigaPathFunc.py:174); this is '
                             'the precision production already ships, recorded '
                             'in log/TODO.log at cos=0.99995 against fp32 with '
                             'a 5.5x speedup. The NaN that appeared alongside '
                             'it needed Token Merging as well; fp16 on its own '
                             'was clean, which is why one shipped and the other '
                             'was removed (log/MILESTONE.log M3).')
    parser.add_argument('--per-slide', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='print the 單片跨層 tables. Worth having only when '
                             'the per-level tables show an H&E / Ki67 split')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    out_dir = Path(args.out or job_result_dir('SlidewinPooling'))
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [f'{p}+{s}' for p in POOLINGS for s in SCORES]
    print(f'bench_slidewin_pooling   {len(args.slides)} slides x levels '
          f'{args.levels}   {args.n_fov} FoV each')
    print(f'poolings  {"  ".join(POOLINGS)}')
    print(f'scores    {"  ".join(SCORES)}'
          f'{"":6}-> {len(arms)} arms, baseline = {BASELINE}\n')

    if args.report_only:
        if not args.csv:
            parser.error('--report-only needs at least one CSV path')
        rows = read_rows(args.csv)
        seen = {r.get('encoder', '') for r in rows}
        if len(seen) > 1:
            parser.error(
                f'these CSVs mix encoders ({", ".join(sorted(seen))}). Every '
                f'table averages over rows, so the merge would print one '
                f'comparison where there are two. Report them separately.')
        report(attach_baseline(rows, BASELINE), arms, BASELINE,
               per_slide=args.per_slide)
        return 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = encoder_config(args.encoder, batch_size=args.batch_size)\
        .with_model(dtype='fp16' if args.fp16 else 'fp32')

    # Which arms this encoder can actually run. POOLINGS above is what the bench
    # WANTS compared; the patch grid decides what is possible, and the two are
    # not the same list across encoders -- 14x14 divides by 7 and 16x16 does
    # not. Narrowed before the model loads, and the dropped ones are printed:
    # a table with four arms where five were asked for still reads as a complete
    # comparison unless something says which is missing.
    poolings, dropped = admissible_poolings(cfg, POOLINGS)
    if not poolings:
        parser.error(f'{args.encoder} admits none of {POOLINGS}; it accepts '
                     f'{", ".join(sorted(cfg.POOLINGS))}')
    if dropped:
        print(f'[arms] {args.encoder} cannot do {", ".join(dropped)} -- '
              f'dropped. Its patch grid admits {", ".join(sorted(cfg.POOLINGS))}')
    arms = [f'{p}+{s}' for p in poolings for s in SCORES]

    encoder = cfg.build(device)
    print(f'device={device}  encoder={args.encoder}  '
          f'model_spec={encoder.model_spec}  '
          f'dtype={encoder.cfg.model.dtype}  batch={args.batch_size}  '
          f'id={encoder.identity_id()}')
    print(f'arms      {len(arms)}, baseline = {BASELINE}\n')

    if not run_gates(gate_tiles(args.slides[0]), encoder, poolings):
        print('\nGATE FAILURE -- stopping before the run spends hours '
              'producing numbers that could not mean anything')
        return 1

    hest_method = HestSegConfig().build(device)
    rng = np.random.default_rng(args.seed)

    all_rows, failed = [], []
    for path in args.slides:
        stem = Path(path).stem
        print(f'\n{"=" * 78}\n{stem}\n{"=" * 78}', flush=True)
        slide = SafeSlide(str(path))
        try:
            mask = TissuesRegionsMask.from_wsi(
                slide, ds=args.mask_ds, method=hest_method,
                seg_chunk_px=int(args.seg_chunk_px), stitch_overlap=128,
                level_rule='nearest')
            mask.filter_regions(min_ratio=args.min_region_ratio)
            mask.merge_overlapping()
            print(f'  mask  tissue {mask.tissue_fraction() * 100:.1f}%  '
                  f'{len(mask.tissue_regions)} regions', flush=True)
            if not mask.tissue_regions:
                continue

            for level in args.levels:
                if level >= slide.level_count:
                    continue
                all_rows.extend(run_slide_level(
                    slide, stem, level, mask, args, encoder, poolings, rng))
        except Exception as exc:                          # noqa: BLE001
            print(f'  {type(exc).__name__}: {exc}')
            failed.append(stem)
        finally:
            slide.close()

    print(f'\n{"=" * 78}\nwriting to {out_dir}')
    # The encoder is in the NAME as well as in every row: the name stops a
    # second encoder's run from overwriting the first one's numbers, the column
    # stops the two from being averaged together afterwards. Neither does the
    # other's job.
    write_csv(all_rows, out_dir / f'slidewin_pooling_{args.encoder}.csv')
    if all_rows:
        report(attach_baseline(all_rows, BASELINE), arms, BASELINE,
               per_slide=args.per_slide)
    if failed:
        print(f'\n{len(failed)} slide(s) failed: {", ".join(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
