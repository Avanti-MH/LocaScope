#!/usr/bin/env python3
"""Does a different token pooling, or a different window score, retrieve the
right window better than production does?

    python utilities/bench_modules/bench_window_retrieval.py --encoder uni2
    python utilities/bench_modules/bench_window_retrieval.py --report-only \
        /work/u26130998/result/WindowRetrievalBench/uni2/window_retrieval.csv

Stage 2 as it actually runs -- mask, regions, sliding window -- with two axes
laid over it. GigaPath emits 197 tokens per tile and production keeps one, the
CLS; `SlidingWindowSimilarity` then turns the query's per-tile cosines into one
window score with an arithmetic mean. Both of those are choices, neither has
been measured against an alternative at the window level, and retrieval's
largest failure bucket is "the truth was never proposed" (32.3% of 1404 shots,
result/BenchLocaScope).

    pooling   cls  cls_avg  cls_std  rings3  grid2x2      (aiNNModel pooling_kinds)
    score     mean  geomean  min                          (how R_q x C_q cosines
                                                           become one number; mean and
                                                           geomean are SlidingWinSimRot's,
                                                           min is this bench's alone)

Five by three is fifteen arms; `cls` + `mean` IS production and is the
baseline every other arm is measured against.


═══════════════════════════════════════════════════════════════════════════════
 WHAT RUNS
═══════════════════════════════════════════════════════════════════════════════

The worked numbers in the sections from VOCABULARY on assume seven slides,
three levels, 100 FoVs each and top-left answers; this section is what runs:

  slides     the first `--n-wsi` of each `--datasets` id's recorded `--split`
             (MakeSplit; the split is shuffled when made), as every bench takes
             them (`BenchCommon.run_slides`). `--shard I/N` takes every N-th
             of the slides, for N processes on N cards.
  levels     each slide's own pyramid levels up to `--max-ds` (the FoV recipe's
             `max_ds` by default; `--fov`, FovSupply.FOV_RECIPES).
  FoVs       `--n-fov` (the recipe's `n_per_rung`) per (slide, level), all of a
             slide's levels in ONE draw on the hest masks of --fov-mask-cache-job,
             photographed through `FovSupply.cached` (`BenchCommon.supply_for`):
             the draw and the photos' record are cache entries, and the same
             flags give the pipeline bench's and the stage-1 bench's photos. A
             level too coarse for the tissue holds no FoV and is skipped.
  answer     the window the query's TILE GRID covers once the shot is turned
             upright, as the pipeline bench places it (`query_grid_centre`, the
             same function): the shot is turned by `matching_rotation`, whole
             tiles are cut from the TURNED photo's top-left, and the centre of
             that grid goes back to the slide through `Render.output_to_level0`
             at the angle and scale the shot was taken at. The window is the one
             of the turned shot's size (the sides swap at 90 / 270). At rotation 0,
             scale 1 and no jitter this is the FoV's top-left, which the bench
             checks on every such FoV.
  rotation   as the stage searches: every shot is scored at all four quarter
             turns (`SlidingWinSimRot.ROTATIONS`), the reference tiles encoded
             once and shared by the four. The truth is the window at the shot's
             `matching_rotation`; a rank counts the windows of ALL four
             rotations that score above it, as `nearest_window` does.
             `--rotation` / `--scale-min` / `--scale-max` still replace the
             recipe's gap (every quarter turn, scale 0.90-1.15 under 'bench') to
             restrict the FoVs; they are not needed to isolate the pooling.
  arms       `--arms cls cls_avg-M rings3`: each pooling says which reference
             grid it lives on -- no suffix is main + the offset grid, `-M` is
             main tiles only (capital M; `-m` is refused). A run with only `-M`
             arms never encodes an offset tile. `cls` on the full grid is the
             baseline and is added if missing. `raw` / `raw-M` is the patch
             tokens unpooled (256 per tile, prefix left out): a tile pair scores
             the mean of the 256 per-position cosines. Asked for by name only.
  metric     the main tables score every arm on the MAIN windows only, so an `-M`
             arm and a full-grid arm meet on the same candidates and the same
             truth (the main window nearest the shot): `rank_main` is that rank
             among main windows, `rank_overlap` repeats it, `pool` counts them.
             A pooling and its `-M` twin therefore agree there by construction;
             the log says on how many FoVs they do not.
  recall     what the offset grid adds is a separate table, for the full-grid
             arms only. A shot has TWO answers, the main window and the offset
             window nearest it (`grid_answers`); each is ranked in the pool it
             belongs to, and the CSV keeps all four ranks:
               rank_main_in_main / pool          main answer, main windows
               rank_offset_in_offset / pool_offset   offset answer, offset windows
               rank_main_in_all, rank_offset_in_all  both, main + offset windows
                                                 (pool_all)
             recall_main@k, recall_offset@k are those ranks <= k. recall_all@k
             is min(rank_main_in_all, rank_offset_in_all) <= k: the better-ranked
             of the two answers, in the mixed pool. The `main:offset` line says,
             among the shots that hit, which answer ranked first (a tie goes to
             main). Two targets are easier to hit than one, so a `random` line
             (expected recall of a random ranking on the same pools) sits beside
             them: the rise from main to all is read against it.
  features   nothing of a slide's tile features is kept. QGF (the query's tile
             features) is computed once per shot; RGF (the reference's) is
             streamed one tile row at a time, its cosines against QGF added into
             the windows that row belongs to, and dropped. Memory does not grow
             with the slide. `source` in the CSV is `stream`. Masks come from
             `--mask-cache-job`'s cache when it has them.


═══════════════════════════════════════════════════════════════════════════════
 VOCABULARY -- every term used below, and where the number comes from
═══════════════════════════════════════════════════════════════════════════════

  查詢 / FoV        one photograph. A camera (Render) at a random position inside a tissue
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

  wins / losses / ties
                    paired against the baseline, per query, on rank_truth:
                        wins:   rank_truth(arm) <  rank_truth(baseline)
                        losses: rank_truth(arm) >  rank_truth(baseline)
                        ties:   equal
  decided           wins + losses. The sign test's real sample size -- ties do
                    not inform it, and at k=1 a great many queries tie.
  win%              wins / decided. '-' when decided is 0.

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

They differ only in the grouping key. The paired block (wins losses ties decided win% Q1
med Q3, one column per K_FRACTIONS entry) is IDENTICAL in all four.

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
             a PCA of the features found the three H&E slides carry mpp on PC1
             while two of the four Ki67 slides carry it on PC2. No fixed
             rank@k -- it mixes a 250x pool range inside one slide.

  全部       none              1 table    n = 21 x --n-fov      CONCLUSION
             One row per arm. No fixed rank@k, no absolute ranks.


═══════════════════════════════════════════════════════════════════════════════
 THE GATES, WHICH RUN BEFORE ANY GPU HOUR IS SPENT
═══════════════════════════════════════════════════════════════════════════════

STAGE 2 TABLES. Beside the CSV, every arm writes the stage 2 entry the pipeline bench
writes, in this job's cache, under `stage1=oracle` (the bench
scores a FoV at the level it was placed at): `output` (the first 100 windows,
picked by the stage's own `candidate_set`), `truth` (with its rank and the pools),
`tile_sims` (10 windows) and `truth_sim`. See "The stage 2 tables an arm writes".
Every arm writes them, `min` and the `-M` arms too (their entries have ids of
their own, the stage having no setting for them).

Nothing here fails loudly. A broken coordinate mapping produces "no pooling
improves retrieval", which reads as a finding. So three checks run first, each
taking seconds, each able to pass only if the machinery means something:

  baseline is production   pooling_kinds(...,'cls') against .features().
                           Without it every arm is compared to a baseline that
                           is not the shipped feature. test_encoders pins the
                           same equality for all three encoders (`features() is
                           the cls slot`); it is repeated here because this
                           bench's whole claim is relative to it, and because a
                           cheap assertion belongs in front of an expensive run
                           rather than only in a test somebody may not have run.
                           The two are separate definitions on purpose -- see
                           gate_baseline_is_production below.

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
                           derived bound, not a tolerance.

And one gate on the run itself: `decoy`, the percentile of a uniformly random
window, which must sit at 0.50. If the coordinate mapping is wrong the truth
window is effectively random, and this is what says so.

Rotated FoVs are scored as the stage scores them: the query turned to each of
four rotations, the truth at the shot's matching rotation (see "rotation" above),
so a rotated shot no longer fails for a reason that has nothing to do with
pooling. A fifth check, `truth centre`, pins that placement to the pipeline's.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from typing import Optional
import json
import math
import contextlib
import os
import types
import queue
import resource
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('utilities', 'aiNNModel', 'query_sim', ''):     # '' = the root: stage packages
    _p = str(_ROOT / _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _paths                                                   # noqa: E402
_paths.setup_import_paths()

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

import Cache                                                     # noqa: E402
from PatchingLib import (FeaturesMap, PatchGrid, region_grids,  # noqa: E402
                         QueryPatchContainer)
from CpuBudget import CpuBudget                                  # noqa: E402
from SlideReader import SlideReader                              # noqa: E402
from SafeSlide import SafeSlide                                  # noqa: E402
from TissueMaskConfig import (MASK_RECIPES, MaskMaker,           # noqa: E402
                              add_mask_args, mask_cfg_from_args)
from TileEncoderFunc import (add_encoder_args, admissible_poolings,  # noqa: E402
                             encoder_cfg_from_args, encoder_config,
                             encoder_names, pooling_kinds)
from FovSupply import FOV_RECIPES, FovSupply, add_fov_args       # noqa: E402
from ReadGeometry import ReadRect, ReadSpec, levels_up_to         # noqa: E402
from ConfigArgs import config_from_args, describe                # noqa: E402
from ConfigIdentity import enc, environment, short_id             # noqa: E402
from dump_function.RetrievalReport import (K_FIXED, K_FRACTIONS,  # noqa: E402,F401
                                           attach_baseline, frac_label, grid_table,
                                           group_by, group_levels, k_at, pct,
                                           print_level_heading, report, truth_rank)
from stage2_retrieval.SlidingWinSimRot import (GEOMEAN_FLOOR, ROTATIONS,      # noqa: E402
                                               SlidingWinSimRot,
                                               SlidingWinSimRotConfig,
                                               WINDOW_SCORES, SlidingWindowSimilarity,
                                               window_score)
from stage2_retrieval.StageInterface import Candidate, CandidateSet  # noqa: E402
from camera import Render                                        # noqa: E402
from _paths import encoder_tag, job_result_dir                   # noqa: E402
from BenchCommon import (ORACLE, ROLES, TRUTH_VERSION, Stage, Tables,  # noqa: E402
                         fmt_each, fmt_rate, fmt_time, matching_rotation,
                         query_grid_centre, require_gpu_if_allocated, run_slides,
                         status, supply_for)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG -- everything a run depends on, in one place. Edit here.
#
#  Each value below is the one the run uses, unless the command line says
#  otherwise: every field of the sampler, camera and encoder configs is a flag
#  (`--sampler-richness-caps`, `--camera-noise-sigma`, `--encoder-batch-size`, ...; see
#  ConfigArgs), and the shorthand flags (`--n-fov`, `--seed`, `--rotation`, `--scale-min`,
#  `--scale-max`, `--batch-size`, `--fp16`) name one field each. What
#  a run really used is printed at its start, and everything that changes a
#  number is in the parts id, so a changed value here can never be resumed into
#  parts made with another.
# ══════════════════════════════════════════════════════════════════════════════

# ── which slides ─────────────────────────────────────────────────────────────
DATASETS         = ('bracs/test', 'ki67_with_photo')   # AccessDatasets ids
SPLIT            = 'val'        # the recorded split (MakeSplit) the slides come from
N_WSI            = 10           # the first N of each dataset's split, as every bench
MASK_CACHE_JOB   = None         # None: this job's own; 'MppRoutingHead' to reuse
FOV_MASK_CACHE_JOB = 'MppRoutingHead'   # the hest masks the FoVs are placed on

# ── the FoVs and the mask ────────────────────────────────────────────────────
# Where the FoVs go, the camera that takes them and the coarsest level are a FoV
# recipe (`--fov`, default `FovSupply.FOV_RECIPES['bench']`); the mask is a mask
# recipe (`--seg`, default `MASK_RECIPES['hest']`). Every field of both is a flag.

# ── the encoder and what is compared ─────────────────────────────────────────
# The encoder's own config is built from its registry entry once its NAME is
# known, because importing every implementation up front would point two of them
# at the wrong weight cache (see --encoder). Its fields are all `--encoder-*`
# flags, and the run prints all of them.
ENCODER    = 'uni2'         # which implementation
HEAD       = ''             # which exit of the model; '' for its own
DTYPE      = 'fp16'         # the forward pass under fp16 autocast
BATCH_SIZE = 2048           # tiles per forward pass
BLOCK_ROWS = 8              # main tile rows per block read (SlideReader.read_grid)
ARMS       =('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2', 'raw-M')
SCORES     = ('mean', 'geomean', 'min')

TILE = 256
HALF_TILE = TILE // 2

#: Rows of the comparison. Every one is a `pooling_kinds` mode; the slot counts
#: are 1, 2, 2, 4, 5, so the concatenated descriptors are 1536 .. 7680 wide.
POOLINGS = ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2')

#: The model's own patch tokens, unpooled: 256 of them per tile for UNI2, the 9
#: prefix tokens (cls and registers) left out. Two tiles' similarity is the mean
#: over the 256 positions of the cosine of the tokens at that position, which is
#: `concat_slots` over 256 slots. Not in `POOLINGS`: it is asked for by name
#: (`--arms raw-M`), because its descriptor is 393,216 wide.
RAW = 'raw'

#: Columns. All three reduce the SAME [R_q, C_q] tensor of per-tile cosines, so
#: they cost no capture and no forward pass -- see `combine`.
# SCORES is in the CONFIG block above.

#: Production. Every other arm is measured against this one, per query.
BASELINE = 'cls+mean'

#: Scores only this bench measures: the stage has `mean` and `geomean`
#: (SlidingWinSimRot.WINDOW_SCORES, imported above with GEOMEAN_FLOOR, so the
#: bench and the stage score a window one way) and no `min`.
BENCH_ONLY_SCORES = {
    'min': lambda per_tile: per_tile.float().amin(dim=(-2, -1)),
}
assert set(SCORES) <= set(WINDOW_SCORES) | set(BENCH_ONLY_SCORES), \
    'SCORES names a score neither the stage nor this bench has'

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

#: What a full-grid arm's row carries besides `rank_main` and `pool` (see
#: `recall_ranks`). Empty on a main-only arm, which has no offset windows.
RECALL_COLUMNS = ('rank_offset_in_offset', 'pool_offset', 'rank_main_in_all',
                  'rank_offset_in_all', 'pool_all')

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

    They stay on the GPU, and so does the result. `reduce` is applied inside
    the encoder's batch loop and returns the concatenated slots where they were
    computed: `stream_windows` scores them there and drops them. A caller that
    keeps descriptors for a whole slide is the one that moves them to the host.
    Pooling on the device means no token crosses PCIe only to be discarded.

    `patches` is a list of tiles or a uint8 batch `[N, T, T, 3]` (what
    `SlideReader.read_grid` yields); `TileEncoder` takes either.

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
            if name == RAW:
                slots = tokens[:, int(spec['num_prefix']):]     # patch tokens only
            else:
                slots = pooling_kinds(tokens, name, spec)
            flat = concat_slots(slots)
            widths[name] = flat.shape[1]
            parts.append(flat)
        return torch.cat(parts, dim=1)        # stays on the device

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
    """[..., R_q, C_q] per-tile cosines -> one score per window, once per score
    in `SCORES`.

    mean     the stage's own (`SlidingWinSimRot.window_score`), what production
             ships. Strong tiles carry a window even when the rest disagree --
             an OR over the query's tiles.
    geomean  the stage's own: exp(mean(log(x))). Multiplicative, so one tile
             near zero drags the window down however good the others are -- an
             AND. "Use AND instead of OR" and "multiply instead of add" are the
             same change; a linear combiner cannot be an AND.
    min      this bench's alone, the hardest AND: one tile can veto the window.
    """
    return {name: (window_score(per_tile, name) if name in WINDOW_SCORES
                   else BENCH_ONLY_SCORES[name](per_tile)) for name in SCORES}


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
    got = pooled_descriptors(sample, encoder, poolings)   # on the device, as `tokens` is
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


def gate_recall_ranks() -> tuple:
    """`recall_ranks` on a hand-made map whose answers are known by construction.

    One region, two rotations. At rotation 0: main windows 3x3, offset windows
    2x2, the answers at main (1, 1) and offset (0, 1). At rotation 90: main 2x2
    and offset 1x1, whose windows are the other rotation's candidates and count
    in every pool but are not the answer. The expected ranks are written down,
    not computed by the code under test:
      main 0.6    above it: main 0.9, 0.8 | 0.99, 0.7 (rot 90) ; offset 0.95, 0.7,
                  0.65 (rot 0) and 0.65 (rot 90)
      offset 0.7  above it: offset 0.95 ; main 0.9, 0.8 | 0.99 (0.7 is not above)
    Run with the answers at rotation 0, and again at 90 with the maps swapped so
    the answer must be read from the rotation `found` names."""
    main = torch.tensor([[0.9, 0.5, 0.1], [0.2, 0.6, 0.3], [0.8, 0.4, 0.0]])
    ovlp = torch.tensor([[0.95, 0.7], [0.65, 0.05]])
    main90 = torch.tensor([[0.99, 0.1], [0.7, 0.2]])
    ovlp90 = torch.tensor([[0.65]])
    found = {'main_rc': (1, 1), 'ovlp_rc': (0, 1), 'rot': 0}
    got = recall_ranks({0: ([main], [ovlp]), 90: ([main90], [ovlp90])}, 0, found)
    want = {'rank_main_in_main': 5, 'pool': 13, 'rank_offset_in_offset': 2,
            'pool_offset': 5, 'rank_main_in_all': 9, 'rank_offset_in_all': 5,
            'pool_all': 18}
    swapped = recall_ranks({0: ([main90], [ovlp90]), 90: ([main], [ovlp])}, 0,
                           dict(found, rot=90))
    return got, got == want and swapped == want


def gate_truth_centre() -> tuple:
    """`answers_for` against an index image, for the four rotations.

    A shot of 1024 x 1440 px is turned with `np.rot90` the way the stage turns
    it, cut into whole tiles from the turned photo's top-left, and the original
    pixel under the centre of that grid is read off an image of its own
    coordinates -- nothing derived. The camera here is the identity map, so the
    window `answers_for` places must stand centred on that pixel. The decoy is
    the centre of the grid cut from the UNTURNED photo, which is what this bench
    used before: it has to be far from the truth at 180 and 270, or the gate
    could not tell the two apart. Returns ((worst |diff|, decoy), ok)."""
    class Identity:
        def output_to_level0(self, x0, y0, u, v, rot_deg=0.0, scale=1.0):
            return u, v
    height, width = 1024, 1440
    ys, xs = np.indices((height, width))
    region = types.SimpleNamespace(x=0, y=0)
    worst, decoy = 0.0, 0.0
    for rot_deg in (0, 90, 180, 270):
        k = matching_rotation(rot_deg) // 90
        turned_x, turned_y = np.rot90(xs, k), np.rot90(ys, k)
        grid_h = (turned_x.shape[0] // TILE) * TILE
        grid_w = (turned_x.shape[1] // TILE) * TILE
        cy, cx = grid_h // 2, grid_w // 2
        centre = [(turned_x[r, c], turned_y[r, c])
                  for r in (cy - 1, cy) for c in (cx - 1, cx)]
        want = (float(np.mean([p[0] for p in centre])) + 0.5,
                float(np.mean([p[1] for p in centre])) + 0.5)
        found = answers_for(Identity(), 0, 0, {'rot_deg': rot_deg, 'angle_jitter': 0.0,
                                               'scale': 1.0}, region, 1.0,
                            (height, width, 3))
        got = (found['x_n'] + found['cols_t'] * HALF_TILE,
               found['y_n'] + found['rows_t'] * HALF_TILE)
        worst = max(worst, abs(got[0] - want[0]), abs(got[1] - want[1]))
        decoy = max(decoy, abs((width // TILE) * TILE / 2.0 - want[0]),
                    abs((height // TILE) * TILE / 2.0 - want[1]))
    return (worst, decoy), worst < 1e-6 and decoy >= HALF_TILE


def gate_scored_windows(seed: int = 0) -> tuple:
    """`ScoredWindows` picks the windows the stage picks. Made-up per-tile maps
    for two regions (12 x 10 and 9 x 9 tiles), four rotations, both lattices:
    the real `SlidingWinSimRot` reduces them with its score and picks; a
    `ScoredWindows` is handed the reduced scores and picks. The two lists --
    window, rank, score -- must be the same, for `mean` and for `geomean`.
    Returns (windows compared, ok)."""
    import types
    from PatchingLib import region_grids
    generator = torch.Generator().manual_seed(seed)
    regions = [types.SimpleNamespace(x=1000, y=2000, w=12 * TILE, h=10 * TILE),
               types.SimpleNamespace(x=40000, y=9000, w=9 * TILE, h=9 * TILE)]
    grids = region_grids(regions, ds=1.0, level=0, tile_size=TILE, overlap=True)
    compared, same = 0, True
    for score in ('mean', 'geomean'):
        sims = {}
        for rot, (r_q, c_q) in ((0, (4, 5)), (90, (5, 4)), (180, (4, 5)), (270, (5, 4))):
            per_region = []
            for grid in grids:
                rows, cols = grid.lattice_dims('main')
                o_rows, o_cols = grid.lattice_dims('offset')
                per_region.append((
                    torch.rand(rows - r_q + 1, cols - c_q + 1, r_q, c_q, generator=generator),
                    torch.rand(o_rows - r_q + 1, o_cols - c_q + 1, r_q, c_q,
                               generator=generator)))
            sims[rot] = per_region
        cfg = SlidingWinSimRotConfig(
            encoder=encoder_config('gigapath'), tile_size=TILE, overlap=True, k=15,
            min_sep_tiles=1.0, score=score)
        real = SlidingWinSimRot(cfg, 'cpu')
        real.grids, real.level, real.ds, real.sim_maps_by_rot = list(grids), 0, 1.0, sims
        scored = ScoredWindows(cfg, 'cpu')
        scored.grids, scored.level, scored.ds = list(grids), 0, 1.0
        scored.sim_maps_by_rot = {
            rot: [(window_score(a, score), window_score(b, score)) for a, b in per_region]
            for rot, per_region in sims.items()}
        want = [(c.key(), round(c.score, 6)) for c in real.candidate_set(15).candidates]
        got = [(c.key(), round(c.score, 6)) for c in scored.candidate_set(15).candidates]
        compared += len(want)
        same = same and want == got and len(want) == 15
    return compared, same


def gate_raw_descriptor(seed: int = 0) -> tuple:
    """The `raw` descriptor's cosine is the mean over patch positions of the
    per-position cosine, and the 9 prefix tokens do not enter it. Made-up tokens
    of UNI2's shape, 9 prefix + 256 patch. Returns ((max |diff|, prefix leak), ok)."""
    generator = torch.Generator().manual_seed(seed)
    prefix, cells, dim = 9, 256, 64
    a = torch.randn(4, prefix + cells, dim, generator=generator)
    b = torch.randn(4, prefix + cells, dim, generator=generator)
    desc = lambda t: concat_slots(t[:, prefix:])                       # noqa: E731
    per_position = F.cosine_similarity(a[:, prefix:], b[:, prefix:], dim=-1).mean(-1)
    delta = float((per_position - (desc(a) * desc(b)).sum(-1)).abs().max())
    moved = a.clone()
    moved[:, :prefix] = torch.randn(4, prefix, dim, generator=generator)
    leak = float((desc(a) - desc(moved)).abs().max())
    return (delta, leak), delta < 1e-5 and leak == 0.0


def tissue_spots(mask, width0: int, height0: int, block_w0: int, block_h0: int,
                 limit: int = 5) -> list:
    """`[(label, x0, y0)]`: top-lefts, level-0, of blocks of `block_w0` x
    `block_h0` centred on the middle of the `limit` largest tissue regions, kept
    inside the slide. Where a gate should look, since the middle of a slide can
    be blank glass. Empty without a mask or without regions."""
    if mask is None:
        return []
    regions = sorted(mask.tissue_regions, key=lambda r: r.w * r.h, reverse=True)
    out = []
    for r in regions[:limit]:
        cx, cy = r.x + r.w // 2, r.y + r.h // 2
        x0 = min(max(0, cx - block_w0 // 2), max(0, width0 - block_w0))
        y0 = min(max(0, cy - block_h0 // 2), max(0, height0 - block_h0))
        out.append((f'region {r.index} centre', int(x0), int(y0)))
    return out


def gate_tiles(path: str, n: int = 32, mask=None) -> list:
    """A few real tiles, read directly: from the middle of the largest tissue
    region when a `mask` says where it is, else from the middle of the slide.

    The gates need pixels the encoder will actually see, and on a slide whose
    tissue is sparse the middle can be blank glass. They must not pay for them
    either: reading a whole tissue region is, on a level-0 BRACS slide, the
    4083-second case in log/TODO.log. One
    2048x1024 read is enough and costs nothing.
    """
    slide = SafeSlide(path)
    try:
        width, height = slide.dimensions
        cols, rows = 8, (n + 7) // 8
        spots = tissue_spots(mask, width, height, cols * TILE, rows * TILE, limit=1)
        if spots:
            _, x0, y0 = spots[0]
        else:
            x0 = max(0, width // 2 - cols * TILE // 2)
            y0 = max(0, height // 2 - rows * TILE // 2)
        image, _ = slide.read_region_valid((x0, y0), 0,
                                           (cols * TILE, rows * TILE))
        return [image[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
                for r in range(rows) for c in range(cols)][:n]
    finally:
        slide.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Reference tiles
#
#  Read by `SlideReader.read_grid` in blocks of rows, never a whole level-0
#  region (tens of GB). Where every tile sits is `PatchGrid`'s alone --
#  `lattice_dims` and `tile_origin` are its methods, checked below against
#  the PatchInfo it makes, and `gate_row_reads` checks that a strip read from
#  a level-0 origin agrees with the same rows read in one piece.
# ══════════════════════════════════════════════════════════════════════════════

LATTICES = ('main', 'offset')


# ══════════════════════════════════════════════════════════════════════════════
#  Windows scored while the reference streams
#
#  A window's score needs only the cosine of each of the query's tiles against
#  the reference tile it lands on, so the reference tiles are never kept: a tile
#  row is encoded, its cosines are added into the windows that row belongs to,
#  and the row is dropped. Tile (y, x) is the query's tile (i, j) for the window
#  whose top-left is (y - i, x - j), which is what `_sim_tensors` writes as a
#  shifted slice; `gate_window_stream` checks the two agree.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  Where the time goes
# ══════════════════════════════════════════════════════════════════════════════

#: The phases of one (slide, level), in the order they run. A phase's name is a
#: key; what it is called in the report, and what is counted in it, is `STAGES`
#: and `STREAM_SPLIT` below.
PLACE, QUERY, STREAM, RANK = 'place', 'query', 'stream', 'rank'
TABLES = 'tables'
READ, ENCODE, COSINE, ACCUM = 'read', 'encode', 'cosine', 'accumulate'

#: `(label, phase, what its speed counts)` of the four stages of a level, and of
#: the parts stage 3 divides into when `--timing` is on. `photo` is a FoV; `tile`
#: is a tile of the query or of the reference; '' is a part with no count worth
#: a speed.
STAGES = (
    ('place the photos on the grid', PLACE, 'photo'),
    ('encode the query tiles, every rotation', QUERY, 'tile'),
    ('stream the reference level', STREAM, 'tile'),
    ('rank the photos', RANK, 'photo'),
    ('write the stage 2 tables', TABLES, 'photo'),
)
STREAM_SPLIT = (
    ('wait for the reader', READ, 'tile'),
    ('encode the reference tiles', ENCODE, 'tile'),
    ('cosines', COSINE, ''),
    ('add to the window scores', ACCUM, ''),
)


class PhaseTimer:
    """Wall-clock seconds spent in the named phases of ONE (slide, level), and
    how many things each phase handled, so a time reads as seconds per item.

    A GPU call returns before the GPU has finished, so a clock around one would
    time only the launch. A phase therefore waits for the GPU at both ends. The
    four stages do (a few waits per level). The split of stage 3 -- one phase
    per tile row, thousands of them -- does only when `fine` is set
    (`--timing`): those waits make the run a little slower, which is what it
    costs to see where the time is. Without `fine`, the split is not measured
    and the report says so."""

    def __init__(self, fine: bool = False) -> None:
        self.fine: bool = fine
        self.seconds: dict[str, float] = {}
        self.items: dict[str, int] = {}
        self._open: dict[str, float] = {}

    @staticmethod
    def _now() -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def start(self, name: str, fine: bool = False) -> None:
        if fine and not self.fine:
            return
        self._open[name] = self._now()

    def stop(self, name: str, items: int = 0) -> None:
        if name not in self._open:
            return
        began: float = self._open.pop(name)
        self.seconds[name] = self.seconds.get(name, 0.0) + (self._now() - began)
        self.items[name] = self.items.get(name, 0) + items

    @contextlib.contextmanager
    def phase(self, name: str, items: int = 0, fine: bool = False):
        self.start(name, fine)
        try:
            yield
        finally:
            self.stop(name, items)

    def report(self, level: int, n_photos: int) -> str:
        """The level's time, step by step: how long it took, how fast it went
        (photos, or tiles, a second or a minute) and how long one photo took in
        it, then the whole level the same way. With `fine`, the reference
        stream is split into its parts, indented under it. The whole level is
        encoded once however many photos stand on it, so the more photos, the
        less each one costs; the report says so because the figures under
        'one photo' are the level's time divided by this level's photos."""
        total: float = sum(self.seconds.get(k, 0.0) for _, k, _ in STAGES)

        def line(label: str, key: str, unit: str, indent: int = 0) -> str:
            sec: float = self.seconds.get(key, 0.0)
            count: float = n_photos if unit == 'photo' else self.items.get(key, 0)
            speed: str = fmt_rate(count, sec, unit) if unit else ''
            return (f'  {" " * indent}{label:<{42 - indent}}{fmt_time(sec):>10}'
                    f'{speed:>18}{fmt_each(n_photos, sec, "photo"):>16}')

        lines: list[str] = [
            f'  L{level}  {n_photos} photo through this level in {fmt_time(total)}'
            f'  (the level is encoded once; each photo added costs less)',
            f'  {"":<42}{"time":>10}{"speed":>18}{"one photo":>16}']
        for label, key, unit in STAGES:
            lines.append(line(label, key, unit))
            if key == STREAM and self.fine:
                parts: float = 0.0
                for sub_label, sub_key, sub_unit in STREAM_SPLIT:
                    lines.append(line(sub_label, sub_key, sub_unit, indent=3))
                    parts += self.seconds.get(sub_key, 0.0)
                self.seconds['other'] = max(0.0, self.seconds.get(STREAM, 0.0) - parts)
                lines.append(line('other (Python, host)', 'other', '', indent=3))
        if not self.fine:
            lines.append('     (run with --timing to split the reference stream)')
        lines.append(f'  {"whole level":<42}{fmt_time(total):>10}'
                     f'{fmt_rate(n_photos, total, "photo"):>18}'
                     f'{fmt_each(n_photos, total, "photo"):>16}')
        return '\n'.join(lines)


class WindowAccumulator:
    """Window scores of ONE lattice of ONE region for `n_shots` queries at once,
    filled one reference tile row at a time. Mean, geomean and min are all
    running reductions, so each row is added and forgotten."""

    def __init__(self, rows: int, cols: int, n_shots: int, rows_q: int,
                 cols_q: int, device):
        self.rows_q, self.cols_q = rows_q, cols_q
        self.h_out, self.w_out = rows - rows_q + 1, cols - cols_q + 1
        # a region smaller than the query holds no window (`_sim_tensors` too)
        self.live = self.h_out >= 1 and self.w_out >= 1
        if self.live:
            shape = (n_shots, self.h_out, self.w_out)
            self.total = torch.zeros(shape, device=device)
            self.logs = torch.zeros(shape, device=device)
            self.low = torch.full(shape, float('inf'), device=device)

    def add_row(self, y: int, cos: torch.Tensor) -> None:
        """`cos` is [S, cols, R_q, C_q]: reference row `y`'s tiles against every
        query tile. Query row i puts this row in window row y - i."""
        if not self.live:
            return
        for i in range(self.rows_q):
            r = y - i
            if not 0 <= r < self.h_out:
                continue
            for j in range(self.cols_q):
                c = cos[:, j:j + self.w_out, i, j]                  # [S, w_out]
                self.total[:, r] += c
                self.logs[:, r] += torch.log(c.clamp_min(GEOMEAN_FLOOR))
                self.low[:, r] = torch.minimum(self.low[:, r], c)

    def scores(self):
        """{score: [S, h_out, w_out]}, or None when the region holds no window."""
        if not self.live:
            return None
        n = self.rows_q * self.cols_q
        return {'mean': self.total / n, 'geomean': torch.exp(self.logs / n),
                'min': self.low}


def row_cosines(features: torch.Tensor, qgf: torch.Tensor) -> torch.Tensor:
    """[cols, D] reference tiles x [S, R_q, C_q, D] query tiles (both unit
    vectors) -> [S, cols, R_q, C_q] cosines."""
    return torch.einsum('wd,sijd->swij', features, qgf)


def prefetched(fn, items, depth: int):
    """`fn(item)` for each item, in order, with up to `depth` results already
    computed while the caller works on the one before.

    The caller encodes a row on the GPU while this reads the next ones, so the
    slide read and the strip cut (CPU, and openslide drops the GIL for the read)
    overlap with the forward pass instead of alternating with it. ONE thread does
    every read, so the slide handle is never used from two threads. Results come
    back in item order, which is all the accumulators need: nothing here changes
    a number. `depth` 0 calls `fn` inline.

    A failure in `fn` is raised in the caller at the item that failed; leaving
    the loop early (an error downstream) stops the thread."""
    if depth <= 0:
        for item in items:
            yield fn(item)
        return
    box: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def put(value) -> bool:
        while not stop.is_set():
            try:
                box.put(value, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def work():
        try:
            for item in items:
                if not put((None, fn(item))):
                    return
        except BaseException as exc:                     # raised in the caller
            put((exc, None))
            return
        put((None, done))

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        while True:
            exc, value = box.get()
            if exc is not None:
                raise exc
            if value is done:
                return
            yield value
    finally:
        stop.set()
        thread.join()


def stream_windows(slide, regions, grids, ds: float, level: int, arm_specs,
                   qgf: dict, n_shots: int, dims: dict, encoder,
                   device, workers: int = 0, block_rows: int = BLOCK_ROWS,
                   timer: PhaseTimer | None = None) -> tuple:
    """`({rotation: {arm name: {lattice: [WindowAccumulator per region]}}},
    n_tiles)`. `timer` (`--timing`) splits the time into reading, encoding,
    cosines and accumulating. `qgf` is `{rotation: {pooling: [S, R_q, C_q, D]}}` and `dims`
    `{rotation: (R_q, C_q)}`: the query turned to each rotation is its own
    kernel, but a reference tile is encoded once for all of them.

    The reference is read by `SlideReader.read_grid`: `block_rows` main rows of a region
    per read, with the offset rows inside them -- read only when some arm is on
    the full grid -- in `workers` DataLoader workers that read ahead of the
    encoder. Each lattice's tiles of a block are encoded once for every arm
    that reads that lattice; the descriptors stay on the device, are scored a
    row at a time and dropped. An accumulator sums its rows, so the order the
    blocks arrive in changes no number."""
    timer = timer if timer is not None else PhaseTimer()
    need_offset = any(g == 'all' for _, g in arm_specs)
    lattices = ('main', 'offset') if need_offset else ('main',)
    acc = {rot: {token_name(b, g): {} for b, g in arm_specs} for rot in dims}
    users = {lat: [(b, g) for b, g in arm_specs if lat == 'main' or g == 'all']
             for lat in lattices}
    for region, grid in zip(regions, grids):
        for lattice in lattices:
            rows, cols = grid.lattice_dims(lattice)
            for b, g in users[lattice]:
                for rot, (rows_q, cols_q) in dims.items():
                    acc[rot][token_name(b, g)].setdefault(lattice, []).append(
                        WindowAccumulator(rows, cols, n_shots, rows_q, cols_q, device))
    reader = SlideReader(slide, workers=workers).read_grid(
        regions, grids, ds, tile=TILE, offset=need_offset,
        block_rows=block_rows, level=level)
    n_tiles = 0
    blocks = iter(reader)
    while True:
        with timer.phase(READ, fine=True):          # the reader's workers read ahead:
            block = next(blocks, None)              # this is only the wait for one
        if block is None:
            break
        for lattice, tiles, n_rows, cols in (
                ('main', block.main, block.main_rows, block.cols),
                ('offset', block.offset, block.offset_rows, block.cols - 1)):
            if lattice not in lattices or n_rows == 0 or cols <= 0:
                continue
            row_bases = list(dict.fromkeys(b for b, _ in users[lattice]))
            with timer.phase(ENCODE, len(tiles), fine=True):
                pooled = pooled_descriptors(tiles, encoder, row_bases)
            timer.items[READ] = timer.items.get(READ, 0) + len(tiles)
            n_tiles += len(tiles)
            for b in row_bases:
                per_row = pooled[b].to(device).reshape(n_rows, cols, -1)
                for i in range(n_rows):
                    for rot in dims:
                        with timer.phase(COSINE, 1, fine=True):
                            cos = row_cosines(per_row[i], qgf[rot][b])
                        with timer.phase(ACCUM, 1, fine=True):
                            for base, g in users[lattice]:
                                if base == b:
                                    acc[rot][token_name(base, g)][lattice][
                                        block.region].add_row(block.row0 + i, cos)
    return acc, n_tiles


def finalize_windows(acc: dict) -> dict:
    """`{rotation: {arm name: {lattice: [scores dict or None per region]}}}`."""
    return {rot: {name: {lat: [a.scores() for a in accs]
                         for lat, accs in by_lat.items()}
                  for name, by_lat in by_arm.items()}
            for rot, by_arm in acc.items()}


def shot_maps(final_arm: dict, score: str, shot: int, n_regions: int) -> tuple:
    """`(main maps, offset maps)`, one [H_out, W_out] tensor per region, for one
    shot -- the shape `rank_in_main` and `recall_ranks` read. A lattice with no
    windows, or an arm not on the offset lattice, gives empty tensors."""
    def per_region(lattice):
        out = []
        for r in range(n_regions):
            got = final_arm.get(lattice)
            scores = got[r] if got is not None else None
            out.append(scores[score][shot] if scores is not None
                       else torch.empty(0))
        return out
    return per_region('main'), per_region('offset')


def gate_window_stream(seed: int = 0, rows_q: int = 4, cols_q: int = 5) -> tuple:
    """The row-by-row windows against `SlidingWindowSimilarity` on a made-up
    region (9 x 11 main tiles, 8 x 10 offset ones, a `rows_q` x `cols_q` query,
    4 x 5 by default and 5 x 4 for the query turned a quarter, 3 shots), and
    against a decoy: the same rows added one column off. The decoy has to be far
    from the truth or agreement would say nothing. Returns
    ((worst |diff|, decoy |diff|), ok)."""
    rows, cols, n_shots, dim = 9, 11, 3, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator().manual_seed(seed)
    unit = lambda *shape: F.normalize(torch.randn(*shape, generator=gen), dim=-1)  # noqa: E731
    wsi_grid = PatchGrid.from_size(cols * TILE, rows * TILE, TILE, overlap=True)
    main_f = unit(rows, cols, dim)
    off_f = unit(rows - 1, cols - 1, dim)
    flat = torch.zeros(len(wsi_grid), dim)
    for r in range(rows):
        for c in range(cols):
            flat[wsi_grid.flat_index_for_main(r, c)] = main_f[r, c]
    for r in range(rows - 1):
        for c in range(cols - 1):
            flat[wsi_grid.flat_index_for_overlap(r, c)] = off_f[r, c]
    wsi_map = FeaturesMap(wsi_grid, flat)
    q_grid = PatchGrid.from_size(cols_q * TILE, rows_q * TILE, TILE, overlap=False)
    queries = [unit(rows_q * cols_q, dim) for _ in range(n_shots)]
    reference = []
    for q in queries:
        main_sim, off_sim = SlidingWindowSimilarity(FeaturesMap(q_grid, q), wsi_map,
                                                    device)
        reference.append((combine(main_sim), combine(off_sim)))
    qgf = torch.stack([FeaturesMap(q_grid, q).main_feature_grid()
                       for q in queries]).to(device)

    def run(shift):
        got = []
        for lattice_f, lat_rows, lat_cols in ((main_f, rows, cols),
                                              (off_f, rows - 1, cols - 1)):
            acc = WindowAccumulator(lat_rows, lat_cols, n_shots, rows_q, cols_q,
                                    device)
            for y in range(lat_rows):
                cos = row_cosines(lattice_f[y].to(device), qgf)
                acc.add_row(y, torch.roll(cos, shift, dims=1) if shift else cos)
            got.append(acc.scores())
        return got

    def worst(got):
        return max(float((got[k][name][s].cpu() - reference[s][k][name].cpu()).abs().max())
                   for k in (0, 1) for s in range(n_shots) for name in SCORES)

    exact, decoy = worst(run(0)), worst(run(1))
    return (exact, decoy), exact < 1e-5 and decoy > 1e-3


def gate_lattice_geometry() -> tuple:
    """`PatchGrid.lattice_dims` and `.tile_origin` against every `PatchInfo` the grid
    makes, at the downsamples the bench meets (2x and 4x pyramids, a slightly
    off 4, a coarse one). Pure geometry, no slide."""
    checked = 0
    for ds in (1.0, 2.0, 4.00002, 16.0):
        region = SimpleNamespace(x=3000, y=5000, w=int(2000 * ds + 37),
                                 h=int(1500 * ds + 11))
        (grid,) = region_grids([region], ds=ds, level=0, tile_size=TILE,
                               overlap=True)
        for lattice, infos in (('main', grid.main_patch_infos),
                               ('offset', grid.overlap_patch_infos)):
            rows, cols = grid.lattice_dims(lattice)
            if rows * cols != len(infos):
                return f'ds {ds:g} {lattice}: {rows}x{cols} != {len(infos)} tiles', False
            for info in infos:
                if (info.x, info.y) != grid.tile_origin(lattice, info.row,
                                                      info.col):
                    return (f'ds {ds:g} {lattice} ({info.row},{info.col}): '
                            f'PatchGrid says {(info.x, info.y)}'), False
                checked += 1
    return f'{checked} tiles, 4 downsamples', True


#: A strip is read from a level-0 integer, so at a level whose downsample is not
#: a whole number (BRACS: 4.000022) its origin sits up to half a level-0 pixel
#: from where the whole-region read would put it, and the pixels differ by a
#: grey level or three. That is accepted, and measured against a decoy rather
#: than a tolerance: the same rows shifted by ONE whole level pixel. The strip
#: has to be at least this many times closer to the block than the decoy is.
ROW_READ_RATIO = 10


#: Where on the slide `gate_row_reads` looks when the mask offers nothing, or
#: nothing there has texture: fractions of the room a 1024 x 640 block has, in
#: order. Blank glass measures nothing, so a position is tried until one does.
ROW_READ_POSITIONS = ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75),
                      (0.75, 0.75), (0.5, 0.25), (0.5, 0.75), (0.25, 0.5),
                      (0.75, 0.5))


def gate_row_reads(path: str, mask=None, max_ds: float = 16.0) -> tuple:
    """Rows read as strips against the same rows read as one block, on the
    slide's own levels: the main lattice (two rows) and the offset lattice
    (shifted by half a tile in both directions).

    A level is measured at the first place where a shift of one level pixel
    changes the pixels (blank glass would agree with anything): the middle of
    each of the largest tissue regions of `mask` first, then `ROW_READ_POSITIONS`.
    A level with no such place is reported as not measured. Returns
    `(measured, flat, ok)`: `measured` is `[(ds, where, mean|diff|, max|diff|,
    mean|decoy diff|)]`, `flat` the downsamples nothing could be measured at,
    and `ok` needs at least one measured level and every one of them to pass."""
    slide = SafeSlide(path)
    try:
        width0, height0 = slide.dimensions
        need_w, need_h = 4 * TILE, 3 * TILE
        measured, flat = [], []
        for level in levels_up_to(slide.level_downsamples, max_ds):
            ds = float(slide.level_downsamples[level])
            w_lv, h_lv = slide.level_dimensions[level]
            if w_lv < need_w + TILE or h_lv < need_h + TILE:
                continue
            room_x = max(0, width0 - int(need_w * ds))
            room_y = max(0, height0 - int(need_h * ds))
            spots = tissue_spots(mask, width0, height0, int(need_w * ds),
                                 int((2 * TILE + HALF_TILE) * ds))
            spots += [(f'({fx:g}, {fy:g})', int(fx * room_x), int(fy * room_y))
                      for fx, fy in ROW_READ_POSITIONS]
            for where, x0, y0 in spots:
                block = slide.read_region_rgb((x0, y0), level,
                                              (need_w, 2 * TILE + HALF_TILE))
                b16 = block.astype(np.int16)
                # what a one-pixel shift does to these very rows: the decoy
                decoy = []
                for row in range(2):
                    ref = b16[row * TILE:(row + 1) * TILE]
                    decoy.append(np.abs(b16[row * TILE + 1:(row + 1) * TILE + 1]
                                        - ref))
                ref_off = b16[HALF_TILE:HALF_TILE + TILE,
                              HALF_TILE:HALF_TILE + need_w - TILE]
                decoy.append(np.abs(b16[HALF_TILE:HALF_TILE + TILE,
                                        HALF_TILE + 1:HALF_TILE + 1 + need_w - TILE]
                                    - ref_off))
                decoy = np.concatenate([d.ravel() for d in decoy])
                if decoy.mean() < 1.0:                  # flat: a shift is invisible
                    continue
                real = []
                for row in range(2):
                    strip = slide.read_region_rgb(
                        (x0, int(round(y0 + row * TILE * ds))), level,
                        (need_w, TILE))
                    real.append(np.abs(strip.astype(np.int16)
                                       - b16[row * TILE:(row + 1) * TILE]))
                shifted = slide.read_region_rgb(
                    (int(round(x0 + HALF_TILE * ds)),
                     int(round(y0 + HALF_TILE * ds))), level,
                    (need_w - TILE, TILE))
                real.append(np.abs(shifted.astype(np.int16) - ref_off))
                real = np.concatenate([r.ravel() for r in real])
                measured.append((ds, where, float(real.mean()), int(real.max()),
                                 float(decoy.mean())))
                break
            else:
                flat.append(ds)
        ok = bool(measured) and all(m * ROW_READ_RATIO <= d
                                    for _, _, m, _, d in measured)
        return measured, flat, ok
    finally:
        slide.close()


def run_gates(patches, encoder, poolings, path: str = '', mask=None) -> bool:
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

    print('[gate] recall ranks', end='  ', flush=True)
    got, ok_recall = gate_recall_ranks()
    print(f'{"OK" if ok_recall else f"FAIL  got {got}"}')

    print('[gate] truth centre == the turned query\'s grid', end='  ', flush=True)
    (delta_truth, decoy_truth), ok_truth = gate_truth_centre()
    print(f'max|Δ| {delta_truth:.2e} px  unturned grid is {decoy_truth:.0f} px off  '
          f'{"OK" if ok_truth else "FAIL"}')

    print('[gate] stage 2 tables pick the stage\'s windows', end='  ', flush=True)
    n_picked, ok_picked = gate_scored_windows()
    print(f'{n_picked} windows, `ScoredWindows` and `SlidingWinSimRot` pick the same  '
          f'{"OK" if ok_picked else "FAIL"}')

    print('[gate] raw descriptor', end='  ', flush=True)
    (delta, leak), ok_raw = gate_raw_descriptor()
    print(f'cosine == mean per-position cosine, max|Δ|={delta:.2e}  '
          f'prefix leak {leak:.0e}  {"OK" if ok_raw else "FAIL"}')

    print('[gate] window stream == SlidingWindowSimilarity', end='  ', flush=True)
    (exact, decoy), ok_stream = gate_window_stream()
    (exact_t, decoy_t), ok_turned = gate_window_stream(rows_q=5, cols_q=4)
    ok_stream = ok_stream and ok_turned
    print(f'max|Δ| {exact:.2e}  decoy (one column off) {decoy:.2e}  '
          f'turned query 5x4: {exact_t:.2e} / {decoy_t:.2e}  '
          f'{"OK" if ok_stream else "FAIL"}')

    print('[gate] lattice geometry', end='  ', flush=True)
    said, ok_lattice = gate_lattice_geometry()
    print(f'{said}  {"OK" if ok_lattice else "FAIL"}')

    ok_rows = True
    if path:
        print('[gate] row reads: strip vs block, against a one-pixel shift')
        levels, flat, ok_rows = gate_row_reads(path, mask)
        for ds, where, mean, worst, decoy in levels:
            print(f'         ds {ds:<10.6g} at {where:<18}  mean|Δ| {mean:.4f}  '
                  f'max|Δ| {worst}/255  (one pixel off: mean|Δ| {decoy:.2f})')
        for ds in flat:
            print(f'         ds {ds:<10.6g} no textured area at any of the '
                  f'tissue regions or {len(ROW_READ_POSITIONS)} positions -- not measured')
        print(f'         strip at least {ROW_READ_RATIO}x closer than the decoy on '
              f'{len(levels)} level(s)  {"OK" if ok_rows else "FAIL"}')
    return (ok_base and ok_reduce and ok_concat and ok_grid and ok_recall
            and ok_truth and ok_picked and ok_raw and ok_stream and ok_lattice and ok_rows)


# ══════════════════════════════════════════════════════════════════════════════
#  One (slide, level)
# ══════════════════════════════════════════════════════════════════════════════

def region_of(regions, x0: int, y0: int, w: int, h: int):
    """Index of the region that holds the rectangle, or None. Regions are found
    by GEOMETRY: the placer's regions and this bench's are two lists made by two
    filters, so their indices do not line up."""
    rect = ReadRect(int(x0), int(y0), int(w), int(h))
    for i, r in enumerate(regions):
        if rect.inside(r.x, r.y, r.x + r.w, r.y + r.h):
            return i
    return None


def answers_for(camera, x: int, y: int, params: dict, region, ds: float,
                shape) -> dict:
    """The two grid points nearest the window the TURNED query's tile grid
    covers, at the rotation that turns the shot upright.

    Stage 2 turns the shot by `matching_rotation` and cuts whole tiles from the
    turned photo's top-left, so the window it stands for is the centre of THAT
    grid -- `BenchCommon.query_grid_centre`, the function the pipeline bench's
    truth uses -- and the window's sides are the turned shot's (`rows_t x
    cols_t`, swapped at 90 / 270). `found` carries them, with the rotation:
    `rot`, `rows_t`, `cols_t`. At rotation 0 and scale 1 this is the FoV's
    top-left exactly, which the caller checks.

    `query_grid_centre` inverts the shot's full angle, jitter included; lens
    distortion and the stage shift are not inverted (`output_to_level0`).
    """
    rot = matching_rotation(params['rot_deg'])
    height, width = shape[:2]
    rows_t, cols_t = ((height, width) if (rot // 90) % 2 == 0
                      else (width, height))
    rows_t, cols_t = rows_t // TILE, cols_t // TILE
    cx0, cy0 = query_grid_centre(camera, x, y, params, shape, TILE)
    cx_n = (cx0 - region.x) / ds
    cy_n = (cy0 - region.y) / ds
    x_tl, y_tl = cx_n - cols_t * HALF_TILE, cy_n - rows_t * HALF_TILE
    found = grid_answers(x_tl, y_tl)
    found['x_n'], found['y_n'] = x_tl, y_tl
    found['rot'], found['rows_t'], found['cols_t'] = rot, rows_t, cols_t
    return found


def window_exists(grid, found: dict, rows_q: int, cols_q: int) -> bool:
    """Are both answer windows inside this region's sliding-window output?

    H_out = R_wsi - R_q + 1 on the main grid, and one less on each axis for the
    overlap grid, which is what `SlidingWindowSimilarity` returns.
    """
    main_h = grid.grid_rows - rows_q + 1
    main_w = grid.grid_cols - cols_q + 1
    ovlp_h, ovlp_w = main_h - 1, main_w - 1
    mr, mc = found['main_rc']
    orow, ocol = found['ovlp_rc']
    return (0 <= mr < main_h and 0 <= mc < main_w
            and 0 <= orow < ovlp_h and 0 <= ocol < ovlp_w)


# ══════════════════════════════════════════════════════════════════════════════
#  Arms: which pooling, on which lattices
# ══════════════════════════════════════════════════════════════════════════════

#: An arm says which lattices it is scored on by its NAME: `cls_avg-M` is the
#: main lattice only, `cls_avg` is main + offset. Only the capital M is
#: recognised; `-m` is refused, so there is one spelling.
MAIN_SUFFIX = '-M'


def parse_token(token: str, allowed) -> tuple:
    """`'cls_avg-M'` -> `('cls_avg', 'main')`; `'cls_avg'` -> `('cls_avg', 'all')`."""
    if token.endswith('-m'):
        raise ValueError(f'{token!r}: the main-only suffix is -M (capital), '
                         f'not -m')
    grid = 'main' if token.endswith(MAIN_SUFFIX) else 'all'
    base = token[:-len(MAIN_SUFFIX)] if grid == 'main' else token
    if base not in allowed:
        raise ValueError(f'{token!r}: {base!r} is not one of {list(allowed)}')
    return base, grid


def token_name(base: str, grid: str) -> str:
    return base + (MAIN_SUFFIX if grid == 'main' else '')


def _pools(maps_by_rot: dict) -> tuple:
    """`(main pools, offset pools)`: every region's windows of every rotation, one
    flat tensor each -- the candidates the stage ranks together."""
    main_pools, ovlp_pools = [], []
    for main_maps, ovlp_maps in maps_by_rot.values():
        main_pools += [m.flatten() for m in main_maps if m.numel()]
        ovlp_pools += [m.flatten() for m in ovlp_maps if m.numel()]
    return main_pools, ovlp_pools


def recall_ranks(maps_by_rot: dict, region: int, found: dict) -> dict:
    """The two answers of one query, ranked in the pools they can be judged in.

    Only meaningful for a full-grid arm, whose maps hold main AND offset windows.
    `maps_by_rot` is `{rotation: (main maps, offset maps)}`: the two answers are
    the windows at `found['rot']`, and every pool holds the windows of all the
    rotations, as the stage ranks them. `rank_main_in_main` is what
    `rank_in_main` returns; the other three are what the offset grid adds:

        main answer   in the main windows      rank_main_in_main    pool
        offset answer in the offset windows    rank_offset_in_offset pool_offset
        both          in main + offset         rank_main_in_all
                                               rank_offset_in_all   pool_all
    """
    main_maps, ovlp_maps = maps_by_rot[found['rot']]
    mr, mc = found['main_rc']
    orow, ocol = found['ovlp_rc']
    main_answer = float(main_maps[region][mr, mc])
    ovlp_answer = float(ovlp_maps[region][orow, ocol])
    main_pools, ovlp_pools = _pools(maps_by_rot)
    both = main_pools + ovlp_pools
    count = lambda pools: int(sum(int(m.numel()) for m in pools))     # noqa: E731
    return {'rank_main_in_main': rank_of(main_answer, main_pools),
            'pool': count(main_pools),
            'rank_offset_in_offset': rank_of(ovlp_answer, ovlp_pools),
            'pool_offset': count(ovlp_pools),
            'rank_main_in_all': rank_of(main_answer, both),
            'rank_offset_in_all': rank_of(ovlp_answer, both),
            'pool_all': count(both)}


def rank_in_main(maps_by_rot: dict, region: int, found: dict) -> tuple:
    """`(rank, pool)` of the nearest MAIN window (at `found['rot']`) among the
    main windows of every rotation.

    This is what an arm on a main-only grid can be scored on, so it is what every
    arm is scored on: an `all` arm's offset windows are left out of the pool, and
    the two kinds of arm meet on the same candidates and the same truth."""
    main_maps, _ = maps_by_rot[found['rot']]
    mr, mc = found['main_rc']
    answer = float(main_maps[region][mr, mc])
    pools, _ = _pools(maps_by_rot)
    return rank_of(answer, pools), int(sum(int(m.numel()) for m in pools))


def truth_score(maps_by_rot: dict, region: int, found: dict, grid: str) -> float:
    """The score of the truth window under one arm: the main window nearest the
    shot for a main-only arm, the nearer of the main and offset windows for a
    full-grid arm -- the rule `nearest_window` applies. The number the pipeline
    bench's stage-2 `truth_score` is, for the same FoV and recipe."""
    main_maps, ovlp_maps = maps_by_rot[found['rot']]
    if grid == 'main' or found['truth'] == 'main':
        mr, mc = found['main_rc']
        return float(main_maps[region][mr, mc])
    orow, ocol = found['ovlp_rc']
    return float(ovlp_maps[region][orow, ocol])


# ══════════════════════════════════════════════════════════════════════════════
#  The stage 2 tables an arm writes
#
#  An arm on the full grid whose score the stage has (`mean`, `geomean`) IS a
#  `SlidingWinSimRot`: a pooling, a score, the same windows. So it gets the stage 2
#  entry the pipeline bench writes -- the same tables, the same place in the tree
#  (`.../render=/stage1=oracle/stage2/`, this job's) -- and the demo page, stage 3
#  and the plots read a window bench's arm as they read a pipeline run's:
#
#    output     the first STAGE2_K windows of the FoV, every rotation, lattice
#               and region together, picked by the stage's own `candidate_set`
#    truth      the window the FoV stands for, its score, and where it ranks
#               among all the windows and among the main ones (+ the pools)
#    tile_sims  the first STAGE2_TILE_SIMS windows' per-tile cosines
#    truth_sim  the truth window's
#
#  The window bench scores a FoV at the level it was placed at, so the entry is
#  under `stage1=oracle`. EVERY arm the run has is written, whatever it is: an arm
#  the stage can be set to is the pipeline's own entry (its id is the config's),
#  any other -- `min`, a `-M` arm, a pooling only this bench computes -- an entry
#  of its own, with an id made from what it is. A new arm needs nothing added.
# ══════════════════════════════════════════════════════════════════════════════

STAGE2_K: int = 100            # windows kept per FoV: the stage's recipe `k`
STAGE2_TILE_SIMS: int = 10     # windows, besides the truth one, with their per-tile cosines


class ScoredWindows(SlidingWinSimRot):
    """A `SlidingWinSimRot` whose window scores were computed before it was given
    them: the bench streams the reference and keeps one score per window, not
    the per-tile similarity maps the retriever reduces. Its `sim_maps_by_rot`
    holds those scores (`[rows, cols]` per rotation, region and lattice) and
    `_window_scores` yields them as they are, so `candidate_set` -- the stage's
    own picking, its per-grid top k and its `min_sep_tiles` -- runs unchanged on
    them. The stage's code is not copied and not touched."""

    def _window_scores(self, sim_maps):
        for ri, (main, offset) in enumerate(sim_maps):
            for lattice, scores in (('main', main), ('offset', offset)):
                if scores.numel():
                    yield ri, lattice, scores


class Stage2Arm:
    """One arm on its way to a stage 2 entry. `entry_id` names the entry and `rec`
    is its record. An arm the stage can be set to (the full grid, a pooling the
    encoder names, `mean` or `geomean`) is the pipeline's own `slidewin` entry,
    its id the config's. Any other arm -- `min`, a `-M` arm, a pooling only this
    bench computes -- has an id made from what it is (encoder, pooling, lattices,
    score, mask): its own entry, in the same tables, that no pipeline run asks
    for. `tables` is what to fill; None when the entry is already there."""

    def __init__(self, base: str, grid: str, score: str, entry_id: str,
                 rec: dict, cfg: SlidingWinSimRotConfig) -> None:
        self.base: str = base
        self.grid: str = grid
        self.score: str = score
        self.name: str = token_name(base, grid)
        self.entry_id: str = entry_id
        self.rec: dict = rec
        self.cfg: SlidingWinSimRotConfig = cfg
        self.tables: Optional[Tables] = None


def stage2_arms(args, arm_specs) -> list[Stage2Arm]:
    """Every arm of the run as a stage 2 entry to write (`Stage2Arm`)."""
    seg: str = args.mask_cfg.seg_id()
    region: str = args.mask_cfg.region_id()
    out: list[Stage2Arm] = []
    base: str
    grid: str
    for base, grid in arm_specs:
        score: str
        for score in SCORES:
            placeholder = SlidingWinSimRotConfig(
                encoder=args.encoder_cfg, tile_size=TILE, overlap=True, k=STAGE2_K,
                min_sep_tiles=1.0, score='mean')
            cfg = None
            if grid == 'all' and base != RAW and score in WINDOW_SCORES:
                try:
                    cfg = SlidingWinSimRotConfig(
                        encoder=dataclasses.replace(args.encoder_cfg, pooling=base),
                        tile_size=TILE, overlap=True, k=STAGE2_K, min_sep_tiles=1.0,
                        score=score)
                except ValueError:                   # the encoder has no such pooling
                    cfg = None
            if cfg is not None:
                stage = Stage(2, 'slidewin', f'{args.encoder}-{base.replace("_", "-")}-{score}',
                              cfg, SlidingWinSimRot, {}, also_id=(seg, region))
                rec = dict(stage.record(stage1=ORACLE, seg=seg, region=region),
                           truth=TRUTH_VERSION, limit=0)
                out.append(Stage2Arm(base, grid, score, stage.id, rec, cfg))
                continue
            hexid: str = short_id([args.encoder_cfg.identity_id(), base, grid, score,
                                   seg, region, 'window-arm'])
            label: str = f'window-{args.encoder}-{token_name(base, grid).replace("_", "-")}-{score}-{hexid}'
            rec = dict(id=hexid, parts=[f'pooling={base}', f'grid={grid}', f'score={score}'],
                       versions={'window_arm': ROW_VERSION},
                       upstream={'stage1': ORACLE, 'seg': seg, 'region': region},
                       env=environment(), label=label, truth=TRUTH_VERSION, limit=0)
            out.append(Stage2Arm(base, grid, score, label, rec, placeholder))
    return out


def stage2_tables(slide, lv: int, ds: float, grids, valid, final, qgf, dims,
                  arm_specs, args, encoder, device, base_addr,
                  seconds_each: float) -> None:
    """Write the stage 2 entry of every arm that has none yet, for one (slide,
    level). `valid` are the FoVs `run_slide_level` scored, `final` their window
    scores (`finalize_windows`), `qgf` their query descriptors per rotation."""
    arms: list[Stage2Arm] = stage2_arms(args, arm_specs)
    entry = base_addr.at(stage1=ORACLE).entry('stage2')
    todo: list[Stage2Arm] = []
    for arm in arms:
        if status(entry, arm.entry_id, arm.rec, ROLES[2]) != 'hit':
            arm.tables = Tables(entry, arm.entry_id, arm.rec, ROLES[2])
            todo.append(arm)
    if not todo:
        print(f'  L{lv}  stage 2 tables: all {len(arms)} arms are already there', flush=True)
        return
    n_regions: int = len(grids)
    frame: dict = dict(level=lv, ds=ds, irretrievable_lvl='', alter_lvl='')
    #: what the per-tile cosines are asked of: (arm, FoV, rank or None for the truth, window)
    wanted: list[tuple] = []
    queries: dict = {}

    for s_idx, (fov_id, shot, ri, found) in enumerate(valid):
        qc = QueryPatchContainer(shot[1])
        qc.extract_all(TILE, overlap=True)
        queries[s_idx] = qc
        for arm in todo:
            maps = {rot: shot_maps(final[rot][arm.name], arm.score, s_idx, n_regions)
                    for rot in dims}
            ret = ScoredWindows(arm.cfg, 'cpu')
            ret.grids, ret.level, ret.ds = list(grids), lv, ds
            ret.sim_maps_by_rot = {rot: list(zip(*m)) for rot, m in maps.items()}
            cs = ret.candidate_set(STAGE2_K)
            if not len(cs):
                arm.tables.add('output', fov_id, [dict(frame, error='no window', t_s=0.0)])
                continue
            arm.tables.add('output', fov_id, [dict(r, **frame, error='', t_s=round(seconds_each, 3))
                                              for r in cs.rows(qc)])
            # the window the FoV stands for: the nearer of its two grid answers --
            # on a main-only arm the main one, the only lattice it scored
            if arm.grid == 'all':
                ranks = recall_ranks(maps, ri, found)
            else:
                rank_m, pool_m = rank_in_main(maps, ri, found)
                ranks = {'rank_main_in_main': rank_m, 'pool': pool_m,
                         'rank_main_in_all': rank_m, 'rank_offset_in_all': rank_m,
                         'pool_all': pool_m}
            lattice, (row, col) = (('main', found['main_rc'])
                                   if arm.grid != 'all' or found['truth'] == 'main'
                                   else ('offset', found['ovlp_rc']))
            rot = found['rot']
            truth_score = float((maps[rot][0] if lattice == 'main' else maps[rot][1])[ri][row, col])
            window = Candidate(ri, lattice, row, col, rot, truth_score)
            box = CandidateSet(candidates=(window,), level=lv, ds=ds, grids=tuple(grids),
                               irretrievable_lvl=(), alter_lvl=()).rows(qc)[0]
            arm.tables.add('truth', fov_id, [dict(
                truth_region=ri, truth_lattice=lattice, truth_row=row, truth_col=col,
                truth_rotation=rot, truth_score=truth_score,
                truth_dist_l0=found['d_main' if lattice == 'main' else 'd_overlap'] * ds,
                truth_main_dist_l0=found['d_main'] * ds,
                truth_x0=box['x0'], truth_y0=box['y0'], truth_w0=box['w0'], truth_h0=box['h0'],
                truth_level=lv,
                truth_rank_all=(ranks['rank_main_in_all'] if lattice == 'main'
                                else ranks['rank_offset_in_all']),
                truth_rank_main_in_all=ranks['rank_main_in_all'],
                pool_all=ranks['pool_all'], pool_main=ranks['pool'],
                truth_rank_main=ranks['rank_main_in_main'])])
            wanted.append((arm, s_idx, fov_id, None, window))
            wanted += [(arm, s_idx, fov_id, rank, c)
                       for rank, c in enumerate(cs.candidates[:STAGE2_TILE_SIMS], 1)]

    # the tiles under those windows, read and encoded once each, for every arm
    keys: dict = {}
    for arm, s_idx, _, _, c in wanted:
        if arm.base == RAW:
            continue
        rows_w, cols_w = dims[c.rotation]
        for i in range(rows_w):
            for j in range(cols_w):
                keys.setdefault((c.region_index, c.lattice, c.row + i, c.col + j), len(keys))
    bases: list[str] = list(dict.fromkeys(arm.base for arm in todo if arm.base != RAW))
    table: dict = {}
    if keys:
        points = [grids[ri].tile_origin_l0(lat, r, c) for (ri, lat, r, c) in keys]
        reader = SlideReader(slide, workers=args.budget.workers)
        for batch, index in reader.read_points(points, ReadSpec(TILE, TILE), ds, level=lv,
                                               batch=256):
            got = pooled_descriptors(batch, encoder, bases)
            for b in bases:
                if b not in table:
                    table[b] = torch.empty(len(keys), got[b].shape[1], device=device)
                table[b][index.to(device)] = got[b].float().to(device)

    worst: float = 0.0
    for arm, s_idx, fov_id, rank, c in wanted:
        if arm.base == RAW:
            continue                      # 393,216 numbers a tile: its tile tables are left empty
        rows_w, cols_w = dims[c.rotation]
        idx = torch.tensor([[keys[(c.region_index, c.lattice, c.row + i, c.col + j)]
                             for j in range(cols_w)] for i in range(rows_w)], device=device)
        cos = (table[arm.base][idx] * qgf[c.rotation][arm.base][s_idx]).sum(-1).cpu().numpy()
        rows = [dict(q_row=i, q_col=j, ref_row=c.row + i, ref_col=c.col + j,
                     cosine=float(cos[i, j]))
                for i in range(rows_w) for j in range(cols_w)]
        if rank is None:
            arm.tables.add('truth_sim', fov_id, rows)
        else:
            arm.tables.add('tile_sims', fov_id, [dict(rank=rank, **r) for r in rows])
            if arm.score == 'mean':               # the window's score is the mean of these
                worst = max(worst, abs(float(cos.mean()) - c.score))
    for arm in todo:
        arm.tables.write()
    print(f'  L{lv}  stage 2 tables: {len(todo)} arms x {len(valid)} FoV -> {entry.dir}  '
          f'(re-scored from the tile cosines: largest |diff| to the streamed score {worst:.1e})',
          flush=True)


def run_slide_level(slide, dataset, stem, level, mask, args, encoder, arms,
                    bases, device, supply, shots, job: str) -> list:
    """Every FoV of one (slide, level), scored by every arm. `shots` are this
    level's `(index, meta, image, params)` out of the slide's one draw
    (`supply`, the draw every bench takes its FoVs from); `index` is the FoV's
    row in the draw, and is the row's `fov_id`."""
    ds = float(slide.level_downsamples[level])
    if not shots:
        # No region holds a FoV this size at this level -- a coarse level on a
        # small tissue section. Not an error: say so.
        print(f'  L{level} (ds {ds:g}): no FoV position -- skipped', flush=True)
        return []
    timer = PhaseTimer(fine=bool(getattr(args, 'timing', False)))
    camera = supply.camera_for(shots[0][1].ds)
    bank = {index: (meta, image, params) for index, meta, image, params in shots}
    rows_q = camera.output_h // TILE
    cols_q = camera.output_w // TILE
    # the query's tile grid at each rotation the stage searches: the sides swap
    # at 90 / 270
    dims = {rot: ((rows_q, cols_q) if (rot // 90) % 2 == 0 else (cols_q, rows_q))
            for rot in ROTATIONS}

    # The regions and their grids come from geometry alone. Which FoVs have an
    # answer window is decided here, before any tile is encoded.
    timer.start(PLACE)
    lv, ds = SlideReader(slide).native_scale(ds=ds)
    regions = mask.patchable(TILE * ds).tissue_regions
    grids = region_grids(regions, ds=ds, level=lv, tile_size=TILE, overlap=True)
    valid, n_no_region, n_no_window = [], 0, 0
    for fov_id, shot in bank.items():
        meta, _, params = shot
        gx, gy = meta.fov_rect[0], meta.fov_rect[1]
        i = region_of(regions, gx, gy, camera.rect_w_l0,
                      camera.rect_h_l0)
        if i is None:
            n_no_region += 1
            continue
        found = answers_for(camera, gx, gy, params, regions[i], ds, shot[1].shape)
        if not window_exists(grids[i], found, found['rows_t'], found['cols_t']):
            n_no_window += 1
            continue
        if (float(params['rot_deg']) % 360 == 0 and float(params['scale']) == 1.0
                and float(params.get('angle_jitter', 0.0)) == 0.0):
            # The one place this arithmetic can be checked against something
            # else: an upright, unscaled, unjittered shot has its window at the
            # FoV's own top-left. If the centre mapping is wrong it is wrong here first.
            want_x = (gx - regions[i].x) / ds
            want_y = (gy - regions[i].y) / ds
            if abs(found['x_n'] - want_x) > 1.5 or abs(found['y_n'] - want_y) > 1.5:
                raise AssertionError(
                    f'{stem} L{level} FoV {fov_id}: the window centre maps to '
                    f'({found["x_n"]:.1f}, {found["y_n"]:.1f}) but an upright '
                    f'shot starts at ({want_x:.1f}, {want_y:.1f})')
        valid.append((fov_id, shot, i, found))
    timer.stop(PLACE, len(bank))
    if not valid:
        print(f'  L{level}  0/{len(bank)} FoV with an answer window -- skipped',
              flush=True)
        return []

    timer.start(QUERY)
    # QGF: the query's tile features, once per shot, pooling and rotation. The
    # shot is turned the way the stage turns it (`np.rot90`, counter-clockwise)
    # and cut into whole tiles from the turned photo's top-left.
    per_base = {rot: {b: [] for b in bases} for rot in dims}
    for _, shot, _, _ in valid:
        for rot, (want_rows, want_cols) in dims.items():
            turned = shot[1] if rot == 0 else np.rot90(shot[1], k=rot // 90)
            container = QueryPatchContainer(turned)
            container.extract_all(TILE, overlap=False)   # only the main kernel is
            if (container.grid.grid_rows, container.grid.grid_cols) != (want_rows, want_cols):
                raise AssertionError(
                    f'{stem} L{level}: a query turned {rot} cuts into '
                    f'{container.grid.grid_rows}x{container.grid.grid_cols} tiles, '
                    f'the camera says {want_rows}x{want_cols}')
            pooled = pooled_descriptors(list(container), encoder, bases)  # ever used
            for b in bases:
                per_base[rot][b].append(
                    FeaturesMap(container.grid, pooled[b]).main_feature_grid())
    qgf = {rot: {b: torch.stack(v).to(device) for b, v in by_base.items()}
           for rot, by_base in per_base.items()}
    timer.stop(QUERY, len(valid) * sum(r * c for r, c in dims.values()))

    # RGF: streamed one tile row at a time, scored as it goes.
    timer.start(STREAM)
    acc, n_tiles = stream_windows(slide, regions, grids, ds, level, arms, qgf,
                                  len(valid), dims, encoder, device,
                                  workers=args.budget.workers,
                                  block_rows=args.block_rows, timer=timer)
    timer.stop(STREAM, n_tiles)
    timer.start(RANK)
    final = finalize_windows(acc)
    del acc

    rows = []
    for s_idx, (fov_id, shot, i, found) in enumerate(valid):
        for base, grid in arms:
            name = token_name(base, grid)
            for score in SCORES:
                maps = {rot: shot_maps(final[rot][name], score, s_idx, len(regions))
                        for rot in dims}
                if grid == 'all':
                    ranks = recall_ranks(maps, i, found)
                    rank_main, pool = ranks['rank_main_in_main'], ranks['pool']
                    extra = {k: ranks[k] for k in RECALL_COLUMNS}
                else:
                    rank_main, pool = rank_in_main(maps, i, found)
                    extra = {k: '' for k in RECALL_COLUMNS}
                rows.append({
                    # First column on purpose: `--report-only` refuses a CSV set
                    # that mixes encoders on this key.
                    'encoder': encoder_tag(args.encoder, args.head),
                    'dataset': dataset, 'slide': stem, 'level': level,
                    'ds': ds, 'fov_id': fov_id, 'pool': pool,
                    'd_main': round(found['d_main'], 2),
                    'd_overlap': round(found['d_overlap'], 2),
                    'white_frac': round(float(shot[0].score), 4),
                    'bucket': shot[0].bucket,
                    'rot_deg': float(shot[2]['rot_deg']),
                    'scale': round(float(shot[2]['scale']), 4),
                    'source': 'stream', 'grid': grid,
                    'arm': f'{name}+{score}',
                    # rank_overlap repeats rank_main ON PURPOSE: the shared
                    # `truth_rank` reads `rank_main` when the main point is the
                    # nearer one and `rank_overlap` otherwise, and here both are
                    # the main-only rank, so every table below is on main only.
                    'rank_main': rank_main, 'rank_overlap': rank_main,
                    # where the truth window is (the rotation that turns the shot
                    # upright) and what it scores: the number the pipeline bench's
                    # stage-2 `truth` row holds for the same FoV and recipe
                    'truth_rot': found['rot'],
                    'truth_score': round(truth_score(maps, i, found, grid), 6),
                    **extra})

    timer.stop(RANK, len(rows))
    base = [r for r in rows if r['arm'] == BASELINE]
    truths = sorted(truth_rank(r) for r in base)
    # The truth's mean percentile. A uniformly random window sits at 0.500, so
    # near 0.5 means the coordinate mapping is broken and every arm is ranking
    # noise.
    truth_pct = float(np.mean([truth_rank(r) / max(1, r['pool']) for r in base])
                      ) if base else float('nan')
    skipped = (f'  skipped: {n_no_region} outside every region, '
               f'{n_no_window} with an answer window off the grid'
               if n_no_region or n_no_window else '')
    print(f'  L{level}  {len(base)}/{len(bank)} FoV   baseline rank_truth med '
          f'{truths[len(truths) // 2] if truths else 0:,}   truth_pctile '
          f'{truth_pct:.4f} (random=0.5000){skipped}', flush=True)
    timer.start(TABLES)
    stage2_tables(slide, lv, ds, grids, valid, final, qgf, dims, arms, args, encoder,
                  device, supply.render_address.on(job),
                  timer.seconds.get(STREAM, 0.0) / max(1, len(valid)))
    del qgf
    timer.stop(TABLES, len(valid))
    print(timer.report(level, len(valid)), flush=True)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Derived metrics -- everything reads only the stored integers
#
#  They live in utilities/dump_function/RetrievalReport.py because bench_tile_retrieval asks
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


def read_rows(paths, quiet: bool = False) -> list:
    integers = {'level', 'fov_id', 'pool', 'rank_main', 'rank_overlap', 'truth_rot',
                *RECALL_COLUMNS}
    floats = {'d_main', 'd_overlap', 'white_frac', 'ds', 'rot_deg', 'scale',
              'truth_score'}
    rows = []
    for path in paths:
        with open(path, newline='') as handle:
            for raw in csv.DictReader(handle):
                row = {}
                for key, value in raw.items():
                    if key in integers or key in floats:
                        row[key] = (value if value == '' else
                                    int(float(value)) if key in integers
                                    else float(value))
                    else:
                        row[key] = value
                rows.append(row)
        if not quiet:
            print(f'  read {path}')
    return rows


#: The code that turns configs into a row (ConfigIdentity rule 3); bump it when
#: the same configs would produce other numbers, and say why in TODO.log.
ROW_VERSION = 2      # 2: also the stage 2 tables (1: every rotation searched, the truth at the turned query's grid)


def config_id(args, arm_specs, configs: dict) -> str:
    """The id of everything that decides a row's numbers. The parts of a run
    live under `parts-<this>/`, so a resume can only ever pick up parts made
    with the same arms, FoVs, slides, seed, camera, mask and encoder; a
    different setting gets a different directory rather than being mixed in.

    `configs` maps a name to each resolved config (sampler, camera, mask,
    encoder); each enters by its own `identity_id`, so its NOT_IDENTITY fields
    do not split a run and every identity field does."""
    parts = [f'{name}={enc(cfg.identity_id())}'
             for name, cfg in sorted(configs.items())]
    parts += [f'arms={enc(sorted(f"{b}/{g}" for b, g in arm_specs))}',
              f'datasets={enc(list(args.datasets))}', f'n_wsi={enc(args.n_wsi)}',
              f'split={enc(args.split)}', f'max_ds={enc(args.max_ds)}',
              f'sensor={enc(list(args.sensor))}', f'version={enc(ROW_VERSION)}']
    return short_id(parts)


def part_path(parts_dir: Path, stem: str, level: int) -> Path:
    return parts_dir / f'{stem.replace(os.sep, "_")}__L{level}.csv'


def write_part(rows: list, path: Path) -> None:
    """One (slide, level)'s rows, written whole or not at all: to a temporary
    name, then renamed. A part that exists is a finished one, which is what lets
    a run be resumed. A level with no FoV writes an empty file -- finished, and
    nothing to report."""
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', newline='') as handle:
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(handle, fieldnames=keys, restval='')
            writer.writeheader()
            writer.writerows(rows)
    os.replace(tmp, path)


def peak_memory_line() -> str:
    """Peak host RSS of the whole process so far (it only ever rises) and the GPU
    peak since the last call."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2   # KB -> GB
    gpu = ''
    if torch.cuda.is_available():
        gpu = f'  GPU peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB'
        torch.cuda.reset_peak_memory_stats()
    return f'peak RSS {rss:.1f} GB (whole process so far){gpu}'


def random_recall(pool_main: int, pool_offset: int, pool_all: int, k: int) -> tuple:
    """Expected (main, offset, all) recall@k of a RANDOM ranking on these pools.

    One target in a pool of P is in the top k with probability min(1, k/P). Two
    targets, both in the mixed pool, are both outside it with probability
    (P-k)(P-k-1) / (P(P-1)), so `all` starts near twice `main` before any
    scoring happens -- the number the rise from main to all is read against."""
    one = lambda pool: min(1.0, k / max(1, pool))                     # noqa: E731
    miss = (max(0, pool_all - k) * max(0, pool_all - k - 1)
            / max(1, pool_all * (pool_all - 1)))
    return one(pool_main), one(pool_offset), 1.0 - miss


#: Candidate counts a full-grid arm's rows carry beyond `pool` (the main windows):
#: what the level tables show as extra columns.
LEVEL_EXTRAS = (('offset windows', 'pool_offset'),
                ('main+offset windows', 'pool_all'))


def recall_block(rows: list, arms, emit=print) -> None:
    """recall_main / recall_offset / recall_all @k, per full-grid arm, one level.

    `rows` are the full-grid rows of one level. Pools differ between slides, so
    a fixed k is the axis (it is also what stage 3 pays for), not a fraction.
    One grid, a group of rows per arm: the three recalls, then the share of the
    hits that the main window answered first. `random` is the recall of a random
    ranking on the same pools."""
    by_arm = group_by(rows, ['arm'])
    first = next((by_arm[(a,)] for a in arms if (a,) in by_arm), None)
    if first is None:
        return
    body, rules = [], []
    rand = [[random_recall(r['pool'], r['pool_offset'], r['pool_all'], k)
             for r in first] for k in K_FIXED]
    for j, name in enumerate(('main', 'offset', 'all')):
        body.append(['random' if j == 0 else '', f'recall_{name}']
                    + [pct(float(np.mean([v[j] for v in rand[i]]))).strip()
                       for i in range(len(K_FIXED))])
    rules.append(len(body) - 1)
    ties = 0
    for arm in arms:
        subset = by_arm.get((arm,))
        if not subset:
            continue
        rm = np.array([r['rank_main'] for r in subset])
        ro = np.array([r['rank_offset_in_offset'] for r in subset])
        am = np.array([r['rank_main_in_all'] for r in subset])
        ao = np.array([r['rank_offset_in_all'] for r in subset])
        best = np.minimum(am, ao)
        ties += int((am == ao).sum())
        for j, (label, ranks) in enumerate((('recall_main', rm),
                                            ('recall_offset', ro),
                                            ('recall_all', best))):
            body.append([arm if j == 0 else '', label]
                        + [pct(float((ranks <= k).mean())).strip() for k in K_FIXED])
        shares = []
        for k in K_FIXED:
            hit = best <= k
            if not hit.any():
                shares.append('-')
                continue
            from_main = float((am[hit] <= ao[hit]).mean()) * 100
            shares.append(f'{from_main:.0f}:{100 - from_main:.0f}')
        body.append(['', 'all: main:offset'] + shares)
        rules.append(len(body) - 1)
    grid_table(['arm', 'metric'] + [f'k={k}' for k in K_FIXED], body,
               aligns='ll' + 'r' * len(K_FIXED), rule_after=rules, emit=emit)
    n = sum(len(v) for v in by_arm.values())
    emit(f'main and offset answers tied in the mixed pool on {ties}/{n} '
         f'(arm, FoV) rows; a tie counts for main')


def recall_report(rows: list, arms, emit=print) -> None:
    """What the offset grid adds: recall of the two answers, per ds.

    Full-grid arms only: a main-only arm has no offset windows to rank. Per level
    and not over all levels, because k means something different in a pool of a
    few hundred windows and in one of a hundred thousand."""
    full = [r for r in rows if r.get('grid') == 'all'
            and r.get('rank_offset_in_offset') not in ('', None)]
    if not full:
        emit('\n(no full-grid arm ran: no main vs offset vs all recall table)')
        return
    emit(f'\n{"=" * 90}\nrecall -- main vs offset vs all (main + offset), '
         f'full-grid arms\n{"=" * 90}')
    for level, subset in group_levels(full).items():
        print_level_heading(subset, level, emit=emit, extras=LEVEL_EXTRAS)
        emit('')
        recall_block(subset, arms, emit=emit)


def assemble(parts_dir: Path, out_dir: Path, arms, per_slide: bool) -> bool:
    """Every part of a run -> `window_retrieval.csv` and the tables. False when
    there is nothing to assemble."""
    parts = sorted(parts_dir.glob('*.csv'))
    all_rows = read_rows(parts, quiet=True)
    print(f'assembling {len(parts)} part(s) from {parts_dir}')
    write_csv(all_rows, out_dir / 'window_retrieval.csv')
    if not all_rows:
        return False
    report(attach_baseline(all_rows, BASELINE), arms, BASELINE,
           per_slide=per_slide, level_extras=LEVEL_EXTRAS)
    recall_report(all_rows, arms)
    return True


def main() -> int:
    # allow_abbrev=False: the config flags are prefixes of one another, and an
    # abbreviation would set a different field without a word (see ConfigArgs).
    parser = argparse.ArgumentParser(
        description='pooling x window-score, measured through stage 2',
        allow_abbrev=False)
    parser.add_argument('csv', nargs='*',
                        help='with --report-only, the CSVs to re-tabulate')
    parser.add_argument('--report-only', action='store_true',
                        help='rebuild every table from stored ranks. No GPU, '
                             'no WSI, no model -- every metric is derived from '
                             'rank_main and rank_overlap, so changing the k '
                             'list or the grouping never costs a GPU hour.')

    # ── which slides ──────────────────────────────────────────────────────────
    # The slides and their FoVs are every bench's (BenchCommon.run_slides,
    # supply_for): the first --n-wsi of each dataset's recorded --split, one
    # draw per slide across its levels, so a FoV here is the same photo the
    # pipeline bench and the stage-1 bench score under the same flags.
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS),
                        help='AccessDatasets ids')
    parser.add_argument('--split', default=SPLIT, choices=['val', 'test'],
                        help='the recorded split (MakeSplit) slides come from')
    parser.add_argument('--n-wsi', type=int, default=N_WSI,
                        help='the first N slides of each dataset\'s split')
    parser.add_argument('--seed', type=int, default=None,
                        help="= --sampler-seed (the recipe's own by default)")
    parser.add_argument('--fov-mask-cache-job', default=FOV_MASK_CACHE_JOB,
                        help='the hest masks the FoVs are placed on')
    parser.add_argument('--draw-cache-job', default=None,
                        help='whose cache the FoV draws are in. Default: this job')
    parser.add_argument('--render-cache-job', default=None,
                        help='whose cache the photo record is in. Default: this job')

    # ── which FoVs, through which camera: one flag per field ─────────────────
    # The four below name one field each; everything else is `--sampler-*`
    # (richness, overlap and inherit included) or `--camera-*`.
    parser.add_argument('--n-fov', type=int, default=None,
                        help="= --sampler-n-per-rung (the recipe's own by "
                             "default): FoVs per (slide, level); a slide that "
                             "cannot give them says how many it did")
    parser.add_argument('--rotation', type=int, choices=(0, 90, 180, 270),
                        default=None, help='= --camera-rotation-choices with one '
                        'value: only shots taken at this rotation. Every shot is '
                        'searched at all four rotations either way')
    parser.add_argument('--scale-min', type=float, default=None,
                        help='= the low end of --camera-scale-range')
    parser.add_argument('--scale-max', type=float, default=None,
                        help='= the high end of --camera-scale-range')
    # --fov, --max-ds, --sampler-*, --camera-*: the FoV recipe and its fields.
    add_fov_args(parser)
    add_mask_args(parser, default=None)              # None: MASK_RECIPES['hest']
    parser.add_argument('--mask-cache-job', default=MASK_CACHE_JOB,
                        help='whose mask cache to read and fill: result/cache/'
                             '<this>/. Default: this job')

    # ── which arms ────────────────────────────────────────────────────────────
    parser.add_argument('--arms', nargs='+', default=None,
                        help="which poolings to compare, each on its own lattices: "
                             "`cls_avg` is main + offset tiles, `cls_avg-M` is "
                             "main tiles only (-M, capital; -m is refused). "
                             "Default: CONFIG ARMS. `cls` on the full grid is the "
                             "baseline and is added if missing")
    parser.add_argument(
        '--encoder', default=ENCODER, choices=encoder_names(),
        help='which tile encoder (TileEncoderFunc._IMPLEMENTATIONS). Arms this encoder cannot do '
             'are dropped by name and printed; see admissible_poolings.')
    parser.add_argument(
        '--head', default=HEAD,
        help="which exit of the model, empty for its own default. Only CONCH "
             "has two, and it needs --head trunk to run here at all: its "
             "default attentional pooler hands back ONE 512-d vector, so every "
             "arm this bench compares is inadmissible and it stops before "
             "loading anything. trunk is the bare ViT, the same shape GigaPath "
             "and UNI2 have, which is what makes the three comparable. The "
             "head reaches identity_id and the output filename, because two "
             "heads are two different vectors of two different widths.")
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'= --encoder-batch-size (CONFIG {BATCH_SIZE}): tiles '
                             f'per forward pass, affordable because the token '
                             f'path runs under fp16 autocast')
    parser.add_argument('--read-workers', type=int, default=None,
                        help='DataLoader workers reading the reference grid ahead '
                             'of the encoder. Default: CpuBudget -- this process\'s '
                             'share of the cpus (cpus / shards) minus one, the rest '
                             'being torch threads. Changes no number')
    parser.add_argument('--timing', action='store_true',
                        help='split the reference stream\'s time into reading, '
                             'encoding, cosines and accumulating (the GPU is waited '
                             'for around each, so the run is a little slower)')
    parser.add_argument('--block-rows', type=int, default=BLOCK_ROWS,
                        help=f'(CONFIG {BLOCK_ROWS}): main tile rows per read. '
                             f'Changes no number')
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction,
                        default=None,
                        help=f'= --encoder-model-dtype fp16 / fp32 (CONFIG '
                             f'{DTYPE}). Run the forward pass under fp16 autocast. Output '
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
    parser.add_argument('--shard', default=None, metavar='I/N',
                        help='process only every N-th slide, starting at I '
                             '(0-based), so N processes on N cards share one '
                             'run. They write into the SAME parts directory '
                             '(a slide belongs to one shard, so no file is '
                             'written twice) and none of them assembles: run '
                             '--assemble once they are all done')
    parser.add_argument('--assemble', action='store_true',
                        help='no model, no GPU: read the parts of the run these '
                             'arguments describe, write window_retrieval.csv and '
                             'print the tables. What a run does at its end, on '
                             'its own -- after sharded processes, or after a '
                             'killed run that left parts')
    parser.add_argument('--gates-only', action='store_true',
                        help='run the gates on the first slide and stop: the '
                             'checks that take seconds, before a run of hours')
    parser.add_argument(
        '--out', default=None,
        help='output directory, used verbatim. Default: '
             'result/<SLURM_JOB_NAME or WindowRetrievalBench>/<encoder>/ -- the '
             'encoder level is added only to that derived path, so name it '
             'yourself when you pass one.')
    # The encoder's fields are flags too, and which fields exist depends on WHICH
    # encoder: so the name is read first, its config is built, and only then are
    # its flags added and the whole line parsed.
    pre_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre_parser.add_argument('--encoder', default=ENCODER)
    pre_parser.add_argument('--head', default=HEAD)
    pre_parser.add_argument('--report-only', action='store_true')
    pre, _ = pre_parser.parse_known_args()
    encoder_base = None
    if not pre.report_only:
        if pre.encoder not in encoder_names():
            parser.error(f'--encoder {pre.encoder!r}: one of {encoder_names()}')
        encoder_base = encoder_config(
            pre.encoder, batch_size=BATCH_SIZE,
            **({'head': pre.head} if pre.head else {})).with_model(dtype=DTYPE)
        add_encoder_args(parser, encoder_base)
    args = parser.parse_args()

    # The tag is a directory, not a filename suffix, added only to the derived
    # path. An explicit --out is used verbatim.
    tag = encoder_tag(args.encoder, args.head)
    out_dir = Path(args.out or job_result_dir('WindowRetrievalBench', encoder=tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    fov = FOV_RECIPES[args.fov]
    if fov.rungs != 'native':
        parser.error(f'--fov {args.fov} names its rungs {fov.rungs}; this bench '
                     f'scores each slide level by level, so it takes a recipe '
                     f"of native levels")
    args.max_ds = args.max_ds if args.max_ds is not None else fov.max_ds
    args.sensor = tuple(fov.sensor)
    print(f'bench_window_retrieval   datasets {args.datasets}  '
          f'{args.n_wsi} slides each  ds <= {args.max_ds:g}')
    print(f'scores    {"  ".join(SCORES)}   baseline = {BASELINE}\n')

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
        # The arms are whatever the CSV holds, in the order they first appear.
        report_arms = list(dict.fromkeys(r['arm'] for r in rows))
        report(attach_baseline(rows, BASELINE), report_arms, BASELINE,
               per_slide=args.per_slide, level_extras=LEVEL_EXTRAS)
        recall_report(rows, report_arms)
        return 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    require_gpu_if_allocated(device)
    # The run's configs, each resolved once: what the recipes hold, then the
    # flag that names one field, then the field's own flag. A bad value is
    # refused here by the config's own checks, before anything is built.
    try:
        sampler_base = fov.sampler
        if args.n_fov is not None:
            sampler_base = dataclasses.replace(sampler_base,
                                               n_per_rung=args.n_fov)
        if args.seed is not None:
            sampler_base = dataclasses.replace(sampler_base, seed=args.seed)
        sampler_cfg = config_from_args(args, sampler_base, 'sampler')

        camera_base = fov.gap
        if args.rotation is not None:
            camera_base = dataclasses.replace(camera_base,
                                              rotation_choices=(args.rotation,))
        if args.scale_min is not None or args.scale_max is not None:
            low, high = camera_base.scale_range
            camera_base = dataclasses.replace(camera_base, scale_range=(
                low if args.scale_min is None else args.scale_min,
                high if args.scale_max is None else args.scale_max))
        camera_cfg = config_from_args(args, camera_base, 'camera')

        encoder_cfg = encoder_base
        if args.batch_size is not None:
            encoder_cfg = dataclasses.replace(encoder_cfg,
                                              batch_size=args.batch_size)
        if args.fp16 is not None:
            encoder_cfg = encoder_cfg.with_model(
                dtype='fp16' if args.fp16 else 'fp32')
        encoder_cfg = encoder_cfg_from_args(args, encoder_cfg)
        mask_cfg = mask_cfg_from_args(args, base=MASK_RECIPES['hest'])
    except ValueError as exc:
        parser.error(str(exc))
    cfg = encoder_cfg
    args.encoder_cfg, args.mask_cfg = encoder_cfg, mask_cfg   # what the stage 2 tables' ids are made of

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
    try:
        if args.arms:
            tokens = list(args.arms)
        else:
            tokens, unfit = [], []
            for token in ARMS:
                name = token[:-len(MAIN_SUFFIX)] if token.endswith(MAIN_SUFFIX) else token
                (tokens if name in poolings or name == RAW else unfit).append(token)
            if unfit:
                print(f'[arms] {args.encoder} cannot do {", ".join(unfit)} -- '
                      f'left out of CONFIG ARMS')
        arm_specs = list(dict.fromkeys(
            parse_token(t, list(poolings) + [RAW]) for t in tokens))
    except ValueError as exc:
        parser.error(str(exc))
    if ('cls', 'all') not in arm_specs:
        arm_specs.insert(0, ('cls', 'all'))
        print(f'[arms] {BASELINE} is the baseline every other arm is paired '
              f'against, so `cls` on the full grid was added')
    bases = list(dict.fromkeys(b for b, _ in arm_specs))
    arms = [f'{token_name(b, g)}+{s}' for b, g in arm_specs for s in SCORES]
    print(f'arms      {"  ".join(token_name(b, g) for b, g in arm_specs)}')

    # Only arguments decide where a run's parts are, so --assemble answers here,
    # before a slide is listed or a model is built.
    # the FoV recipe as this run resolved it -- what supply_for draws with
    fov_run = dataclasses.replace(fov, sampler=sampler_cfg, gap=camera_cfg,
                                  max_ds=args.max_ds)
    configs = {'sampler': sampler_cfg, 'camera': camera_cfg, 'mask': mask_cfg,
               'encoder': encoder_cfg}
    print(f'FoV       {sampler_cfg.n_per_rung} per (slide, level), one draw per '
          f'slide across its levels, seed {sampler_cfg.seed}')
    print(f'mask      {"the hest recipe" if mask_cfg == MASK_RECIPES["hest"] else "NOT the hest recipe: a new seg_id, made again"}'
          f'  ({mask_cfg.seg_id()})')
    print('configuration this run uses (CONFIG, then flags):')
    for name, config in configs.items():
        for line in describe(config, name):
            print(f'  {line}')
    print()
    parts_dir = out_dir / f'parts-{config_id(args, arm_specs, configs)}'
    if args.assemble:
        return 0 if assemble(parts_dir, out_dir, arms, args.per_slide) else 1

    selected = run_slides(args)
    shard = None
    if args.shard:
        try:
            index, _, count = args.shard.partition('/')
            shard = (int(index), int(count))
            if not 0 <= shard[0] < shard[1]:
                raise ValueError
        except ValueError:
            parser.error(f'--shard is I/N with 0 <= I < N, got {args.shard!r}')
        total = len(selected)
        selected = selected[shard[0]::shard[1]]
        print(f'shard {shard[0]}/{shard[1]}: {len(selected)} of {total} slides')
    # The shards of one run share the job's cpus; each takes its share and no
    # more. Left to torch, two shards on 8 cpus ran 16 threads and encoded 7x
    # slower than one (CpuBudget's docstring has the numbers).
    args.budget = CpuBudget.for_job(processes=shard[1] if shard else 1,
                                    workers=args.read_workers).apply()
    print(args.budget.line())
    print(f'slides ({len(selected)}):')
    width = max((len(d) for d, _, _ in selected), default=0) + 2
    for dataset, name, _ in selected:
        print(f'  {dataset:<{width}}{name}')

    job = Cache.job_name('WindowRetrievalBench')
    fov_masks = MaskMaker(MASK_RECIPES['hest'], args.fov_mask_cache_job, device)
    mask_job = args.mask_cache_job or job
    print(f'masks     {mask_cfg.seg_id()}  cache {mask_job}')

    encoder = cfg.build(device)
    if RAW in bases and getattr(encoder.model_spec, 'kind', None) != 'tokens':
        parser.error(f'raw arms need an encoder with patch tokens, and '
                     f'{args.encoder} gives {encoder.model_spec}')
    print(f'device={device}  encoder={tag}  '
          f'model_spec={encoder.model_spec}  '
          f'dtype={encoder.cfg.model.dtype}  batch={cfg.batch_size}  '
          f'id={encoder.identity_id()}')
    print(f'{len(arms)} arms (pooling x score), baseline = {BASELINE}\n')

    if not selected:
        print('no slides for this process (more shards than slides): nothing to do')
        return 0
    failed = []
    with MaskMaker(mask_cfg, mask_job, device) as masks:
        # The gates look at tissue, so the first slide's mask comes first. It is
        # the mask the run needs for that slide anyway, and is cached.
        first = SafeSlide(str(selected[0][2]))
        try:
            first_mask, _ = masks.mask(first)
        finally:
            first.close()
        if not run_gates(gate_tiles(selected[0][2], mask=first_mask), encoder,
                         [b for b in bases if b != RAW], selected[0][2],
                         first_mask):
            print('\nGATE FAILURE -- stopping before the run spends hours '
                  'producing numbers that could not mean anything')
            return 1
        if args.gates_only:
            return 0

        parts_dir.mkdir(parents=True, exist_ok=True)
        print(f'parts     {parts_dir}  '
              f'({len(list(parts_dir.glob("*.csv")))} already there: '
              f'a resume skips them)')
        for dataset, stem, path in selected:
            print(f'\n{"=" * 78}\n{stem}  ({dataset})\n{"=" * 78}', flush=True)
            slide = SafeSlide(str(path))
            try:
                mask, hit = masks.mask(slide)
                print(f'  mask  {"from the cache" if hit else "segmented now"}  '
                      f'tissue {mask.tissue_fraction() * 100:.1f}%  '
                      f'{len(mask.tissue_regions)} regions', flush=True)
                if not mask.tissue_regions:
                    continue
                todo = []
                for level in levels_up_to(slide.level_downsamples, args.max_ds):
                    if part_path(parts_dir, stem, level).exists():
                        print(f'  L{level}  already done -- skipped', flush=True)
                    else:
                        todo.append(level)
                if not todo:
                    continue
                # the slide's FoVs, every level in one draw, as every bench
                # takes them; rendered once and split by level
                supply = supply_for(str(path), fov_run, fov_masks, args, job)
                by_level = {}
                if supply is not None:
                    for shot in supply.shots(workers=args.budget.workers):
                        by_level.setdefault(int(shot[1].level), []).append(shot)
                for level in todo:
                    part = part_path(parts_dir, stem, level)
                    rows = run_slide_level(
                        slide, dataset, stem, level, mask, args, encoder,
                        arm_specs, bases, device, supply, by_level.get(level, []), job)
                    write_part(rows, part)
                    print(f'  L{level}  {len(rows)} rows -> {part.name}   '
                          f'{peak_memory_line()}', flush=True)
            except Exception as exc:                          # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f'  {type(exc).__name__}: {exc}')
                failed.append(stem)
            finally:
                slide.close()

    print(f'\n{"=" * 78}')
    if shard is None:
        # The encoder is in the NAME as well as in every row: the name stops a
        # second encoder's run from overwriting the first one's numbers, the
        # column stops the two from being averaged together afterwards.
        assemble(parts_dir, out_dir, arms, args.per_slide)
    else:
        print(f'shard {shard[0]}/{shard[1]} done. Once every shard has finished, '
              f'run the same command with --assemble.')
    if failed:
        print(f'\n{len(failed)} slide(s) failed: {", ".join(failed)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
