#!/usr/bin/env python3
"""The stage 2 retriever bench: which way of finding the window puts the truth first?

    python utilities/bench_modules/bench_stage2_retriever.py --encoder uni2 \
        --arms cls cls_avg uni2:BagOfTiles::cls_avg+gap uni2:BagOfTiles::rings3+quad \
               uni2:TileVote::grid2x2+exact uni2:ClusterBow::cls_avg
    python utilities/bench_modules/bench_stage2_retriever.py --encoder uni2 \
        --rerank-on uni2:SlidingWindow::cls+mean --rerankers Dual::cls_avg+dual CSLS::grid2x2+csls \
                    gigapath:Corr::cls_avg+sinkhorn
    python utilities/bench_modules/bench_stage2_retriever.py --report-only \
        /work/u26130998/result/Stage2RetrieverBench/uni2/stage2_retriever.csv \
        /work/u26130998/result/Stage2RetrieverBench/gigapath/stage2_retriever.csv

This was the window bench, and it compared pooling x score because the sliding window was the only
first phase there was. An ARM is now a RETRIEVER and then any number of RERANKERS, named
`[encoder:]Method::pooling[+operation][@K]` joined by `::` (stage2_retrieval/ArmSpec.py):

    uni2:SlidingWindow::cls_avg+mean              the window's tile cosines averaged
    uni2:BagOfTiles::rings3+quad                  a window as its four quadrants' mean vectors
    uni2:TileVote::cls_avg+hnsw                   the query's tiles vote, nearest tiles by an hnsw graph
    uni2:ClusterBow::cls_avg                      a window as a bag of tile clusters (no operation: no `+`)
    uni2:BagOfTiles::cls_avg+gap@100::gigapath:CSLS::grid2x2+csls
                                                  ... the first 100 of that list, which gigapath reads
                                                  through its grid2x2 tiles and corrects for hubness

`+operation` is a choice inside the method; a method with none has no `+`, and one that takes no pooling
writes `raw`. `@K` is what the segment's consumer reads: the next phase's `topk_in`, or, on the last
segment, the length the arm's rows are cut at. An arm is the chain of stage-2 retrievers
`ArmSpec.config_of_arm` makes -- the configs `--stage2 arm:<arm>` runs in the pipeline -- and each phase is
the pipeline's own stage 2 entry (RetrieverArms.py). A reranker encodes for itself what its pooling and
encoder need, or reuses the phase before it when that is the very same features. `--arms cls_avg` is the
older shorthand for the sliding window with every window score.

EVERY RETRIEVER ARM HAS TWO MODES, measured on the same FoVs from the same score map:

  ranked       every window of the level ranked by the arm's score, no suppression (NMS off) and no list
               in the way, only the answer's own window counting: the search itself, judged strictly.
               A rank can be 76,435.
  listed       what the stage hands on: the best `--k` windows, a window dropped when a better one of its
               rotation lies within `--min-sep-tiles` (NMS on). The window an answer names may have been
               dropped for one within that distance, which stands for it (an answer is its window or one of
               its rotation within the suppression radius); with none it is rank pool + 1, a miss at every k
               and f%. A shorter K is a prefix of the list, so the report reads recall@K for K = 30 ... 1000
               off one list.

A ranked arm's recall@K is a list of K with no suppression, so up to `--k` the two modes differ by the
suppression and its tolerance alone. A reranker reorders a given list, so it has a listed mode only. The
report pairs every arm with
`SlidingWindow::cls+mean` and, apart, every reranker with the list it was given.

The sliding window is streamed: one pass over the level scores every pooling and every window score at
once, keeping nothing of a slide's features. Every other retriever and every reranker runs in the
ArmEngine on hosts that encode a level once per (encoder, pooling). An arm keeps nothing but its stage 2
entries (one per phase, named as the pipeline names them, `stage2_retrieval/Naming.py`): its rows are read off
them (`slide_rows`), by the one definition of an answer and its ranks (`BenchCommon.answers`, `ranked_ranks`,
`listed_places`), into one `stage2_retriever.csv`. One arm is redone by removing its entries. An arm's name
carries its encoder, so the CSVs of two runs (`--encoder uni2`,
`--encoder gigapath`) can be reported together. `SlidingWindow::cls+mean` is the baseline and is always run.

The streamed sliding window: stage 2 as it actually runs -- mask, regions, sliding window -- with two axes
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

  slides     the first `--n-wsi` of each dataset a `--datasets` id draws from
             (a virtual dataset is shuffled when drawn), as every bench takes
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
             window nearest it (`BenchCommon.answers`); each is ranked in the pool it
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
scores a FoV at the level it was placed at): `output` (the first `--k` windows,
picked by the stage's own `candidate_set`), `truth` (both answers, with their ranks and the
pools), `tile_sims` (10 windows) and `truth_sim`. See "The stage 2 tables an arm writes".
Every arm writes them, `min` and the `-M` arms too (their entries have ids of
their own, the stage having no setting for them).

The other arms (BagOfTiles, TileVote, ClusterBow, WinEmbed and every reranker) write one
entry per phase (`ArmTables.py`, named and recorded as the pipeline does), the tables that phase
holds (`BenchCommon.phase_roles`): a first phase `output` and `truth`, a reranker `output` and
`rerank_scores`. An entry holds every level of its slide, so it is
written when the slide's last level is done; a slide whose entries are missing, or stop short
of a level holding a FoV the arms are scored on, runs all those levels again. Demo and plots
read them as they read the pipeline's, and the bench reads its rows off them.

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
from typing import Any, NamedTuple, Optional, Sequence
import json
import math
import contextlib
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
from ConfigIdentity import environment, short_id                  # noqa: E402
from dump_function.RetrievalReport import (K_FIXED, K_FRACTIONS,  # noqa: E402,F401
                                           attach_baseline, frac_label, grid_table,
                                           group_by, group_levels, k_at, pct,
                                           print_level_heading, print_paired, report, truth_rank)
from stage2_retrieval.RetrieverSlidingWinSimRot import (GEOMEAN_FLOOR, TILE_SIM_ROWS,   # noqa: E402
                                                        RetrieverSlidingWinSimRot,
                                                        RetrieverSlidingWinSimRotConfig,
                                                        WINDOW_SCORES, SlidingWindowSimilarity,
                                                        window_score, window_score_maps)
from stage2_retrieval.Context import top_candidates             # noqa: E402
from stage2_retrieval.MultiPhases import MultiPhasesConfig      # noqa: E402
from stage2_retrieval.Retriever import EncoderPool              # noqa: E402
from stage2_retrieval.StageInterface import ROTATIONS, Candidate, CandidateSet  # noqa: E402
from camera import Render                                        # noqa: E402
from _paths import encoder_tag, job_result_dir                   # noqa: E402
from BenchCommon import (ANSWER_LATTICES, ORACLE, TRUTH_VERSION, Answer, Answers, Maps, Pools,  # noqa: E402
                         Tables, answers, answers_of_row, fmt_each, fmt_rate, fmt_time, listed_of_rows,
                         listed_places, pools_of,
                         matching_rotation, phase_ids, phase_record, phase_roles, placed_estimate, query_grid_centre,
                         ranked_ranks, require_gpu_if_allocated, run_slides, supply_for, truth_row)
from ConfigIdentity import file_fingerprint                      # noqa: E402
from stage2_retrieval.Naming import entry_name, label as window_label   # noqa: E402
from ArmTables import SlideTables, entries_complete, entry_levels   # noqa: E402
from RetrieverArms import (FAMILY_METHOD, FAMILY_POOLING, FAMILY_RERANK,  # noqa: E402
                           MODE_LISTED, MODE_RANKED, ArmEngine, LevelResult,
                           ListQuery, arm_texts)
from stage2_retrieval.ArmSpec import (SEP, ArmSpec, expand, is_arm_text,  # noqa: E402
                                      parse_arm, parse_segment)
from stage1_estimation.StageInterface import weights_path        # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG -- everything a run depends on, in one place. Edit here.
#
#  Each value below is the one the run uses, unless the command line says
#  otherwise: every field of the sampler, camera and encoder configs is a flag
#  (`--sampler-richness-caps`, `--camera-noise-sigma`, `--encoder-batch-size`, ...; see
#  ConfigArgs), and the shorthand flags (`--n-fov`, `--seed`, `--rotation`, `--scale-min`,
#  `--scale-max`, `--batch-size`, `--fp16`) name one field each. What
#  a run really used is printed at its start, and everything that changes a
#  number is in the stage 2 entries' ids, so a changed value here can never be
#  read off entries made with another.
# ══════════════════════════════════════════════════════════════════════════════

# ── which slides ─────────────────────────────────────────────────────────────
DATASETS         = ('@legacy-val',)   # AccessDatasets ids: registered, or virtual (`@<name>`)
N_WSI            = 10           # the first N of each dataset an id draws from, as every bench
MASK_CACHE_JOB   = None         # None: this job's own; 'Masks' to reuse
FOV_MASK_CACHE_JOB = 'Masks'              # the hest masks the FoVs are placed on

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
PRODUCTION_K       = 1000   # listed mode: the longest list an arm hands on; a shorter K is its prefix
PRODUCTION_MIN_SEP = 1.0    # listed mode: suppression radius in tiles (min_sep_tiles)
#: the list lengths the listed recall curve is read at (a K past a level's pool is the whole level)
K_CURVE = (30, 100, 200, 300, 500, 1000)
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
BASELINE = 'cls+mean'          # the sliding window's own name for it; the arm is `<encoder>:SlidingWindow::cls+mean`


def arm_prefix(encoder: str) -> str:
    """What every streamed sliding-window arm of an encoder starts with: `uni2:SlidingWindow::`."""
    return f'{encoder}:SlidingWindow{SEP}'


def baseline_of(encoder: str) -> str:
    """`uni2:SlidingWindow::cls+mean`: the baseline every arm is paired against."""
    return arm_prefix(encoder) + BASELINE


def canonical_arm(arm: str, encoder: str) -> str:
    """An arm name in the bench's grammar. An older CSV names its sliding-window arms `cls_avg+geomean`;
    they are `<encoder>:SlidingWindow::cls_avg+geomean`."""
    return arm if SEP in arm else arm_prefix(encoder) + arm

#: Every score is the stage's own (SlidingWinSimRot.WINDOW_SCORES, imported above with GEOMEAN_FLOOR), so
#: the bench and the pipeline score a window one way and a pipeline run can be set to any of them.
assert set(SCORES) <= set(WINDOW_SCORES), 'SCORES names a score the stage does not have'

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
#: `BenchCommon.ranked_ranks`). Empty on a main-only arm, which has no offset windows.
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
    min      the stage's own too, the hardest AND: one tile can veto the window.
    """
    return {name: window_score(per_tile, name) for name in SCORES}


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
    """`BenchCommon.answers` on a real grid: each answer is at most half a tile from the point on each axis (so
    at most 128 sqrt(2) away), and the nearer of the two at most 128 -- the union of the lattices is a square
    lattice turned 45 degrees (`MAX_TRUTH_DISTANCE`). Points anywhere a 5 x 4 tile window of both lattices fits
    round them, in a region whose origin is not a multiple of a tile."""
    region = types.SimpleNamespace(x=1001, y=2003, w=40 * TILE, h=40 * TILE)
    grids = region_grids([region], ds=1.0, level=0, tile_size=TILE, overlap=True)
    frame = CandidateSet(candidates=(), level=0, ds=1.0, grids=tuple(grids), irretrievable_lvl=(), alter_lvl=())
    rng = np.random.default_rng(seed)
    margin = 5 * HALF_TILE + TILE
    worst_main = worst_ovlp = worst_truth = 0.0
    axis_ok = True
    for x, y in rng.uniform(margin, 40 * TILE - margin, size=(n, 2)):
        found = answers(frame, (4, 5), (region.x + float(x), region.y + float(y)), 0)
        if found is None or found.offset is None:
            axis_ok = False
            continue
        axis_ok = axis_ok and found.within(HALF_TILE + 0.5)
        worst_main = max(worst_main, found.main.dist_l0)
        worst_ovlp = max(worst_ovlp, found.offset.dist_l0)
        worst_truth = max(worst_truth, min(found.main.dist_l0, found.offset.dist_l0))
    # half a level-0 pixel of slack: a tile's read starts at a rounded level-0 point (`tile_origin_l0`)
    ok = (axis_ok and worst_main <= MAX_SINGLE_GRID_DISTANCE + 1.0
          and worst_ovlp <= MAX_SINGLE_GRID_DISTANCE + 1.0
          and worst_truth <= MAX_TRUTH_DISTANCE + 1.0)
    return (worst_main, worst_ovlp, worst_truth), ok


def gate_recall_ranks() -> tuple:
    """`BenchCommon.ranked_ranks` on hand-made maps whose answers' ranks are known by construction.

    One region, two rotations. At rotation 0: main windows 3x3, offset windows
    2x2, the answers at main (1, 1) and offset (0, 1). At rotation 90: main 2x2
    and offset 1x1, whose windows are the other rotation's candidates and count
    in every pool but are not the answer. The expected ranks are written down,
    not computed by the code under test:
      main 0.6    above it: main 0.9, 0.8 | 0.99, 0.7 (rot 90) ; offset 0.95, 0.7,
                  0.65 (rot 0) and 0.65 (rot 90)
      offset 0.7  above it: offset 0.95 ; main 0.9, 0.8 | 0.99 (0.7 is not above)
    Run with the answers at rotation 0, and again at 90 with the maps swapped so
    the answer must be read from the rotation it names."""
    main = torch.tensor([[0.9, 0.5, 0.1], [0.2, 0.6, 0.3], [0.8, 0.4, 0.0]])
    ovlp = torch.tensor([[0.95, 0.7], [0.65, 0.05]])
    main90 = torch.tensor([[0.99, 0.1], [0.7, 0.2]])
    ovlp90 = torch.tensor([[0.65]])

    def at(rot: int) -> Answers:
        return Answers(Answer('main', 0, 1, 1, rot, (0, 0, 0.0, 0.0), 0.0),
                       Answer('offset', 0, 0, 1, rot, (0, 0, 0.0, 0.0), 0.0))

    want = {'rank_main': 5, 'rank_main_in_all': 9, 'rank_offset_in_offset': 2, 'rank_offset_in_all': 5}
    maps: Maps = {(0, 0, 'main'): main, (0, 0, 'offset'): ovlp, (90, 0, 'main'): main90, (90, 0, 'offset'): ovlp90}
    swapped: Maps = {(0, 0, 'main'): main90, (0, 0, 'offset'): ovlp90, (90, 0, 'main'): main,
                     (90, 0, 'offset'): ovlp}
    pools: Pools = pools_of(maps)
    got = ranked_ranks(maps, at(0), pools)
    return got, (got == want and ranked_ranks(swapped, at(90), pools_of(swapped)) == want
                 and (pools.main, pools.offset, pools.all) == (13, 5, 18))


def gate_truth_centre() -> tuple:
    """`query_grid_centre` against an index image, for the four rotations.

    A shot of 1024 x 1440 px is turned with `np.rot90` the way the stage turns
    it, cut into whole tiles from the turned photo's top-left, and the original
    pixel under the centre of that grid is read off an image of its own
    coordinates -- nothing derived. The camera here is the identity map, so the
    point the answers are found around must be that pixel. The decoy is
    the centre of the grid cut from the UNTURNED photo, which is what this bench
    used before: it has to be far from the truth at 180 and 270, or the gate
    could not tell the two apart. Returns ((worst |diff|, decoy), ok)."""
    class Identity:
        def output_to_level0(self, x0, y0, u, v, rot_deg=0.0, scale=1.0):
            return u, v
    height, width = 1024, 1440
    ys, xs = np.indices((height, width))
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
        got = query_grid_centre(Identity(), 0, 0, {'rot_deg': rot_deg, 'angle_jitter': 0.0, 'scale': 1.0},
                                (height, width, 3), TILE)
        worst = max(worst, abs(got[0] - want[0]), abs(got[1] - want[1]))
        decoy = max(decoy, abs((width // TILE) * TILE / 2.0 - want[0]),
                    abs((height // TILE) * TILE / 2.0 - want[1]))
    return (worst, decoy), worst < 1e-6 and decoy >= HALF_TILE


def gate_scored_windows(seed: int = 0) -> tuple:
    """`streamed_list` picks the windows the stage picks. Made-up per-tile maps
    for two regions (12 x 10 and 9 x 9 tiles), four rotations, both lattices:
    the stage reduces them with its score and picks (`window_score_maps`,
    `top_candidates`, as `RetrieverSlidingWinSimRot.list_windows` does); the
    streamed pass's picker is handed scores reduced one window at a time and
    picks. The two lists -- window, rank, score -- must be the same, for `mean`,
    `geomean` and `min`. Returns (windows compared, ok)."""
    import types
    from PatchingLib import region_grids
    generator = torch.Generator().manual_seed(seed)
    regions = [types.SimpleNamespace(x=1000, y=2000, w=12 * TILE, h=10 * TILE),
               types.SimpleNamespace(x=40000, y=9000, w=9 * TILE, h=9 * TILE)]
    grids = region_grids(regions, ds=1.0, level=0, tile_size=TILE, overlap=True)
    compared, same = 0, True
    for score in ('mean', 'geomean', 'min'):
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
        frame = CandidateSet(candidates=(), level=0, ds=1.0, grids=tuple(grids), irretrievable_lvl=(),
                             alter_lvl=())
        stage = top_candidates(window_score_maps(sims, score), frame, 15, 1.0, Candidate)
        reduced = {rot: ([window_score(a, score) for a, _ in per_region],
                         [window_score(b, score) for _, b in per_region])
                   for rot, per_region in sims.items()}
        streamed = streamed_list(reduced, list(grids), 0, 1.0, 15, 1.0)
        want = [(c.key(), round(c.score, 6)) for c in stage.candidates]
        got = [(c.key(), round(c.score, 6)) for c in streamed.candidates]
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
    shot -- the shape `streamed_list` and `flat_maps` read. A lattice with no
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
    print(f'{n_picked} windows, the streamed picker and the stage pick the same  '
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


def scores_of(args, base: str, grid: str) -> tuple:
    """The window scores arm `base` on `grid` is reported with: all of `SCORES` for a bare pooling
    (`cls_avg`), the ones named for `cls_avg+mean`. The streamed pass computes all of them either
    way; this is which rows are kept."""
    return tuple(getattr(args, 'score_filter', {}).get((base, grid), SCORES))


def flat_maps(maps_by_rot: dict, lattices: Sequence[str] = ANSWER_LATTICES) -> Maps:
    """The streamed pass's window scores, `{rotation: (main maps, offset maps)}` one `[rows, cols]` per region,
    as a first phase hands them on (`window_scores`): one map per `(rotation, region, lattice)` of `lattices`,
    in the order the stage lists them (rotation, region, main before offset), leaving out a region that holds
    no window or a lattice that was not scored."""
    out: Maps = {}
    for rot, (mains, offsets) in maps_by_rot.items():
        for ri in range(len(mains)):
            for lattice, m in (('main', mains[ri]), ('offset', offsets[ri] if ri < len(offsets) else None)):
                if lattice in lattices and m is not None and m.numel():
                    out[(rot, ri, lattice)] = m
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  The stage 2 tables an arm writes
#
#  A streamed arm whose pooling the encoder names IS a `RetrieverSlidingWinSimRot`:
#  a pooling, a lattice (`-M`: the main one only), a score, the same windows. So it
#  gets the stage 2 entry the pipeline writes for that config -- the same name and
#  record (`BenchCommon.phase_ids`, `phase_record`), the same tables, the same
#  place in the tree (`.../render=/stage1=oracle/stage2/`, this job's) -- and the
#  demo page, stage 3, the plots and a pipeline run read a window bench's arm as
#  they read a pipeline run's:
#
#    output     the first `--k` windows of the FoV, every rotation, lattice
#               and region together, picked as the stage picks them (`streamed_list`)
#    truth      the FoV's answers -- the main window and, on an arm that scored
#               the offset lattice, the offset one: each one's window, box,
#               distance, score, and rank among its lattice's windows and among
#               all of them (+ the pools)
#    tile_sims  the first STAGE2_TILE_SIMS windows' per-tile cosines
#    truth_sim  the answers', by `lattice`
#
#  The window bench scores a FoV at the level it was placed at, so the entry is
#  under `stage1=oracle`. EVERY arm the run has is written: a `raw` arm, a pooling
#  the stage has no reading of, an entry of its own with an id made from what it is.
# ══════════════════════════════════════════════════════════════════════════════

#: windows, besides the truth one, with their per-tile cosines: the stage's own `tile_sims` length
STAGE2_TILE_SIMS: int = TILE_SIM_ROWS


def streamed_list(maps_by_rot: dict, grids: list, lv: int, ds: float, k: int, min_sep: float) -> CandidateSet:
    """The list the stage makes of a FoV's window scores: `maps_by_rot` is `{rotation: (main maps, offset
    maps)}`, one `[rows, cols]` per region (empty where a region holds no window or the lattice was not
    scored), put in the order the stage lists them (rotation, region, main before offset) and picked by the
    stage's own `top_candidates` -- its per-map top k and its suppression."""
    frame = CandidateSet(candidates=(), level=lv, ds=ds, grids=tuple(grids), irretrievable_lvl=(), alter_lvl=())
    return top_candidates(flat_maps(maps_by_rot), frame, k, min_sep, Candidate)


class Stage2Arm:
    """One streamed arm on its way to a stage 2 entry. `entry_id` names the entry and `rec` is its record:
    the pipeline's own for the sliding window of this config (`cfg`), or, for a `raw` arm (`cfg` None), an id
    made from what it is (encoder, lattices, score, mask, the cut), an entry no pipeline run asks for.
    `tables` is what to fill; None when the entry is already there."""

    def __init__(self, base: str, grid: str, score: str, entry_id: str, rec: dict,
                 cfg: Optional[RetrieverSlidingWinSimRotConfig]) -> None:
        self.base: str = base
        self.grid: str = grid
        self.score: str = score
        self.name: str = token_name(base, grid)
        self.entry_id: str = entry_id
        self.rec: dict = rec
        self.cfg: Optional[RetrieverSlidingWinSimRotConfig] = cfg
        self.tables: Optional[Tables] = None


def window_config(args, base: str, grid: str, score: str) -> Optional[RetrieverSlidingWinSimRotConfig]:
    """The sliding window of a streamed arm, as the stage holds it; None for a pooling the stage has no reading
    of (`raw`, or one the encoder does not name)."""
    if base == RAW or score not in WINDOW_SCORES:
        return None
    try:
        return RetrieverSlidingWinSimRotConfig(
            encoder=dataclasses.replace(args.encoder_cfg, pooling=base), tile_size=TILE, overlap=(grid == 'all'),
            raw_topk=args.k, min_sep_tiles=args.min_sep_tiles, score=score)
    except ValueError:                   # the encoder has no such pooling
        return None


def stage2_arms(args, arm_specs) -> list[Stage2Arm]:
    """Every streamed arm of the run as a stage 2 entry to write (`Stage2Arm`)."""
    seg: str = args.mask_cfg.seg_id()
    region: str = args.mask_cfg.region_id()
    out: list[Stage2Arm] = []
    for base, grid in arm_specs:
        for score in SCORES:
            cfg: Optional[RetrieverSlidingWinSimRotConfig] = window_config(args, base, grid, score)
            if cfg is not None:
                name, _ = phase_ids(MultiPhasesConfig(phases=(cfg,)), (seg, region))[0]
                rec = dict(phase_record(cfg, name, '', stage1=ORACLE, seg=seg, region=region),
                           truth=TRUTH_VERSION, limit=0)
                out.append(Stage2Arm(base, grid, score, name, rec, cfg))
                continue
            hexid: str = short_id([args.encoder_cfg.identity_id(), base, grid, score,
                                   seg, region, 'window-arm', f'k={args.k}', f'min_sep={args.min_sep_tiles}'])
            label: str = entry_name(window_label('SlidingWindow', args.encoder, base, score, 0, grid == 'main'), hexid)
            rec = dict(id=hexid, parts=[f'pooling={base}', f'grid={grid}', f'score={score}'],
                       versions={'window_arm': ROW_VERSION},
                       upstream={'stage1': ORACLE, 'seg': seg, 'region': region},
                       env=environment(), label=label, truth=TRUTH_VERSION, limit=0)
            out.append(Stage2Arm(base, grid, score, label, rec, None))
    return out


def stage2_tables(slide, lv: int, ds: float, grids, valid, final, qgf, dims,
                  arm_specs, args, encoder, device, sink: Optional[SlideTables],
                  seconds_each: float) -> None:
    """Put this level's rows into the slide's stage 2 tables (`sink`; None when the
    slide's entries are already complete). They are written when the slide's last
    level is done (`SlideTables.flush`), because an entry holds every level.
    `valid` are the FoVs `place_shots` placed and their answers, `final` their window
    scores (`finalize_windows`), `qgf` their query descriptors per rotation. The truth
    is `BenchCommon.truth_row`'s, as the pipeline's: both answers placed, the ones of
    the lattices the arm scored ranked."""
    arms: list[Stage2Arm] = stage2_arms(args, arm_specs)
    if sink is None:
        print(f'  L{lv}  stage 2 tables: all {len(arms)} arms are already there', flush=True)
        return
    todo: list[Stage2Arm] = [a for a in arms if sink.has(a.entry_id)]     # the arms whose entry is written now
    for arm in todo:
        arm.tables = sink.tables(arm.entry_id)
    n_regions: int = len(grids)
    frame: dict = dict(level=lv, ds=ds, irretrievable_lvl='', alter_lvl='')
    #: what the per-tile cosines are asked of: (arm, FoV, rank or None for an answer, window)
    wanted: list[tuple] = []
    #: the baseline's ranked rank and pool, FoV by FoV: the run's check that it ranks something at all
    baseline: list[tuple] = []

    for s_idx, (fov_id, shot, found) in enumerate(valid):
        qc = QueryPatchContainer(shot[1])
        qc.extract_all(TILE, overlap=True)
        for arm in todo:
            maps = {rot: shot_maps(final[rot][arm.name], arm.score, s_idx, n_regions)
                    for rot in dims}
            cs = streamed_list(maps, list(grids), lv, ds, args.k, args.min_sep_tiles)
            if not len(cs):
                arm.tables.add('output', fov_id, [dict(frame, error='no level holds the query', t_s=0.0)])
                continue
            arm.tables.add('output', fov_id, [dict(r, **frame, n_listed=len(cs), error='',
                                                   t_s=round(seconds_each, 3))
                                              for r in cs.rows(qc)])
            lattices: tuple = ANSWER_LATTICES if arm.grid == 'all' else ('main',)
            row: dict = truth_row(found, flat_maps(maps, lattices))
            arm.tables.add('truth', fov_id, [dict(row, truth_level=lv)])
            wanted += [(arm, s_idx, fov_id, None, Candidate(a.region, a.lattice, a.row, a.col, a.rotation, 0.0))
                       for a in found if a.lattice in lattices]
            wanted += [(arm, s_idx, fov_id, rank, c)
                       for rank, c in enumerate(cs.candidates[:STAGE2_TILE_SIMS], 1)]
            if (arm.base, arm.grid, arm.score) == ('cls', 'all', 'mean'):
                baseline.append((row['truth_rank_main'], row['pool_main']))

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
            arm.tables.add('truth_sim', fov_id, [dict(lattice=c.lattice, **r) for r in rows])
        else:
            arm.tables.add('tile_sims', fov_id, [dict(rank=rank, **r) for r in rows])
            if arm.score == 'mean':               # the window's score is the mean of these
                worst = max(worst, abs(float(cos.mean()) - c.score))
    if baseline:
        # The truth's mean percentile. A uniformly random window sits at 0.500, so near 0.5 means the
        # coordinate mapping is broken and every arm is ranking noise.
        ranks = sorted(r for r, _ in baseline)
        pct = float(np.mean([r / max(1, p) for r, p in baseline]))
        print(f'  L{lv}  baseline {baseline_of(args.encoder)}: rank_main med {ranks[len(ranks) // 2]:,}   '
              f'truth_pctile {pct:.4f} (random=0.5000)', flush=True)
    print(f'  L{lv}  stage 2 tables: {len(todo)} arms x {len(valid)} FoV -> {sink.entry.dir}  '
          f'(re-scored from the tile cosines: largest |diff| to the streamed score {worst:.1e}; '
          f'written when the slide is done)', flush=True)


class Placed(NamedTuple):
    """Where a level's FoVs stand on the slide, from geometry alone: the scale, the
    regions and their grids, the camera, the query's tile grid at each rotation, and the
    FoVs that have answer windows as `(fov_id, shot, answers)`."""
    ds: float
    lv: int
    regions: list
    grids: list
    camera: Any
    dims: dict
    valid: list
    n_shots: int
    n_no_region: int
    n_no_window: int


def place_shots(slide: SafeSlide, stem: str, level: int, mask: Any, supply: Any,
                shots: Sequence[Any], quiet: bool = False) -> Optional[Placed]:
    """Which of a level's FoVs have answer windows, and which they are (`BenchCommon.answers`, the stage's
    own grid). None, said aloud unless `quiet`, when no FoV does. Every arm of the run -- pooling or list --
    is scored on exactly the FoVs this returns, so the arms pair.

    `shots` are `(index, meta, image, params)`; the image may be None: placing a FoV reads its position and
    the camera's parameters only, and the photo's size is the camera's. A FoV is placed when its rectangle
    is inside a tissue region and each answer is at most half a tile from the point its grid stands for on
    each axis: there the lattice point nearest the point has its window. Farther, the region lacks it, and
    the nearest window there is stands in for another place."""
    ds = float(slide.level_downsamples[level])
    if not shots:
        # No region holds a FoV this size at this level -- a coarse level on a
        # small tissue section. Not an error: say so.
        if not quiet:
            print(f'  L{level} (ds {ds:g}): no FoV position -- skipped', flush=True)
        return None
    camera = supply.camera_for(shots[0][1].ds)
    rows_q = camera.output_h // TILE
    cols_q = camera.output_w // TILE
    # the query's tile grid at each rotation the stage searches: the sides swap
    # at 90 / 270
    dims = {rot: ((rows_q, cols_q) if (rot // 90) % 2 == 0 else (cols_q, rows_q))
            for rot in ROTATIONS}

    # The regions and their grids come from geometry alone. Which FoVs have an
    # answer window is decided here, before any tile is encoded.
    lv, ds = SlideReader(slide).native_scale(ds=ds)
    regions = mask.patchable(TILE * ds).tissue_regions
    grids = region_grids(regions, ds=ds, level=lv, tile_size=TILE, overlap=True)
    frame = CandidateSet(candidates=(), level=lv, ds=ds, grids=tuple(grids), irretrievable_lvl=(), alter_lvl=())
    shape = (camera.output_h, camera.output_w)
    half = HALF_TILE * ds + 0.5               # half a tile, and half a pixel for a tile's rounded read point
    valid, n_no_region, n_no_window = [], 0, 0
    for fov_id, meta, image, params in shots:
        gx, gy = meta.fov_rect[0], meta.fov_rect[1]
        if region_of(regions, gx, gy, camera.rect_w_l0, camera.rect_h_l0) is None:
            n_no_region += 1
            continue
        rot = matching_rotation(params['rot_deg'])
        centre = query_grid_centre(camera, gx, gy, params, shape, TILE)
        found = answers(frame, dims[rot], centre, rot)
        if found is None or found.offset is None or not found.within(half):
            n_no_window += 1
            continue
        if (float(params['rot_deg']) % 360 == 0 and float(params['scale']) == 1.0
                and float(params.get('angle_jitter', 0.0)) == 0.0):
            # The one place this arithmetic can be checked against something
            # else: an upright, unscaled, unjittered shot's tile grid starts at the
            # FoV's own top-left. If the centre mapping is wrong it is wrong here first.
            start = (centre[0] - cols_q * HALF_TILE * ds, centre[1] - rows_q * HALF_TILE * ds)
            if abs(start[0] - gx) > 1.5 * ds or abs(start[1] - gy) > 1.5 * ds:
                raise AssertionError(
                    f'{stem} L{level} FoV {fov_id}: the query grid starts at '
                    f'({start[0]:.1f}, {start[1]:.1f}) but an upright shot starts at ({gx}, {gy})')
        valid.append((fov_id, (meta, image, params), found))
    if not valid:
        if not quiet:
            print(f'  L{level}  0/{len(shots)} FoV with an answer window -- skipped', flush=True)
        return None
    return Placed(ds, lv, regions, grids, camera, dims, valid, len(shots), n_no_region, n_no_window)


def run_slide_level(slide, dataset, stem, level, mask, args, encoder, arms,
                    bases, device, supply, shots, sink: Optional[SlideTables]) -> None:
    """Every FoV of one (slide, level), scored by every streamed arm, into the stage 2
    tables (`stage2_tables`). `shots` are this level's `(index, meta, image, params)`
    out of the slide's one draw (`supply`, the draw every bench takes its FoVs from);
    `index` is the FoV's row in the draw, and is its `fov_id`."""
    timer = PhaseTimer(fine=bool(getattr(args, 'timing', False)))
    timer.start(PLACE)
    placed = place_shots(slide, stem, level, mask, supply, shots)
    timer.stop(PLACE, len(shots))
    if placed is None:
        return
    ds, lv, regions, grids, camera, dims, valid = (
        placed.ds, placed.lv, placed.regions, placed.grids, placed.camera, placed.dims, placed.valid)
    skipped = (f'   skipped: {placed.n_no_region} outside every region, '
               f'{placed.n_no_window} with an answer window off the grid'
               if placed.n_no_region or placed.n_no_window else '')
    print(f'  L{level}  {len(valid)}/{placed.n_shots} FoV placed{skipped}', flush=True)

    timer.start(QUERY)
    # QGF: the query's tile features, once per shot, pooling and rotation. The
    # shot is turned the way the stage turns it (`np.rot90`, counter-clockwise)
    # and cut into whole tiles from the turned photo's top-left.
    per_base = {rot: {b: [] for b in bases} for rot in dims}
    for _, shot, _ in valid:
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
    timer.stop(RANK, len(valid))
    timer.start(TABLES)
    stage2_tables(slide, lv, ds, grids, valid, final, qgf, dims, arms, args, encoder,
                  device, sink, timer.seconds.get(STREAM, 0.0) / max(1, len(valid)))
    del qgf
    timer.stop(TABLES, len(valid))
    print(timer.report(level, len(valid)), flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  The retriever path: arms that are not the streamed sliding window (RetrieverArms.py)
# ══════════════════════════════════════════════════════════════════════════════

def list_queries(args: argparse.Namespace, level: int, slide: SafeSlide, placed: Placed) -> list[ListQuery]:
    """The level's FoVs as the retriever-path arms take them: the FoVs the pooling arms are scored on
    (`place_shots`), each with the point its truth is placed around."""
    out: list[ListQuery] = []
    fov_id: int
    for fov_id, shot, found in placed.valid:
        meta, image, params = shot
        centre = query_grid_centre(placed.camera, meta.fov_rect[0], meta.fov_rect[1], params,
                                   image.shape, TILE)
        out.append(ListQuery(
            fov_id=fov_id, image=image,
            estimate=placed_estimate(float(meta.ds), level, slide.base_mpp),
            centre=(float(centre[0]), float(centre[1])), rotation=found.main.rotation,
            level=level, ds=placed.ds))
    return out


def run_arms_level(engine: ArmEngine, slide: SafeSlide, stem: str, level: int, mask: Any,
                   args: argparse.Namespace, supply: Any, shots: Sequence[Any], sink: SlideTables) -> None:
    """Every phase of `engine` whose entry `sink` holds, on every FoV of one (slide, level) the pooling arms
    are scored on: each one's tables (`BenchCommon.write_phase`)."""
    placed: Optional[Placed] = place_shots(slide, stem, level, mask, supply, shots)
    if placed is None:
        return
    queries: list[ListQuery] = list_queries(args, level, slide, placed)
    started: float = time.perf_counter()
    res: LevelResult = engine.run(queries, level, sink)
    print(f'  L{level}  {len(engine.arm_names())} arms: {res.searched} searches over {len(queries)} FoV'
          f'{f", {res.skipped} skipped (not searched at this level)" if res.skipped else ""}   '
          f'{time.perf_counter() - started:.0f}s', flush=True)


def placed_levels(slide: SafeSlide, stem: str, levels: Sequence[int], mask: Any, supply: Any) -> Optional[set]:
    """The levels holding a FoV the arms are scored on, placed from the draw and the render table -- no photo
    rendered, no tile read. None when the render table is not written yet (the slide never ran)."""
    path = supply.render_entry.path('render', supply.microscope.cfg.identity_id(), '.csv')
    if not path.is_file():
        return None
    with open(path, newline='') as handle:
        params = {int(r['index']): {k: float(r[k]) for k in ('rot_deg', 'angle_jitter', 'scale')}
                  for r in csv.DictReader(handle)}
    metas = {i: s.meta for i, s in enumerate(supply.sampler)}
    out: set = set()
    for level in levels:
        shots = [(i, m, None, params[i]) for i, m in metas.items() if int(m.level) == level and i in params]
        if place_shots(slide, stem, level, mask, supply, shots, quiet=True) is not None:
            out.add(level)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  The rows: read off the stage 2 entries
#
#  An arm's rows are not kept anywhere but the cache: every one is read off its entries, by the one
#  definition of an answer and its ranks (BenchCommon). A ranked row is its first phase's truth (the ranks
#  it holds); a listed row is the answers' places in its last phase's list (`listed_places`), judged on its
#  first phase's answers. The FoV's own columns come from the draw and the render table.
# ══════════════════════════════════════════════════════════════════════════════

class ArmRows(NamedTuple):
    """How one arm's rows are read off the cache: its text, the entry of its first phase (whose truth judges
    it) and of its last (whose list it is), its family, the `source` its rows say, its lattices (`grid`), the
    length its list is cut at (None: all it holds), and whether it has a ranked mode (a first phase that
    scored every window)."""
    text: str
    first: str
    last: str
    family: str
    source: str
    grid: str
    cut: Optional[int]
    ranked: bool


def arm_rows_of(streamed_arms: Sequence[Stage2Arm], streamed_rows: dict, engine: Optional[ArmEngine]
                ) -> list[ArmRows]:
    """Every arm of the run that has rows: the streamed ones (`streamed_rows`, arm text -> entry) and the
    engine's."""
    by_entry: dict = {a.entry_id: a for a in streamed_arms}
    out: list[ArmRows] = [ArmRows(text, entry_id, entry_id, FAMILY_POOLING, 'stream', by_entry[entry_id].grid,
                                  None, True)
                          for text, entry_id in streamed_rows.items()]
    if engine is not None:
        for text in engine.arm_names():
            spec = parse_arm(text)
            reranker: bool = len(spec.segments) > 1
            out.append(ArmRows(text, engine.first_entry_of(text), engine.entry_of(text),
                               FAMILY_RERANK if reranker else FAMILY_METHOD, 'retriever', 'all',
                               spec.segments[-1].cut or None,
                               not reranker and spec.retriever.method != 'SlidingWindow'))
    return out


def _entry_table(entry: Any, entry_id: str, role: str) -> dict:
    """`{FoV index: rows}` of one table of an entry; empty when the entry has no such table."""
    path = entry.path(role, entry_id, '.csv')
    out: dict = {}
    if not path.is_file():
        return out
    with open(path, newline='') as handle:
        for r in csv.DictReader(handle):
            out.setdefault(int(r['index']), []).append(r)
    return out


def slide_rows(entry: Any, arms: Sequence[ArmRows], *, encoder: str, dataset: str, stem: str, supply: Any,
               min_sep: float) -> tuple[list, int]:
    """`(rows, FoVs two entries place differently)` of one slide: for every arm and every FoV its first phase's
    truth ranks, a ranked row (when the arm has that mode) and a listed row. The answers come from that truth
    (`answers_of_row`); every first phase places them by the same geometry, and a FoV they disagree on is
    counted, not silently compared."""
    path = supply.render_entry.path('render', supply.microscope.cfg.identity_id(), '.csv')
    with open(path, newline='') as handle:
        render = {int(r['index']): r for r in csv.DictReader(handle)}
    metas = {i: s.meta for i, s in enumerate(supply.sampler)}
    tables: dict = {}

    def table(entry_id: str, role: str) -> dict:
        if (entry_id, role) not in tables:
            tables[(entry_id, role)] = _entry_table(entry, entry_id, role)
        return tables[(entry_id, role)]

    rows: list = []
    seen: dict = {}                                   # FoV -> its main answer's window, as the first entry put it
    differ: set = set()
    for arm in arms:
        truth: dict = table(arm.first, 'truth')
        lists: dict = table(arm.last, 'output')
        for fov, (t, *_) in sorted(truth.items()):
            found: Optional[Answers] = answers_of_row(t)
            if found is None or t.get('pool_main', '') == '' or fov not in lists:
                continue                              # not searched at its own level: nothing to rank
            if seen.setdefault(fov, (found.main.key, found.main.box)) != (found.main.key, found.main.box):
                differ.add(fov)
            pools = Pools(main=int(float(t['pool_main'])),
                          offset=int(float(t['pool_all'])) - int(float(t['pool_main'])))
            listed_rows = sorted((r for r in lists[fov] if r.get('rank', '') != ''), key=lambda r: int(r['rank']))
            first_row = lists[fov][0]
            ds = float(first_row['ds'])
            full: bool = arm.grid == 'all'
            judged: Answers = found if full else found.main_only()
            score = t.get('truth_main_score', '')
            columns = dict(
                # First column on purpose: `--report-only` refuses a CSV set that mixes encoders on this key.
                encoder=encoder, dataset=dataset, slide=stem, level=int(float(t['truth_level'])), ds=ds,
                fov_id=fov, pool=pools.main, d_main=round(found.main.dist_l0 / ds, 2),
                d_overlap=round(found.offset.dist_l0 / ds, 2) if found.offset is not None else '',
                white_frac=round(float(metas[fov].score), 4), bucket=metas[fov].bucket,
                rot_deg=float(render[fov]['rot_deg']), scale=round(float(render[fov]['scale']), 4),
                source=arm.source, grid=arm.grid, family=arm.family, arm=arm.text,
                truth_rot=found.main.rotation, truth_score=round(float(score), 6) if score != '' else '',
                seconds=float(first_row.get('t_s') or 0.0))
            blank = {k: '' for k in RECALL_COLUMNS}

            def row_of(mode: str, ranks: dict, list_len: int) -> dict:
                recall = ({**{k: ranks[k] for k in RECALL_COLUMNS if k in ranks},
                           'pool_offset': pools.offset, 'pool_all': pools.all} if full else blank)
                # rank_overlap repeats rank_main ON PURPOSE: the shared `truth_rank` reads it when the offset
                # point is the nearer one, and THE answer is the main one
                return dict(columns, mode=mode, list_len=list_len, rank_main=ranks['rank_main'],
                            rank_overlap=ranks['rank_main'], **recall)

            if arm.ranked:
                stored = {k: int(float(t[f'truth_{k}'])) for k in ('rank_main', 'rank_main_in_all',
                                                                   'rank_offset_in_offset', 'rank_offset_in_all')
                          if t.get(f'truth_{k}', '') != ''}
                rows.append(row_of(MODE_RANKED, stored, pools.all))
            cut_rows = listed_rows if arm.cut is None else listed_rows[:arm.cut]
            listed = listed_of_rows(cut_rows)
            ranks = listed_places(listed, judged, pools, float(min_sep) * TILE * ds)
            rows.append(row_of(MODE_LISTED, ranks, len(listed)))
    return rows, len(differ)


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
                'list_len', *RECALL_COLUMNS}
    floats = {'d_main', 'd_overlap', 'white_frac', 'ds', 'rot_deg', 'scale',
              'truth_score', 'seconds'}
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


#: The code that turns a streamed arm with no pipeline config (a `raw` arm, `min`, `-M`) into its stage 2 entry, in
#: that entry's record (ConfigIdentity rule 3). Bump it only for a change made to be compared (another formula,
#: another way of scoring); a bug fix keeps it: remove the entries the bug had made wrong (TODO.log names them)
#: and the next run writes those again. Say why in TODO.log either way. How a row is read off an entry is
#: `BenchCommon`'s, and a change there is `TRUTH_VERSION`'s.
ROW_VERSION = 2      # 2: also the stage 2 tables (1: every rotation searched, the truth at the turned query's grid)


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


#: The modes' names in an older CSV.
OLD_MODES: dict[str, str] = {'whole': MODE_RANKED, 'production': MODE_LISTED}


def mode_of(row: dict) -> str:
    """An older CSV has no `mode` and is all ranked, or names the modes `whole` and `production`."""
    mode: str = row.get('mode') or MODE_RANKED
    return OLD_MODES.get(mode, mode)


def family_of(row: dict) -> str:
    """An older CSV has no `family` and is all streamed sliding-window arms."""
    return row.get('family') or FAMILY_POOLING


def normalise(rows: list) -> list:
    """The rows with every arm in the bench's grammar: an older CSV's `cls_avg+geomean` is
    `<encoder>:SlidingWindow::cls_avg+geomean`, named by the encoder its rows say."""
    for r in rows:
        r['arm'] = canonical_arm(r['arm'], r.get('encoder') or 'uni2')
        r['mode'] = mode_of(r)
    return rows


def report_arms(rows: list) -> list[str]:
    """The arms a report shows: the streamed sliding windows, then the other retrievers, then the
    rerankers, each in the order they first appear."""
    arms: list[str] = []
    for family in (FAMILY_POOLING, FAMILY_METHOD, FAMILY_RERANK):
        arms += dict.fromkeys(r['arm'] for r in rows if family_of(r) == family)
    return list(dict.fromkeys(arms))


def pick_baseline(rows: list, requested: str = '') -> str:
    """The arm every other is paired with: the one asked for, else `uni2:SlidingWindow::cls+mean`, else
    the first encoder's. The FoVs are the same whatever the encoder, so any encoder's arm can be the one."""
    names = {r['arm'] for r in rows if mode_of(r) == MODE_RANKED}
    if requested:
        if requested not in names:
            raise SystemExit(f'--baseline {requested!r} is not an arm of these rows')
        return requested
    for encoder in ['uni2'] + [r.get('encoder') or 'uni2' for r in rows]:
        if baseline_of(encoder) in names:
            return baseline_of(encoder)
    raise SystemExit('no `cls+mean` sliding window in these rows to pair the others with')


def check_pools(rows: list, baseline: str, emit=print) -> int:
    """Every arm of a FoV is ranked in the same pools. The retriever-path arms count theirs off the
    retriever's maps and the streamed arms off the streamed windows; they must be equal, and a FoV where
    they are not is a place the two do not agree on. Returns how many."""
    base = {(r['slide'], r['level'], r['fov_id']): (r['pool'], r.get('pool_offset'), r.get('pool_all'))
            for r in rows if r['arm'] == baseline and mode_of(r) == MODE_RANKED}
    bad = 0
    total = 0
    for r in rows:
        if family_of(r) == FAMILY_POOLING:
            continue
        want = base.get((r['slide'], r['level'], r['fov_id']))
        if want is None:
            continue
        total += 1
        bad += int((r['pool'], r.get('pool_offset'), r.get('pool_all')) != want)
    if total:
        emit(f'pools: {total - bad}/{total} retriever-path rows are ranked in the same pools as the '
             f'baseline' + ('' if not bad else f'  -- {bad} are NOT: those FoVs are not comparable'))
    return bad


def listed_report(rows: list, arms, emit=print) -> None:
    """What a listed arm's list could not rank. An answer the list does not hold has rank pool + 1, a miss
    at every k and f%, so each arm's ceiling is the share of FoVs whose answer is listed at all -- here,
    in the bench's own terms -- and the lists' length."""
    by_arm = group_by([r for r in rows if r.get('list_len') not in ('', None)], ['arm'])
    body = []
    for arm in arms:
        subset = by_arm.get((arm,))
        if not subset:
            continue
        main = np.array([r['rank_main'] <= r['pool'] for r in subset])      # the answer nearer the shot
        if all(r.get('pool_all') not in ('', None) for r in subset):
            either = np.array([min(r['rank_main_in_all'], r['rank_offset_in_all']) <= r['pool_all']
                               for r in subset])
            either_text = pct(float(either.mean())).strip()
        else:
            either_text = '-'
        lengths = np.array([r['list_len'] for r in subset], dtype=float)
        body.append([arm, len(subset), f'{np.median(lengths):.0f}', pct(float(main.mean())).strip(),
                     either_text])
    if not body:
        return
    emit(f'\n{"=" * 90}\nlisted -- how much of the answer a list holds at all\n{"=" * 90}')
    grid_table(['arm', 'FoVs', 'list length (med)', 'nearer answer listed', 'either answer listed'],
               body, aligns='lrrrr', emit=emit)
    emit('(listed = the window itself, or one of its rotation within the suppression radius -- the one that '
         'kept its place; an answer with none is rank pool + 1: a miss at every k and every f%)')


def curve_report(rows: list, arms, emit=print) -> None:
    """How the recall of a listed arm's list grows with its length: `recall_all@K` for each K of `K_CURVE`,
    per level, for each arm. A list holds its best K as a prefix, so one list gives the whole curve; a K
    past a level's pool is the level, and an answer not listed is never a hit however large K is. The
    place to read which K covers the most of the answer without making the list needlessly long."""
    full = [r for r in rows if r.get('rank_offset_in_offset') not in ('', None)]
    if not full:
        return
    emit(f'\n{"=" * 90}\nrecall_all@K of a listed arm -- how long a list has to be\n{"=" * 90}')
    for level, subset in group_levels(full).items():
        print_level_heading(subset, level, emit=emit, extras=LEVEL_EXTRAS)
        by_arm = group_by(subset, ['arm'])
        body = []
        for arm in arms:
            rs = by_arm.get((arm,))
            if not rs:
                continue
            best = np.array([min(r['rank_main_in_all'], r['rank_offset_in_all']) for r in rs])
            pool = np.array([r['pool_all'] for r in rs])
            body.append([arm] + [pct(float((best <= np.minimum(k, pool)).mean())).strip() for k in K_CURVE]
                        + [f'{np.median([r["list_len"] for r in rs]):.0f}'])
        grid_table(['arm'] + [f'K={k}' for k in K_CURVE] + ['list (med)'], body,
                   aligns='l' + 'r' * (len(K_CURVE) + 1), emit=emit)


def rerank_report(rows: list, emit=print) -> None:
    """Each reranker against the list it was given: every arm `A::R` paired with `A`, query by query, in
    the bench's own terms (wins / losses / ties, win%, ratio, top@f%). Both hold the same windows, so the
    ceiling is shared and what differs is only where the answer stands in the list."""
    children: dict[str, list[str]] = {}
    for arm in report_arms(rows):
        tokens = arm.split(SEP)
        if len(tokens) > 2 and any(family_of(r) == FAMILY_RERANK for r in rows if r['arm'] == arm):
            children.setdefault(SEP.join(tokens[:-2]), []).append(arm)
    if not children:
        return
    emit(f'\n{"=" * 90}\nrerankers against the list they were given (each paired with its own retriever)\n{"=" * 90}')
    for parent, kids in children.items():
        subset = [r for r in rows if r['arm'] in [parent, *kids]]
        paired = attach_baseline(subset, parent)
        if not paired:
            continue
        for level, part in group_levels(paired).items():
            print_level_heading(part, level, emit=emit, extras=LEVEL_EXTRAS)
            print_paired(part, f'  {parent}  ->', [parent, *kids], parent, emit=emit)


MODE_HEADINGS = {
    MODE_RANKED: 'RANKED (NMS off) -- every window of the level ranked by the arm\'s score, no suppression: '
                 'the search itself, only the answer\'s own window counting',
    MODE_LISTED: 'LISTED (NMS on) -- the longest list an arm hands on, with suppression; a shorter K is its '
                 'prefix. An answer is its window or a window within the suppression radius (the one '
                 'that kept its place); with none it is rank pool + 1'}


def tabulate(rows: list, per_slide: bool, baseline: str = '', emit=print) -> None:
    """The whole report: one block per mode, each paired with the baseline; for the listed mode also the
    recall curve over list length and each reranker against its own retriever."""
    rows = normalise(rows)
    base: str = pick_baseline(rows, baseline)
    emit(f'baseline (every arm is paired with it): {base}')
    check_pools(rows, base, emit=emit)
    for mode in (MODE_RANKED, MODE_LISTED):
        subset = [r for r in rows if mode_of(r) == mode]
        if not subset:
            continue
        emit(f'\n{"#" * 90}\n# {MODE_HEADINGS[mode]}\n{"#" * 90}')
        arms = report_arms(subset)
        report(attach_baseline(subset, base), arms, base,
               per_slide=per_slide, level_extras=LEVEL_EXTRAS, emit=emit)
        if mode == MODE_RANKED:
            recall_report(subset, arms, emit=emit)      # main vs offset vs all: a list has no lattice of its own
        if mode == MODE_LISTED:
            curve_report(subset, arms, emit=emit)
            listed_report(subset, arms, emit=emit)
            rerank_report(subset, emit=emit)


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
    # supply_for): the first --n-wsi of each dataset a --datasets id draws from,
    # one draw per slide across its levels, so a FoV here is the same photo the
    # pipeline bench and the stage-1 bench score under the same flags.
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS),
                        help='AccessDatasets ids: a registered dataset or a virtual one, `@<name>`')
    parser.add_argument('--n-wsi', type=int, default=N_WSI,
                        help='the first N slides of each dataset an id draws from')
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
                        help="the arms: `[encoder:]Method::pooling[+op]`, a retriever and then rerankers "
                             "joined by `::` (stage2_retrieval/ArmSpec.py), e.g. "
                             "`uni2:SlidingWindow::cls_avg+mean`, `uni2:BagOfTiles::rings3+quad`, "
                             "`uni2:TileVote::cls_avg+exact::Dual::cls_avg+dual@100`. A bare pooling "
                             "(`cls_avg`, `cls_avg-M`: main lattice only) is the sliding window with every "
                             "window score; `cls_avg+geomean` one of them. Every retriever of a run is "
                             "this run's --encoder. Default: CONFIG ARMS. `SlidingWindow::cls+mean` is the "
                             "baseline and is added if missing")
    parser.add_argument('--rerankers', nargs='*', default=[], metavar='RERANKER',
                        help='rerankers (`[encoder:]Method::pooling[+op][@K]`, e.g. `Dual::cls_avg+dual` '
                             '`gigapath:CSLS::grid2x2+csls@100`), each put after every arm of --rerank-on: '
                             'k retrievers x h rerankers are k x h arms. A reranker encodes for itself what '
                             'its pooling and encoder need. Only a listed-mode list can be reranked')
    parser.add_argument('--rerank-on', nargs='*', default=[], metavar='ARM',
                        help='the retriever arms (`uni2:SlidingWindow::cls+mean`, ...) the --rerankers go '
                             'after; each is also an arm of its own, so a reranker is read against it')
    parser.add_argument('--baseline', default='',
                        help='the arm every other is paired with in the report (default `uni2:SlidingWindow'
                             '::cls+mean`, else the first encoder\'s)')
    parser.add_argument('--k', type=int, default=PRODUCTION_K,
                        help=f'listed mode: the longest list an arm hands on (CONFIG {PRODUCTION_K}). '
                             f'A shorter K is its prefix, so one list gives the recall at every K; a level '
                             f'with fewer windows than K gives them all')
    parser.add_argument('--min-sep-tiles', type=float, default=PRODUCTION_MIN_SEP,
                        help=f'listed mode: a window is dropped when a better one of its '
                             f'rotation lies within this many tiles (CONFIG {PRODUCTION_MIN_SEP}, '
                             f'the stage\'s own); 0 keeps the raw top-k')
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
                             'run. They write into the SAME cache (a slide '
                             'belongs to one shard, so no entry is written '
                             'twice) and none of them writes the CSV: run '
                             '--assemble once they are all done')
    parser.add_argument('--assemble', action='store_true',
                        help='no model, no GPU: read every slide\'s rows off the '
                             'stage 2 entries the run these arguments describe '
                             'wrote, write stage2_retriever.csv and print the '
                             'tables. What a run does at its end, on its own -- '
                             'after sharded processes, or after a killed run')
    parser.add_argument('--gates-only', action='store_true',
                        help='run the gates on the first slide and stop: the '
                             'checks that take seconds, before a run of hours')
    parser.add_argument(
        '--out', default=None,
        help='output directory, used verbatim. Default: '
             'result/<SLURM_JOB_NAME or Stage2RetrieverBench>/<encoder>/ -- the '
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
    out_dir = Path(args.out or job_result_dir('Stage2RetrieverBench', encoder=tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    fov = FOV_RECIPES[args.fov]
    if fov.rungs != 'native':
        parser.error(f'--fov {args.fov} names its rungs {fov.rungs}; this bench '
                     f'scores each slide level by level, so it takes a recipe '
                     f"of native levels")
    args.max_ds = args.max_ds if args.max_ds is not None else fov.max_ds
    args.sensor = tuple(fov.sensor)
    print(f'bench_stage2_retriever   datasets {args.datasets}  '
          f'{args.n_wsi} slides each  ds <= {args.max_ds:g}')
    print(f'scores    {"  ".join(SCORES)}   baseline = {baseline_of(args.encoder)}\n')

    if args.report_only:
        if not args.csv:
            parser.error('--report-only needs at least one CSV path')
        # Any number of CSVs, of any encoders: an arm's name carries its encoder (an older CSV's is
        # made into the bench's grammar), so a table holds the arms of several runs side by side.
        rows = read_rows(args.csv)
        tabulate(rows, args.per_slide, args.baseline)
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
        # An arm is `[encoder:]Method::pooling[+op]`, a retriever and then rerankers (ArmSpec.py). The
        # sliding window is scored in one streamed pass over the level (every pooling and score at once,
        # keeping nothing of the slide's features); every other arm goes through the ArmEngine.
        # An older bare token -- `cls_avg`, `cls_avg+geomean`, `cls_avg-M` -- is the sliding window.
        arm_specs: list = []                    # (pooling, grid) of the streamed sliding windows
        score_filter: dict = {}                 # their scores: None = all
        engine_specs: list[ArmSpec] = []

        def stream(pooling: str, op: str) -> None:
            spec = parse_token(pooling, list(poolings) + [RAW])
            if spec not in arm_specs:
                arm_specs.append(spec)
            if not op:
                score_filter[spec] = None                                  # every score
            elif op not in SCORES:
                raise ValueError(f'{op!r} is not a window score ({", ".join(SCORES)})')
            elif score_filter.get(spec, ()) is not None:
                score_filter[spec] = tuple(dict.fromkeys(score_filter.get(spec, ()) + (op,)))

        def take(spec: ArmSpec, rerank_on: bool = False) -> None:
            if spec.retriever.encoder != args.encoder:
                raise ValueError(f'{spec.text()}: this run is --encoder {args.encoder}, and its retrievers '
                                 f'are that encoder\'s. Run {spec.retriever.encoder}\'s arms with '
                                 f'--encoder {spec.retriever.encoder} (a reranker may name another encoder)')
            if len(spec.segments) == 1 and spec.retriever.method == 'SlidingWindow':
                if spec.retriever.cut and not rerank_on:        # a list to rerank may be cut: its rerankers read the cut one
                    raise ValueError(f'{spec.text()}: a sliding-window arm on its own is always cut at --k '
                                     f'(its list for K = 30 ... is read off the prefixes); drop the '
                                     f'@{spec.retriever.cut}, or set --k')
                stream(spec.retriever.pooling, spec.retriever.op)
            else:
                engine_specs.append(spec)
                if spec.retriever.method == 'SlidingWindow':
                    stream(spec.retriever.pooling, spec.retriever.op)         # the list it starts from is scored too

        for token in tokens:
            if is_arm_text(token):
                take(parse_arm(token, args.encoder))
            else:
                name, _, op = token.partition('+')
                stream(name, op)
        on: list[ArmSpec] = [parse_arm(t, args.encoder) for t in args.rerank_on]
        for spec in on:
            take(spec, rerank_on=True)
        for spec in expand(on, [parse_segment(t) for t in args.rerankers]):
            take(spec)
        if args.rerankers and not on:
            raise ValueError('--rerankers needs --rerank-on: the retriever arms they go after')
        engine_specs = list({s.text(): s for s in engine_specs}.values())
    except ValueError as exc:
        parser.error(str(exc))
    if ('cls', 'all') not in arm_specs:
        arm_specs.insert(0, ('cls', 'all'))
        score_filter[('cls', 'all')] = ('mean',)
        print(f'[arms] {baseline_of(args.encoder)} is the baseline every other arm is paired '
              f'against, so it was added')
    elif score_filter.get(('cls', 'all'), ()) is not None and 'mean' not in score_filter[('cls', 'all')]:
        score_filter[('cls', 'all')] += ('mean',)
        print(f'[arms] {baseline_of(args.encoder)} is the baseline, so it was added to the `cls` arm')
    args.score_filter = {spec: tuple(s for s in SCORES if s in (SCORES if chosen is None else chosen))
                         for spec, chosen in score_filter.items()}
    bases = list(dict.fromkeys(b for b, _ in arm_specs))
    arms = [f'{arm_prefix(args.encoder)}{token_name(b, g)}+{s}' for b, g in arm_specs for s in scores_of(args, b, g)]
    print(f'arms      {"  ".join(arms)}')
    if engine_specs:
        print(f'engine    {len(engine_specs)} arm(s): {"  ".join(s.text() for s in engine_specs)}')
    print(f'listed: the {args.k} best windows, suppression {args.min_sep_tiles:g} tile(s)')

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
    # Every arm keeps nothing but its stage 2 entries; its rows are read off them (`slide_rows`), a slide at a
    # time, and the CSV is written from those. An arm is redone by removing its entries.
    streamed_arms = stage2_arms(args, arm_specs)
    streamed_wanted: dict[str, dict] = {a.entry_id: a.rec for a in streamed_arms}
    # the streamed arms that have rows (the scores asked for): arm text -> entry name
    streamed_rows: dict[str, str] = {
        f'{arm_prefix(args.encoder)}{token_name(a.base, a.grid)}+{a.score}': a.entry_id
        for a in streamed_arms if a.score in scores_of(args, a.base, a.grid)}
    engine = None
    engine_wanted: dict[str, dict] = {}                   # entry name -> its record (the entries it writes)
    if engine_specs:
        try:
            engine = ArmEngine(engine_specs, raw_topk=args.k, min_sep=args.min_sep_tiles, device=device,
                               encoders=EncoderPool(device), seg=args.mask_cfg.seg_id(),
                               region=args.mask_cfg.region_id())
        except (ValueError, NotImplementedError) as exc:
            parser.error(str(exc))
        engine_wanted = engine.wanted()
    # entry name -> the tables it holds: a streamed arm is a sliding window's first phase
    entry_roles: dict[str, tuple] = {a.entry_id: phase_roles(RetrieverSlidingWinSimRot, True, True)
                                     for a in streamed_arms}
    if engine is not None:
        entry_roles.update(engine.roles())
    arm_rows = arm_rows_of(streamed_arms, streamed_rows, engine)

    selected = run_slides(args)
    shard = None
    if args.shard and not args.assemble:
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

    job = Cache.job_name('Stage2RetrieverBench')
    fov_masks = MaskMaker(MASK_RECIPES['hest'], args.fov_mask_cache_job, device)
    mask_job = args.mask_cache_job or job
    print(f'masks     {mask_cfg.seg_id()}  cache {mask_job}')
    rows: list = []
    failed: list = []
    differ: int = 0

    def read_rows_of(dataset: str, stem: str, supply: Any, entry: Any) -> None:
        """The slide's rows, read off its entries, into the run's."""
        nonlocal differ
        got, apart = slide_rows(entry, arm_rows, encoder=tag, dataset=dataset, stem=stem, supply=supply,
                                min_sep=args.min_sep_tiles)
        rows.extend(got)
        differ += apart
        print(f'  {len(got)} rows read off the stage 2 entries'
              + (f'   {apart} FoV whose answers two entries place differently: their rows compare '
                 f'other windows' if apart else ''), flush=True)

    if args.assemble:
        # No model, no GPU: every slide's rows read off the entries the run (or its shards) wrote.
        for dataset, stem, path in selected:
            print(f'\n{stem}  ({dataset})', flush=True)
            try:
                supply = supply_for(str(path), fov_run, fov_masks, args, job)
                if supply is not None:
                    read_rows_of(dataset, stem, supply,
                                 supply.render_address.on(job).at(stage1=ORACLE).entry('stage2'))
            except Exception as exc:                          # noqa: BLE001
                print(f'  {type(exc).__name__}: {exc}')
                failed.append(stem)
        return finish_run(rows, failed, differ, None, None, out_dir, args)

    encoder = cfg.build(device)
    if RAW in bases and getattr(encoder.model_spec, 'kind', None) != 'tokens':
        parser.error(f'raw arms need an encoder with patch tokens, and '
                     f'{args.encoder} gives {encoder.model_spec}')
    print(f'device={device}  encoder={tag}  '
          f'model_spec={encoder.model_spec}  '
          f'dtype={encoder.cfg.model.dtype}  batch={cfg.batch_size}  '
          f'id={encoder.identity_id()}')
    print(f'{len(arms)} streamed sliding-window arms, baseline = {baseline_of(args.encoder)}')
    if engine is not None:
        engine.encoders.offer(encoder)                  # the streamed pass's encoder: loaded once for both
        engine.read_workers = args.budget.workers
        print(f'engine    {"  ".join(engine.arm_names())}')
    print()

    if not selected:
        print('no slides for this process (more shards than slides): nothing to do')
        return 0
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
                # the slide's FoVs, every level in one draw (cached), and with them where its
                # stage 2 entries are
                supply = supply_for(str(path), fov_run, fov_masks, args, job)
                if supply is None:
                    continue
                entry = supply.render_address.on(job).at(stage1=ORACLE).entry('stage2')
                # An entry holds every level of its slide and is written when the last level is done
                # (ArmTables): one that is missing, or does not reach every level holding a FoV the arms are
                # scored on, runs ALL those levels again. Those levels are placed from the render table, no
                # photo rendered; a slide that never ran has none, and runs every level.
                levels = list(levels_up_to(slide.level_downsamples, args.max_ds))
                want: Optional[set] = placed_levels(slide, stem, levels, mask, supply)
                run_levels = levels if want is None else sorted(want)

                def short(wanted: dict) -> set:
                    return {n for n, rec in wanted.items()
                            if want is None or not entries_complete(entry, {n: rec}, want, entry_roles)}

                streamed_bad: set = short(streamed_wanted)
                engine_bad: set = short(engine_wanted)
                if streamed_bad or engine_bad:
                    print(f'  entries to write: {len(streamed_bad)} of the pooling arms, {len(engine_bad)} of the '
                          f'others -- levels {run_levels}', flush=True)
                    sink = (SlideTables(entry, {i: streamed_wanted[i] for i in streamed_bad}, entry_roles)
                            if streamed_bad else None)
                    sink_arms = (SlideTables(entry, {n: engine_wanted[n] for n in engine_bad}, entry_roles)
                                 if engine_bad else None)
                    # rendered once and split by level
                    by_level: dict = {}
                    for shot in supply.shots(workers=args.budget.workers):
                        by_level.setdefault(int(shot[1].level), []).append(shot)
                    if engine_bad:
                        engine.build(slide, mask)
                    for level in run_levels:
                        shots = by_level.get(level, [])
                        if streamed_bad:
                            run_slide_level(slide, dataset, stem, level, mask, args, encoder,
                                            arm_specs, bases, device, supply, shots, sink)
                        if engine_bad:
                            run_arms_level(engine, slide, stem, level, mask, args, supply, shots, sink_arms)
                        print(f'  L{level}  {peak_memory_line()}', flush=True)
                    for label, tables in (('pooling arms', sink), ('other arms', sink_arms)):
                        if tables is None:
                            continue
                        written, empty = tables.flush()
                        print(f'  stage 2 entries of the {label}: {written} written -> {tables.entry.dir}'
                              + (f'   {len(empty)} left out (no rows: the arm failed or found nothing)'
                                 if empty else ''), flush=True)
                else:
                    print(f'  every entry is there (levels {run_levels})', flush=True)
                read_rows_of(dataset, stem, supply, entry)
            except Exception as exc:                          # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f'  {type(exc).__name__}: {exc}')
                failed.append(stem)
            finally:
                slide.close()
    return finish_run(rows, failed, differ, shard, engine, out_dir, args)


def finish_run(rows: list, failed: list, differ: int, shard: Optional[tuple], engine: Optional[ArmEngine],
               out_dir: Path, args: argparse.Namespace) -> int:
    """The CSV and the tables of the slides read (all of them, unless this is one shard of a run: the last
    step is then `--assemble`), and what failed."""
    print(f'\n{"=" * 78}')
    if shard is None:
        # The encoder is in the NAME as well as in every row: the name stops a
        # second encoder's run from overwriting the first one's numbers, the
        # column stops the two from being averaged together afterwards.
        write_csv(rows, out_dir / 'stage2_retriever.csv')
        if rows:
            tabulate(rows, args.per_slide, args.baseline)
    else:
        print(f'shard {shard[0]}/{shard[1]} done. Once every shard has finished, '
              f'run the same command with --assemble.')
    if differ:
        print(f'\n{differ} FoV(s) whose answers two entries place differently: one geometry should place them, '
              f'and does not -- look at those entries before reading their rows')
    if engine is not None and engine.failures:
        print(f'\n{len(engine.failures)} arm(s) failed and have no rows (the rest of the run is intact):')
        for arm, why in engine.failures.items():
            print(f'  {arm}   {why}')
    if failed:
        print(f'\n{len(failed)} slide(s) failed: {", ".join(failed)}')
    return 1 if failed or differ or (engine is not None and engine.failures) else 0


if __name__ == '__main__':
    sys.exit(main())
