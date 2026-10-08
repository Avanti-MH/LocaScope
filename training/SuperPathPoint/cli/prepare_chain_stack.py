#!/usr/bin/env python3
"""Make F/R/C's own-shaped stacks ready against a real slide. plan.md 2.1.

    python training/SuperPathPoint/cli/prepare_chain_stack.py \
        --wsi <path> --wsi-stem <stem>              # one slide
    python training/SuperPathPoint/cli/prepare_chain_stack.py
        # no --wsi -- all 12 slides of the corpus (see _CORPUS_SLIDES)

Decides which corpus each axis reads, then calls that axis's `from_own()` --
the enumeration and lazy read logic lives in `ChainStack.py`
(`FStack`/`RStack`/`CStack.from_own`), not here. A corpus whose draw of a slide
is not in the cache yet is drawn by that call (`Corpora.Corpus.draw`).

ONE JOB, 12 SLIDES -- `--wsi`/`--wsi-stem` OMITTED, NOT A FLAG.
================================================================================
Omitting BOTH runs `_CORPUS_SLIDES` -- the same 12 slides `BuildMaskStore.sh`
builds masks for and `stageA` covers -- one after another in the same
process. Passing `--wsi` means exactly one slide.

A FAILED SLIDE DOES NOT STOP THE BATCH. Each slide's `_run_slide` runs in its
own `try/except` when there is more than one; the end-of-run summary says
which failed and why, and the process exits nonzero if any did. A single
`--wsi` raises straight through -- that path is for ad hoc debugging, where a
stack trace is more useful than a summary line.

RESUMABLE: a slide whose draws are cached is not re-sampled, so a walltime
kill costs only the slides not yet reached.

THREE CORPORA, THREE SHAPES (common/Corpora.py holds their knobs)
================================================================
    F  stageB-fOwn   inherited chains, share=1.0     -- FStack.from_own
    R  stageA        no chains, share=0, full ladder -- RStack.from_own
    C  stageB-cOwn   no chains, share=0, single rung -- CStack.from_own

R does not get a corpus of its own: stageA already IS a batch of independent,
multi-rung real tiles, which is what R's own needs.

THE CORPUS IS COMPUTED, NOT FOUND. Each axis's address is
`Corpora.corpus_of(recipe, ...)` -- known before anything runs, from the same
config the draw samples with, so there is nothing to guess and nothing to
drift: a reader that looked on disk for "the only sampler_id" would read
stageA as F the moment stageA existed and F's own did not.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'utilities'))
import _paths                                                     # noqa: E402
_paths.setup_import_paths('SuperPathPoint')

from cli import (add_chainstack_args, add_corpus_args,  # noqa: E402
                 chainstack_root, corpus_from_args)

import AccessDatasets                                              # noqa: E402
from SafeSlide import SafeSlide                                   # noqa: E402
from SurvivalAnalysis import ChainStack                           # noqa: E402
from common.Corpora import AXIS_RECIPE, Corpus                    # noqa: E402

#: The 12-slide corpus (spec.md 13), by NAME -- `AccessDatasets.locate(name)`
#: resolves the path. MUST STAY IN SYNC with `BuildMaskStore.sh`'s `SLIDES`
#: array (bash, so it hardcodes paths). Each name is what the mask and
#: draw caches key the slide by.
_CORPUS_SLIDE_NAMES = [
    'BRACS_1228', 'BRACS_1476', 'BRACS_1936', 'BRACS_1579', 'BRACS_1284',
    'S1104233,G7E,110208', 'S1104360,G7E,110208', 'S1151088,G7E,111220',
    'S1103520,G7E,110126', 'S1140701,G7E,111018',
    'BRACS_1598',                 # held out
    'S1103627,G7E,110127',        # held out
]
_CORPUS_SLIDES = [(name, AccessDatasets.locate(name).path)
                 for name in _CORPUS_SLIDE_NAMES]


def add_axis_corpus_args(ap) -> None:
    """`--r-rungs`: what `axis_corpus` reads besides `add_corpus_args`'s
    flags, `--rungs` and `--c-rungs`. Shared by every CLI that reads the three
    axes, so they address the same draws."""
    ap.add_argument('--r-rungs', type=float, nargs='+', default=None,
                    help="the rungs stageA is drawn over -- part of its "
                         "address. Default: DsLadder's")


def axis_rungs(axis: str, args):
    """The rungs `axis`'s corpus is drawn over -- part of its address. F over
    the rungs it must be complete over; C over its mother alone; R reads
    stageA, over DsLadder's default (None)."""
    return {'F': list(args.rungs), 'R': args.r_rungs,
            'C': [max(args.c_rungs)]}[axis]


def axis_corpus(axis: str, args) -> Corpus:
    """`axis`'s own corpus, computed from its recipe (`AXIS_RECIPE`)."""
    return corpus_from_args(args, AXIS_RECIPE[axis], axis_rungs(axis, args))


def _run_slide(wsi_path: str, wsi_stem: str, args) -> dict:
    """Everything one slide needs: each requested axis's corpus (drawn on its
    first read), then one `from_own()` read as a sanity check. Returns
    `{axis: corpus key}` for the requested axes."""
    corpora = {}
    for axis in args.axes:
        corpora[axis] = axis_corpus(axis, args)
        print(f'{axis}: {corpora[axis].key}', flush=True)

    if 'F' in args.axes:
        t0 = time.perf_counter()
        own = ChainStack.FStack.from_own(
            corpora['F'], wsi_stem, tile=args.tile, rungs=args.rungs)
        print(f'F: {len(own)} complete chains ({time.perf_counter()-t0:.2f}s '
              f'to enumerate)', flush=True)
        if len(own):
            i0 = next(iter(own))
            t0 = time.perf_counter()
            stack = own[i0]
            print(f'   chain {i0}: {sorted(stack)} rungs read in '
                  f'{(time.perf_counter()-t0)*1000:.0f} ms', flush=True)

    if 'R' in args.axes:
        t0 = time.perf_counter()
        own = ChainStack.RStack.from_own(
            corpora['R'], wsi_stem, args.rungs, tile=args.tile,
            cache_root=chainstack_root(args))
        print(f'R: {len(own)} own tiles ({time.perf_counter()-t0:.2f}s to '
              f'enumerate)', flush=True)
        if len(own):
            t0 = time.perf_counter()
            stack = own[0]
            print(f'   tile 0: {sorted(stack)} rungs derived in '
                  f'{(time.perf_counter()-t0)*1000:.0f} ms', flush=True)

    if 'C' in args.axes:
        t0 = time.perf_counter()
        with SafeSlide(wsi_path) as wsi:
            forest = ChainStack.CStack.from_own(
                corpora['C'], wsi_stem, args.c_rungs, wsi,
                tile=args.tile, cache_root=chainstack_root(args))
            print(f'C: {len(forest)} trees, geometry built in '
                  f'{time.perf_counter()-t0:.2f}s (no wsi touched yet)',
                  flush=True)
            if len(forest):
                t0 = time.perf_counter()
                mother, mother_image, groups_by_ds, images_by_ds = forest[0]
                n = 1 + sum(len(v) for v in images_by_ds.values())
                print(f'   tree 0: mother ds {mother.ds:g}, {n} real tiles '
                      f'read in {time.perf_counter()-t0:.2f}s', flush=True)

    return {axis: corpora[axis].key for axis in args.axes}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wsi-name', default=None,
                    help='look up --wsi/--wsi-stem via AccessDatasets -- '
                         'takes priority over setting them directly')
    ap.add_argument('--wsi', default=None,
                    help='one slide only. Omit together with --wsi-stem to '
                         'build the whole 12-slide corpus instead '
                         '(_CORPUS_SLIDES)')
    ap.add_argument('--wsi-stem', default=None)
    add_corpus_args(ap, corpus=False)
    ap.add_argument('--rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help='F chain completeness, F own draw + R output '
                         'rungs')
    ap.add_argument('--c-rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help="C's mother is always the coarsest of these")
    add_axis_corpus_args(ap)
    ap.add_argument('--axes', nargs='+', default=['F', 'R', 'C'],
                    choices=['F', 'R', 'C'])
    add_chainstack_args(ap, 'PrepareChainStack', on=False)   # R/C's own tiles: off
    args = ap.parse_args()

    if args.wsi_name:
        entry = AccessDatasets.locate(args.wsi_name)
        args.wsi, args.wsi_stem = entry.path, entry.name

    if (args.wsi is None) != (args.wsi_stem is None):
        ap.error('--wsi and --wsi-stem must be given together (or both '
                 'omitted, for the full 12-slide corpus)')

    if args.wsi is not None:
        slides = [(args.wsi_stem, args.wsi)]
    else:
        slides = _CORPUS_SLIDES
        print(f'no --wsi given -- building the full {len(slides)}-slide '
              f'corpus (spec.md 13, _CORPUS_SLIDES)', flush=True)

    batch = len(slides) > 1
    results = {}
    for wsi_stem, wsi_path in slides:
        if batch:
            print(f'\n---- {wsi_stem} ----', flush=True)
        try:
            results[wsi_stem] = _run_slide(wsi_path, wsi_stem, args)
        except Exception as exc:
            if not batch:
                raise
            print(f'FAILED: {exc}', flush=True)
            results[wsi_stem] = exc

    if not batch:
        return 0

    print('\n======== summary ========')
    n_failed = 0
    for wsi_stem, r in results.items():
        if isinstance(r, Exception):
            n_failed += 1
            print(f'  {wsi_stem}: FAILED -- {r}')
        else:
            done = ', '.join(f'{a}={key}' for a, key in r.items())
            print(f'  {wsi_stem}: {done}')
    print(f'{len(slides) - n_failed}/{len(slides)} slides OK')
    return 1 if n_failed else 0


if __name__ == '__main__':
    sys.exit(main())
