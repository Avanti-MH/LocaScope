#!/usr/bin/env python3
"""Make F/R/C's own-shaped stacks ready against a real slide. plan.md 2.1.

    python training/SuperPathPoint/cli/prepare_chain_stack.py \
        --wsi <path> --wsi-stem <stem>              # one slide
    python training/SuperPathPoint/cli/prepare_chain_stack.py
        # no --wsi -- all 12 slides of the corpus (see _CORPUS_SLIDES)

Decides which corpus each axis reads, makes sure it exists, then calls that
axis's `from_own()` -- the enumeration and lazy read logic lives in
`ChainStack.py` (`FStack`/`RStack`/`CStack.from_own`), not here.

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

RESUMABLE: a slide whose own corpus already has finished rungs is not
re-sampled, so a walltime kill costs only the slides not yet reached.

THREE CORPORA, THREE SHAPES (common/Corpora.py holds their knobs)
================================================================
    F  stageB-fOwn   inherited chains, share=1.0     -- FStack.from_own
    R  stageA        no chains, share=0, full ladder -- RStack.from_own
    C  stageB-cOwn   no chains, share=0, single rung -- CStack.from_own

F's and C's own corpora are sampled HERE, in process, through the same
`extract_pretiles._extract_slide` the stageA script uses -- no subprocess and
no second copy of the knobs. R does not get its own extraction: stageA
already IS a batch of independent, multi-rung real tiles, which is what R's
own needs, and it is the human-run training corpus (ExtractPreTiles.sh). R
reads it and never re-cuts it.

THE CORPUS IS COMPUTED, NOT FOUND. Each axis's directory is
`Corpora.corpus_of(recipe, ...)` -- known before anything runs, from the same
config the extraction samples with. Two earlier versions guessed instead: v1
took "the only sampler_id on disk for this slide" and silently read stageA as
F the moment stageA existed and F's own did not (0 chains, reported as a clean
result); v2 inspected what was on disk and duplicated F's and C's knobs a
second time. Addressing by the config leaves nothing to guess and nothing to
drift.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_HERE, '..', '..', '..')
for _p in (os.path.join(_REPO_ROOT, 'utilities'), os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _HERE not in sys.path:            # extract_pretiles.py lives right here
    sys.path.insert(0, _HERE)

from cli import (add_pretile_args, corpus_from_args,  # noqa: E402
                 job_result_dir, mask_root, setup_import_paths)

setup_import_paths()

import torch                                                      # noqa: E402

import AccessDatasets                                              # noqa: E402
from SafeSlide import SafeSlide                                   # noqa: E402
from Store import PreTileCorpus                                   # noqa: E402
from SurvivalAnalysis import ChainStack                           # noqa: E402
from TissueMaskConfig import MASK_RECIPES, MaskMaker              # noqa: E402
from common.Corpora import AXIS_RECIPE, recipe_config             # noqa: E402
import extract_pretiles                                            # noqa: E402

#: The 12-slide corpus (spec.md 13), by NAME -- `AccessDatasets.locate(name)`
#: resolves the path. MUST STAY IN SYNC with `BuildMaskStore.sh`'s `SLIDES`
#: array (bash, so it hardcodes paths). Each name is what the mask and
#: pre-tile caches key the slide by.
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
    """`--r-rungs` and `--{f,r,c}-corpus`: what `axis_corpus` reads besides
    `add_pretile_args`'s flags, `--rungs` and `--c-rungs`. Shared by every CLI
    that reads the three axes, so they address the same directories."""
    ap.add_argument('--r-rungs', type=float, nargs='+', default=None,
                    help="the rungs stageA was cut over -- part of its "
                         "address. Default: DsLadder's, which is what "
                         'ExtractPreTiles.sh cuts')
    for axis in ('f', 'r', 'c'):
        ap.add_argument(f'--{axis}-corpus', default=None,
                        help='a corpus key (printed by extract_pretiles) to '
                             'read instead of the computed one')


def axis_rungs(axis: str, args):
    """The rungs `axis`'s corpus is cut over -- part of its address. F is cut
    over the rungs it must be complete over; C over its mother alone; R reads
    stageA, which ExtractPreTiles.sh cuts over DsLadder's default (None)."""
    return {'F': list(args.rungs), 'R': args.r_rungs,
            'C': [max(args.c_rungs)]}[axis]


def axis_corpus(axis: str, args) -> PreTileCorpus:
    """`axis`'s own corpus: `--<axis>-corpus KEY` if given, else computed."""
    key = getattr(args, f'{axis.lower()}_corpus')
    if key:
        from cli import pretile_root                               # noqa: PLC0415
        return PreTileCorpus.from_key(pretile_root(args), key)
    return corpus_from_args(args, AXIS_RECIPE[axis], axis_rungs(axis, args))


def _extract_own(axis: str, corpus: PreTileCorpus, wsi_path: str, args,
                 masks: MaskMaker, rows: list = None) -> None:
    """Sample `axis`'s own corpus for one slide, in process, with the recipe's
    own config -- the object whose `sampler_id()` is in `corpus`'s address,
    not a reconstruction of it.

    `rows`, if given, is EXTENDED with the rows `_extract_slide` returns: a
    mutable accumulator the caller owns, as `failures` is. `main()`'s summary
    CSV is the one reader.
    """
    cfg = recipe_config(AXIS_RECIPE[axis])
    if cfg.sampler_id() != corpus.sampler_id:
        raise RuntimeError(
            f'{axis}: --{axis.lower()}-corpus names sampler {corpus.sampler_id}, '
            f'but the {AXIS_RECIPE[axis]} recipe is {cfg.sampler_id()}. An '
            f'explicit corpus is read, not extracted -- it has to exist')
    print(f'[{AXIS_RECIPE[axis]}] no finished rung yet -- sampling directly '
          f'into {corpus.key} ...', flush=True)
    failures = []
    with SafeSlide(wsi_path) as wsi:
        new_rows = extract_pretiles._extract_slide(
            wsi, masks, cfg, corpus, axis_rungs(axis, args), tile=args.tile,
            n=cfg.n_per_rung, overwrite=False, failures=failures)
    if rows is not None:
        rows.extend(new_rows)
    if failures:
        raise RuntimeError(f'{AXIS_RECIPE[axis]} extraction failed: {failures}')


def _other_corpora(corpus: PreTileCorpus, wsi_stem: str):
    """Sibling corpora of this slide under the same mask, with finished rungs:
    `{set directory name: [ds...]}`."""
    base = corpus.root / corpus.seg_id / wsi_stem
    mine = corpus.set_dir(wsi_stem)
    found = {}
    for f_dir in base.glob('*/f*'):
        if f_dir == mine:
            continue
        rungs = sorted(float(d.name[2:]) for d in f_dir.glob('ds*')
                       if (d / 'index.csv').exists())
        if rungs:
            found[f'{f_dir.parent.name}/{f_dir.name}'] = rungs
    return found


def ensure_corpus(axis: str, corpus: PreTileCorpus, wsi_stem: str,
                  wsi_path: str, args, masks: MaskMaker,
                  rows: list = None) -> PreTileCorpus:
    """`corpus`, extracting this slide's part of it first if it has no
    finished rung yet.

    R DOES NOT AUTO-EXTRACT. `stageA` is the separate, human-run training
    corpus (ExtractPreTiles.sh). If it is missing for this slide, that is a
    gap to fill with THAT script, not something to reproduce here as a side
    effect of wanting R's own tiles. And if something multi-rung already sits
    beside the computed address, the likelier story is that the stageA recipe
    here drifted from the one that cut it -- which a fresh extraction would
    paper over with a third corpus -- so that is named rather than guessed at.
    """
    if corpus.rung_dirs(wsi_stem):
        return corpus
    if axis == 'R':
        wide = {k: v for k, v in _other_corpora(corpus, wsi_stem).items()
                if len(v) > 1}
        hint = (f' Multi-rung corpora that DO exist for it: {wide}. If one is '
                f'stageA, the recipe in common/Corpora.py has drifted from '
                f'the one that cut it -- fix the recipe, or pass --r-corpus '
                f'with its key.' if wide else '')
        raise RuntimeError(
            f'stageA has no finished rung for {wsi_stem} at {corpus.key}. R '
            f'reads stageA and does not extract it -- run ExtractPreTiles.sh '
            f'for this slide first.{hint}')
    _extract_own(axis, corpus, wsi_path, args, masks, rows)
    if not corpus.rung_dirs(wsi_stem):
        raise RuntimeError(
            f'sampled {AXIS_RECIPE[axis]} for {wsi_stem} but no finished rung '
            f'appeared at {corpus.set_dir(wsi_stem)} -- check the output above')
    return corpus


def _run_slide(wsi_path: str, wsi_stem: str, args, masks: MaskMaker,
               rows: list = None) -> dict:
    """Everything one slide needs: each requested axis's corpus (extracting
    F's or C's own first if missing), then one `from_own()` read as a sanity
    check. Returns `{axis: corpus key}` for the requested axes."""
    corpora = {}
    for axis in args.axes:
        corpora[axis] = ensure_corpus(axis, axis_corpus(axis, args), wsi_stem,
                                      wsi_path, args, masks, rows)
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
            cache_root=args.cache_root)
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
                tile=args.tile, cache_root=args.cache_root)
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
    add_pretile_args(ap)
    ap.add_argument('--rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help='F chain completeness, F own extraction + R output '
                         'rungs')
    ap.add_argument('--c-rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help="C's mother is always the coarsest of these")
    add_axis_corpus_args(ap)
    ap.add_argument('--axes', nargs='+', default=['F', 'R', 'C'],
                    choices=['F', 'R', 'C'])
    ap.add_argument('--cache-root', default=None,
                    help="R/C's own local cache (see RStack.from_own's "
                         'docstring for why it defaults off)')
    ap.add_argument('--out', default=None,
                    help='directory for the summary CSV (default: '
                         "job_result_dir('PrepareChainStack'), NOT "
                         "ExtractPreTiles.sh's own directory -- this is a "
                         'different job, same row shape')
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
    rows = []
    # The device matters: `build(None)` is CPU, which is ~250x slower for the
    # uni2_pca segmenter (see extract_pretiles.main).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with MaskMaker(MASK_RECIPES[args.seg], mask_root(args), device) as masks:
        for wsi_stem, wsi_path in slides:
            if batch:
                print(f'\n---- {wsi_stem} ----', flush=True)
            try:
                results[wsi_stem] = _run_slide(wsi_path, wsi_stem, args,
                                               masks, rows=rows)
            except Exception as exc:
                if not batch:
                    raise
                print(f'FAILED: {exc}', flush=True)
                results[wsi_stem] = exc

    # SAME ROW SHAPE AS `extract_pretiles.py`'s OWN SUMMARY, same filename,
    # a DIFFERENT directory -- `job_result_dir` keys it off SLURM_JOB_NAME,
    # and this script's own jobscript names the job differently, so the two
    # never collide the way running ExtractPreTiles.sh twice under the same
    # name does (see this file's own history of that exact mistake, module
    # docstring). Only rows for extractions THIS run actually performed --
    # a cache HIT contributes none, matching `extract_pretiles.py`'s own
    # semantics of not recording rungs it skipped.
    if rows:
        out_dir = args.out or job_result_dir('PrepareChainStack')
        os.makedirs(out_dir, exist_ok=True)
        summary = os.path.join(out_dir, 'extract_pretiles.csv')
        with open(summary, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        total = sum(r['n_got'] for r in rows)
        gb = sum(r['bytes'] for r in rows) / 1e9
        print(f'\nSaved {summary}   ({len(rows)} cells, {total} pre-tiles, '
             f'{gb:.1f} GB on disk)')

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
