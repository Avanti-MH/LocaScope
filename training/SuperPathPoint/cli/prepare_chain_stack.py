#!/usr/bin/env python3
"""Make F/R/C's own-shaped stacks ready against a real slide. plan.md 2.1.

    python training/SuperPathPoint/cli/prepare_chain_stack.py \
        --wsi <path> --wsi-stem <stem>              # one slide
    python training/SuperPathPoint/cli/prepare_chain_stack.py
        # no --wsi -- all 12 slides of the corpus (see _CORPUS_SLIDES)

Decides which axes to build and which corpus/`sampler_id` each reads from,
then calls that axis's `from_own()` -- the actual enumeration and lazy read
logic lives in `ChainStack.py` (`FStack`/`RStack`/`CStack.from_own`), not
here. This script is the thin orchestrator plan.md 2.1 calls for: for each
axis, make sure its own corpus exists and hand back a ready `from_own()`
result.

ONE JOB, 12 SLIDES (2026-09-06) -- `--wsi`/`--wsi-stem` OMITTED, NOT A FLAG.
================================================================================
`--wsi`/`--wsi-stem` used to be required, so building the whole corpus meant
12 separate `sbatch` submissions. Omitting BOTH now runs `_CORPUS_SLIDES` --
the same 12 slides `BuildMaskStore.sh:74-112` already builds masks for and
`stageA` already covers -- one after another IN THE SAME PROCESS, not a
second script. Passing `--wsi` still means exactly one slide, unchanged.

A FAILED SLIDE DOES NOT STOP THE BATCH. Each slide's `_run_slide` runs in its
own `try/except` when there is more than one -- the same "one bad cell should
not lose everything else" philosophy `extract_pretiles.py`'s `failures` list
already has. The end-of-run summary says which slides failed and why; the
process exits nonzero if any did, so a jobscript's `$?` still tells the
truth. Passing a single `--wsi` keeps the OLD behaviour of raising straight
through -- that path exists for ad hoc single-slide debugging, where a stack
trace is more useful than a summary line.

RESUMABLE THE SAME WAY `ExtractPreTiles.sh` ALREADY IS: `_has_store` is
checked before sampling, so a slide whose own corpus already exists (this
run or an earlier one) is skipped, not re-sampled. A walltime kill partway
through the 12 costs only the slides not yet reached.

OPTION B, NOT A (2026-09-06, reversed a same-day decision): F's and C's own
corpora are sampled DIRECTLY HERE -- `MaskStore`/`TissuesRegionsMask` to load
the slide's tissue mask, then `extract_pretiles._extract_slide` (refactored
this same day to take keyword arguments instead of `argparse.Namespace`,
specifically so this could call it) -- no subprocess, no
`ExtractPreTiles.sh` involvement at all for these two. `ExtractPreTiles.sh`
goes back to being only the `stageA` training-corpus script, unaware this
file exists. The earlier `CORPUS=custom` design (subprocess, knobs passed as
env vars) is gone -- keeping `ExtractPreTiles.sh` itself untouched by this
file's needs was the whole point of reversing it.

THREE DIFFERENT CORPORA, THREE DIFFERENT SHAPES (2026-09-06)
================================================================
    F  stageB-fOwn   inherited chains, share=1.0     -- FStack.from_own
    R  stageA        no chains, share=0, full ladder -- RStack.from_own
    C  stageB-cOwn   no chains, share=0, single rung -- CStack.from_own

R DOES NOT GET ITS OWN EXTRACTION. `stageA` (2026-08-27) already IS a batch
of independent, multi-rung real tiles -- exactly what R's own needs -- so
pointing at it is not a shortcut, it is the design (plan.md 2.1, confirmed
2026-09-06): R's own tile does not need a chain, and stageA already has no
chains either.

COMPUTING `sampler_id` DIRECTLY, NOT GUESSING (2026-09-06, second revision)
================================================================================
Two earlier versions of this file guessed which on-disk store belonged to
which axis instead of just computing the answer:

    v1  "exactly one sampler_id exists for this slide -> it must be this
        axis's own corpus". Silently picked up `stageA`'s sampler_id for F
        the moment `stageA` already existed and F's own corpus did not --
        which is every real slide on the very first run. `chains()` then
        found 0 chains and reported it as a clean result: no error, wrong
        answer. Exactly the failure mode ClaudeRules 8 exists to catch
        before an expensive run, not after one.
    v2  inspect what is ALREADY on disk (does a group have `inherit_id>=0`
        rows? does its `ds` coverage look like a full ladder or one rung?)
        and pick by those properties instead of by count. Correct, but it
        duplicated F's and C's own knobs a SECOND time -- once here (to
        compute what `sampler_id` an axis's corpus WOULD have, which turned
        out to still be needed for the "not found -> extract" branch) and
        once in `ExtractPreTiles.sh`'s `CORPUS` switch -- so the two could
        still drift from each other without either failing loudly.

`_sampler_config_for` is now the ONE place F's and C's own `SamplerConfig`s
are written down. `sampler_id()` is computed from it directly -- no
inspection of what is already on disk needed at all, because the identity
this axis's corpus WOULD have is already known before anything runs. Missing
-> `_extract_own` samples it directly with this SAME object (option B, see
above), so there is no second copy of the numbers for it to disagree with.
`stageA` gets the same treatment -- recomputing its `SamplerConfig` here
(rather than only trusting whatever `sampler_id` happens to be on disk)
means `ensure_sampler_id` can tell a real mismatch (this script's idea of
stageA's config drifted from `ExtractPreTiles.sh`'s own `stageA)` case)
apart from stageA simply not being extracted yet for this slide -- though R
never samples it either way (see `ensure_sampler_id`'s docstring).
"""

from __future__ import annotations

import argparse
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

from _paths import RESULT_DIR, setup_import_paths               # noqa: E402

setup_import_paths()

import AccessDatasets                                              # noqa: E402
import MaskStore                                                  # noqa: E402
import PreTileStore                                              # noqa: E402
from SafeSlide import SafeSlide                                   # noqa: E402
from SurvivalAnalysis import ChainStack                           # noqa: E402
from TileSampler import (SamplerConfig, RichnessConfig,          # noqa: E402
                         OverlapConfig, InheritConfig)
from TissuesRegionsMask import TissuesRegionsMask                 # noqa: E402
import extract_pretiles                                            # noqa: E402

DEFAULT_TILES_ROOT = os.path.join(RESULT_DIR, 'cache', 'tiles')
DEFAULT_MASK_ROOT = extract_pretiles.DEFAULT_MASK_ROOT

#: axis -> which recipe below builds its `SamplerConfig`.
_CORPUS_OF = {'F': 'stageB-fOwn', 'R': 'stageA', 'C': 'stageB-cOwn'}

#: THE ONE DEFINITION of every corpus's knobs (2026-09-06). Used for BOTH
#: computing `sampler_id()` (no extraction needed to know it) and, on a
#: miss, building the `SamplerConfig` `_extract_own` samples with directly
#: -- so there is exactly one place these numbers are written down. `stageA`'s
#: are "the values of 2026-08-27" (`ExtractPreTiles.sh`'s own words) and MUST
#: stay these values or every existing folder's `sampler_id` stops matching.
#: All three take `RichnessConfig`'s own default floors/caps (spec.md 6.5),
#: differing only in `bucket_frame`.
_RECIPES = {
    'stageA': dict(n=100, inherit_share=0.0, inherit_source_rung=None,
                  bucket_frame='per_rung',
                  grid_step=0, max_overlap=0.0, overlapping_share=0.0),
    'stageB-fOwn': dict(n=200, inherit_share=1.0, inherit_source_rung=16.0,
                       bucket_frame='at_inherit',
                       grid_step=128, max_overlap=0.5, overlapping_share=1.0),
    'stageB-cOwn': dict(n=10, inherit_share=0.0, inherit_source_rung=None,
                       bucket_frame='per_rung',
                       grid_step=0, max_overlap=0.0, overlapping_share=0.0),
}
#: 2500 // n, `_sampler_config`'s own formula (`extract_pretiles.py`) --
#: MAX_TRIES itself is a flat 2500 in `ExtractPreTiles.sh`, not per-corpus.
_MAX_TRIES = 2500

#: The 12-slide corpus (spec.md 13), by NAME -- `AccessDatasets.locate(name)`
#: resolves the path, so this file no longer carries a second copy of it.
#: MUST STAY IN SYNC with `BuildMaskStore.sh:74-112`'s `SLIDES` array (that
#: one is bash and needs its own paths, so it still hardcodes them); `stageA`
#: (`ExtractPreTiles.sh`, 2026-08-27) already covers exactly these 12,
#: confirmed against `result/cache/tiles/` on disk. Each name is exactly
#: what `MaskStore`/`PreTileStore` store as `wsi_stem`.
_CORPUS_SLIDE_NAMES = [
    'BRACS_1228', 'BRACS_1476', 'BRACS_1936', 'BRACS_1579', 'BRACS_1284',
    'S1104233,G7E,110208', 'S1104360,G7E,110208', 'S1151088,G7E,111220',
    'S1103520,G7E,110126', 'S1140701,G7E,111018',
    'BRACS_1598',                 # held out
    'S1103627,G7E,110127',        # held out
]
_CORPUS_SLIDES = [(name, AccessDatasets.locate(name).path)
                 for name in _CORPUS_SLIDE_NAMES]


def _sampler_config_for(corpus: str, tile: int) -> SamplerConfig:
    """`_RECIPES[corpus]`'s knobs as a real `SamplerConfig` -- computed
    before anything runs, so `sampler_id()` is known WITHOUT sampling
    anything, and it IS the config `_extract_own` samples with on a miss
    (not a separate reconstruction of it). See the module docstring for why
    this replaced inspecting what is already on disk.
    """
    r = _RECIPES[corpus]
    richness = RichnessConfig(bucket_frame=r['bucket_frame'])
    return SamplerConfig(
        tile=int(tile), n_per_rung=int(r['n']), seed=0, candidates='lattice',
        max_tries_per_tile=max(1, _MAX_TRIES // max(int(r['n']), 1)),
        overlap=OverlapConfig(grid_step=int(r['grid_step']),
                              max_overlap_ratio=float(r['max_overlap']),
                              overlapping_share=float(r['overlapping_share'])),
        richness=richness,
        # stack_kind is ALWAYS 'F' here regardless of share -- every plan
        # extract_pretiles.py builds comes from DsLadder, never
        # resolution_plan, so every plan it ever sees tags itself 'F'
        # (TileSampler.py's native_plans); this is not a place 'R' can occur.
        inherit=InheritConfig(stack_kind='F', share=float(r['inherit_share']),
                              source_rung=r['inherit_source_rung']))


def _has_store(tiles_root, wsi_stem: str, tile: int, sampler_id: str) -> bool:
    if not os.path.isdir(tiles_root):
        return False
    for folder in PreTileStore.find(tiles_root, tile=int(tile),
                                    sampler_id=sampler_id):
        if PreTileStore.load_meta(folder).wsi_stem == wsi_stem:
            return True
    return False


def _extract_own(corpus: str, wsi_path: str, wsi_stem: str, tiles_root: str,
                 tile: int, ds=None, mask_root=DEFAULT_MASK_ROOT) -> None:
    """Sample `corpus`'s own tiles for this one slide -- IN PROCESS, no
    subprocess, no `ExtractPreTiles.sh` (2026-09-06, option B). Mirrors what
    `extract_pretiles.main()` does per slide, just with `cfg` built from
    `_sampler_config_for` instead of argparse, and `ds` narrowed to what THIS
    axis actually needs (F: every rung `extract_pretiles`'s own default
    covers; C: one rung, the mother's).
    """
    cfg = _sampler_config_for(corpus, tile)
    ds = list(ds) if ds is not None else list(extract_pretiles.DEFAULT_RUNGS)
    mask_path = MaskStore.find_one(mask_root, wsi_stem=wsi_stem)
    slide_mask, mask_meta = MaskStore.load(mask_path)
    print(f'[{corpus}] no matching store yet -- sampling directly '
          f'(sampler_id={cfg.sampler_id()}) ...', flush=True)
    failures = []
    with SafeSlide(wsi_path) as wsi:
        trm = TissuesRegionsMask.from_mask(wsi, slide_mask.mask,
                                           slide_mask.origin, slide_mask.span)
        extract_pretiles._extract_slide(
            wsi, trm, slide_mask, mask_meta, cfg,
            tile=tile, pre_tile_factor=extract_pretiles.PRE_TILE_FACTOR,
            ds=ds, n=cfg.n_per_rung, root=tiles_root, overwrite=False,
            stem=wsi_stem, failures=failures)
    if failures:
        raise RuntimeError(f'{corpus} extraction failed for {wsi_stem}: '
                           f'{failures}')


def _warn_if_something_r_shaped_already_exists(tiles_root, wsi_stem: str,
                                               tile: int, sid: str) -> None:
    """SAFETY NET, not the primary check (that is `_has_store` on the
    computed `sid`). `R` reuses `stageA` -- the existing, 2026-08-27, 12-slide
    training corpus -- and if `_RECIPES['stageA']` ever drifts even slightly
    from `ExtractPreTiles.sh`'s own `stageA)` case, `_sampler_config_for`
    computes a `sid` that does NOT match what is actually on disk, and
    `ensure_sampler_id` would silently re-extract this slide under a THIRD,
    spurious `sampler_id` instead of erroring. This looks for anything
    already on disk that LOOKS like stageA (no chains, several `ds` folders)
    under a DIFFERENT `sampler_id` and refuses rather than duplicating it.
    """
    if not os.path.isdir(tiles_root):
        return
    suspects = {}
    for folder in PreTileStore.find(tiles_root, tile=int(tile)):
        meta = PreTileStore.load_meta(folder)
        if meta.wsi_stem != wsi_stem or meta.sampler_id == sid:
            continue
        g = suspects.setdefault(meta.sampler_id, set())
        g.add(float(meta.ds))
    wide = {s: ds for s, ds in suspects.items() if len(ds) >= 3}
    if wide:
        raise RuntimeError(
            f'about to extract a fresh "stageA" for {wsi_stem}, but '
            f'{len(wide)} store(s) that already look like a full ladder '
            f'exist under a DIFFERENT sampler_id: {sorted(wide)}. This '
            f'almost certainly means _RECIPES["stageA"] has drifted from '
            f'ExtractPreTiles.sh\'s own stageA) case rather than stageA '
            f'genuinely never having been extracted for this slide -- fix '
            f'the recipe, or pass --r-sampler-id to point at the right one '
            f'directly, rather than let this duplicate it')


def ensure_sampler_id(axis: str, tiles_root: str, wsi_stem: str, wsi: str,
                      tile: int, rungs, mother_ds,
                      mask_root=DEFAULT_MASK_ROOT) -> str:
    """The `sampler_id` of `axis`'s own corpus for this slide, extracting it
    first if nothing matching is there yet. Computed from `_sampler_config_for`
    directly -- see the module docstring for why this is no longer a guess.

    R DOES NOT AUTO-EXTRACT. `stageA` is the separate, human-run training
    corpus (`ExtractPreTiles.sh`'s own case) -- if it is missing for this
    slide, that is a real gap to go fill with THAT script, not something this
    one should quietly reproduce in-process as a side effect of wanting one
    slide's own tiles for R.

    `rungs` is what F actually extracts -- `_extract_own`'s `ds` for F, not
    `extract_pretiles.DEFAULT_RUNGS`.
    """
    corpus = _CORPUS_OF[axis]
    sid = _sampler_config_for(corpus, tile).sampler_id()
    if _has_store(tiles_root, wsi_stem, tile, sid):
        return sid
    if axis == 'R':
        _warn_if_something_r_shaped_already_exists(tiles_root, wsi_stem,
                                                   tile, sid)
        raise RuntimeError(
            f'stageA has no store for {wsi_stem} under {tiles_root} '
            f'(sampler_id={sid} not found). R reuses stageA and does not '
            f'extract it -- run ExtractPreTiles.sh for this slide first')
    ds = [mother_ds] if axis == 'C' else list(rungs)
    _extract_own(corpus, wsi, wsi_stem, tiles_root, tile, ds=ds,
                mask_root=mask_root)
    if not _has_store(tiles_root, wsi_stem, tile, sid):
        raise RuntimeError(
            f'sampled {corpus} for {wsi_stem} but no store with '
            f'sampler_id={sid} appeared under {tiles_root} -- check the '
            f'output above')
    return sid


def _run_slide(wsi_path: str, wsi_stem: str, args) -> dict:
    """Everything one slide needs: each requested axis's `sampler_id`
    (extracting its own corpus first if missing), then one `from_own()` read
    as a sanity check. Returns `{axis: sampler_id}` for the requested axes --
    `main()`'s batch mode uses it for the end-of-run summary; single-slide
    mode just lets the prints speak for themselves, same as before this was
    split out of `main()`.
    """
    mother_ds = max(args.c_rungs)
    sampler_id = {'F': args.f_sampler_id, 'R': args.r_sampler_id,
                 'C': args.c_sampler_id}
    for axis in args.axes:
        if sampler_id[axis] is None:
            sampler_id[axis] = ensure_sampler_id(
                axis, args.tiles_root, wsi_stem, wsi_path, args.tile,
                args.rungs, mother_ds, mask_root=args.mask_root)
        print(f'{axis}: sampler_id={sampler_id[axis]}', flush=True)

    if 'F' in args.axes:
        t0 = time.perf_counter()
        own = ChainStack.FStack.from_own(
            args.tiles_root, wsi_stem, tile=args.tile, rungs=args.rungs,
            sampler_id=sampler_id['F'])
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
            args.tiles_root, wsi_stem, args.rungs, tile=args.tile,
            sampler_id=sampler_id['R'], cache_root=args.cache_root)
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
                args.tiles_root, wsi_stem, args.c_rungs, wsi,
                tile=args.tile, sampler_id=sampler_id['C'],
                cache_root=args.cache_root)
            print(f'C: {len(forest)} trees, geometry built in '
                  f'{time.perf_counter()-t0:.2f}s (no wsi touched yet)',
                  flush=True)
            if len(forest):
                t0 = time.perf_counter()
                mother, mother_image, groups_by_ds, images_by_ds = forest[0]
                n = 1 + sum(len(v) for v in images_by_ds.values())
                print(f'   tree 0: mother ds {mother.ds:g}, {n} real tiles '
                      f'read in {time.perf_counter()-t0:.2f}s', flush=True)

    return {axis: sampler_id[axis] for axis in args.axes}


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
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help='F chain completeness, F own extraction + R output '
                         'rungs')
    ap.add_argument('--c-rungs', type=float, nargs='+',
                    default=[1.0, 2.0, 4.0, 8.0, 16.0],
                    help="C's mother is always the coarsest of these")
    ap.add_argument('--tiles-root', default=DEFAULT_TILES_ROOT)
    ap.add_argument('--mask-root', default=DEFAULT_MASK_ROOT,
                    help='where build_mask_store.py wrote the masks -- '
                         "needed when F's/C's own corpus has to be sampled")
    ap.add_argument('--axes', nargs='+', default=['F', 'R', 'C'],
                    choices=['F', 'R', 'C'])
    ap.add_argument('--f-sampler-id', default=None)
    ap.add_argument('--r-sampler-id', default=None)
    ap.add_argument('--c-sampler-id', default=None)
    ap.add_argument('--cache-root', default=None,
                    help="R/C's own local cache (see RStack.from_own's "
                         'docstring for why it defaults off)')
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
            done = ', '.join(f'{a}={sid}' for a, sid in r.items())
            print(f'  {wsi_stem}: {done}')
    print(f'{len(slides) - n_failed}/{len(slides)} slides OK')
    return 1 if n_failed else 0


if __name__ == '__main__':
    sys.exit(main())
