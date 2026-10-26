#!/usr/bin/env python3
'''Build what AccessDatasets reads besides the slides themselves: the catalog, and the virtual datasets.

    python utilities/cli/build_cache/make_dataset.py --catalog                    # every registered dataset
    python utilities/cli/build_cache/make_dataset.py --catalog --datasets bracs/test --refresh
    python utilities/cli/build_cache/make_dataset.py --recipe pure                # writes @pure-train, @pure-val, @pure-test
    python utilities/cli/build_cache/make_dataset.py --recipe pure-half           # @pure-val in two: @pure-half-val, @pure-half-test
    python utilities/cli/build_cache/make_dataset.py --catalog --recipe pure         # the catalog first, then that
    python utilities/cli/build_cache/make_dataset.py --list                       # what is written, checked
    python utilities/cli/build_cache/make_dataset.py --bracs-prune                # what slimming BRACS deletes

Both live in AccessDatasets' own cache, `result/cache/<DATASETS_JOB>/`, whatever this job is called: every job
reads the same catalog and the same `@val`.

THE CATALOG is one row per slide of a registered dataset: its header (`base_mpp`, the level downsamples and
sizes) and the labels its layout gives it (BRACS's group and type). Building it opens each slide's header -- not
a pixel -- so it is a Slurm job; a slide already in the catalog is not opened again (`--refresh` reads every one).
`select(..., where=...)` reads it and never opens a slide itself.

A RECIPE is a function in `DATASET_DRAWS` that draws its virtual datasets with `AccessDatasets.select` and returns
`{'@name': Selection}`. Running it writes them all or none: one already written is refused (a virtual dataset is
written once, `save_virtual`), before anything is written. Recipes live here, in code, so how a dataset was drawn
is read and reviewed like any other code, and its record says which recipe made it.
'''
from __future__ import annotations

import argparse
import glob
import os
import random
import re
import sys
from itertools import combinations
from typing import Callable, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _paths import DATASETS_DIR, setup_import_paths                    # noqa: E402
setup_import_paths()

from AccessDatasets import (DATASETS_JOB, Member, Selection, SlideInfo, build_catalog,  # noqa: E402
                            dataset_ids, locate, save_virtual, select, vdataset_entry, virtual_ids,
                            virtual_record)
import AccessDatasets                                               # noqa: E402


# ── the recipes ──────────────────────────────────────────────────────────────

def native_32(slide: SlideInfo) -> bool:
    """The slide has a level of its own at ds 1, 4, 16 and 32: every one of those rungs is read, not resampled
    from a finer level."""
    return slide.native_at((1.0, 4.0, 16.0, 32.0))


#: BRACS's seven lesion types, the `type` label its layout gives each slide.
BRACS_TYPES: Tuple[str, ...] = ('Type_N', 'Type_PB', 'Type_UDH', 'Type_FEA', 'Type_ADH', 'Type_DCIS', 'Type_IC')

#: BRACS slides kept on disk per type, every one native at ds 1, 4, 16 and 32; the rest is deleted
#: (`--bracs-prune` lists them). BRACS train holds 20 Type_N native slides and gives one to test, so it keeps 19.
BRACS_KEEP_TRAIN: int = 20
BRACS_KEEP_TEST: int = 5

#: BRACS test holds 4 Type_N slides native at ds 1-32; this one of BRACS train is its fifth.
BRACS_TRAIN_TO_TEST: Tuple[str, str] = ('bracs/train', 'BRACS_1003675')


def _per_type(counts: Sequence[int]) -> Dict[str, int]:
    return dict(zip(BRACS_TYPES, counts))


#: The BRACS slides of each @pure set, per type (N, PB, UDH, FEA, ADH, DCIS, IC).
PURE_TRAIN_HE: Dict[str, int] = _per_type((14, 15, 14, 14, 14, 14, 15))       # 100
PURE_VAL_HE: Dict[str, int] = _per_type((3, 3, 3, 3, 3, 3, 2))                # 20
PURE_TEST_HE: Dict[str, int] = _per_type((2, 3, 2, 2, 2, 2, 3))               # 16
PURE_HALF_VAL_HE: Dict[str, int] = _per_type((2, 1, 2, 1, 2, 1, 1))           # 10 of PURE_VAL_HE
PURE_HALF_TEST_HE: Dict[str, int] = _per_type((1, 2, 1, 2, 1, 2, 1))          # the other 10


def by_type(sel: Selection, *parts: Dict[str, int], seed: int, at_most: bool = False) -> Tuple[Selection, ...]:
    """Disjoint parts of the BRACS slides of `sel`, each holding `part[type]` slides of every type: each type's
    slides sorted, shuffled by `seed` and cut in the order the parts are given. Asking a type for more than it
    holds is an error, unless `at_most` (then it gives what it has)."""
    picked: List[List[Member]] = [[] for _ in parts]
    for t in BRACS_TYPES:
        pool: List[Member] = sorted(sel.where(lambda s, t=t: s.labels.get('type') == t, label=t).members)
        random.Random(seed).shuffle(pool)
        wanted: int = sum(p.get(t, 0) for p in parts)
        if wanted > len(pool) and not at_most:
            raise ValueError(f'{t}: {wanted} asked, {len(pool)} in {sel.how}')
        at: int = 0
        for i, part in enumerate(parts):
            picked[i] += pool[at:at + part.get(t, 0)]
            at += part.get(t, 0)
    return tuple(Selection(members, f'{sel.how} by_type({[parts[i][t] for t in BRACS_TYPES]}, seed={seed})'
                                    f'{"[" + str(i) + "]" if len(parts) > 1 else ""}')
                 for i, members in enumerate(picked))


def bracs_kept() -> Tuple[Selection, Selection]:
    """The BRACS slides kept on disk, `(train, test)`: `BRACS_KEEP_TRAIN` of each type of BRACS train and
    `BRACS_KEEP_TEST` of BRACS test, every one native at ds 1, 4, 16 and 32, drawn by seed 0; BRACS test's
    Type_N is short of native slides and takes `BRACS_TRAIN_TO_TEST` from train. Drawn again once the rest is
    deleted, it is the same slides: each type then holds exactly what it keeps."""
    moved: Selection = Selection([BRACS_TRAIN_TO_TEST], f'{BRACS_TRAIN_TO_TEST[1]}')
    keep_train: Dict[str, int] = dict.fromkeys(BRACS_TYPES, BRACS_KEEP_TRAIN)
    keep_test: Dict[str, int] = dict.fromkeys(BRACS_TYPES, BRACS_KEEP_TEST)
    (train,) = by_type(select(dataset='bracs/train', where=native_32, label='native 1 4 16 32') - moved,
                       keep_train, seed=0, at_most=True)
    (test,) = by_type(select(dataset='bracs/test', where=native_32, label='native 1 4 16 32'),
                      keep_test, seed=0, at_most=True)
    return train, test | moved


def pure() -> Dict[str, Selection]:
    """Train 200 and val 40, half of each stain, apart, from the slides no test touches (Ki67_pure, BRACS
    train); test 32, half of each stain, from the slides the pipeline is judged on (Ki67_with_photo, BRACS
    test). Every slide is native at ds 1, 4, 16 and 32 (`native_32`). BRACS's slides come from those kept on
    disk (`bracs_kept`), as even across its seven types as the counts allow (`PURE_*_HE`). Each set is shuffled
    once more as a whole, so a prefix of it holds both stains."""
    pool: List[str] = ['ki67_pure', 'bracs/train']
    val_ki67, train_ki67 = select(dataset=pool, stain='Ki67', where=native_32,
                                  label='native 1 4 16 32').split(20, 100, seed=0)
    test_ki67: Selection = select(dataset='ki67_with_photo', where=native_32,
                                  label='native 1 4 16 32').sample(16, seed=0)
    kept_train, kept_test = bracs_kept()
    val_he, train_he = by_type(kept_train, PURE_VAL_HE, PURE_TRAIN_HE, seed=0)
    (test_he,) = by_type(kept_test, PURE_TEST_HE, seed=0)
    return {'@pure-train': (train_ki67 | train_he).sample(200, seed=0),
            '@pure-val': (val_ki67 | val_he).sample(40, seed=0),
            '@pure-test': (test_ki67 | test_he).sample(32, seed=0)}


def legacy() -> Dict[str, Selection]:
    """The split every package read before virtual datasets, drawn again the way it was drawn (`val=10,
    test=rest`, seed 42): the slides of BRACS test and of Ki67_with_photo native at ds 1, 4, 16 and 32
    (`native_32`), each dataset sorted, shuffled and cut val then test (`Selection.split`, the same sort,
    shuffle and cut). Each set holds BRACS's slides first, in their drawn order."""
    val_he, test_he = select(dataset='bracs/test', where=native_32,
                             label='native 1 4 16 32').split(10, 'rest', seed=42)
    val_ki67, test_ki67 = select(dataset='ki67_with_photo', where=native_32,
                                 label='native 1 4 16 32').split(10, 'rest', seed=42)
    return {'@legacy-val': val_he | val_ki67, '@legacy-test': test_he | test_ki67}


def pure_half() -> Dict[str, Selection]:
    """@pure-val cut in two, 10 of each stain apart: @pure-half-val is where methods and thresholds are chosen,
    @pure-half-test a test that keeps its ground truth -- a simulated stand-in for a real test set, which has
    none. BRACS's are cut by type (`PURE_HALF_*_HE`). Each set is shuffled once more as a whole, so a prefix of
    it holds both stains."""
    val_ki67, test_ki67 = select(dataset='@pure-val', stain='Ki67').split(10, 'rest', seed=0)
    val_he, test_he = by_type(select(dataset='@pure-val', stain='HE'), PURE_HALF_VAL_HE, PURE_HALF_TEST_HE, seed=0)
    return {'@pure-half-val': (val_ki67 | val_he).sample(20, seed=0),
            '@pure-half-test': (test_ki67 | test_he).sample(20, seed=0)}


#: name -> the function that draws its virtual datasets.
DATASET_DRAWS: Dict[str, Callable[[], Dict[str, Selection]]] = {
    'pure': pure,
    'legacy': legacy,
    'pure-half': pure_half,
}


# ── what the tool does ───────────────────────────────────────────────────────

def run_recipe(name: str) -> int:
    """Draw recipe `name` and write its virtual datasets, all or none."""
    drawn: Dict[str, Selection] = DATASET_DRAWS[name]()
    there: List[str] = [v for v in drawn if v in virtual_ids()]
    if there:
        print(f'{name}: {there} are written already -- nothing written (a virtual dataset is written once; delete '
              f'its entry in {vdataset_entry().dir} to write it again)', flush=True)
        return 1
    for vid, sel in drawn.items():
        by_source: Dict[str, int] = {}
        for d, _ in sel:
            by_source[d] = by_source.get(d, 0) + 1
        print(f'{vid:12s} {len(sel):5d} slides  {by_source}\n{"":12s} {sel.how}', flush=True)
    for (a, sa), (b, sb) in combinations(drawn.items(), 2):
        shared: int = len(sa & sb)
        if shared:
            print(f'  {a} and {b} share {shared} slides', flush=True)
    for vid, sel in drawn.items():
        print(f'  written {save_virtual(vid, sel, recipe=name)}', flush=True)
    return 0


def list_virtual() -> int:
    """Every virtual dataset written, each checked to be on disk now."""
    worst: int = 0
    for vid in virtual_ids():
        rec = virtual_record(vid)
        try:
            n: int = len(AccessDatasets.list_names(dataset=vid))
            state: str = 'ok'
        except FileNotFoundError as e:
            n, state, worst = rec.get('n', 0), f'BROKEN: {e}', 1
        print(f'{vid:12s} {n:5d} slides  recipe {rec.get("recipe") or "-":12s} {rec.get("created_at", "")}  '
              f'{state}\n{"":12s} {rec.get("how", "")}', flush=True)
    if not virtual_ids():
        print(f'no virtual dataset in {vdataset_entry().dir}')
    return worst


#: BRACS's RoI images, `<split>/<k>_<type>/<slide>_<type>_<n>.png`; the slide is the name's `BRACS_<id>`.
BRACS_ROI_ROOT: str = os.path.join(DATASETS_DIR, 'histoimage.na.icar.cnr.it/BRACS_RoI/latest_version')


def bracs_prune() -> int:
    """List what slimming BRACS deletes, and delete nothing: every BRACS slide not in `bracs_kept`, every RoI of
    such a slide, and every RoI of BRACS val (whose slides are not here at all). Written as one absolute path a
    line next to the virtual datasets, in `bracs_prune/`: `keep_wsi.txt`, `delete_wsi.txt`, `delete_roi.txt`."""
    kept_train, kept_test = bracs_kept()
    kept: set = set((kept_train | kept_test).names())
    wsi: Dict[str, str] = {n: locate(n, dataset=d).path for d in ('bracs/train', 'bracs/test')
                           for _, n in select(dataset=d)}
    rois: List[str] = sorted(glob.glob(os.path.join(BRACS_ROI_ROOT, '*', '*', '*.png')))
    slide_of: Dict[str, str] = {p: re.match(r'(BRACS_\d+)_', os.path.basename(p)).group(1) for p in rois}
    lists: Dict[str, List[str]] = {
        'keep_wsi': sorted(wsi[n] for n in kept),
        'delete_wsi': sorted(p for n, p in wsi.items() if n not in kept),
        'delete_roi': [p for p in rois
                       if slide_of[p] not in kept or os.path.relpath(p, BRACS_ROI_ROOT).split(os.sep)[0] == 'val']}
    out_dir: str = os.path.join(os.path.dirname(vdataset_entry().dir), 'bracs_prune')
    os.makedirs(out_dir, exist_ok=True)
    for role, paths in lists.items():
        out: str = os.path.join(out_dir, f'{role}.txt')
        with open(out, 'w') as handle:
            handle.writelines(p + '\n' for p in paths)
        gib: float = sum(os.path.getsize(p) for p in paths) / 1024 ** 3
        print(f'{role:11s} {len(paths):5d} files {gib:7.1f} GiB  {out}', flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalog', action='store_true',
                    help='write the catalog: a header read for each slide it does not hold yet; with --recipe, '
                         'first')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--recipe', nargs='+', choices=sorted(DATASET_DRAWS), metavar='NAME',
                      help=f'write the virtual datasets of these recipes ({", ".join(sorted(DATASET_DRAWS))})')
    mode.add_argument('--list', action='store_true', help='the virtual datasets written, each checked')
    mode.add_argument('--bracs-prune', action='store_true',
                      help='list the BRACS slides and RoIs slimming deletes (bracs_kept), deleting nothing')
    ap.add_argument('--datasets', nargs='+', choices=dataset_ids(), default=None,
                    help='--catalog: these datasets only (default every registered one)')
    ap.add_argument('--refresh', action='store_true', help='--catalog: read every header again')
    args = ap.parse_args()
    if not (args.catalog or args.recipe or args.list or args.bracs_prune):
        ap.error('say what to do: --catalog, --recipe NAME, --list, --bracs-prune (--catalog and --recipe go '
                 'together)')

    print(f'AccessDatasets cache: result/cache/{DATASETS_JOB}/', flush=True)
    if args.catalog:
        build_catalog(args.datasets, refresh=args.refresh)
    if args.list:
        return list_virtual()
    if args.bracs_prune:
        return bracs_prune()
    return max(run_recipe(name) for name in args.recipe) if args.recipe else 0


if __name__ == '__main__':
    raise SystemExit(main())
