#!/usr/bin/env python3
"""Locate real microscope photos in their own slide: the mainstream run, with
no ground truth. The stage tables of every photo land in the cache, as the
bench's do for synthetic FoVs (utilities/bench_modules/bench_locascope.py).

    python utilities/cli/driver/locate_photo.py \\
        --stage1 knn:gigapath --stage2 slidewin:gigapath --stage3 sift:default

    --slide-index N     only the Nth slide of ki67_with_photo (a Slurm array task)
    --slides NAME ...   only these
    --limit N           only the first N photos of each slide (a smoke run)

SLIDES: every slide of ki67_with_photo -- all of them are real cases, there is
no val / test split here. A slide's photos are the folder `entry.related
['photos']` names.

STAGES: each one is `<method>:<recipe>` from its package's table, and every
recipe field is a `--stageN-<field>` flag, as in the bench. The stages are the
pipeline's (utilities/LocaScopePipeline.py), so a method that honours its
stage's interface runs here without a change.

CACHE: a slide's photos are one `photos=<id>` level of the job's tree (the id
hashes the folder's file names and sizes), under the slide, the mask recipe
and its region prep:

    <job>/slide=/seg=/region=/photos=<id>/
        shots/                      index -> file name, bytes, width, height
        stage1/                     stage1=<s1>/stage2/   stage2=<s2>/stage3/

Each stage's tables for a slide are ONE entry, written when the slide is done,
so a killed job redoes that slide (there is no resume inside one). A rerun
reads the entries that hit and loads no model for them. The tables are the
bench's, without the truth tables a real photo has no ground for:

    stage 1   output  neighbours  probs  votes  prototypes_index
    stage 2   output  tile_sims          (what the retriever offers is written)
    stage 3   output  matches  answer

`answer` is one row per photo: the verified candidate with the largest
`confidence` -- its centre, level-0 and fractional. Where every confidence is
0.0 that is the first candidate's own position, the retrieval's
(stage3_localization/StageInterface.py). Confidence is printed and written as
it is; it is not graded into labels, for the grades' thresholds are not
calibrated (log/TODO.log, 2026-08-08).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # utilities/
import _paths                                                        # noqa: E402
_paths.setup_import_paths()
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bench_modules'))

import Cache                                                         # noqa: E402
from AccessDatasets import list_names, locate                          # noqa: E402
from ConfigIdentity import short_id                                    # noqa: E402
from CpuBudget import CpuBudget                                        # noqa: E402
from TissueMaskConfig import MaskMaker, add_mask_args                  # noqa: E402
from stage2_retrieval.StageInterface import CandidateSet               # noqa: E402
from stage3_localization.StageInterface import LocalizationResultSet   # noqa: E402
from BenchCommon import (ROLES, RunClock, Shot, Stage, fmt_each, fmt_rate,  # noqa: E402
                         fmt_time, parse_run_args, require_gpu_if_allocated, run_from_args, run_slide_stages, status,
                         write_rows)

DATASET: str = 'ki67_with_photo'
PHOTO_SUFFIXES: Tuple[str, ...] = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')

#: The tables of each stage's entry: the bench's, without `truth` and
#: `truth_sim`, and stage 3's with the answer.
ROLES_REAL: Dict[int, Tuple[str, ...]] = {
    1: ROLES[1], 2: ('output', 'tile_sims'), 3: ('output', 'matches', 'answer')}


# ── a slide's photos ─────────────────────────────────────────────────────────

def photo_files(folder: str) -> List[Tuple[int, Path]]:
    """`(index, path)` of every image in `folder`, by index. The index is the
    number the file name ends in (`S1103520_ki67_100.bmp` -> 100), so it does
    not move when a photo is added; a folder whose names do not give one
    distinct number each is numbered in name order instead."""
    files: List[Path] = sorted(p for p in Path(folder).iterdir()
                               if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES)
    numbers: List[Optional[re.Match]] = [re.search(r'(\d+)$', p.stem) for p in files]
    if files and all(numbers) and len({m.group(1) for m in numbers if m}) == len(files):
        return sorted(((int(m.group(1)), p) for m, p in zip(numbers, files) if m),
                      key=lambda item: item[0])
    return list(enumerate(files))


def photo_set_id(files: Sequence[Tuple[int, Path]]) -> str:
    """One id for a folder of photos: each file as `name:bytes`, sorted by name,
    hashed. A photo added, removed, renamed or resized is another set; the
    folder moved to another path is the same."""
    parts: List[str] = sorted(f'{p.name}:{p.stat().st_size}' for _, p in files)
    return short_id(parts)


def write_shots(addr: Cache.Address, files: Sequence[Tuple[int, Path]]) -> None:
    """`shots/`: which file each index is, written once for the set."""
    entry = addr.entry('shots')
    rec: Dict[str, Any] = {'id': 'files', 'parts': [f'n={len(files)}'],
                           'upstream': {}, 'versions': {}, 'env': {}}
    if status(entry, 'files', rec, ('index',)) == 'hit':
        return
    rows: List[Dict[str, Any]] = []
    index: int
    path: Path
    for index, path in files:
        with Image.open(path) as im:
            width: int
            height: int
            width, height = im.size
        rows.append(dict(index=index, photo=path.name, bytes=path.stat().st_size,
                         width=width, height=height))
    with entry.writing('files', rec) as put:
        write_rows(put('index', '.csv'), rows)


def answer_row(rs: LocalizationResultSet) -> Dict[str, Any]:
    """The photo's answer: the verified candidate with the largest confidence,
    its centre. `retrieval_only` is 1 where no candidate was localized (every
    confidence 0.0) and the position is the retrieval's."""
    if not len(rs):
        return dict(error='no candidate verified')
    best = rs.best
    return dict(rank=best.rank + 1, confidence=best.confidence,
                x0=best.center_x0, y0=best.center_y0, level=best.level,
                ds=best.ds, retrieval_only=int(best.confidence == 0.0))


# ── one slide ────────────────────────────────────────────────────────────────

def run_slide(name: str, args: argparse.Namespace,
              stages: Tuple[Stage, Stage, Stage], masks: MaskMaker,
              mask_cfg: Any, own: str, run_clock: RunClock) -> None:
    entry = locate(name, dataset=DATASET)
    folder: Optional[str] = entry.related.get('photos')
    if not folder or not Path(folder).is_dir():
        print(f'== {name}: no photos folder -- skipped', flush=True)
        return
    every: List[Tuple[int, Path]] = photo_files(folder)
    files: List[Tuple[int, Path]] = every[:args.limit] if args.limit else every
    path: str = str(entry.path)
    base: Cache.Address = Cache.Address(
        own, slide=Cache.wsi_stem_of(path), seg=mask_cfg.seg_id(),
        region=mask_cfg.region_id(), photos=photo_set_id(every))
    print(f'== {name}   {len(files)} of {len(every)} photos   {folder}', flush=True)

    def photos() -> Iterator[Shot]:
        index: int
        photo: Path
        for index, photo in files:
            yield Shot(index=index, img=np.array(Image.open(photo).convert('RGB')),
                       label=photo.name)

    def add_answer(shot: Shot, route: str, cs: Optional[CandidateSet],
                   rs: Optional[LocalizationResultSet], tables: Any) -> None:
        """The photo's `answer` row beside stage 3's tables."""
        if cs is None or not len(cs):
            tables.add('answer', shot.index, [dict(error='no candidates')])
        else:
            tables.add('answer', shot.index,
                       [answer_row(rs)] if rs is not None
                       else [dict(error='stage 3 failed')])

    def describe(shot: Shot, route: str, cs: CandidateSet,
                 rs: Optional[LocalizationResultSet]) -> str:
        if rs is None or not len(rs):
            return ''
        best = rs.best
        return (f'  conf={best.confidence:.3f}'
                f'  at=({best.center_x0:.0f}, {best.center_y0:.0f})')

    started: float = time.perf_counter()
    ran: bool = run_slide_stages(
        path, photos(), stages, ['stage1'], masks, base, ROLES_REAL, own,
        limit=args.limit, truth=False, before=lambda: write_shots(base, every),
        after_stage3=add_answer, describe=describe, run_clock=run_clock)
    if ran:
        took: float = time.perf_counter() - started
        print(f'  whole slide, models loaded and cache written too: {len(files)} photo '
              f'in {fmt_time(took)} = {fmt_rate(len(files), took, "photo")} = '
              f'{fmt_each(len(files), took, "photo")}', flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False)
    ap.add_argument('--stage1', default='knn:gigapath', help='<method>:<recipe>')
    ap.add_argument('--stage2', default='slidewin:gigapath', help='<method>:<recipe>')
    ap.add_argument('--stage3', default='sift:default', help='<method>:<recipe>')
    ap.set_defaults(route='stage1')     # a real photo has no placed level: no oracle
    add_mask_args(ap)
    ap.add_argument('--mask-cache-job', default='MppRoutingHead',
                    help="whose mask cache the --seg mask is read from and "
                         'written to')
    ap.add_argument('--slides', nargs='*', default=[],
                    help=f'slide names; default every slide of {DATASET}')
    ap.add_argument('--slide-index', type=int, default=None,
                    help='only the Nth slide of the list (a Slurm array task)')
    ap.add_argument('--limit', type=int, default=0,
                    help='only the first N photos of each slide (a smoke run); '
                         'recorded in the entries, so a full run does not hit them')
    ap.add_argument('--batch-size', type=int, default=None,
                    help="stage 2 encoder's batch (not identity)")
    ap.add_argument('--multi-gpu', action='store_true')
    ap.add_argument('--device', default='auto')
    args: argparse.Namespace = parse_run_args(ap)

    names: List[str] = list(args.slides) or list_names(dataset=DATASET)
    if args.slide_index is not None:
        if not 0 <= args.slide_index < len(names):
            print(f'[skip] --slide-index {args.slide_index} but {len(names)} slides')
            return 0
        names = [names[args.slide_index]]

    import torch                                                     # noqa: PLC0415
    device: torch.device = torch.device(
        ('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'auto' else args.device)
    require_gpu_if_allocated(device, args.device)
    budget = CpuBudget.for_job(processes=1).apply()
    print(f'device     : {device}   {budget.line()}', flush=True)

    stages, routes, mask_cfg = run_from_args(
        args, device, batch_size=args.batch_size, multi_gpu=args.multi_gpu,
        read_workers=budget.workers)
    for s in stages:
        print(f'stage {s.n}    : {s.id}', flush=True)
    own: str = Cache.job_name('LocatePhoto')
    masks: MaskMaker = MaskMaker(mask_cfg, args.mask_cache_job, device)
    print(f'slides     : {len(names)} of {DATASET}   mask {args.seg} '
          f'{mask_cfg.seg_id()}/{mask_cfg.region_id()}   cache job {own}', flush=True)

    t_start: float = time.time()
    run_clock: RunClock = RunClock()
    name: str
    for name in names:
        run_slide(name, args, stages, masks, mask_cfg, own, run_clock)
    if run_clock.slides:
        print('\n' + run_clock.report(['stage1']), flush=True)
    print(f'\nTotal wall time: {fmt_time(time.time() - t_start)}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
