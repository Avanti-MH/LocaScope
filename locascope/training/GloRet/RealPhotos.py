'''Real photos as validation FoVs: the photos of `ki67_with_photo` whose place a person confirmed (the checklist's `correct`
rows), read as `Supply.FovResult`s so that the heads are scored on them exactly as on the bench's FoVs (`train.validate`).

A real photo has no truth of its own; the truth is the confirmed place, taken from the real-photo job's cache (`RealTest`):
the checklist row says which photo and where (the stage 3 answer of the arm the person labelled from), the arm's stage 2
table says which window that answer was found at -- its box, its rotation, its level and ds. So the answer of a photo
is that window (and the nearest window of each lattice to its centre, `BenchCommon.answers`), the level is the one the
checklist names (an oracle stage 1, as the bench's `placed_estimate`: stage 1 is 99.5 % right on these photos), and nothing
of the photo's geometry is needed.

The photos are read for monitoring. Nothing here is a training set, and a head is not selected on them.
'''
from __future__ import annotations

import csv
import math
import re
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from locascope import paths as _paths
from locascope.core.slides.AccessDatasets import locate
from locascope.cli.bench.BenchCommon import Answers, answers, placed_estimate
from locascope.core.slides.SafeSlide import SafeSlide
from locascope.core.tiles.TissueMaskConfig import MASK_RECIPES, MaskMaker
from locascope.core.stage2_retrieval.Retriever import RetrievalOutput
from locascope.core.stage2_retrieval.StageInterface import ROTATIONS, CandidateSet

from locascope.training.GloRet.Supply import FovResult, tick

PHOTO_SUFFIXES: Tuple[str, ...] = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
SAME_PX: float = 24.0        # level-0 px: the answer of the arm and the checklist's place are the same place within this


@dataclass(frozen=True)
class RealPhoto:
    """One confirmed photo: where its file is, the level and ds it is searched at, the stage-2 rotation that sets it upright
    and the level-0 centre of the window it was found at."""
    slide: str
    index: int
    path: str
    level: int
    ds: float
    rot: int
    centre: Tuple[float, float]


def photo_paths(folder: str) -> Dict[int, Path]:
    """`{index: file}` of a slide's photos: the index is the number the file name ends in (`locate_photo.photo_files`'s rule);
    a folder whose names do not give each file one number is numbered in name order."""
    files: List[Path] = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES)
    numbers: List[Optional[re.Match]] = [re.search(r'(\d+)$', p.stem) for p in files]
    if files and all(numbers) and len({m.group(1) for m in numbers if m}) == len(files):
        return {int(m.group(1)): p for m, p in zip(numbers, files) if m}
    return dict(enumerate(files))


def slide_photos(name: str, job: str, checklists: str, stage1: str, stage2: str, dataset: str) -> List[RealPhoto]:
    """The confirmed photos of one slide that the arm `stage1 | stage2` of the cache `job` found at the confirmed place
    (the checklist rows labelled from that arm, `correct` = 1, answer within `SAME_PX` of the checklist's x0, y0)."""
    root: Path = Path(_paths.RESULT_DIR) / 'cache' / job / f'slide={name}'
    s1_dirs: List[Path] = sorted(root.glob(f'seg=*/region=*/photos=*/stage1={stage1}-*'))
    ck_file: Path = Path(checklists) / f'{name.replace(",", "_")}_mrxs.csv'
    entry: Any = locate(name, dataset=dataset)
    folder: Optional[str] = entry.related.get('photos')
    if not s1_dirs or not ck_file.is_file() or not folder or not Path(folder).is_dir():
        return []
    s1_dir: Path = s1_dirs[0]
    answer_files: List[Path] = sorted(s1_dir.glob(f'stage2={stage2}-*/stage3/answer_*.csv'))
    output_files: List[Path] = sorted((s1_dir / 'stage2').glob(f'output_{stage2}-*.csv'))
    if not answer_files or not output_files:
        return []
    ck: pd.DataFrame = pd.read_csv(ck_file)
    ck = ck[(ck['correct'] == 1) & ck['arm'].astype(str).str.contains(f'stage2={stage2}-', regex=False)]
    ans: pd.DataFrame = pd.read_csv(answer_files[0]).set_index('index')
    out: pd.DataFrame = pd.read_csv(output_files[0])
    paths: Dict[int, Path] = photo_paths(folder)
    found: List[RealPhoto] = []
    for _, row in ck.iterrows():
        index: int = int(row['index'])
        if index not in ans.index or index not in paths or int(ans.loc[index, 'retrieval_only']) == 1:
            continue
        a: pd.Series = ans.loc[index]
        if math.hypot(float(a['x0']) - float(row['x0']), float(a['y0']) - float(row['y0'])) > SAME_PX:
            continue                                             # the arm's answer is not the confirmed place
        w: pd.DataFrame = out[(out['index'] == index) & (out['rank'] == int(a['rank']))]
        if not len(w):
            continue
        r: pd.Series = w.iloc[0]
        found.append(RealPhoto(slide=name, index=index, path=str(paths[index]), level=int(r['level']), ds=float(r['ds']),
                               rot=int(r['rotation']),
                               centre=(float(r['x0']) + float(r['w0']) / 2.0, float(r['y0']) + float(r['h0']) / 2.0)))
    return found


def choose_photos(names: Sequence[str], total: int, seed: int, **where: Any) -> List[RealPhoto]:
    """`total` photos (0: all) drawn at random from the confirmed photos of the slides `names`, a seeded draw that does not
    depend on the order the slides are given in."""
    every: List[RealPhoto] = [p for n in sorted(names) for p in slide_photos(n, **where)]
    if total and len(every) > total:
        pick: np.ndarray = np.random.default_rng([seed, zlib.crc32(b'real-photos')]).choice(len(every), size=total, replace=False)
        every = [every[i] for i in sorted(pick.tolist())]
    return every


def write_list(photos: Sequence[RealPhoto], path: str) -> None:
    """The photos monitored, one row each, so that a run can be compared with another's on the same photos."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['slide', 'index', 'level', 'ds', 'rotation', 'centre_x0', 'centre_y0', 'path'])
        for p in photos:
            w.writerow([p.slide, p.index, p.level, p.ds, p.rot, p.centre[0], p.centre[1], p.path])


def real_stream(photos: Sequence[RealPhoto], slide_paths: Dict[str, str], make_retriever: Callable[[str], Any], mask_job: str,
                device: Any, workers: int, say: Callable[[str], None] = print, topk: Optional[int] = None,
                make_lister: Optional[Callable[[str], Any]] = None, own_topk: int = 1) -> Iterator[FovResult]:
    """`Supply.fov_stream` for real photos: a `FovResult` for each of `photos` (a photo whose level the retriever cannot
    search, or that has no window at its answer, is passed over, as the bench passes it). `slide_paths` is `{name: file}`.
    The meaning of the fields is `Supply.FovResult`'s; `FovResult.dataset` is `real`."""
    masks: MaskMaker = MaskMaker(MASK_RECIPES['hest'], mask_job, device)
    by_slide: Dict[str, List[RealPhoto]] = {}
    for p in photos:
        by_slide.setdefault(p.slide, []).append(p)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for name in sorted(by_slide):
            opened: float = time.perf_counter()
            wsi: SafeSlide = SafeSlide(slide_paths[name])
            mask: Any = masks.mask(wsi)[0]
            retriever: Any = make_retriever(name)
            retriever.build(wsi, mask)
            lister: Optional[Any] = make_lister(name) if make_lister is not None else None
            if lister is not None:
                lister.build(wsi, mask)
            tick('open_and_mask', time.perf_counter() - opened)
            encoded: set = set()                                   # the levels whose encoding has been timed (as `Supply` does)
            say(f'== real {name}: {len(by_slide[name])} photos')
            mine: List[RealPhoto] = sorted(by_slide[name], key=lambda p: (p.level, p.index))
            images: Dict[int, Any] = {p.index: pool.submit(lambda f: np.array(Image.open(f).convert('RGB')), p.path)
                                      for p in mine}
            for p in mine:
                waited: float = time.perf_counter()
                img: np.ndarray = images.pop(p.index).result()
                tick('photo_wait', time.perf_counter() - waited)
                estimate = placed_estimate(p.ds, p.level, wsi.base_mpp)
                searched: float = time.perf_counter()
                try:
                    out: RetrievalOutput = retriever.retrieve(estimate, img, topk=own_topk if lister is not None else topk)
                    listed: Optional[RetrievalOutput] = lister.retrieve(estimate, img, topk=topk) if lister is not None else None
                except Exception as exc:                                  # noqa: BLE001
                    say(f'  [{p.index}] L{p.level}: {type(exc).__name__}: {exc} -- skipped')
                    continue
                finally:
                    tick('retrieve', time.perf_counter() - searched)
                for who, seen in ((retriever, encoded), (lister, encoded)):
                    for lv, (seconds, tiles) in dict(getattr(who, 'level_encodes', None) or {}).items():
                        if (id(who), lv) not in seen:
                            seen.add((id(who), lv))
                            tick('level_encode', seconds)
                            tick('level_tiles', tiles)
                cs: CandidateSet = out.candidates
                own: Optional[CandidateSet] = None
                if listed is not None:
                    if not len(listed.candidates) or listed.candidates.level != cs.level:
                        continue
                    own, cs = cs, replace(cs, candidates=listed.candidates.candidates)
                if not len(cs) or cs.level != p.level:
                    continue
                made = out.find('level_features')
                if not made:
                    say(f'  [{p.index}] L{p.level}: the first phase made no level features -- skipped')
                    continue
                src: Any = made[-1].value
                tiles_extra = out.find('tiles')
                scored = (listed if listed is not None else out).find('window_scores')
                queries: Dict[int, torch.Tensor] = {rot: src.query_grid(rot, src.pooling).detach().clone() for rot in ROTATIONS}
                rows, cols = int(queries[p.rot].shape[0]), int(queries[p.rot].shape[1])
                found: Optional[Answers] = answers(cs, (rows, cols), p.centre, p.rot)
                if found is None:
                    continue
                tile_l0: float = float(cs.grids[0].tile_size) * float(cs.ds)
                yield FovResult(slide=name, dataset='real', level=p.level, index=p.index, cs=cs, source=src, found=found,
                                rot=int(p.rot), centre=(float(p.centre[0]), float(p.centre[1])), tile_l0=tile_l0,
                                tol_px=float(getattr(retriever.cfg, 'min_sep_tiles', 1.0)) * tile_l0, queries=queries,
                                own=own, tiles=(tiles_extra[-1].value if tiles_extra else None),
                                scores=(scored[-1].value if scored else None))


__all__ = ['RealPhoto', 'choose_photos', 'photo_paths', 'real_stream', 'slide_photos', 'write_list']
