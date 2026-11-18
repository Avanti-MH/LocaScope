'''Real photos, to be scored beside the simulated validation: observed, never selected on.

A list (`real_val/<name>.tsv`, made once by `real_val/make_list.py`) holds, for each photo, where it stands
on its slide: stage 3's homography `H` (photo px -> px of the stage 3 crop at `level`), the crop's level-0 corner and the
level and ds the crop was cut at. `placed` turns a row into a `Visits.Placed`, so the photo goes through the very same
`Visits.searched` a render does, and `real_stream` is `Visits.fov_stream` for them. The conditions are the stage 2 of
`M-realtest`: the photo as it is, its level known, its scale not (`placed_estimate`: the query is not rescaled).

    photos = read_list('gigapath-winembed_found50_missed50')                                 # once, before the first epoch
    for res in real_stream(photos, make_retriever, mask_job, device, label_cfg, workers, say, topk):
        ...                                                      # a FovResult, as the simulated validation's
'''
from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from math import sqrt
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from locascope.cli.bench.BenchCommon import Truth, query_grid_centre
from locascope.core.slides.AccessDatasets import locate
from locascope.core.slides.SafeSlide import SafeSlide
from locascope.core.tiles.TissueMaskConfig import MASK_RECIPES, MaskMaker
from locascope.training.RerankHead.Labels import LabelConfig
from locascope.training.RerankHead.Visits import FovResult, Placed, searched, tick

OBSERVE: str = 'real-photo (observe only)'      # THE label: the log title and the wandb prefix
DATASET: str = 'ki67_with_photo'
LISTS: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'real_val')     # <LISTS>/<name>.tsv
CENTRE_TOL_PX: float = 2.0                      # level-0 px: H(photo centre) against the list's center_x0, center_y0
#: d(photo px) / d(px of np.rot90(photo, rot // 90)), by rot (the table in `BenchCommon.query_grid_centre`, differentiated)
QUARTER: Dict[int, np.ndarray] = {
    0: np.array([[1.0, 0.0], [0.0, 1.0]]), 90: np.array([[0.0, -1.0], [1.0, 0.0]]),
    180: np.array([[-1.0, 0.0], [0.0, -1.0]]), 270: np.array([[0.0, 1.0], [-1.0, 0.0]])}
COLUMNS: Tuple[str, ...] = ('slide', 'index', 'group', 'level', 'ds', *(f'h{i}{j}' for i in range(3) for j in range(3)),
                            'crop_x0', 'crop_y0', 'center_x0', 'center_y0', 'height', 'width', 'path')


@dataclass(frozen=True)
class RealPhoto:
    slide: str
    index: int
    group: str                                  # the list's: 'found' | 'missed'
    path: str
    level: int
    ds: float
    h: Tuple[float, ...]                        # h00..h22: photo px -> px of the stage 3 crop, at `level`
    crop: Tuple[float, float]                   # crop_x0, crop_y0 (level-0)
    centre: Tuple[float, float]                 # center_x0, center_y0: stage 3's answer, the photo's centre (level-0)
    size: Tuple[int, int]                       # the photo's (height, width)

    def matrix(self) -> np.ndarray:
        return np.asarray(self.h, dtype=np.float64).reshape(3, 3)

    def to_level0(self, u: float, v: float) -> Tuple[float, float]:
        """The level-0 point photo pixel (u, v) stands on."""
        q: np.ndarray = self.matrix() @ np.array([u, v, 1.0])
        return (self.crop[0] + float(q[0] / q[2]) * self.ds, self.crop[1] + float(q[1] / q[2]) * self.ds)

    def output_to_level0(self, x: int, y: int, u: float, v: float, rot_deg: float = 0.0,
                         scale: float = 1.0) -> Tuple[float, float]:
        """Stands where `Render` stands in `query_grid_centre`: the homography already holds the turn and the scale."""
        return self.to_level0(u, v)

    def centre_error(self) -> float:
        """Level-0 px between where H puts the photo's centre and the list's centre."""
        h_img, w_img = self.size
        x, y = self.to_level0(w_img / 2.0, h_img / 2.0)
        return float(np.hypot(x - self.centre[0], y - self.centre[1]))


def list_path(name: str) -> str:
    return os.path.join(LISTS, f'{name}.tsv')


def read_list(name: str) -> List[RealPhoto]:
    """The photos of the list `name`, sorted by (slide, level, index). Read ONCE, before the first epoch. Raises when a row's
    determinant is <= 0, its `centre_error()` is over CENTRE_TOL_PX, or its file is missing: the list is made right or not used."""
    photos: List[RealPhoto] = []
    with open(list_path(name), newline='') as f:
        rows: csv.DictReader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        missing: List[str] = [c for c in COLUMNS if c not in (rows.fieldnames or [])]
        if missing:
            raise ValueError(f'{list_path(name)}: no column {missing}')
        for row in rows:
            p: RealPhoto = RealPhoto(
                slide=row['slide'], index=int(row['index']), group=row['group'], path=row['path'],
                level=int(row['level']), ds=float(row['ds']),
                h=tuple(float(row[f'h{i}{j}']) for i in range(3) for j in range(3)),
                crop=(float(row['crop_x0']), float(row['crop_y0'])),
                centre=(float(row['center_x0']), float(row['center_y0'])),
                size=(int(row['height']), int(row['width'])))
            where: str = f'{list_path(name)}: {p.slide} [{p.index}]'
            if not os.path.isfile(p.path):
                raise FileNotFoundError(f'{where}: no photo file {p.path}')
            if float(np.linalg.det(p.matrix()[:2, :2])) <= 0.0:
                raise ValueError(f'{where}: the determinant of H is not positive')
            if p.centre_error() > CENTRE_TOL_PX:
                raise ValueError(f'{where}: H puts the centre {p.centre_error():.2f} px off the list\'s (> {CENTRE_TOL_PX})')
            photos.append(p)
    return sorted(photos, key=lambda p: (p.slide, p.level, p.index))


def upright(p: RealPhoto) -> Tuple[int, float]:
    """`(rot, scale)`: the stage-2 rotation that sets the photo upright on the scan, and level px per photo px."""
    a: np.ndarray = p.matrix()[:2, :2]
    s: float = sqrt(float(np.linalg.det(a)))
    rot: int = min(QUARTER, key=lambda r: float(np.linalg.norm(a @ QUARTER[r] / s - np.eye(2))))
    return rot, s


def placed(p: RealPhoto) -> Placed:
    """Where the photo stands, for `Visits.searched`. The grid's centre in photo px is `query_grid_centre`'s, the one
    definition the simulated FoVs use."""
    rot, s = upright(p)
    params: Dict[str, float] = {'rot_deg': (360 - rot) % 360, 'scale': 1.0}      # matching_rotation(...) gives `rot` back
    h_img, w_img = p.size
    return Placed(truth=Truth(level=p.level, ds=p.ds, rot=rot, centre=partial(query_grid_centre, p, 0, 0, params, p.size)),
                  fov_centre=p.to_level0(w_img / 2.0, h_img / 2.0), fov_w0=w_img * s * p.ds, fov_h0=h_img * s * p.ds)


def _read_photo(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))


def real_stream(photos: Sequence[RealPhoto], make_retriever: Callable[[str], Any], mask_job: str, device: Any,
                label_cfg: LabelConfig, workers: int, say: Callable[[str], None], topk: Optional[int]) -> Iterator[FovResult]:
    """`Visits.fov_stream` for real photos: a slide at a time, its photos in (level, index) order, each through `Visits.searched`."""
    masks: MaskMaker = MaskMaker(MASK_RECIPES['hest'], mask_job, device)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for name, group in groupby(photos, key=lambda p: p.slide):           # `photos` is sorted
            mine: List[RealPhoto] = list(group)
            say(f'== {DATASET} {name}')
            t0: float = time.perf_counter()
            wsi: SafeSlide = SafeSlide(str(locate(name, dataset=DATASET).path))
            mask: Any = masks.mask(wsi)[0]
            tick('open_and_mask', time.perf_counter() - t0)
            retriever: Any = make_retriever(name)
            retriever.build(wsi, mask)
            encoded: set = set()                                  # the levels whose encoding has been timed
            images: Dict[int, Any] = {p.index: pool.submit(_read_photo, p.path) for p in mine}
            for p in mine:
                t0 = time.perf_counter()
                img: np.ndarray = images.pop(p.index).result()
                tick('photo_wait', time.perf_counter() - t0)
                res: Optional[FovResult] = searched(retriever, wsi, name, DATASET, p.index, img, placed(p),
                                                    label_cfg, topk, say, encoded)
                if res is not None:
                    yield res


__all__ = ['OBSERVE', 'DATASET', 'LISTS', 'RealPhoto', 'list_path', 'read_list', 'upright', 'placed', 'real_stream']
