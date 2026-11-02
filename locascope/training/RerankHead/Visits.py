'''Synthetic FoVs through the first phase, made as they are used.

Nothing here is stored. A slide is opened, its tissue mask read, a draw of FoV
positions made over every level, and -- one LEVEL at a time, a visit -- the level
is encoded (by the retriever, the first time it is searched), photographs are
rendered, and each goes through the retriever, whose candidates and features are
handed to the caller one FoV after the next. The next slide replaces it all.

    stream = fov_stream(slides, make_retriever, fov, mask_job, device, label_cfg, ...)
    for res in stream:                       # a FovResult
        data = prepare_fov(res, ...)         # BEFORE asking for the next one

A `FovResult` is only valid until the next one is asked for: `res.source` (the
retriever's `level_features`, a `GridSource`) holds this query's features.
`prepare_fov` copies what a step needs.

WHAT A "VISIT" COSTS. Encoding a level is the expensive part (about two to three
minutes for a level of a few hundred thousand tiles; one slide's levels, about
four), and everything after is cheap, so `per_visit` FoVs are taken at once and a
caller trains on them several times before the level is dropped.
'''
from __future__ import annotations

import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from locascope.core.slides.SafeSlide import SafeSlide
from locascope.core.slides.SlideReader import SlideReader
from locascope.core.tiles.TileSampler import PlanSpec
from locascope.core.tiles.TissueMaskConfig import MASK_RECIPES, MaskMaker
from locascope.core.query_sim.FovSupply import FovRecipe, FovSupply
from locascope.core.query_sim.camera import Render
from locascope.cli.bench.BenchCommon import Answers, Truth, answers, matching_rotation, placed_estimate, query_grid_centre
from locascope.core.stage2_retrieval.Context import WindowBatch, window_boxes
from locascope.core.stage2_retrieval.RerankerNeural import NeuralInputs, neural_inputs, pair_features
from locascope.core.stage2_retrieval.Retriever import RetrievalOutput
from locascope.core.stage2_retrieval.StageInterface import CandidateSet

from locascope.training.RerankHead.Labels import Boxes, LabelConfig, Labels, choose_subset, label_windows, truth_boxes


# ══════════════════════════════════════════════════════════════════════════════
#  where the time goes, and what is kept between epochs
# ══════════════════════════════════════════════════════════════════════════════

_TIMERS: Dict[str, float] = defaultdict(float)
_TIMERS_LOCK: threading.Lock = threading.Lock()


def tick(name: str, seconds: float) -> None:
    """Add `seconds` to the timer `name` (any thread)."""
    with _TIMERS_LOCK:
        _TIMERS[name] += float(seconds)


def take_timers() -> Dict[str, float]:
    """The timers since the last call, and zero them."""
    with _TIMERS_LOCK:
        out: Dict[str, float] = dict(_TIMERS)
        _TIMERS.clear()
    return out


class MemoryFeatureStore:
    """The encoded levels of ONE slide, kept on the host between epochs: the retriever's `feature_store`
    (`Context.encode_level`). The features of a level do not change from one epoch to the next (the slide, the
    encoder and the pooling are the same; only the FoVs are new), and encoding them is most of the time. `save`
    copies a level's map to the host; `load` hands it back when the level, ds, tile cut, lattices and regions are
    the same, and the retriever moves it to the device."""

    def __init__(self) -> None:
        self._maps: Dict[Tuple[Any, ...], Any] = {}
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def _key(regions: Sequence[Any], ds: float, level: int, tile_size: int, overlap: bool) -> Tuple[Any, ...]:
        return (int(level), float(ds), int(tile_size), bool(overlap),
                tuple((int(r.x), int(r.y), int(r.w), int(r.h)) for r in regions))

    def load(self, regions: Sequence[Any], *, ds: float, level: int, tile_size: int, overlap: bool) -> Optional[Any]:
        found: Optional[Any] = self._maps.get(self._key(regions, ds, level, tile_size, overlap))
        self.hits += found is not None
        self.misses += found is None
        return found

    def save(self, features: Any) -> None:
        key: Tuple[Any, ...] = self._key(features.regions, features.ds, features.level, features.tile_size,
                                         features.overlap)
        self._maps[key] = features.to('cpu')

    def nbytes(self) -> int:
        return sum(int(m.features.numel()) * int(m.features.element_size())
                   for maps in self._maps.values() for m in maps.maps)


@dataclass
class FovResult:
    """One FoV and what the retriever made of it. Valid until the next one."""
    slide: str
    dataset: str
    level: int
    ds: float
    index: int
    img: np.ndarray
    cs: CandidateSet
    retriever: Any
    out: Any                      # what the retriever handed back (a chain's: its last phase's)
    source: Any                   # the first phase's `level_features`: this query's and this level's features
    where: Boxes
    boxes: np.ndarray             # [K, 4] candidate boxes
    labels: Labels
    queries: Dict[int, torch.Tensor]   # rotation -> the query's main tile grid, copied


def shape_fn(source: Any) -> Callable[[int], Tuple[int, int]]:
    """`(rows, cols)` of a window at a rotation, from the query as the retriever cut it (`source`, a
    `GridSource`)."""
    def shape_of(rotation: int) -> Tuple[int, int]:
        g: Any = source.query.containers[rotation].grid
        return int(g.grid_rows), int(g.grid_cols)
    return shape_of


def fov_stream(slides: Sequence[Tuple[str, str, str]], make_retriever: Callable[[str], Any],
               fov: FovRecipe, mask_job: str, device: Any, label_cfg: LabelConfig,
               per_visit: int, seed: int, epoch: int = 0, levels: Optional[Sequence[int]] = None,
               workers: int = 1, say: Callable[[str], None] = print,
               topk: Optional[int] = None) -> Iterator[FovResult]:
    """`slides` is `(dataset, name, path)`. For each, in the order given: the
    slide's FoVs of each level (all of them in a random order per level, at most
    `per_visit` each), through a fresh retriever, its list cut to `topk` (None:
    the whole list). FoVs whose level the retriever could not search (it went
    finer: the query did not fit) are passed over, as the bench passes them: their
    truth was never on offer."""
    masks: MaskMaker = MaskMaker(MASK_RECIPES['hest'], mask_job, device)
    # the photos of a level are taken by `workers` threads at once (`FovSupply.photo` is the same picture on
    # any thread) while the first of them is searched
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        yield from _fov_stream(slides, make_retriever, fov, masks, pool, label_cfg, per_visit, seed, epoch, levels,
                               say, topk)


def _fov_stream(slides: Sequence[Tuple[str, str, str]], make_retriever: Callable[[str], Any], fov: FovRecipe,
                masks: MaskMaker, pool: ThreadPoolExecutor, label_cfg: LabelConfig, per_visit: int, seed: int,
                epoch: int, levels: Optional[Sequence[int]], say: Callable[[str], None],
                topk: Optional[int]) -> Iterator[FovResult]:
    """`fov_stream` once its tools are made."""
    for dataset, name, path in slides:
        say(f'== {dataset} {name}')
        t0: float = time.perf_counter()
        wsi: SafeSlide = SafeSlide(path)
        mask: Any = masks.mask(wsi)[0]
        reader: SlideReader = SlideReader(wsi)
        tick('open_and_mask', time.perf_counter() - t0)
        microscope: Render = Render(reader, fov.sensor, fov.gap, ds=1.0, seed=fov.sampler.seed)
        plan: PlanSpec = PlanSpec('ladder', fov.rungs_for(reader.level_downsamples), camera=microscope.spec)
        t0 = time.perf_counter()
        try:
            supply: FovSupply = FovSupply(microscope, plan, fov.sampler, mask)
            positions: List[Tuple[int, Any]] = [(i, s.meta) for i, s in enumerate(supply.sampler)]
        except RuntimeError as exc:
            say(f'  no FoV position at any level -- skipped ({str(exc).splitlines()[0]})')
            continue
        tick('draw_positions', time.perf_counter() - t0)
        retriever: Any = make_retriever(name)
        retriever.build(wsi, mask)
        encoded: set = set()                                  # the levels whose encoding has been timed
        by_level: Dict[int, List[Tuple[int, Any]]] = {}
        for i, meta in positions:
            by_level.setdefault(int(meta.level), []).append((i, meta))
        order: List[int] = sorted(by_level)
        rng: np.random.Generator = np.random.default_rng([seed, epoch, zlib.crc32(name.encode())])
        rng.shuffle(order)
        for level in order:
            if levels is not None and level not in levels:
                continue
            chosen: List[int] = rng.permutation(len(by_level[level]))[:per_visit].tolist()
            photos: Dict[int, Any] = {j: pool.submit(supply.photo, by_level[level][j][1]) for j in chosen}
            for j in chosen:
                index, meta = by_level[level][j]
                t0 = time.perf_counter()
                img, params = photos[j].result()
                tick('photo_wait', time.perf_counter() - t0)
                geom: Dict[str, float] = supply.geometry(meta)
                truth: Truth = Truth(
                    level=int(geom['level']), ds=float(meta.ds), rot=matching_rotation(int(params['rot_deg'])),
                    centre=partial(query_grid_centre, supply.camera_for(meta.ds), geom['x0'], geom['y0'],
                                   params, img.shape))
                t0 = time.perf_counter()
                try:
                    out: RetrievalOutput = retriever.retrieve(
                        placed_estimate(truth.ds, truth.level, wsi.base_mpp), img, topk=topk)
                except Exception as exc:                                  # noqa: BLE001
                    say(f'  [{index}] L{level}: {type(exc).__name__}: {exc} -- skipped')
                    continue
                finally:
                    tick('retrieve', time.perf_counter() - t0)
                for lv, (seconds, tiles) in dict(getattr(retriever, 'level_encodes', {}) or {}).items():
                    if lv not in encoded:                     # the encoding (or the read back) is inside `retrieve`
                        encoded.add(lv)
                        tick('level_encode', seconds)
                        tick('level_tiles', tiles)
                cs: CandidateSet = out.candidates
                if not len(cs) or cs.level != truth.level:
                    continue
                # the first phase's features: the frame and the query the list was made in (a chain's rerankers
                # may offer features of their own, nearer)
                src: Any = out.find('level_features')[-1].value
                shape_of: Callable[[int], Tuple[int, int]] = shape_fn(src)
                tile: int = int(cs.grids[0].tile_size)
                rows, cols = shape_of(truth.rot)
                # the answers, as the benches define them; a FoV with none (no main window near enough) is passed
                # over, as the bench passes it
                found: Optional[Answers] = answers(cs, (rows, cols), truth.centre(tile), truth.rot)
                if found is None:
                    continue
                where: Boxes = truth_boxes(truth.centre(tile), truth.rot, rows, cols, tile, cs.ds,
                                           (geom['center_x0'], geom['center_y0']), geom['w0'], geom['h0'],
                                           label_cfg, [(a.key, a.origin) for a in found])
                boxes: np.ndarray = window_boxes(cs, shape_of)
                tol_px: float = float(getattr(retriever.cfg, 'min_sep_tiles', 1.0)) * tile * cs.ds
                labels: Labels = label_windows(boxes, np.array([c.rotation for c in cs.candidates]),
                                               [c.key() for c in cs.candidates], where, tile * cs.ds, label_cfg,
                                               tol_px)
                queries: Dict[int, torch.Tensor] = {
                    rot: src.query_grid(rot, src.pooling).detach().clone() for rot in (0, 90, 180, 270)}
                yield FovResult(slide=name, dataset=dataset, level=level, ds=float(cs.ds), index=index, img=img,
                                cs=cs, retriever=retriever, out=out, source=src, where=where, boxes=boxes,
                                labels=labels,
                                queries=queries)


@dataclass
class FovData:
    """What a training step needs of one FoV, copied out of the retriever: the
    tensors of a SUBSET of its candidates and their labels, on the device."""
    ni: NeuralInputs
    pos: torch.Tensor
    neg: torch.Tensor
    covers: torch.Tensor
    rot_ok: torch.Tensor
    loc: torch.Tensor
    loc_mask: torch.Tensor
    near: torch.Tensor
    near_target: torch.Tensor
    pair: Optional[torch.Tensor]
    slide: str
    level: int
    index: int
    n_total: int


def _t(a: np.ndarray, device: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return torch.as_tensor(a, device=device, dtype=dtype)


def prepare_fov(res: FovResult, pooling: str, n_cand: int, hard_share: float,
                rng: np.random.Generator, need_pair: bool = False) -> Optional[FovData]:
    """The training tensors of one FoV, or None when no candidate in its list is
    a positive (the truth was never proposed: nothing to rank)."""
    lab: Labels = res.labels
    if not lab.pos.any():
        return None
    sel: np.ndarray = choose_subset(lab, n_cand, hard_share, rng)
    cands: List[Any] = [res.cs.candidates[i] for i in sel]
    batch: WindowBatch = WindowBatch.build(res.cs, res.source, pooling, candidates=cands)
    device: Any = batch.groups[0].q.device
    ni: NeuralInputs = neural_inputs(
        batch, torch.tensor([c.score for c in cands], device=device),
        torch.tensor(sel + 1, device=device), [c.rotation for c in cands], [c.lattice for c in cands])
    pair: Optional[torch.Tensor] = None
    if need_pair:
        centres: np.ndarray = np.stack([(res.boxes[sel, 0] + res.boxes[sel, 2]) / 2.0,
                                        (res.boxes[sel, 1] + res.boxes[sel, 3]) / 2.0], axis=1)
        pair = pair_features(_t(centres, device, torch.float32),
                             _t(np.array([c.rotation for c in cands]), device),
                             float(res.cs.grids[0].tile_size) * float(res.cs.ds))
    return FovData(ni=ni, pos=_t(lab.pos[sel], device), neg=_t(lab.neg[sel], device),
                   covers=_t(lab.covers[sel], device), rot_ok=_t(lab.rot_ok[sel], device),
                   loc=_t(lab.loc[sel], device, torch.float32), loc_mask=_t(lab.loc_mask[sel], device),
                   near=_t(lab.near[sel], device), near_target=_t(lab.soft[sel], device, torch.float32),
                   pair=pair, slide=res.slide, level=res.level, index=res.index, n_total=len(res.cs))


def full_inputs(res: FovResult, pooling: str) -> Tuple[NeuralInputs, Optional[torch.Tensor]]:
    """The tensors of ALL of a FoV's candidates (evaluation scores the whole
    list), and the pair features for a model that wants them."""
    cands: List[Any] = list(res.cs.candidates)
    batch: WindowBatch = WindowBatch.build(res.cs, res.source, pooling)
    device: Any = batch.groups[0].q.device
    ni: NeuralInputs = neural_inputs(batch, torch.tensor([c.score for c in cands], device=device),
                                     torch.arange(1, len(cands) + 1, device=device),
                                     [c.rotation for c in cands], [c.lattice for c in cands])
    centres: np.ndarray = np.stack([(res.boxes[:, 0] + res.boxes[:, 2]) / 2.0,
                                    (res.boxes[:, 1] + res.boxes[:, 3]) / 2.0], axis=1)
    pair: torch.Tensor = pair_features(_t(centres, device, torch.float32),
                                       _t(np.array([c.rotation for c in cands]), device),
                                       float(res.cs.grids[0].tile_size) * float(res.cs.ds))
    return ni, pair


__all__ = ['FovResult', 'FovData', 'fov_stream', 'prepare_fov', 'full_inputs', 'shape_fn']
