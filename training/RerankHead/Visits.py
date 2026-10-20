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

import zlib
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from SafeSlide import SafeSlide
from SlideReader import SlideReader
from TileSampler import PlanSpec
from TissueMaskConfig import MASK_RECIPES, MaskMaker
from FovSupply import FovRecipe, FovSupply
from camera import Render
from BenchCommon import Truth, matching_rotation, placed_estimate, query_grid_centre
from stage2_retrieval.Context import WindowBatch, window_boxes
from stage2_retrieval.RerankerNeural import NeuralInputs, neural_inputs, pair_features
from stage2_retrieval.Retriever import RetrievalOutput
from stage2_retrieval.StageInterface import CandidateSet

from training.RerankHead.Labels import Boxes, LabelConfig, Labels, choose_subset, label_windows, truth_boxes


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


def fov_stream(slides: Sequence[Tuple[str, str, str]], make_retriever: Callable[[], Any],
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
    for dataset, name, path in slides:
        say(f'== {dataset} {name}')
        wsi: SafeSlide = SafeSlide(path)
        mask: Any = masks.mask(wsi)[0]
        reader: SlideReader = SlideReader(wsi)
        microscope: Render = Render(reader, fov.sensor, fov.gap, ds=1.0, seed=fov.sampler.seed)
        plan: PlanSpec = PlanSpec('ladder', fov.rungs_for(reader.level_downsamples), camera=microscope.spec)
        try:
            supply: FovSupply = FovSupply(microscope, plan, fov.sampler, mask)
            positions: List[Tuple[int, Any]] = [(i, s.meta) for i, s in enumerate(supply.sampler)]
        except RuntimeError as exc:
            say(f'  no FoV position at any level -- skipped ({str(exc).splitlines()[0]})')
            continue
        retriever: Any = make_retriever()
        retriever.build(wsi, mask)
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
            for j in chosen:
                index, meta = by_level[level][j]
                img, params = supply.photo(meta)
                geom: Dict[str, float] = supply.geometry(meta)
                truth: Truth = Truth(
                    level=int(geom['level']), ds=float(meta.ds), rot=matching_rotation(int(params['rot_deg'])),
                    centre=partial(query_grid_centre, supply.camera_for(meta.ds), geom['x0'], geom['y0'],
                                   params, img.shape))
                try:
                    out: RetrievalOutput = retriever.retrieve(
                        placed_estimate(truth.ds, truth.level, wsi.base_mpp), img, topk=topk)
                except Exception as exc:                                  # noqa: BLE001
                    say(f'  [{index}] L{level}: {type(exc).__name__}: {exc} -- skipped')
                    continue
                cs: CandidateSet = out.candidates
                if not len(cs) or cs.level != truth.level:
                    continue
                # the first phase's features: the frame and the query the list was made in (a chain's rerankers
                # may offer features of their own, nearer)
                src: Any = out.find('level_features')[-1].value
                shape_of: Callable[[int], Tuple[int, int]] = shape_fn(src)
                tile: int = int(cs.grids[0].tile_size)
                rows, cols = shape_of(truth.rot)
                where: Boxes = truth_boxes(truth.centre(tile), truth.rot, rows, cols, tile, cs.ds,
                                           (geom['center_x0'], geom['center_y0']), geom['w0'], geom['h0'],
                                           label_cfg)
                boxes: np.ndarray = window_boxes(cs, shape_of)
                labels: Labels = label_windows(boxes, np.array([c.rotation for c in cs.candidates]), where,
                                               tile * cs.ds, label_cfg)
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
    pair: Optional[torch.Tensor]
    slide: str
    level: int
    index: int
    n_total: int


def _t(a: np.ndarray, device: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return torch.as_tensor(a, device=device, dtype=dtype)


def prepare_fov(res: FovResult, pooling: str, label_cfg: LabelConfig, n_cand: int, hard_share: float,
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
    covers: np.ndarray = lab.iou[sel] >= label_cfg.pos_iou
    return FovData(ni=ni, pos=_t(lab.pos[sel], device), neg=_t(lab.neg[sel], device), covers=_t(covers, device),
                   rot_ok=_t(lab.rot_ok[sel], device), loc=_t(lab.loc[sel], device, torch.float32),
                   loc_mask=_t(lab.loc_mask[sel], device), pair=pair, slide=res.slide, level=res.level,
                   index=res.index, n_total=len(res.cs))


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
