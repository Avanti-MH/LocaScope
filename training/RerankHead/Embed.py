'''Training and scoring `WinEmbedder`, the learned window vector.

Every window of a visit's level is a candidate for every one of its FoVs, at each
of the four rotations the query is tried at, and the loss is the retrieval itself:
a softmax over ALL of them, in which the windows that cover the truth at the right
rotation are the positives and the negatives are the windows disjoint from the
photo and the windows that cover the truth at a wrong rotation. A window that
overlaps the photo some other way is in neither (`Labels.py`).

The level's tiles are projected once per step and cut into windows of each shape;
the queries of the step's FoVs share that pool.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from stage2_retrieval.StageInterface import CandidateSet

from training.RerankHead.Labels import Boxes, LabelConfig, label_windows
from training.RerankHead.Losses import pool_softmax_loss
from training.RerankHead.Metrics import first_hit_rank
from training.RerankHead.Visits import FovResult, shape_fn

ROTATIONS: Tuple[int, ...] = (0, 90, 180, 270)


@dataclass
class EmbedFov:
    """The part of a FoV the embedding's loss needs: its query's tile grids and
    where it was placed. Nothing of the retriever's per-query state."""
    queries: Dict[int, torch.Tensor]
    where: Boxes
    slide: str
    level: int
    index: int


@dataclass
class EmbedVisit:
    """A level of one slide, and the FoVs placed on it."""
    grids: Dict[Tuple[int, str], torch.Tensor]                 # (region, lattice) -> [R, C, F]
    origin: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]]   # -> (x of each column, y of each row), level-0
    tile_l0: float                                             # a tile's side in level-0 px
    fovs: List[EmbedFov]


def visit_of(res: FovResult, pooling: str) -> EmbedVisit:
    """A new visit from the first FoV of a level: the retriever's grids of this level
    (references; they stay valid after the retriever moves on) and the level-0 origin
    of every row and column of each."""
    grids: Dict[Tuple[int, str], torch.Tensor] = {}
    origin: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}
    for region, pg in enumerate(res.cs.grids):
        for lattice in ('main', 'offset'):
            g: torch.Tensor = res.source.reference_grid(region, lattice, pooling)
            if not g.numel():
                continue
            grids[(region, lattice)] = g
            xs: List[int] = [pg.tile_origin_l0(lattice, 0, c)[0] for c in range(int(g.shape[1]))]
            ys: List[int] = [pg.tile_origin_l0(lattice, r, 0)[1] for r in range(int(g.shape[0]))]
            origin[(region, lattice)] = (torch.tensor(xs, dtype=torch.float64, device=g.device),
                                         torch.tensor(ys, dtype=torch.float64, device=g.device))
    return EmbedVisit(grids=grids, origin=origin,
                      tile_l0=float(res.cs.grids[0].tile_size) * float(res.cs.ds), fovs=[])


def fov_of(res: FovResult) -> EmbedFov:
    return EmbedFov(queries=res.queries, where=res.where, slide=res.slide, level=res.level, index=res.index)


def _iou_and_disjoint(x0: torch.Tensor, y0: torch.Tensor, w: float, h: float, truth: np.ndarray,
                      fov: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """For windows whose top-left corners are `(x0[None, :], y0[:, None])` and whose
    size is `w x h`: their IoU with the truth box `[Ho, Wo]` and whether they are
    disjoint from the FoV box."""
    ix: torch.Tensor = (torch.minimum(x0 + w, torch.tensor(truth[2], device=x0.device))
                        - torch.maximum(x0, torch.tensor(truth[0], device=x0.device))).clamp(min=0)
    iy: torch.Tensor = (torch.minimum(y0 + h, torch.tensor(truth[3], device=x0.device))
                        - torch.maximum(y0, torch.tensor(truth[1], device=x0.device))).clamp(min=0)
    inter: torch.Tensor = ix * iy
    t_area: float = float((truth[2] - truth[0]) * (truth[3] - truth[1]))
    iou: torch.Tensor = inter / (w * h + t_area - inter)
    fx: torch.Tensor = (torch.minimum(x0 + w, torch.tensor(fov[2], device=x0.device))
                        - torch.maximum(x0, torch.tensor(fov[0], device=x0.device))).clamp(min=0)
    fy: torch.Tensor = (torch.minimum(y0 + h, torch.tensor(fov[3], device=x0.device))
                        - torch.maximum(y0, torch.tensor(fov[1], device=x0.device))).clamp(min=0)
    return iou, (fx * fy) <= 0


def embed_loss(model: nn.Module, visit: EmbedVisit, fovs: Sequence[EmbedFov], pos_iou: float,
               temperature: float) -> torch.Tensor:
    """The mean over `fovs` of the pool softmax loss (see the module docstring)."""
    tiles: Dict[Tuple[int, str], torch.Tensor] = {key: model.tiles(g.float()) for key, g in visit.grids.items()}
    windows: Dict[Tuple[Tuple[int, str], Tuple[int, int]], torch.Tensor] = {}
    total: Optional[torch.Tensor] = None
    for fov in fovs:
        logits: List[torch.Tensor] = []
        pos: List[torch.Tensor] = []
        keep: List[torch.Tensor] = []
        for rot in ROTATIONS:
            qg: torch.Tensor = fov.queries[rot].float()
            a, b = int(qg.shape[0]), int(qg.shape[1])
            e_q: torch.Tensor = model.query(qg)
            for key, g in visit.grids.items():
                if g.shape[0] < a or g.shape[1] < b:
                    continue
                slot = (key, (a, b))
                if slot not in windows:
                    windows[slot] = model.windows_from(tiles[key], (a, b))
                w_emb: torch.Tensor = windows[slot]
                ho, wo = int(w_emb.shape[0]), int(w_emb.shape[1])
                xs, ys = visit.origin[key]
                iou, disjoint = _iou_and_disjoint(xs[None, :wo], ys[:ho, None], b * visit.tile_l0, a * visit.tile_l0,
                                                  fov.where.truth, fov.where.fov)
                covers: torch.Tensor = iou >= pos_iou
                right: bool = rot == fov.where.rotation
                p: torch.Tensor = covers if right else torch.zeros_like(covers)
                n: torch.Tensor = (disjoint | (covers & (not right))) & ~p
                logits.append((w_emb @ e_q).reshape(-1).float())
                pos.append(p.reshape(-1))
                keep.append((p | n).reshape(-1))
        loss: torch.Tensor = pool_softmax_loss(torch.cat(logits), torch.cat(pos), torch.cat(keep), temperature)
        total = loss if total is None else total + loss
    assert total is not None
    return total / len(fovs)


@torch.no_grad()
def embed_rank(model: nn.Module, res: FovResult, pooling: str, label_cfg: LabelConfig, state: Dict[str, Any],
               k: int = 100, min_sep_tiles: float = 1.0) -> float:
    """The first hit's rank when the embedding alone proposes the windows (its top
    `k`, the retriever's own separation rule), or `inf` when none of them is a hit.
    `state` is one dict per scoring pass: it keeps the embeddings (`RetrieverWinEmbed.WindowEmbeddings`), and
    so the embedding of the level's windows, across the FoVs of a visit."""
    from dataclasses import replace                                               # noqa: PLC0415
    from stage2_retrieval.Context import top_candidates, window_boxes             # noqa: PLC0415
    from stage2_retrieval.RetrieverWinEmbed import WindowEmbeddings               # noqa: PLC0415
    from stage2_retrieval.StageInterface import Candidate                         # noqa: PLC0415
    model.eval()
    if state.get('model') is not model:
        embeddings: WindowEmbeddings = WindowEmbeddings('', 256)
        embeddings.model, embeddings.ckpt = model, {'pooling': pooling}
        state.update(model=model, embeddings=embeddings, slide=None)
    embeddings = state['embeddings']
    if state['slide'] != res.slide:
        embeddings.level, embeddings.vectors = None, {}   # a new slide: the windows of the last are not its
        state['slide'] = res.slide
    maps = embeddings.window_maps(res.source, len(res.cs.grids), res.level)
    cs: CandidateSet = top_candidates(maps, replace(res.cs, candidates=()), k, min_sep_tiles, Candidate)
    if not len(cs):
        return float('inf')
    boxes: np.ndarray = window_boxes(cs, shape_fn(res.source))
    lab = label_windows(boxes, np.array([c.rotation for c in cs.candidates]), res.where,
                        float(res.cs.grids[0].tile_size) * float(cs.ds), label_cfg)
    return first_hit_rank(-np.arange(len(cs), dtype=np.float64), lab.hit)
