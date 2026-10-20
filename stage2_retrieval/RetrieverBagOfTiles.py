"""F1 -- a window as the mean of its tiles: a first-phase retriever.

    r   = RetrieverBagOfTiles(BAGOFTILES_RECIPES['uni2-avg-gap'], device).build(wsi, mask)
    out = r.retrieve(est_mpp_result, shot_img)

The sliding window compares a window with the query tile by tile. This compares their MEAN feature vectors
(`gap`), or the means of their four quadrants (`quad`, which keeps the layout, so a turned window does not
match): a truth that falls between two lattices' windows loses less, because nothing is aligned tile to
tile. Every window's mean is a box filter over the level's feature grid (`avg_pool2d`, stride 1), a slice
of channels at a time, so the level is never copied per window. No training.

It encodes its own level and query (`Context.WindowSearch`) with the encoder and pooling its config names,
and offers them downstream as `level_features`, with the score of every window as `window_scores`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ConfigIdentity import register
from SafeSlide import SafeSlide
from TileEncoderFunc import TileEncoder, TileEncoderConfig, encoder_config
from stage1_estimation.StageInterface import EstMppResult
from stage2_retrieval.Context import (LATTICES, GridSource, LevelFeatures, QueryFeatures, WindowSearch,
                                      first_phase_output, top_candidates)
from stage2_retrieval.Retriever import EncoderPool, RetrievalOutput, Retriever, RetrieverConfig
from stage2_retrieval.StageInterface import ROTATIONS, Candidate, CandidateSet

MODES: Tuple[str, ...] = ('gap', 'quad')


@register('retriever_bagoftiles')
@dataclass(frozen=True)
class RetrieverBagOfTilesConfig(RetrieverConfig):
    """The encoder (pooling included) and tile cut it reads the slide with, whether it slides the offset
    lattice too, how many windows it lists (`raw_topk`) and how far apart two at one rotation must be
    (`min_sep_tiles`); `mode` the mean of the whole window (`gap`) or of its four quadrants (`quad`).
    `channels` is how many feature channels are filtered at a time (memory, not identity)."""
    encoder:       TileEncoderConfig
    tile_size:     int   = 256
    overlap:       bool  = True
    raw_topk:      int   = 1000
    min_sep_tiles: float = 1.0
    mode:          str   = 'gap'
    channels:      int   = 256

    BASELINE = {'tile_size': 256, 'overlap': True, 'raw_topk': 1000, 'min_sep_tiles': 1.0, 'mode': 'gap'}
    NOT_IDENTITY = ('channels',)
    VERSION = 1

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, got {self.mode!r}')
        if self.channels < 1 or self.raw_topk < 1:
            raise ValueError('channels and raw_topk must be >= 1')


#: Named retrievers, every field written out; `--stage2 bagoftiles:<name>`.
BAGOFTILES_RECIPES: Dict[str, RetrieverBagOfTilesConfig] = {
    'uni2-avg-gap': RetrieverBagOfTilesConfig(
        encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, mode='gap', channels=256),
    'uni2-avg-quad': RetrieverBagOfTilesConfig(
        encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, mode='quad', channels=256),
}


def box_means(grid: torch.Tensor, query: torch.Tensor, cells: List[Tuple[int, int, int, int]],
              channels: int) -> List[torch.Tensor]:
    """For each cell `(a0, a1, b0, b1)` (rows `a0:a1`, columns `b0:b1` of the
    window), the cosine, at every window position of `grid` `[R, C, F]`, between
    the mean feature of the window's cell and the query's `[a, b, F]` over the
    same cell. Each is `[R - a + 1, C - b + 1]`; windows are `a x b` tiles.

    Channels are filtered `channels` at a time; the dot product and the two norms
    are accumulated across the slices, so the cosine is exact."""
    a, b = int(query.shape[0]), int(query.shape[1])
    ho, wo = int(grid.shape[0]) - a + 1, int(grid.shape[1]) - b + 1
    dots: List[torch.Tensor] = [torch.zeros((ho, wo), device=grid.device) for _ in cells]
    norm_w: List[torch.Tensor] = [torch.zeros((ho, wo), device=grid.device) for _ in cells]
    norm_q: List[float] = [0.0] * len(cells)
    for s in range(0, int(grid.shape[-1]), channels):
        x: torch.Tensor = grid[..., s:s + channels].float().permute(2, 0, 1).unsqueeze(0)       # [1, c, R, C]
        qs: torch.Tensor = query[..., s:s + channels].float()
        for i, (a0, a1, b0, b1) in enumerate(cells):
            pooled: torch.Tensor = F.avg_pool2d(x, (a1 - a0, b1 - b0), stride=1)[0]             # [c, R-ah+1, C-bh+1]
            mean_w: torch.Tensor = pooled[:, a0:a0 + ho, b0:b0 + wo]
            mean_q: torch.Tensor = qs[a0:a1, b0:b1].mean(dim=(0, 1))
            dots[i] += (mean_w * mean_q[:, None, None]).sum(dim=0)
            norm_w[i] += (mean_w * mean_w).sum(dim=0)
            norm_q[i] += float((mean_q * mean_q).sum())
    return [d / (nw.sqrt() * (nq ** 0.5) + 1e-12) for d, nw, nq in zip(dots, norm_w, norm_q)]


def cells(mode: str, a: int, b: int) -> List[Tuple[int, int, int, int]]:
    """The cells of an `a x b` window a mode compares: the whole window, or its four quadrants."""
    if mode == 'gap':
        return [(0, a, 0, b)]
    ha, hb = max(a // 2, 1), max(b // 2, 1)
    return [(0, ha, 0, hb), (0, ha, hb, b), (ha, a, 0, hb), (ha, a, hb, b)]


def window_maps(source: GridSource, n_regions: int, mode: str, channels: int
                ) -> Dict[Tuple[int, int, str], torch.Tensor]:
    """The score of every window of the level, per `(rotation, region, lattice)`: the whole ranking."""
    maps: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for rot in ROTATIONS:
        qg: torch.Tensor = source.query_grid(rot, source.pooling)
        a, b = int(qg.shape[0]), int(qg.shape[1])
        for region in range(n_regions):
            for lattice in LATTICES:
                grid: torch.Tensor = source.reference_grid(region, lattice, source.pooling)
                if not grid.numel() or grid.shape[0] < a or grid.shape[1] < b:
                    continue
                parts: List[torch.Tensor] = box_means(grid, qg, cells(mode, a, b), int(channels))
                maps[(rot, region, lattice)] = torch.stack(parts).mean(dim=0)
    return maps


class RetrieverBagOfTiles(Retriever):
    """See the module docstring."""

    UPSTREAM = (EstMppResult,)

    def __init__(self, cfg: RetrieverBagOfTilesConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False, read_workers: int = 0,
                 feature_store: Optional[Any] = None) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: RetrieverBagOfTilesConfig = cfg
        self._encoder: Optional[TileEncoder] = None
        self.search: WindowSearch = WindowSearch(lambda: self.encoder, cfg.tile_size, cfg.overlap, self.device,
                                                 read_workers, feature_store)

    @property
    def encoder(self) -> TileEncoder:
        if self._encoder is None:
            self._encoder = self.encoders.get(self.cfg.encoder)
        return self._encoder

    @property
    def model(self) -> torch.nn.Module:
        return self.encoder.model

    @model.setter
    def model(self, value: Any) -> None:
        """The encoder is the model here."""

    def build(self, wsi: Union[str, Any], mask: Any) -> 'RetrieverBagOfTiles':
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        super().build(wsi, mask)
        self.search.build(wsi, mask)
        return self

    def frame(self, level: int) -> CandidateSet:
        assert self.search.levels is not None
        return self.search.levels.frame(level)

    def run(self, upstream: EstMppResult, query: np.ndarray) -> RetrievalOutput:
        made: Dict[str, Any] = {}

        def scorer(level: LevelFeatures, qf: QueryFeatures, frame: CandidateSet) -> CandidateSet:
            source: GridSource = GridSource(level, qf, self.encoder.feature_pooling)
            made['scores'] = window_maps(source, len(level.grids), self.cfg.mode, self.cfg.channels)
            return top_candidates(made['scores'], frame, self.cfg.raw_topk, self.cfg.min_sep_tiles, Candidate)

        cs, level, qf = self.search.run(int(upstream.chosen_level), query, scorer)
        return first_phase_output(cs, level, qf, self.encoder, self.cfg.tile_size, self.cfg.overlap,
                                  made.get('scores', {}))


__all__ = ['BAGOFTILES_RECIPES', 'MODES', 'RetrieverBagOfTiles', 'RetrieverBagOfTilesConfig', 'box_means',
           'cells', 'window_maps']
