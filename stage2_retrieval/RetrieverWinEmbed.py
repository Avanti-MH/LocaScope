"""F3 -- a window as one learned vector: a first-phase retriever.

    r   = RetrieverWinEmbed(WINEMBED_RECIPES['uni2-avg'], device).build(wsi, mask)
    out = r.retrieve(est_mpp_result, shot_img)

`RerankHeads.WinEmbedder` maps the tile features of an `a x b` window to a unit vector; the query goes
through the same function. Every window of the level is embedded once (a depthwise convolution over the
feature grid: no gather per window), and a query is one matrix-vector product per rotation and grid. What
the sliding window does with 20 cosines per window and a scan of the level, this does with one dot product
-- and the vector is trained for the task (the full-softmax loss of `training/RerankHead`), not read off a
general encoder.

The index is exact. The vectors are what an approximate index (`RetrieverTileVote`'s graph) could be built
over, if a slide were too large to multiply.

It encodes its own level and query (`Context.WindowSearch`) and offers them as `level_features`, with the
score of every window as `window_scores`. The checkpoint must have been trained on features of the pooling
the config's encoder names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ConfigIdentity import register
from SafeSlide import SafeSlide
from TileEncoderFunc import TileEncoder, TileEncoderConfig, encoder_config
from stage1_estimation.StageInterface import EstMppResult, weights_path
from stage2_retrieval.Context import (LATTICES, GridSource, LevelFeatures, QueryFeatures, WindowSearch,
                                      first_phase_output, top_candidates)
from stage2_retrieval.Retriever import EncoderPool, RetrievalOutput, Retriever, RetrieverConfig
from stage2_retrieval.StageInterface import ROTATIONS, Candidate, CandidateSet


@register('retriever_winembed')
@dataclass(frozen=True)
class RetrieverWinEmbedConfig(RetrieverConfig):
    """The encoder (pooling included) and tile cut, a `winembed` checkpoint (`weights`: by name, which is not
    identity; its content is, through the stage's id), the offset lattice, the list's length (`raw_topk`) and
    separation (`min_sep_tiles`). `band` is how many window rows are embedded at a time (memory)."""
    encoder:       TileEncoderConfig
    weights:       str
    tile_size:     int   = 256
    overlap:       bool  = True
    raw_topk:      int   = 1000
    min_sep_tiles: float = 1.0
    band:          int   = 256

    BASELINE = {'tile_size': 256, 'overlap': True, 'raw_topk': 1000, 'min_sep_tiles': 1.0}
    NOT_IDENTITY = ('weights', 'band')
    VERSION = 1

    def __post_init__(self) -> None:
        if self.band < 1 or self.raw_topk < 1:
            raise ValueError('band and raw_topk must be >= 1')


#: Named retrievers, every field written out; `--stage2 winembed:<name>`. The weights are what
#: `training/RerankHead` writes for this encoder and pooling.
WINEMBED_RECIPES: Dict[str, RetrieverWinEmbedConfig] = {
    'uni2-avg': RetrieverWinEmbedConfig(
        encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
        weights='RerankHead/weights/bracs_train+ki67_pure--uni2--cls_avg--winembed--best.pt',
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, band=256),
}


class WindowEmbeddings:
    """A loaded `winembed` checkpoint and the embedding of every window of the level it last embedded."""

    def __init__(self, weights: str, band: int) -> None:
        self.weights: str = weights
        self.band: int = int(band)
        self.model: Optional[torch.nn.Module] = None
        self.ckpt: Optional[Dict[str, Any]] = None
        self.level: Optional[int] = None
        self.vectors: Dict[Tuple[int, str, Tuple[int, int]], torch.Tensor] = {}

    def loaded(self, device: Any, pooling: str) -> torch.nn.Module:
        if self.model is None:
            from Checkpoints import build_rerank_from_checkpoint       # noqa: PLC0415
            self.model, self.ckpt = build_rerank_from_checkpoint(weights_path(self.weights), device)
        assert self.ckpt is not None
        if self.ckpt['pooling'] != pooling:
            raise ValueError(f'{self.weights} reads {self.ckpt["pooling"]!r} features and the encoder gives '
                             f'{pooling!r}')
        return self.model

    @torch.no_grad()
    def embed(self, model: torch.nn.Module, grid: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """`[Ho, Wo, E]` fp16: the embedding of every window of the grid, a band of
        window rows at a time (a window looks at nothing past its own tiles, so a
        band is exactly the rows `r0 .. r0 + band + a - 1` of the grid)."""
        a, _ = shape
        ho: int = int(grid.shape[0]) - a + 1
        parts: List[torch.Tensor] = []
        for r0 in range(0, ho, self.band):
            parts.append(model.windows(grid[r0:r0 + self.band + a - 1].float(), shape).half())
        return torch.cat(parts, dim=0)

    def window_maps(self, source: GridSource, n_regions: int, level: int
                    ) -> Dict[Tuple[int, int, str], torch.Tensor]:
        """The score of every window of the level, per `(rotation, region, lattice)`: the whole ranking."""
        model: torch.nn.Module = self.loaded(source.query_grid(0, source.pooling).device, source.pooling)
        if self.level != level:
            self.level, self.vectors = level, {}
        maps: Dict[Tuple[int, int, str], torch.Tensor] = {}
        for rot in ROTATIONS:
            qg: torch.Tensor = source.query_grid(rot, source.pooling)
            shape: Tuple[int, int] = (int(qg.shape[0]), int(qg.shape[1]))
            with torch.no_grad():
                e_q: torch.Tensor = model.query(qg.float())
            for region in range(n_regions):
                for lattice in LATTICES:
                    grid: torch.Tensor = source.reference_grid(region, lattice, source.pooling)
                    if grid.numel() == 0 or grid.shape[0] < shape[0] or grid.shape[1] < shape[1]:
                        continue
                    slot: Tuple[int, str, Tuple[int, int]] = (region, lattice, shape)
                    if slot not in self.vectors:
                        self.vectors[slot] = self.embed(model, grid, shape)
                    maps[(rot, region, lattice)] = self.vectors[slot].float() @ e_q
        return maps


class RetrieverWinEmbed(Retriever):
    """See the module docstring."""

    UPSTREAM = (EstMppResult,)

    def __init__(self, cfg: RetrieverWinEmbedConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False, read_workers: int = 0,
                 feature_store: Optional[Any] = None) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: RetrieverWinEmbedConfig = cfg
        self._encoder: Optional[TileEncoder] = None
        self.search: WindowSearch = WindowSearch(lambda: self.encoder, cfg.tile_size, cfg.overlap, self.device,
                                                 read_workers, feature_store)
        self.embeddings: WindowEmbeddings = WindowEmbeddings(cfg.weights, cfg.band)

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
        """The encoder is the model here; the checkpoint's content is in the stage's id (`stage_hexid`)."""

    def build(self, wsi: Union[str, Any], mask: Any) -> 'RetrieverWinEmbed':
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        super().build(wsi, mask)
        self.search.build(wsi, mask)
        self.embeddings.level, self.embeddings.vectors = None, {}
        return self

    def frame(self, level: int) -> CandidateSet:
        assert self.search.levels is not None
        return self.search.levels.frame(level)

    def run(self, upstream: EstMppResult, query: np.ndarray) -> RetrievalOutput:
        made: Dict[str, Any] = {}

        def scorer(level: LevelFeatures, qf: QueryFeatures, frame: CandidateSet) -> CandidateSet:
            source: GridSource = GridSource(level, qf, self.encoder.feature_pooling)
            made['scores'] = self.embeddings.window_maps(source, len(level.grids), level.level)
            return top_candidates(made['scores'], frame, self.cfg.raw_topk, self.cfg.min_sep_tiles, Candidate)

        cs, level, qf = self.search.run(int(upstream.chosen_level), query, scorer)
        return first_phase_output(cs, level, qf, self.encoder, self.cfg.tile_size, self.cfg.overlap,
                                  made.get('scores', {}))


__all__ = ['RetrieverWinEmbed', 'RetrieverWinEmbedConfig', 'WINEMBED_RECIPES', 'WindowEmbeddings']
