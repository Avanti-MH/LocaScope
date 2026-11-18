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

The tile features it reads come from one of two sources (`FmTiles`, `CoaeTiles`): the FM's own pooling, or a usage
of a CoAE checkpoint (`cfg.tiles`: `GigaCoAE+<arm>:WinEmbed::<usage>`). The CoAE source offers no `level_features`
(a reranker must not read the network's vectors as the FM's own), only `window_scores`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from locascope.core.config.ConfigIdentity import register
from locascope.core.slides.SafeSlide import SafeSlide
from locascope.core.backbones.TileEncoder import TileEncoder, TileEncoderConfig, encoder_config
from locascope.core.stage1_estimation.StageInterface import EstMppResult, weights_path
from locascope.core.stage2_retrieval.ArmSpec import coae_encoder_name
from locascope.core.stage2_retrieval.Context import (LATTICES, GridSource, LevelFeatures, QueryFeatures, WindowSearch,
                                      feature_key, first_phase_output, top_candidates)
from locascope.core.stage2_retrieval.Retriever import EncoderPool, Extra, RetrievalOutput, Retriever, RetrieverConfig
from locascope.core.stage2_retrieval.RetrieverCoae import CoaeTilesConfig, coae_tile_encoder
from locascope.core.stage2_retrieval.StageInterface import ROTATIONS, Candidate, CandidateSet


@register('retriever_winembed')
@dataclass(frozen=True)
class RetrieverWinEmbedConfig(RetrieverConfig):
    """The encoder (pooling included) and tile cut, a `winembed` checkpoint (`weights`: by name, which is not
    identity; its content is, through the stage's id), the offset lattice, the list's length (`raw_topk`) and
    separation (`min_sep_tiles`). `band` is how many window rows are embedded at a time (memory). `tiles`: None
    reads the FM's pooled features (`encoder`); a `CoaeTilesConfig` reads a usage of a CoAE checkpoint, and `encoder`
    then names the FM that checkpoint reads (pooling 'cls'), as in `RetrieverCoaeConfig`."""
    encoder:       TileEncoderConfig
    weights:       str
    tile_size:     int   = 256
    overlap:       bool  = True
    raw_topk:      int   = 1000
    min_sep_tiles: float = 1.0
    band:          int   = 256
    tiles:         Optional[CoaeTilesConfig] = None

    BASELINE = {'tile_size': 256, 'overlap': True, 'raw_topk': 1000, 'min_sep_tiles': 1.0, 'tiles': None}
    NOT_IDENTITY = ('weights', 'band')
    VERSION = 1

    @property
    def arm_encoder(self) -> str:
        """The encoder name the arm writes: `<Fm>CoAE+<arm>` for a CoAE source; '' for the FM's own (the name falls back to the encoder's)."""
        return '' if self.tiles is None else coae_encoder_name(str(getattr(self.encoder, 'REGISTERED_AS', '')), self.tiles.weights)

    @property
    def arm_pooling(self) -> str:
        """The arm's pooling: the CoAE usage; '' for the FM's own (the pooling falls back to the encoder's)."""
        return '' if self.tiles is None else self.tiles.feature

    def __post_init__(self) -> None:
        if self.band < 1 or self.raw_topk < 1:
            raise ValueError('band and raw_topk must be >= 1')
        if self.tiles is not None and str(self.encoder.pooling) != 'cls':
            raise ValueError(f'the CoAE checkpoint reads the FM\'s CLS: the encoder\'s pooling must be \'cls\', got {self.encoder.pooling!r}')


#: Named retrievers, every field written out; `--stage2 winembed:<name>`. The weights are what
#: `training/RerankHead` writes for this encoder and pooling.
WINEMBED_RECIPES: Dict[str, RetrieverWinEmbedConfig] = {
    'uni2-avg': RetrieverWinEmbedConfig(
        encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
        weights='RerankHead/weights/bracs_train+ki67_pure--uni2--cls_avg--winembed--best.pt',
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, band=256, tiles=None),
    # TEMPORARY: kept until `gigapath-avg` below is checked to give the same answers on the real photos (RealTest); then this recipe,
    # RerankHead.BK/ and the RealTest cache entry of these weights go.
    # gigapath `cls_avg`, the checkpoint of RerankHead job 597492 (val R@30 0.999), kept in RerankHead.BK/ while a new
    # RerankHead job writes RerankHead/weights/ under the same names. The weights' name is not identity, their content is
    # (through the stage's id), so a run on the new ones is another cache entry, not a hit on this one.
    'gigapath-avg-bk': RetrieverWinEmbedConfig(
        encoder=encoder_config('gigapath', pooling='cls_avg').with_model(dtype='fp16'),
        weights='RerankHead.BK/weights/pure-train--gigapath--cls_avg--winembed--best.pt',
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, band=256, tiles=None),
    # gigapath `cls_avg`, the checkpoint the RerankHead job 614471 wrote to RerankHead/weights/ (val R@30 0.997, epoch 19):
    # the current one, `--stage2 winembed:gigapath-avg`. The weights' content is identity (through the stage's id), so it is
    # a cache entry of its own and no hit on `gigapath-avg-bk`'s.
    'gigapath-avg': RetrieverWinEmbedConfig(
        encoder=encoder_config('gigapath', pooling='cls_avg').with_model(dtype='fp16'),
        weights='RerankHead/weights/pure-train--gigapath--cls_avg--winembed--best.pt',
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, band=256, tiles=None),
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
            from locascope.core.networks.Checkpoints import build_rerank_from_checkpoint       # noqa: PLC0415
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


class FmTiles:
    """The tile features of the FM's own pooling (`cfg.encoder`): the source of a config with no `tiles`."""

    def __init__(self, cfg: RetrieverWinEmbedConfig, encoders: EncoderPool, feature_store: Optional[Any]) -> None:
        self.cfg: RetrieverWinEmbedConfig = cfg
        self.encoders: EncoderPool = encoders
        #: what the retriever's `WindowSearch` keeps the level's features in: the one given
        self.store: Optional[Any] = feature_store
        self._encoder: Optional[TileEncoder] = None

    def encoder(self) -> TileEncoder:
        """The FM's tile encoder, taken from the pool on first use."""
        if self._encoder is None:
            self._encoder = self.encoders.get(self.cfg.encoder)
        return self._encoder

    def model(self) -> torch.nn.Module:
        """What `IdentifiedBuild.weights_id` hashes: the FM."""
        return self.encoder().model

    def pooling(self) -> str:
        """The name of the features a level and a query are held in (`GridSource`): the encoder's own."""
        return self.encoder().feature_pooling

    def output(self, cs: CandidateSet, level: Optional[LevelFeatures], qf: Optional[QueryFeatures],
               scores: Dict[Tuple[int, int, str], torch.Tensor]) -> RetrievalOutput:
        """The list and, as extras, the features it was made from (`level_features`) and the score of every window."""
        return first_phase_output(cs, level, qf, self.encoder(), self.cfg.tile_size, self.cfg.overlap, scores)


class CoaeTiles:
    """The tile features of a usage of a CoAE checkpoint (`cfg.tiles`): the source of a config with `tiles`."""

    #: no store: one keyed by the encoder's config (MultiPhases') would hand this retriever the FM's own level features
    store: Optional[Any] = None

    def __init__(self, cfg: RetrieverWinEmbedConfig, device: Any) -> None:
        assert cfg.tiles is not None
        self.cfg: RetrieverWinEmbedConfig = cfg
        self.tiles: CoaeTilesConfig = cfg.tiles
        self.device: Any = device
        self._encoder: Optional[Any] = None

    def encoder(self) -> Any:
        """The network as a tile encoder (`RetrieverCoae.coae_tile_encoder`), built from the checkpoint on first use."""
        if self._encoder is None:
            self._encoder = coae_tile_encoder(self.tiles.weights, self.tiles.feature, self.cfg.encoder, self.cfg.tile_size, self.device)
        return self._encoder

    def model(self) -> torch.nn.Module:
        """What `IdentifiedBuild.weights_id` hashes: the FM the network loaded (the checkpoint's content is in the stage's id)."""
        return self.encoder().net.fm.vit

    def pooling(self) -> str:
        """The arm's word for the usage (`latent`), not `CoaeTileEncoder.feature_pooling` ('coae-latent'): it is the name training
        passes around, writes into the checkpoint (`ckpt['pooling']`) and `WindowEmbeddings.loaded` checks."""
        return self.tiles.feature

    def output(self, cs: CandidateSet, level: Optional[LevelFeatures], qf: Optional[QueryFeatures],
               scores: Dict[Tuple[int, int, str], torch.Tensor]) -> RetrievalOutput:
        """The list and the score of every window; no `level_features` (a reranker must not read the network's vectors as the FM's own)."""
        if not len(cs) or level is None:
            return RetrievalOutput(candidates=cs)
        key: Tuple[Any, ...] = ('coae', self.tiles.feature) + feature_key(self.cfg.encoder, self.cfg.tile_size, self.cfg.overlap)
        return RetrievalOutput(candidates=cs, extras={'window_scores': Extra('window_scores', key, scores)})


class RetrieverWinEmbed(Retriever):
    """See the module docstring."""

    UPSTREAM = (EstMppResult,)

    def __init__(self, cfg: RetrieverWinEmbedConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False, read_workers: int = 0,
                 feature_store: Optional[Any] = None) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: RetrieverWinEmbedConfig = cfg
        self.tiles: Union[FmTiles, CoaeTiles] = (CoaeTiles(cfg, self.device) if cfg.tiles is not None
                                                 else FmTiles(cfg, self.encoders, feature_store))
        self.search: WindowSearch = WindowSearch(self.tiles.encoder, cfg.tile_size, cfg.overlap, self.device,
                                                 read_workers, self.tiles.store)
        self.embeddings: WindowEmbeddings = WindowEmbeddings(cfg.weights, cfg.band)

    @property
    def model(self) -> torch.nn.Module:
        return self.tiles.model()

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
            source: GridSource = GridSource(level, qf, self.tiles.pooling())
            made['scores'] = self.embeddings.window_maps(source, len(level.grids), level.level)
            return top_candidates(made['scores'], frame, self.cfg.raw_topk, self.cfg.min_sep_tiles, Candidate)

        cs, level, qf = self.search.run(int(upstream.chosen_level), query, scorer)
        return self.tiles.output(cs, level, qf, made.get('scores', {}))


__all__ = ['CoaeTiles', 'FmTiles', 'RetrieverWinEmbed', 'RetrieverWinEmbedConfig', 'WINEMBED_RECIPES', 'WindowEmbeddings']
