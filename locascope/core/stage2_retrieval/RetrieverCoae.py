"""CoAE: a first-phase retriever whose tile feature is a usage of the CoAE network (`training/CoAE`), level and query alike.

    r   = RetrieverCoae(cfg, device).build(wsi, mask)
    out = r.retrieve(est_mpp_result, shot_img)

The sliding window is `RetrieverSlidingWinSimRot`'s, untouched: `list_windows`, the per-tile cosines of every window at every rotation
and lattice, the `score` of them, the best `raw_topk` windows `min_sep_tiles` apart. What differs is the tile feature: the checkpoint's
network (`weights`, an epoch of `training/CoAE/train.py`) reads every tile of the level and of the query, and `feature` names which of
its outputs make the vector (`ArmSpec.COAE_PARTS`: each part made unit, joined, made unit again; the plain cosine, no mean taken off).
`cfg.encoder` names the FM the checkpoint reads (pooling 'cls': it names the entry and gives the batch) and nothing else of it is used:
the network loads that FM itself.
In an arm the checkpoint is an encoder of the sliding window (`GigaCoAE+<arm>:SlidingWindow::<feature>+mean`, `ArmSpec.coae_encoder`).

It offers the score of every window (`window_scores`) and the per-tile cosines (`tile_sims`), and the level's and the query's features as
`tile_features` (a `GridSource` named by the arm's usage, `latent`): what a head trained on the network's vectors reads
(`training/RerankHead`; `RetrieverCoae.FEATURES`). It does NOT offer `level_features`: that source would hand a reranker the network's
vectors as the FM's own, which they are not. The level's features are kept only in a store that belongs to this retriever alone
(`single_retriever`: the training's per-slide memory store); any other `feature_store` is taken and not handed on (a store keyed by the
encoder's config, MultiPhases', would give gigapath's own).

`CoaeTilesConfig` and `coae_tile_encoder` are the one place a checkpoint becomes a tile encoder: this retriever and `RetrieverWinEmbed`
(a window embedder that reads a CoAE usage) both build theirs through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from locascope.core.backbones.TileEncoder import TileEncoderConfig
from locascope.core.config.ConfigIdentity import IdentifiedConfig, register
from locascope.core.slides.SafeSlide import SafeSlide
from locascope.core.stage1_estimation.StageInterface import EstMppResult, weights_path
from locascope.core.stage2_retrieval.ArmSpec import COAE_PARTS, coae_encoder_name
from locascope.core.stage2_retrieval.Context import GridSource, LevelFeatures, QueryFeatures, WindowSearch, feature_key
from locascope.core.stage2_retrieval.Retriever import EncoderPool, Extra, RetrievalOutput, Retriever, RetrieverConfig
from locascope.core.stage2_retrieval.RetrieverSlidingWinSimRot import TILE_SIM_ROWS, WINDOW_SCORES, list_windows, tile_sim_rows
from locascope.core.stage2_retrieval.StageInterface import CandidateSet


@register('coae_tiles')
@dataclass(frozen=True)
class CoaeTilesConfig(IdentifiedConfig):
    """A CoAE checkpoint read as a tile encoder: `weights` (by name, which is not identity; its content is, through the stage's id,
    which `weights_of_config` finds by the field's name) and the usage `feature` (a key of `ArmSpec.COAE_PARTS`)."""
    weights: str
    feature: str

    BASELINE = {}
    NOT_IDENTITY = ('weights',)
    VERSION = 1

    def __post_init__(self) -> None:
        if self.feature not in COAE_PARTS:
            raise ValueError(f'no CoAE usage {self.feature!r}; they are {", ".join(COAE_PARTS)}')


def coae_tile_encoder(weights: str, feature: str, fm: TileEncoderConfig, tile_size: int, device: Any) -> Any:
    """The `Validate.CoaeTileEncoder` of the checkpoint `weights` reading the usage `feature`, on `device`. `fm` is the FM the checkpoint
    reads (pooling 'cls': it names the FM and gives the batch); the checkpoint is refused when it reads another FM or tiles of another size."""
    # imported here: core does not import training when it is imported
    from locascope.training.CoAE.Validate import CoaeTileEncoder, coae_of_checkpoint   # noqa: PLC0415
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError('torch.backends.cuda.matmul.allow_tf32 is on: the cosines of a CoAE feature would keep 10 mantissa bits, '
                           'too few for the places\' differences (float32 is the contract)')
    net, ccfg, scale = coae_of_checkpoint(weights_path(weights), device)
    print(f'{weights}: FM weights {ccfg.fm.weights_dtype}', flush=True)
    registered: str = str(getattr(fm, 'REGISTERED_AS', ''))
    if ccfg.fm.encoder != registered or int(ccfg.cofig.tile) != int(tile_size):
        raise ValueError(f'{weights} reads {ccfg.fm.encoder!r} tiles of {ccfg.cofig.tile} px and the config names '
                         f'{registered!r} and {tile_size} px')
    return CoaeTileEncoder(net, scale, device, feature, batch=int(fm.batch_size))


@register('retriever_coae')
@dataclass(frozen=True)
class RetrieverCoaeConfig(RetrieverConfig):
    """The FM the checkpoint reads (`encoder`, pooling 'cls'), the CoAE checkpoint (`weights`: by name, which is not identity; its
    content is, through the stage's id), the usage of the network's outputs that is the tile feature (`feature`, a key of
    `ArmSpec.COAE_PARTS`: the arm's pooling), and the sliding window's own fields: the tile it cuts, whether it also slides the half-tile offset
    lattice, how many windows it lists (`raw_topk`), how far apart two of them at one rotation must be (`min_sep_tiles`, in tiles), and
    the window `score`."""
    encoder:       TileEncoderConfig
    weights:       str
    feature:       str
    tile_size:     int   = 256
    overlap:       bool  = True
    raw_topk:      int   = 1000
    min_sep_tiles: float = 1.0
    score:         str   = 'mean'

    BASELINE = {'tile_size': 256, 'overlap': True, 'raw_topk': 1000, 'min_sep_tiles': 1.0, 'score': 'mean'}
    NOT_IDENTITY = ('weights',)
    VERSION = 1

    @property
    def arm_encoder(self) -> str:
        """The encoder name the arm writes: `<Fm>CoAE+<arm>` (ArmSpec.coae_encoder_name of the FM and the checkpoint)."""
        return coae_encoder_name(str(getattr(self.encoder, 'REGISTERED_AS', '')), self.weights)

    @property
    def arm_pooling(self) -> str:
        """The arm's pooling: the feature."""
        return self.feature

    def __post_init__(self) -> None:
        if self.score not in WINDOW_SCORES:
            raise ValueError(f'no window score {self.score!r}; the scores are {", ".join(WINDOW_SCORES)}')
        if self.raw_topk < 1:
            raise ValueError(f'raw_topk must be >= 1, got {self.raw_topk}')
        if self.feature not in COAE_PARTS:
            raise ValueError(f'no CoAE usage {self.feature!r}; they are {", ".join(COAE_PARTS)}')
        if str(self.encoder.pooling) != 'cls':
            raise ValueError(f'the CoAE checkpoint reads the FM\'s CLS: the encoder\'s pooling must be \'cls\', got {self.encoder.pooling!r}')


class RetrieverCoae(Retriever):
    """See the module docstring. `build` binds a slide; `retrieve` takes stage 1's output, encodes the level it routed to on first
    use (kept for the slide), reads the query's tiles with the same tile encoder, and lists the best `raw_topk` windows."""

    UPSTREAM = (EstMppResult,)
    TABLES = ('tile_sims',)
    FEATURES = 'tile_features'                 # what a trainer reads; NOT 'level_features' (the module docstring's reason stands)

    def __init__(self, cfg: RetrieverCoaeConfig, device: Optional[Any] = None, *, encoders: Optional[EncoderPool] = None,
                 multi_gpu: bool = False, read_workers: int = 0, feature_store: Optional[Any] = None) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: RetrieverCoaeConfig = cfg
        self._features: Optional[Any] = None
        # a store that holds ONE retriever's levels (`single_retriever`: the training's per-slide memory store) is this retriever's own
        # and is handed on; any other (MultiPhases hands its encoder-keyed one to every phase) is NOT: it would give this retriever
        # gigapath's own level features
        own: Optional[Any] = feature_store if getattr(feature_store, 'single_retriever', False) else None
        self.search: WindowSearch = WindowSearch(lambda: self.features, cfg.tile_size, cfg.overlap, self.device, read_workers, own)

    @property
    def features(self) -> Any:
        """The tile encoder of the level AND of the query (`Validate.CoaeTileEncoder`), built from the checkpoint on first use."""
        if self._features is None:
            self._features = coae_tile_encoder(self.cfg.weights, self.cfg.feature, self.cfg.encoder, self.cfg.tile_size, self.device)
        return self._features

    @property
    def model(self) -> Any:
        """What IdentifiedBuild.weights_id hashes: the FM the network loaded (no second copy from the pool); the checkpoint's content
        is in the stage's id (`stage_hexid`)."""
        return self.features.net.fm.vit

    @model.setter
    def model(self, value: Any) -> None:
        """The base class sets `model` to None; the network's FM is the model here."""

    @property
    def level_encodes(self) -> Dict[int, Tuple[float, int]]:
        """`{level: (seconds, tiles)}` of every level encoded since `build`."""
        return self.search.level_encodes

    @property
    def last_level_seconds(self) -> float:
        """Seconds the last `retrieve` spent building levels (0 when cached)."""
        return self.search.last_level_seconds

    def build(self, wsi: Union[str, Any], mask: Any) -> 'RetrieverCoae':
        """Bind a slide and its mask. Cheap: no level is built until one is asked for."""
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        super().build(wsi, mask)
        self.search.build(wsi, mask)
        return self

    def frame(self, level: int) -> CandidateSet:
        """An empty CandidateSet in `level`'s frame (geometry only)."""
        assert self.search.levels is not None
        return self.search.levels.frame(level)

    def run(self, upstream: EstMppResult, query: np.ndarray) -> RetrievalOutput:
        made: Dict[str, Any] = {}

        def scorer(level: LevelFeatures, qf: QueryFeatures, frame: CandidateSet) -> CandidateSet:
            # `list_windows` reads only `cfg.score`, `cfg.raw_topk` and `cfg.min_sep_tiles`, which this config has
            cs, made['sims'], made['scores'] = list_windows(level, qf, frame, self.cfg)   # type: ignore[arg-type]
            return cs

        cs, level, qf = self.search.run(int(upstream.chosen_level), query, scorer)
        if not len(cs) or level is None:
            return RetrievalOutput(candidates=cs)
        key: Tuple[Any, ...] = ('coae', self.cfg.feature) + feature_key(self.cfg.encoder, self.cfg.tile_size, self.cfg.overlap)
        return RetrievalOutput(candidates=cs, extras={
            'window_scores': Extra('window_scores', key, made['scores']),
            'tile_sims': Extra('tile_sims', key, made['sims']),
            # named by the arm's word for the usage (`latent`), not `CoaeTileEncoder.feature_pooling` ('coae-latent'): it is the name
            # training passes around (`--pooling`), writes into the checkpoint and `WindowEmbeddings.loaded` checks
            self.FEATURES: Extra(self.FEATURES, key, GridSource(level, qf, self.cfg.feature))})

    def tables(self, out: RetrievalOutput, qc: Any) -> Dict[str, List[Dict[str, Any]]]:
        """`tile_sims`: for the first `TILE_SIM_ROWS` listed windows, every query tile's cosine with the reference tile under it."""
        if 'tile_sims' not in out.extras:
            return {}
        return {'tile_sims': tile_sim_rows(out.extras['tile_sims'].value, out.candidates.candidates[:TILE_SIM_ROWS])}


__all__ = ['CoaeTilesConfig', 'RetrieverCoae', 'RetrieverCoaeConfig', 'coae_tile_encoder']
