'''Patch tokens for TilePool `attnpool` (`proposal.md` §3.2): the tile encoder read as the tokens of each tile's
patches, in place of the one vector (`cls`, `cls_avg`) a tile is otherwise reduced to.

The level and the query go through the machinery every first phase uses (`Context.WindowSearch`: read the level a
block of tiles at a time, encode, arrange the grids), with ONE change: what the encoder returns for a block of
tiles is each tile's patch tokens flattened, `[N, P * F]` in fp16 (P = 196 and F = 1536 for GigaPath), where it
would be `[N, F]`. Everything downstream is then shaped as it always was -- `[rows, cols, P * F]` grids, windows
gathered from them, the host copy of a level between epochs (`Supply.MemoryFeatureStore`) -- and `GloRet.TileEmbed`
reads the flattened `P * F` as `P` tokens of `F`.

A tile is 0.6 MB of tokens (196 * 1536 * 2 bytes) against 6 KB for `cls_avg`: a whole slide's levels do not fit in
host memory at the size of the training set, which is why training with them is first done on the coarse levels
(`--min-ds`), whose tiles are few.
'''
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import torch

from ReadGeometry import ReadSpec
from stage2_retrieval.Context import GridSource, WindowSearch, encode_query
from stage2_retrieval.Retriever import Extra, RetrievalOutput
from stage2_retrieval.StageInterface import Candidate, CandidateSet

from training.GloRet.Lazy import QuerySource, TileAccess

#: The pooling a patch-token source is read through (`GridSource.pooling`, the weights' file name, `NetCfg.tile_pool`).
POOLING: str = 'attnpool'


class TokenEncoder:
    """A tile encoder as `encode_level` and `encode_query` call it -- `encoder(tiles) -> [N, D]` -- that returns each
    tile's patch tokens, flattened, in fp16. The prefix tokens (the CLS, and the registers of a model that has them)
    are dropped: the pool reads patches."""

    def __init__(self, encoder: Any) -> None:
        self.encoder: Any = encoder
        self.prefix: int = int(encoder.model_spec.num_prefix)

    @property
    def width(self) -> int:
        """P * F: the numbers a tile's patch tokens come to."""
        spec = self.encoder.model_spec
        return int(spec.feat_hw[0]) * int(spec.feat_hw[1]) * int(spec.dim)

    def __call__(self, tiles: Any) -> torch.Tensor:
        tokens: torch.Tensor = self.encoder.tokens(tiles)                    # [N, prefix + P, F] fp32
        patches: torch.Tensor = tokens[:, self.prefix:, :]
        return patches.reshape(patches.shape[0], -1).half()


#: The pooling of a source whose tokens went through a trained, frozen `attnpool` (stage C): one vector a tile.
FROZEN_POOLING: str = 'attnpool_frozen'


def load_pool(path: str, device: Any) -> torch.nn.Module:
    """The attention pool over a tile's patch tokens that the checkpoint `path` (a network trained with TilePool
    `attnpool`: `embed.pool`) learned, on `device`, frozen: `Linear(F, F/4) -> Tanh -> Linear(F/4, 1)`, scoring each
    token; `GloRet.TileEmbed` takes the softmax of the scores over a tile's tokens as the weights of their sum."""
    ckpt: Dict[str, Any] = torch.load(path, map_location='cpu')
    if ckpt['hparams'].get('tile_pool') != POOLING:
        raise ValueError(f'{path} was trained with TilePool {ckpt["hparams"].get("tile_pool")!r}, not {POOLING!r}: '
                         'it has no attention pool to freeze')
    width: int = int(ckpt['hparams']['in_dim'])
    pool: torch.nn.Sequential = torch.nn.Sequential(torch.nn.Linear(width, width // 4), torch.nn.Tanh(),
                                                    torch.nn.Linear(width // 4, 1))
    prefix: str = 'embed.pool.'
    pool.load_state_dict({k[len(prefix):]: v for k, v in ckpt['state'].items() if k.startswith(prefix)})
    for p in pool.parameters():
        p.requires_grad_(False)
    return pool.to(device).eval()


class PooledTokenEncoder:
    """`TokenEncoder` followed by a FROZEN attention pool: each tile's patch tokens summed with the weights the pool
    gives them, `[N, F]` in fp16 -- a tile is one vector again, as with `cls`, so a level of them is kept and read
    like any other (`--pooling attnpool_frozen`, stage C). The weights are `TileEmbed`'s: softmax over the tokens
    of the pool's score of each."""

    def __init__(self, encoder: Any, pool: torch.nn.Module) -> None:
        self.tokens: TokenEncoder = TokenEncoder(encoder)
        self.pool: torch.nn.Module = pool

    @torch.no_grad()
    def __call__(self, tiles: Any) -> torch.Tensor:
        flat: torch.Tensor = self.tokens(tiles)                                  # [N, P * F] fp16
        width: int = int(self.pool[0].in_features)
        patches: torch.Tensor = flat.reshape(flat.shape[0], -1, width).float()   # [N, P, F]
        beta: torch.Tensor = torch.softmax(self.pool(patches).squeeze(-1), dim=-1)
        return (beta.unsqueeze(-1) * patches).sum(dim=1).half()


class PatchSource:
    """What a first phase hands a trainer, without a list: the level's and the query's patch grids, in the frame the
    candidates mean something in. It answers `build(wsi, mask)` and `retrieve(estimate, img, topk=)` as a retriever
    does, so the supply of FoVs takes it for one; its `candidates` are the frame with one placeholder window, since
    it ranks nothing -- the list a LogLoRet reads comes from another retriever (`--list-encoder`)."""

    def __init__(self, encoder: Callable[[], Any], device: Any, tile_size: int = 256, overlap: bool = True,
                 read_workers: int = 0, feature_store: Optional[Any] = None, min_sep_tiles: float = 1.0,
                 pool: Optional[torch.nn.Module] = None, lazy: bool = False) -> None:
        #: stage B: the level is NOT encoded; the query is, and the tiles under chosen windows are, on demand
        #: (Lazy.py), so a fine level whose tokens do not fit the card can be trained on
        self.lazy: bool = bool(lazy)
        self._encoder: Callable[[], Any] = encoder
        self._tokens: Optional[Any] = None
        #: a FROZEN attention pool: the tokens of a tile are pooled as they are encoded (stage C), and the grids
        #: are one vector a tile; without it they are the tokens (stages A and B)
        self.pool: Optional[torch.nn.Module] = pool
        self.pooling: str = FROZEN_POOLING if pool is not None else POOLING
        self.tile_size: int = int(tile_size)
        self.overlap: bool = bool(overlap)
        self.search: WindowSearch = WindowSearch(lambda: self.tokens, tile_size, overlap, device, read_workers,
                                                 feature_store)
        self.cfg: SimpleNamespace = SimpleNamespace(min_sep_tiles=float(min_sep_tiles))

    @property
    def tokens(self) -> Any:
        if self._tokens is None:
            encoder = self._encoder()
            self._tokens = PooledTokenEncoder(encoder, self.pool) if self.pool is not None else TokenEncoder(encoder)
        return self._tokens

    @property
    def level_encodes(self) -> Any:
        return self.search.level_encodes

    def build(self, wsi: Any, mask: Any) -> 'PatchSource':
        self.search.build(wsi, mask)
        return self

    def retrieve_lazily(self, upstream: Any, query: Any) -> RetrievalOutput:
        """`retrieve` without encoding the level: the level is found the way a search finds it (the nearest to the
        routed one that holds the query), the query is encoded, and what is offered is the query's tokens
        (`level_features`, a `QuerySource`) and how to read and encode the tiles of the level (`tiles`)."""
        levels = self.search.levels
        reader = self.search.reader
        if levels is None or reader is None:
            raise RuntimeError('call build(wsi, mask) first')
        state: Dict[str, Any] = {}

        def search(level: int) -> CandidateSet:
            if 'query' not in state:
                state['query'] = encode_query(query, self.tokens, self.tile_size, self.overlap)
            return replace(levels.frame(level), candidates=(Candidate(0, 'main', 0, 0, 0, 0.0),))

        cs: CandidateSet = levels.route(int(upstream.chosen_level), levels.query_dims(query), lambda level: None, search)
        if not len(cs):
            return RetrievalOutput(candidates=cs)
        tile: int = self.tile_size
        encoder = self.tokens

        def read(points: Any) -> Any:
            return reader.read_points(points, ReadSpec(tile, tile), cs.ds, level=cs.level, batch=256)

        key = (self.pooling, tile, self.overlap)
        return RetrievalOutput(candidates=cs, extras={
            'level_features': Extra('level_features', key, QuerySource(state['query'], self.pooling)),
            'tiles': Extra('tiles', key, TileAccess(read, encoder, int(encoder.width)))})

    def retrieve(self, upstream: Any, query: Any, *, topk: Optional[int] = None) -> RetrievalOutput:
        if self.lazy:
            return self.retrieve_lazily(upstream, query)

        def scorer(level: Any, qf: Any, frame: CandidateSet) -> CandidateSet:
            return replace(frame, candidates=(Candidate(0, 'main', 0, 0, 0, 0.0),))

        cs, level, qf = self.search.run(int(upstream.chosen_level), query, scorer)
        if not len(cs) or level is None or qf is None:
            return RetrievalOutput(candidates=cs)
        # `WindowSearch` keeps every level it encoded for the slide; with patch tokens a level is gigabytes on the
        # card, and the FoVs of a slide are visited one level after the next, so the others are let go (a visit that
        # still reads one holds it itself; the next epoch reads it back from the host copy)
        held = getattr(self.search, '_by_level', None)
        if isinstance(held, dict):
            for other in [lv for lv in held if lv != level.level]:
                del held[other]
        key = (self.pooling, self.tile_size, self.overlap)
        return RetrievalOutput(candidates=cs, extras={
            'level_features': Extra('level_features', key, GridSource(level, qf, self.pooling))})


__all__ = ['FROZEN_POOLING', 'POOLING', 'PatchSource', 'PooledTokenEncoder', 'TokenEncoder', 'load_pool']
