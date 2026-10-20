'''The networks of stage 2's second phase (`stage2_retrieval/spec.md`), plain
`nn.Module`s with plain dataclass hyper-parameters like `Pooling` and
`PrototypeGenerators`. Training them is `training/RerankHead/`; running them is
`stage2_retrieval/RerankNeural.py` (a scorer) and `WinEmbed.py` (a retriever).

    CorrNet            the n x n tile correspondence matrix of a window -> a logit
    TileGT             a window's tiles as a graph (a transformer with relative
                       position bias) -> a logit, "right rotation", and where
                       the true place is relative to this window
    CandidateContext   the candidates of one query attending to each other
    TileGTContext      the two together
    WinEmbedder        a window of tile features -> one unit vector, for all the
                       windows of a slide at once (a depthwise convolution)
    StackNet           the columns of the score table -> one score

THE TILE ORDER. A window's tiles are row-major in the window's own frame -- the
frame of the slide, after the query has been turned to match it -- so tile `i`
of the query and tile `i` of the window are the pair the retriever compares, for
a window of either shape (`rows x cols` or `cols x rows`). `pos` `[B, n, 2]`
carries each tile's (row, col) so one batch may mix shapes.
'''
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  the building block
# ══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    '''Pre-norm transformer block whose attention takes an additive bias per
    head. Token 0 is a summary token ("cls"); the others are `n` tiles whose
    pairwise bias is a learned function of their relative position (`max_side`
    > 0: a table over row and column differences, shared by every window shape)
    and/or a bias handed in (`extra_bias`, `[B, H, N, N]`).'''

    def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float, max_side: int = 0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f'dim {dim} must divide by heads {heads}')
        self.heads: int = heads
        self.max_side: int = max_side
        self.ln1: nn.LayerNorm = nn.LayerNorm(dim)
        self.qkv: nn.Linear = nn.Linear(dim, 3 * dim)
        self.out: nn.Linear = nn.Linear(dim, dim)
        self.ln2: nn.LayerNorm = nn.LayerNorm(dim)
        self.ff: nn.Sequential = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(),
                                               nn.Dropout(dropout), nn.Linear(ff_mult * dim, dim))
        self.drop: nn.Dropout = nn.Dropout(dropout)
        self.rel: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(heads, 2 * max_side - 1, 2 * max_side - 1)) if max_side else None)

    def forward(self, x: torch.Tensor, rel_index: Optional[torch.Tensor] = None,
                extra_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''`x` `[B, N, D]`; `rel_index` `[B, n, n, 2]` long, (row, col) differences
        shifted to be >= 0, for the `n = N - 1` tiles after the summary token.'''
        b, n_tok, d = x.shape
        h: int = self.heads
        q, k, v = self.qkv(self.ln1(x)).reshape(b, n_tok, 3, h, d // h).permute(2, 0, 3, 1, 4)
        bias: Optional[torch.Tensor] = None
        if self.rel is not None and rel_index is not None:
            tile_bias: torch.Tensor = self.rel[:, rel_index[..., 0], rel_index[..., 1]]       # [H, B, n, n]
            bias = F.pad(tile_bias.permute(1, 0, 2, 3), (1, 0, 1, 0))                       # [B, H, N, N]
        if extra_bias is not None:
            bias = extra_bias if bias is None else bias + extra_bias
        y: torch.Tensor = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if bias is None else bias.to(q.dtype),
            dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.out(y.transpose(1, 2).reshape(b, n_tok, d)))
        return x + self.drop(self.ff(self.ln2(x)))


# ══════════════════════════════════════════════════════════════════════════════
#  CorrNet
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CorrNetCfg:
    n_tiles: int = 20
    width: int = 64
    dropout: float = 0.1


class CorrNet(nn.Module):
    '''`C[i, j]`, the cosine of query tile `i` with window tile `j`, as an image:
    one channel per query tile, the window's grid as space, and beside it one
    channel per query tile marking where that tile itself would sit. Three
    3x3 convolutions and a pooled head; a descendant of CVNet's correlation
    network, small because `n` is 20 rather than a whole feature map.'''

    def __init__(self, cfg: CorrNetCfg) -> None:
        super().__init__()
        self.cfg: CorrNetCfg = cfg
        n, w = cfg.n_tiles, cfg.width
        self.net: nn.Sequential = nn.Sequential(
            nn.Conv2d(2 * n, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GELU())
        self.head: nn.Sequential = nn.Sequential(nn.Linear(2 * w, w), nn.GELU(),
                                                 nn.Dropout(cfg.dropout), nn.Linear(w, 1))

    def forward(self, corr: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        '''`corr` `[B, n, n]`; the window is `rows x cols` tiles (`rows * cols == n`).
        Returns the logit `[B]`.'''
        b, n, _ = corr.shape
        image: torch.Tensor = corr.reshape(b, n, rows, cols)
        own: torch.Tensor = torch.eye(n, device=corr.device, dtype=corr.dtype).reshape(1, n, rows, cols).expand(b, -1, -1, -1)
        x: torch.Tensor = self.net(torch.cat([image, own], dim=1))
        return self.head(torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  TileGT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TileGTCfg:
    in_dim: int = 3072
    n_tiles: int = 20
    proj: int = 256
    dim: int = 256
    layers: int = 4
    heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_side: int = 8
    n_scalar: int = 1
    n_global: int = 8


class TileGT(nn.Module):
    '''A window as a graph whose nodes are its `n` tile pairs.

    A node is the query tile and the window tile under it, projected by one
    shared layer and combined as `[q, w, q*w, |q-w|]`, plus the node's row and
    column of the correspondence matrix (how it compares with EVERY tile of the
    window: a shift shows as a better match one tile over), its own scalars and
    its place in the window. A summary node carries the window-level scalars.
    Attention among the nodes has a learned bias by relative position, so it is
    graph attention on the grid graph, without being told the grid is square.

    Outputs: `logit` (is this the place), `rot` (is this the right rotation),
    `loc` (where the true place is from this window's centre, in tiles) and
    `embed` (the summary node, for `CandidateContext`).'''

    def __init__(self, cfg: TileGTCfg) -> None:
        super().__init__()
        self.cfg: TileGTCfg = cfg
        d, p = cfg.dim, cfg.proj
        self.project: nn.Sequential = nn.Sequential(nn.LayerNorm(cfg.in_dim), nn.Linear(cfg.in_dim, p))
        self.fuse: nn.Linear = nn.Linear(4 * p, d)
        self.corr_row: nn.Linear = nn.Linear(cfg.n_tiles, d)
        self.corr_col: nn.Linear = nn.Linear(cfg.n_tiles, d)
        self.scalar: nn.Linear = nn.Linear(cfg.n_scalar, d)
        self.row_emb: nn.Embedding = nn.Embedding(cfg.max_side, d)
        self.col_emb: nn.Embedding = nn.Embedding(cfg.max_side, d)
        self.cls: nn.Parameter = nn.Parameter(torch.zeros(1, 1, d))
        self.glob: nn.Linear = nn.Linear(cfg.n_global, d)
        self.blocks: nn.ModuleList = nn.ModuleList(
            [Block(d, cfg.heads, cfg.ff_mult, cfg.dropout, cfg.max_side) for _ in range(cfg.layers)])
        self.norm: nn.LayerNorm = nn.LayerNorm(d)
        self.score: nn.Linear = nn.Linear(d, 1)
        self.rot: nn.Linear = nn.Linear(d, 1)
        self.loc: nn.Linear = nn.Linear(d, 2)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, q: torch.Tensor, w: torch.Tensor, pos: torch.Tensor, scalars: torch.Tensor,
                corr: torch.Tensor, glob: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''`q`, `w` `[B, n, F]`; `pos` `[B, n, 2]` long (row, col) of each tile in
        its window; `scalars` `[B, n, S]`; `corr` `[B, n, n]`; `glob` `[B, G]`.'''
        cfg = self.cfg
        qp, wp = self.project(q), self.project(w)
        x: torch.Tensor = self.fuse(torch.cat([qp, wp, qp * wp, (qp - wp).abs()], dim=-1))
        x = x + self.corr_row(corr) + self.corr_col(corr.transpose(1, 2)) + self.scalar(scalars)
        x = x + self.row_emb(pos[..., 0]) + self.col_emb(pos[..., 1])
        head: torch.Tensor = self.cls.expand(x.shape[0], -1, -1) + self.glob(glob).unsqueeze(1)
        x = torch.cat([head, x], dim=1)
        rel: torch.Tensor = pos[:, :, None, :] - pos[:, None, :, :] + (cfg.max_side - 1)        # [B, n, n, 2]
        for block in self.blocks:
            x = block(x, rel_index=rel)
        summary: torch.Tensor = self.norm(x[:, 0])
        return {'logit': self.score(summary).squeeze(-1), 'rot': self.rot(summary).squeeze(-1),
                'loc': self.loc(summary), 'embed': summary}


# ══════════════════════════════════════════════════════════════════════════════
#  the candidates of a query, together
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ContextCfg:
    dim: int = 256
    layers: int = 2
    heads: int = 8
    dropout: float = 0.1
    n_pair: int = 4


class CandidateContext(nn.Module):
    '''The `K` candidates of one query attend to each other, with a bias made
    from how each pair relates (`pair` `[K, K, n_pair]`: offset in tiles, same
    rotation, near). A plateau of equal scores and a lone peak look different
    from here; which is the true place is learned, not assumed.'''

    def __init__(self, cfg: ContextCfg) -> None:
        super().__init__()
        self.cfg: ContextCfg = cfg
        self.pair: nn.Sequential = nn.Sequential(nn.Linear(cfg.n_pair, 32), nn.GELU(), nn.Linear(32, cfg.heads))
        self.blocks: nn.ModuleList = nn.ModuleList(
            [Block(cfg.dim, cfg.heads, 4, cfg.dropout) for _ in range(cfg.layers)])
        self.norm: nn.LayerNorm = nn.LayerNorm(cfg.dim)
        self.score: nn.Linear = nn.Linear(cfg.dim, 1)

    def forward(self, embed: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        '''`embed` `[K, D]`, `pair` `[K, K, n_pair]` -> logit `[K]`.'''
        bias: torch.Tensor = self.pair(pair).permute(2, 0, 1).unsqueeze(0)               # [1, H, K, K]
        x: torch.Tensor = embed.unsqueeze(0)
        for block in self.blocks:
            x = block(x, extra_bias=bias)
        return self.score(self.norm(x[0])).squeeze(-1)


class TileGTContext(nn.Module):
    '''`TileGT` on every candidate, then `CandidateContext` over them.
    `forward` is for ONE query's candidates (`B == K`): the trunk's outputs plus
    the context's `logit`.'''

    def __init__(self, trunk: TileGTCfg, context: ContextCfg) -> None:
        super().__init__()
        if trunk.dim != context.dim:
            raise ValueError(f'trunk dim {trunk.dim} != context dim {context.dim}')
        self.trunk: TileGT = TileGT(trunk)
        self.context: CandidateContext = CandidateContext(context)

    def forward(self, q: torch.Tensor, w: torch.Tensor, pos: torch.Tensor, scalars: torch.Tensor,
                corr: torch.Tensor, glob: torch.Tensor, pair: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = self.trunk(q, w, pos, scalars, corr, glob)
        out['trunk_logit'] = out['logit']
        out['logit'] = self.context(out['embed'], pair)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  a window as one vector
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EmbedCfg:
    in_dim: int = 3072
    dim: int = 512
    shapes: Tuple[Tuple[int, int], ...] = ((4, 5), (5, 4))


class WinEmbedder(nn.Module):
    '''A window of tile features to a unit vector, for every window of a slide
    at once. Each tile is projected (`Linear` + `LayerNorm`) and the window is
    the sum of its tiles weighted by a learned weight per place and channel -- a
    depthwise convolution whose kernel is the window -- then a norm and a small
    MLP. One kernel per window shape.

    Nothing looks past a window's edge, so the embedding of a window of the
    slide and of a query laid out the same way are the same function of the same
    tiles: `windows(grid, shape)[r, c]` is `query(grid[r:r+a, c:c+b])`. (A
    convolution that mixed each tile with its neighbours would see the
    reference's surroundings and not the query's.)'''

    def __init__(self, cfg: EmbedCfg) -> None:
        super().__init__()
        self.cfg: EmbedCfg = cfg
        self.proj: nn.Sequential = nn.Sequential(nn.Linear(cfg.in_dim, cfg.dim), nn.LayerNorm(cfg.dim))
        self.kernels: nn.ParameterDict = nn.ParameterDict({
            f'{a}x{b}': nn.Parameter(torch.randn(cfg.dim, a, b) / math.sqrt(a * b)) for a, b in cfg.shapes})
        self.post: nn.Sequential = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim),
                                                 nn.GELU(), nn.Linear(cfg.dim, cfg.dim))

    def tiles(self, grid: torch.Tensor) -> torch.Tensor:
        '''`[R, C, F]` -> `[1, dim, R, C]`: every tile projected, once, for all the
        window shapes that will be cut from it.'''
        return self.proj(grid).permute(2, 0, 1).unsqueeze(0)

    def windows_from(self, x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        '''`tiles(...)` -> `[R - a + 1, C - b + 1, dim]`, unit vectors: the
        embedding of every `a x b` window (empty if the grid is smaller).'''
        a, b = shape
        y: torch.Tensor = F.conv2d(x, self.kernels[f'{a}x{b}'].unsqueeze(1), groups=self.cfg.dim)
        return F.normalize(self.post(y[0].permute(1, 2, 0)), dim=-1)

    def windows(self, grid: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        '''`[R, C, F]` -> `[R - a + 1, C - b + 1, dim]`, unit vectors: the
        embedding of every `a x b` window of the grid (empty if it is smaller).'''
        return self.windows_from(self.tiles(grid), shape)

    def query(self, tiles: torch.Tensor) -> torch.Tensor:
        '''`[a, b, F]` -> `[dim]`: the embedding of the one window the query is.'''
        return self.windows(tiles, (int(tiles.shape[0]), int(tiles.shape[1])))[0, 0]


# ══════════════════════════════════════════════════════════════════════════════
#  the score table to one score
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StackCfg:
    n_features: int = 8
    hidden: int = 64
    dropout: float = 0.1


class StackNet(nn.Module):
    '''A small MLP on the columns of the score table (rank-normalised per query by
    the caller), one score per candidate.'''

    def __init__(self, cfg: StackCfg) -> None:
        super().__init__()
        self.cfg: StackCfg = cfg
        h: int = cfg.hidden
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(cfg.n_features, h), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(h, h), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
