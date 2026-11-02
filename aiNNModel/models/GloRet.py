'''The two architectures of `stage2_retrieval/proposal.md`, as plain `nn.Module`s.

    GloRet     (Arch 1, §5)  FM tile features -> TileEmbed -> ContextEn -> GlobalEn -> bi-encoder cosine head.
                             A window and a query are encoded alone and meet in one inner product, so the
                             window vectors do not depend on the query. `Align` (no GlobalEn) compares the
                             contextualised tiles one by one instead.
    LogLoRet   (Arch 2, §6)  the same TileEmbed and ContextEn on both sides, then two parallel branches -- the
                             GlobalEn cosine and a bidirectional cross-attention with a no-match option -- into
                             an evidence extractor and an MLP: one score per (query, window).

THE TILE ORDER. A window's tiles are row-major in its own frame, the query turned to match it: tile `i` of the
query and tile `i` of the window are the pair compared (§2). `shape = (rows, cols)` says how to lay the `N`
tiles out; the position of tile `i` is `(i // cols, i % cols)`, and everything that depends on position (the
relative bias, the neighbourhood of a spatially masked attention or a GNN, the CNN's grid) is made from it, so
a network takes any shape.

The names are the proposal's: ContextEn `SelfAttn | GNN | SptSelfAttn | Identity | CLSTrans` (§4), GlobalEn
`AttnPool | MaskedMean | CNN | GraphPool | Align | Identity` (§5.2), TilePool `cls | cls_avg | attnpool`
(§3.2), no-match `dustbin | sinkhorn | sigmoid` (§6.4), evidence aggregation `mean | attn | robust` (§6.5).
`combo_problem` says which pairs of ContextEn and GlobalEn exist.
'''
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CONTEXT_EN: Tuple[str, ...] = ('SelfAttn', 'GNN', 'SptSelfAttn', 'Identity', 'CLSTrans')
GLOBAL_EN: Tuple[str, ...] = ('AttnPool', 'MaskedMean', 'CNN', 'GraphPool', 'Align', 'Identity')
#: `attnpool_frozen`: the tile's patch tokens pooled by an `attnpool` that was trained (stages A and B) and is no longer, so the
#: network reads one vector a tile like `cls` and the pooled grids can be kept (stage C).
TILE_POOLS: Tuple[str, ...] = ('cls', 'cls_avg', 'cls_std', 'grid2x2', 'attnpool', 'attnpool_frozen')
NO_MATCH: Tuple[str, ...] = ('dustbin', 'sinkhorn', 'sigmoid')
EVIDENCE_AGG: Tuple[str, ...] = ('mean', 'attn', 'robust')

#: the features one tile gives per evidence channel of a head (`LogLoRetNet`): aligned logit, margin, bidirectional
#: agreement, coverage, collapse, dustbin rate (each way), entropy
N_HEAD_EVIDENCE: int = 8


def combo_problem(context: str, global_en: str, arch: int) -> Optional[str]:
    '''Why `(context, global_en)` is not a method of architecture `arch` (1: GloRet, 2: LogLoRet), or None when
    it is. `CLSTrans` carries its own global output (so its GlobalEn is `Identity`, and only it has that);
    `GraphPool` is the GNN's; `Align` has no vector to cosine and is Arch 1's alone.'''
    if context not in CONTEXT_EN:
        return f'ContextEn {context!r} is not one of {CONTEXT_EN}'
    if global_en not in GLOBAL_EN:
        return f'GlobalEn {global_en!r} is not one of {GLOBAL_EN}'
    if (context == 'CLSTrans') != (global_en == 'Identity'):
        return 'CLSTrans goes with GlobalEn Identity, and Identity only with CLSTrans'
    if global_en == 'GraphPool' and context != 'GNN':
        return 'GraphPool pools the GNN\'s graph and goes with ContextEn GNN only'
    if global_en == 'Align' and arch != 1:
        return 'Align compares tiles one by one with no global vector: it is GloRet\'s alone'
    return None


@dataclass(frozen=True)
class NetCfg:
    '''The hyper-parameters of both networks. `in_dim` is the width of a tile's feature (the encoder's
    pooling: 1536 for `cls`, 3072 for `cls_avg` of gigapath or UNI2) or, for TilePool `attnpool`, of a patch
    token. `d` is the width after `W_in` (§4.1), `d_r` of the retrieval vectors (§5.3).'''
    in_dim: int = 3072
    tile_pool: str = 'cls_avg'
    context: str = 'SelfAttn'
    global_en: str = 'AttnPool'
    d: int = 512
    d_r: int = 512
    layers: int = 2
    heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    pos_hidden: int = 32           # the relative-position bias is a small MLP of the offset (§3.3)
    max_rel: int = 8               # offsets are clipped to this many tiles
    spt_radius: int = 1            # SptSelfAttn and the GNN look at tiles within this Chebyshev distance
    tau_init: float = 0.05
    # Arch 2
    no_match: str = 'dustbin'
    evidence_agg: str = 'attn'
    d_u: int = 64                  # the width of the match feature
    score_hidden: int = 128
    sinkhorn_iters: int = 10
    log_n: bool = True             # the log-probabilities are read against uniform: + log(N) (or N + 1)
    recon: str = ''                # '' | 'ae' | 'vae': a decoder on each narrowing projection (in_dim > d, d > d_r), §9.3 a

    def check(self, arch: int) -> None:
        problem: Optional[str] = combo_problem(self.context, self.global_en, arch)
        if problem is not None:
            raise ValueError(problem)
        if self.tile_pool not in TILE_POOLS:
            raise ValueError(f'tile_pool {self.tile_pool!r} is not one of {TILE_POOLS}')
        if self.d % self.heads:
            raise ValueError(f'd {self.d} must divide by heads {self.heads}')
        if self.recon not in ('', 'ae', 'vae'):
            raise ValueError(f'recon {self.recon!r} is not one of ("", "ae", "vae")')
        if self.no_match not in NO_MATCH or self.evidence_agg not in EVIDENCE_AGG:
            raise ValueError(f'no_match {self.no_match!r} / evidence_agg {self.evidence_agg!r}')


# ══════════════════════════════════════════════════════════════════════════════
#  geometry of a window of tiles
# ══════════════════════════════════════════════════════════════════════════════

def tile_positions(shape: Tuple[int, int], device: Optional[torch.device] = None) -> torch.Tensor:
    '''`[N, 2]` long, the (row, col) of each of the `rows * cols` tiles, row-major.'''
    rows, cols = shape
    r = torch.arange(rows, device=device).repeat_interleave(cols)
    c = torch.arange(cols, device=device).repeat(rows)
    return torch.stack([r, c], dim=1)


def neighbour_mask(shape: Tuple[int, int], radius: int, device: Optional[torch.device] = None) -> torch.Tensor:
    '''`[N, N]` bool, tile `j` within `radius` (Chebyshev) of tile `i`, the tile itself included.'''
    pos = tile_positions(shape, device)
    return (pos[:, None, :] - pos[None, :, :]).abs().amax(dim=-1) <= radius


def edge_features(shape: Tuple[int, int], device: Optional[torch.device] = None) -> torch.Tensor:
    '''`[N, N, 11]` edge attributes of the grid graph (§4.2): the offset (dr, dc), its length and the direction
    as a one-hot over the eight neighbours (all zero for the tile itself and for tiles farther away).'''
    pos = tile_positions(shape, device).float()
    delta = pos[None, :, :] - pos[:, None, :]                                    # [N, N, 2] from i to j
    length = delta.norm(dim=-1, keepdim=True)
    direction = torch.zeros(delta.shape[0], delta.shape[1], 8, device=pos.device)
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for k, (dr, dc) in enumerate(dirs):
        direction[..., k] = ((delta[..., 0] == dr) & (delta[..., 1] == dc)).float()
    return torch.cat([delta, length, direction], dim=-1)


class RelBias(nn.Module):
    '''§3.3, the default: `B_ij = f_pos(offset_ij)`, a small MLP of the offset in tiles, one value per head. Shared
    by every window of a shape, so it is made once for the shape: `[heads, N, N]`. `cross` takes the offset from a
    tile of the first set to a tile of the second.'''

    def __init__(self, heads: int, hidden: int, max_rel: int) -> None:
        super().__init__()
        self.max_rel: int = max_rel
        self.net: nn.Sequential = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, heads))

    def forward(self, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        pos = tile_positions(shape, device).float()
        delta = (pos[None, :, :] - pos[:, None, :]).clamp(-self.max_rel, self.max_rel) / self.max_rel   # [N, N, 2]
        return self.net(delta).permute(2, 0, 1)                                                          # [h, N, N]


# ══════════════════════════════════════════════════════════════════════════════
#  what a narrowing keeps (§9.3 a)
# ══════════════════════════════════════════════════════════════════════════════

RECON: Tuple[str, ...] = ('', 'ae', 'vae')


def narrows(d_in: int, d_out: int, mode: str) -> bool:
    '''Whether a projection `d_in -> d_out` gets the auxiliary decoder: the mode is set AND it really narrows. With
    no compression (or no mode) there is nothing to keep and the loss does not exist.'''
    return mode in ('ae', 'vae') and d_in > d_out


class Decoded(nn.Module):
    '''A projection `x -> z` (`d_in -> d_out`) with the decoder that mirrors it (`z -> d_in`, a linear map as the
    projection is) and the loss that makes the narrowing keep what it dropped: in training the decoder rebuilds the
    (detached) input from `z` and the cosine to it is the reconstruction loss (`recon`). With `mode` `vae` the decoder
    is given a draw `z + exp(logvar / 2) * eps` instead (the mean `z` and `logvar` read from `x`), and the KL of
    `N(z, exp(logvar))` against `N(0, I)` is the second loss (`kl`). What the network passes on is always `z`: the
    forward result does not depend on the mode. Without a narrowing there is no decoder and no loss.'''

    def __init__(self, d_in: int, d_out: int, mode: str) -> None:
        super().__init__()
        self.mode: str = mode if narrows(d_in, d_out, mode) else ''
        self.proj: nn.Linear = nn.Linear(d_in, d_out)
        self.dec: Optional[nn.Linear] = nn.Linear(d_out, d_in) if self.mode else None
        self.logvar: Optional[nn.Linear] = nn.Linear(d_in, d_out) if self.mode == 'vae' else None
        self._recon: List[torch.Tensor] = []
        self._kl: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        if self.mode and self.training and torch.is_grad_enabled():
            draw = z
            if self.logvar is not None:
                lv = self.logvar(x).float().clamp(-10.0, 10.0)
                draw = z + torch.exp(0.5 * lv).to(z.dtype) * torch.randn_like(z)
                self._kl.append(-0.5 * (1.0 + lv - z.float().square() - lv.exp()).mean())
            assert self.dec is not None
            self._recon.append((1.0 - F.cosine_similarity(self.dec(draw).float(), x.detach().float(), dim=-1)).mean())
        return z

    def take(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        '''The mean reconstruction loss and KL of the forward passes since the last call (None for none), and forget them.'''
        recon = torch.stack(self._recon).mean() if self._recon else None
        kl = torch.stack(self._kl).mean() if self._kl else None
        self._recon, self._kl = [], []
        return recon, kl


def take_aux(net: nn.Module) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    '''The reconstruction loss and the KL of every narrowing of `net` since the last call, summed over them (None:
    no narrowing was decoded: no compression, no mode, evaluation, or no gradient).'''
    recon: Optional[torch.Tensor] = None
    kl: Optional[torch.Tensor] = None
    for m in net.modules():
        if isinstance(m, Decoded):
            r, k = m.take()
            recon = r if recon is None else (recon if r is None else recon + r)
            kl = k if kl is None else (kl if k is None else kl + k)
    return recon, kl



# ══════════════════════════════════════════════════════════════════════════════
#  TilePool, ContextEn
# ══════════════════════════════════════════════════════════════════════════════

class TileEmbed(nn.Module):
    '''§3: the tile's feature to a token. `cls` and `cls_avg` arrive pooled by the encoder `[B, N, F]`; `attnpool`
    arrives as the patch tokens `[B, N, P, F]` -- or a tile's tokens flattened, `[B, N, P * F]`, the way the level and
    the query grids carry them -- and is pooled by the learned attention (§3.2). Then `LN -> W_in`.'''

    def __init__(self, cfg: NetCfg) -> None:
        super().__init__()
        self.cfg: NetCfg = cfg
        self.pool: Optional[nn.Sequential] = None
        if cfg.tile_pool == 'attnpool':
            self.pool = nn.Sequential(nn.Linear(cfg.in_dim, cfg.in_dim // 4), nn.Tanh(),
                                      nn.Linear(cfg.in_dim // 4, 1))
        self.norm: nn.LayerNorm = nn.LayerNorm(cfg.in_dim)
        self.proj: Decoded = Decoded(cfg.in_dim, cfg.d, cfg.recon)         # W_in, with its decoder when it narrows

    def pooled(self, tokens: torch.Tensor) -> torch.Tensor:
        '''`[..., P, F] -> [..., F]`: the attention pool alone (the weights are the softmax over the tokens of the
        pool's score of each).'''
        assert self.pool is not None
        beta = torch.softmax(self.pool(tokens).squeeze(-1), dim=-1)
        return (beta.unsqueeze(-1) * tokens).sum(dim=-2)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.pool is not None and not (feats.ndim == 3 and feats.shape[-1] == self.cfg.in_dim):
            # (a `[B, N, F]` input is a tile already pooled -- a level pooled once to be scored without a gradient)
            if feats.ndim == 3:                                            # [B, N, P * F]: a tile's tokens flattened
                if feats.shape[-1] % self.cfg.in_dim:
                    raise ValueError(f'TilePool attnpool: {feats.shape[-1]} is not a whole number of tokens of '
                                     f'{self.cfg.in_dim}')
                feats = feats.reshape(feats.shape[0], feats.shape[1], -1, self.cfg.in_dim)
            if feats.ndim != 4:
                raise ValueError('TilePool attnpool reads the patch tokens [B, N, P, F] (or [B, N, P * F])')
            feats = self.pooled(feats)
        elif self.pool is None and feats.ndim != 3:
            raise ValueError('TilePool cls / cls_avg reads the pooled features [B, N, F]')
        return self.proj(self.norm(feats))


class Block(nn.Module):
    '''A pre-norm transformer block (§4.1) whose attention takes an additive per-head bias `[h, N, N]` (or
    `[B, h, N, N]`) and a boolean `[N, N]` mask of the pairs that may attend, and a `[B, N]` mask of valid tiles.'''

    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.heads: int = heads
        self.ln1: nn.LayerNorm = nn.LayerNorm(d)
        self.qkv: nn.Linear = nn.Linear(d, 3 * d)
        self.out: nn.Linear = nn.Linear(d, d)
        self.ln2: nn.LayerNorm = nn.LayerNorm(d)
        self.ff: nn.Sequential = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout),
                                               nn.Linear(ff_mult * d, d))
        self.drop: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None, allow: Optional[torch.Tensor] = None,
                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        mask: Optional[torch.Tensor] = None
        if bias is not None:
            mask = bias if bias.ndim == 4 else bias.unsqueeze(0)
        if allow is not None or valid is not None:
            keep = torch.ones(1, 1, n, n, dtype=torch.bool, device=x.device)
            if allow is not None:
                keep = keep & allow[None, None]
            if valid is not None:
                keep = keep & valid[:, None, None, :]
            fill = torch.zeros(keep.shape, dtype=x.dtype, device=x.device).masked_fill(~keep, float('-inf'))
            mask = fill if mask is None else mask.to(x.dtype) + fill
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask.to(q.dtype),
                                           dropout_p=self.drop.p if self.training else 0.0)
        x = x + self.drop(self.out(y.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class GineLayer(nn.Module):
    '''One round of message passing on the grid graph (§4.2, GINE): `x_i <- x_i + MLP((1 + eps) x_i + sum over
    the neighbours j of ReLU(x_j + W_e e_ij))`, then a feed-forward. The neighbours are the tiles within
    `radius`, the tile itself excluded; the edge attributes are `edge_features`.'''

    def __init__(self, d: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1: nn.LayerNorm = nn.LayerNorm(d)
        self.edge: nn.Linear = nn.Linear(11, d)
        self.eps: nn.Parameter = nn.Parameter(torch.zeros(1))
        self.mlp: nn.Sequential = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ln2: nn.LayerNorm = nn.LayerNorm(d)
        self.ff: nn.Sequential = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout),
                                               nn.Linear(ff_mult * d, d))
        self.drop: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, shape: Tuple[int, int], radius: int) -> torch.Tensor:
        n = x.shape[1]
        adj = neighbour_mask(shape, radius, x.device) & ~torch.eye(n, dtype=torch.bool, device=x.device)
        h = self.ln1(x)
        e = self.edge(edge_features(shape, x.device))                                         # [N, N, d]
        msg = F.relu(h[:, None, :, :] + e[None])                                              # [B, i, j, d]
        agg = (msg * adj[None, :, :, None].to(msg.dtype)).sum(dim=2)                          # [B, N, d]
        x = x + self.drop(self.mlp((1.0 + self.eps) * h + agg))
        return x + self.drop(self.ff(self.ln2(x)))


class ContextEn(nn.Module):
    '''§4. Takes the tile tokens `[B, N, d]` of windows of one `shape`, returns `(X', g)`: the contextualised tokens
    `[B, N, d]` and, for `CLSTrans` alone, the global vector `[B, d]` of its CLS token (else None).'''

    def __init__(self, cfg: NetCfg) -> None:
        super().__init__()
        self.cfg: NetCfg = cfg
        self.kind: str = cfg.context
        self.bias: Optional[RelBias] = None
        self.blocks: nn.ModuleList = nn.ModuleList()
        if self.kind in ('SelfAttn', 'SptSelfAttn', 'CLSTrans'):
            self.bias = RelBias(cfg.heads, cfg.pos_hidden, cfg.max_rel)
            self.blocks = nn.ModuleList([Block(cfg.d, cfg.heads, cfg.ff_mult, cfg.dropout)
                                         for _ in range(cfg.layers)])
        elif self.kind == 'GNN':
            self.blocks = nn.ModuleList([GineLayer(cfg.d, cfg.ff_mult, cfg.dropout) for _ in range(cfg.layers)])
        self.cls: Optional[nn.Parameter] = nn.Parameter(torch.randn(1, 1, cfg.d) * 0.02) if self.kind == 'CLSTrans' else None
        self.norm: nn.LayerNorm = nn.LayerNorm(cfg.d)

    def forward(self, x: torch.Tensor, shape: Tuple[int, int], valid: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.kind == 'Identity':
            return x, None
        if self.kind == 'GNN':
            for layer in self.blocks:
                x = layer(x, shape, self.cfg.spt_radius)
            return self.norm(x), None
        bias = self.bias(shape, x.device) if self.bias is not None else None
        if self.kind == 'CLSTrans':
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
            bias = F.pad(bias, (1, 0, 1, 0))                                            # the CLS token has no offset
            if valid is not None:
                valid = F.pad(valid, (1, 0), value=True)
            for block in self.blocks:
                x = block(x, bias, None, valid)
            x = self.norm(x)
            return x[:, 1:], x[:, 0]
        allow = neighbour_mask(shape, self.cfg.spt_radius, x.device) if self.kind == 'SptSelfAttn' else None
        for block in self.blocks:
            x = block(x, bias, allow, valid)
        return self.norm(x), None


# ══════════════════════════════════════════════════════════════════════════════
#  GlobalEn
# ══════════════════════════════════════════════════════════════════════════════

def masked_mean(x: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
    '''Mean of `x` `[B, N, ...]` over the valid tiles `[B, N]` (all of them when None).'''
    if valid is None:
        return x.mean(dim=1)
    w = valid.to(x.dtype)
    while w.ndim < x.ndim:
        w = w.unsqueeze(-1)
    return (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)


class AttnPool(nn.Module):
    '''§5.2: `e_i = w^T tanh(W x_i)`, `alpha = softmax_i(e)` over the valid tiles, `g = sum_i alpha_i x_i`.'''

    def __init__(self, d: int) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        e = self.net(x).squeeze(-1)
        if valid is not None:
            e = e.masked_fill(~valid, float('-inf'))
        return (torch.softmax(e, dim=1).unsqueeze(-1) * x).sum(dim=1)


class GlobalEn(nn.Module):
    '''§5.2. `[B, N, d]` -> the window's vector `g` `[B, d]`; None for `Align`. `g_cls` is the `CLSTrans`
    context's own output, which `Identity` returns.'''

    def __init__(self, cfg: NetCfg) -> None:
        super().__init__()
        self.kind: str = cfg.global_en
        self.cfg: NetCfg = cfg
        d = cfg.d
        self.pool: Optional[AttnPool] = AttnPool(d) if self.kind in ('AttnPool', 'GraphPool') else None
        self.graph: Optional[GineLayer] = GineLayer(d, cfg.ff_mult, cfg.dropout) if self.kind == 'GraphPool' else None
        self.cnn: Optional[nn.Sequential] = (
            nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, d, 3, padding=1), nn.GELU())
            if self.kind == 'CNN' else None)

    def forward(self, x: torch.Tensor, shape: Tuple[int, int], valid: Optional[torch.Tensor] = None,
                g_cls: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if self.kind == 'Align':
            return None
        if self.kind == 'Identity':
            if g_cls is None:
                raise ValueError('GlobalEn Identity reads the CLS output of ContextEn CLSTrans')
            return g_cls
        if self.kind == 'MaskedMean':
            return masked_mean(x, valid)
        if self.kind == 'AttnPool':
            return self.pool(x, valid)
        if self.kind == 'GraphPool':
            return self.pool(self.graph(x, shape, self.cfg.spt_radius), valid)
        rows, cols = shape
        grid = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], rows, cols)
        return self.cnn(grid).mean(dim=(2, 3))


# ══════════════════════════════════════════════════════════════════════════════
#  Arch 1
# ══════════════════════════════════════════════════════════════════════════════

class GloRetNet(nn.Module):
    '''Architecture 1 (§5). `encode` turns the tile features of windows of one shape into what they are compared
    by -- unit vectors `[B, d_r]`, or for `Align` the contextualised tokens `[B, N, d]`; `query` and `window`
    differ only in the projection (`W_q`, `W_k`, which may differ: §7.1). `score` is `q . k / tau` for vectors, and
    for tokens the mean over the tiles of the cosine of tile `i` with tile `i`, over `tau`.'''

    def __init__(self, cfg: NetCfg) -> None:
        super().__init__()
        cfg.check(1)
        self.cfg: NetCfg = cfg
        self.embed: TileEmbed = TileEmbed(cfg)
        self.context: ContextEn = ContextEn(cfg)
        self.global_en: GlobalEn = GlobalEn(cfg)
        self.aligned: bool = cfg.global_en == 'Align'
        self.w_q: Optional[Decoded] = None if self.aligned else Decoded(cfg.d, cfg.d_r, cfg.recon)
        self.w_k: Optional[Decoded] = None if self.aligned else Decoded(cfg.d, cfg.d_r, cfg.recon)
        self.log_tau: nn.Parameter = nn.Parameter(torch.tensor(math.log(cfg.tau_init)))

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def tokens(self, feats: torch.Tensor, shape: Tuple[int, int], valid: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''`(X', g_cls)` of windows `feats` `[B, N, F]` (or `[B, N, P, F]` for attnpool).'''
        return self.context(self.embed(feats), shape, valid)

    def encode(self, feats: torch.Tensor, shape: Tuple[int, int], role: str = 'window',
               valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        x, g_cls = self.tokens(feats, shape, valid)
        if self.aligned:
            return x
        g = self.global_en(x, shape, valid, g_cls)
        assert g is not None
        proj = self.w_q if role == 'query' else self.w_k
        return F.normalize(proj(g), dim=-1)

    def score(self, q: torch.Tensor, k: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''`q` `[Q, ...]` against `k` `[B, ...]` -> `[Q, B]`.'''
        if not self.aligned:
            return (q @ k.transpose(0, 1)) / self.tau
        cos = torch.einsum('qnd,bnd->qbn', F.normalize(q, dim=-1), F.normalize(k, dim=-1))      # [Q, B, N]
        if valid is None:
            return cos.mean(dim=-1) / self.tau
        w = valid.to(cos.dtype)[None]                                                           # [1, B, N]
        return (cos * w).sum(dim=-1) / w.sum(dim=-1).clamp(min=1.0) / self.tau

    def pair_score(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        '''`[B]`, query `i` against window `i` only (what a candidate list needs): the same number `score` gives
        on the diagonal.'''
        if not self.aligned:
            return (q * k).sum(dim=-1) / self.tau
        return (F.normalize(q, dim=-1) * F.normalize(k, dim=-1)).sum(dim=-1).mean(dim=-1) / self.tau

    def forward(self, query: torch.Tensor, windows: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        '''Scores `[Q, B]` of queries `[Q, N, F]` against windows `[B, N, F]`.'''
        return self.score(self.encode(query, shape, 'query'), self.encode(windows, shape, 'window'))


# ══════════════════════════════════════════════════════════════════════════════
#  Arch 2
# ══════════════════════════════════════════════════════════════════════════════

def sinkhorn_plan(scores: torch.Tensor, dustbin: torch.Tensor, iters: int) -> torch.Tensor:
    '''§6.4: the optimal-transport plan between the `N` tiles of one window and the `N` of the other, with a
    dustbin row and column (`dustbin`, a scalar) that take what has no match: `[.., N, N] -> [.., N+1, N+1]`, in
    probability. Rows and columns of tiles have mass 1; the dustbin has mass `N` (log-domain iterations).'''
    n = scores.shape[-1]
    pad_col = dustbin.expand(*scores.shape[:-1], 1)
    pad_row = dustbin.expand(*scores.shape[:-2], 1, n + 1)
    z = torch.cat([torch.cat([scores, pad_col], dim=-1), pad_row], dim=-2)
    log_mu = torch.cat([torch.zeros(n), torch.full((1,), math.log(n))]).to(z)
    log_u = torch.zeros_like(log_mu).expand(*z.shape[:-2], n + 1)
    log_v = torch.zeros_like(log_u)
    for _ in range(iters):
        log_u = log_mu - torch.logsumexp(z + log_v.unsqueeze(-2), dim=-1)
        log_v = log_mu - torch.logsumexp(z + log_u.unsqueeze(-1), dim=-2)
    return (z + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).exp()


class TileAgg(nn.Module):
    '''§6.5, what makes one number of a window from the number each of its tiles gave. It has to work for any `N`
    (so it is never an MLP over the tiles): `mean` over the valid tiles, `attn` (a learned weight per tile from its
    own channels, softmax over the tiles) or `robust` (the mean without the lowest and highest tenth).
    `[B, N, h, k] -> [B, h, k]`.'''

    def __init__(self, mode: str, heads: int, channels: int) -> None:
        super().__init__()
        self.mode: str = mode
        self.w: Optional[nn.Parameter] = nn.Parameter(torch.zeros(heads, channels)) if mode == 'attn' else None

    def forward(self, v: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == 'mean':
            return masked_mean(v, valid)
        if self.mode == 'attn':
            e = (v * self.w).sum(dim=-1)                                         # [B, N, h]
            if valid is not None:
                e = e.masked_fill(~valid.unsqueeze(-1), float('-inf'))
            return (torch.softmax(e, dim=1).unsqueeze(-1) * v).sum(dim=1)
        srt = v.sort(dim=1).values
        cut = v.shape[1] // 10
        return srt[:, cut:v.shape[1] - cut].mean(dim=1)


class CrossAttn(nn.Module):
    '''§6.3, one set of weights used both ways. `one_way(xa, xb)` asks, for each tile `i` of `xa`, how it matches
    each tile `j` of `xb`: the content logits `S0 = Q_a K_b^T / sqrt(d_h)` and `S = S0 + B^x`, where `B^x` is the
    learned bias of the offset between the two tiles (the query is turned to the window's frame, so the offsets
    are aligned). The values `V_b` are read back by the match `A` the caller computes.'''

    def __init__(self, cfg: NetCfg) -> None:
        super().__init__()
        self.cfg: NetCfg = cfg
        self.heads: int = cfg.heads
        self.ln: nn.LayerNorm = nn.LayerNorm(cfg.d)
        self.q: nn.Linear = nn.Linear(cfg.d, cfg.d)
        self.k: nn.Linear = nn.Linear(cfg.d, cfg.d)
        self.v: nn.Linear = nn.Linear(cfg.d, cfg.d)
        self.out: nn.Linear = nn.Linear(cfg.d, cfg.d)
        self.bias: RelBias = RelBias(cfg.heads, cfg.pos_hidden, cfg.max_rel)

    def split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, d = t.shape
        return t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2)

    # `logits` is these three in a row; the pieces are apart so that what depends on one side only can be made once:
    # the queries of the side that asks (`ask`), the keys and values of the side that is asked (`answer`), the logits (`score`).
    def ask(self, xa: torch.Tensor) -> torch.Tensor:
        '''`[B, h, N, d_h]` the queries of the tiles of `xa` (own layer norm, as in `logits`).'''
        return self.split(self.q(self.ln(xa)))

    def answer(self, xb: torch.Tensor, values: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''`(K_b, V_b)` `[B, h, N, d_h]` of the tiles of `xb` (one layer norm for both); `V_b` is None without `values`.'''
        h = self.ln(xb)
        return self.split(self.k(h)), (self.split(self.v(h)) if values else None)

    def score(self, qa: torch.Tensor, kb: torch.Tensor, shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        '''`(S0, S)` `[B, h, N, N]` content logits, and with the position bias.'''
        s0 = qa @ kb.transpose(-1, -2) / math.sqrt(qa.shape[-1])
        return s0, s0 + self.bias(shape, qa.device).unsqueeze(0)

    def logits(self, xa: torch.Tensor, xb: torch.Tensor, shape: Tuple[int, int]
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''`(S0, S, V_b)`: `[B, h, N, N]` content logits, with the position bias, and `[B, h, N, d_h]`.'''
        kb, vb = self.answer(xb)
        s0, s = self.score(self.ask(xa), kb, shape)
        assert vb is not None
        return s0, s, vb

    def read(self, a: torch.Tensor, vb: torch.Tensor) -> torch.Tensor:
        '''`C = Concat_h(A V) W_O`: `a` `[B, h, N, N]` (the tile-to-tile part of the match) -> `[B, N, d]`.'''
        b, h, n, dh = vb.shape
        return self.out((a @ vb).transpose(1, 2).reshape(b, n, h * dh))


class QuerySide(NamedTuple):
    '''What `LogLoRetNet.query_side` makes of the query: its contextualised tiles `x` `[1 or B, N, d]`, its global vector `g`
    `[1 or B, d]`, and the queries `q` and keys `k` `[1 or B, h, N, d_h]` of the cross-attention.'''
    x: torch.Tensor
    g: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor


class WindowSide(NamedTuple):
    '''What `LogLoRetNet.window_side` makes of `B` windows: the same as `QuerySide` for a query, and the values `v` too.'''
    x: torch.Tensor
    g: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


class LogLoRetNet(nn.Module):
    '''Architecture 2 (§6): one score for a query and a window, from the global cosine, the match evidence of the
    cross-attention both ways, and a match feature. `forward(fq, fw, shape)` takes the query tiles `fq` `[1 or B, N,
    F]` and the window tiles `fw` `[B, N, F]` (the query, shared by all the candidates, is contextualised once) and
    returns `score` `[B]`, with `A_fw`, `A_wf` `[B, h, N, N+1]` (the dustbin last, zeros without one) when the
    losses of §9.2 ask for the maps.

    Two choices that are mine and not in the proposal: the aligned logit and the margin are read off the CONTENT
    logits (without `B^x`, which already prefers the diagonal and would make every window look aligned), and the
    log-probabilities are read against uniform (`+ log N`, or `N + 1` with a dustbin): a softmax over more tiles
    gives every tile a smaller share, and the scores of windows of different shapes have to be comparable.'''

    def __init__(self, cfg: NetCfg) -> None:
        super().__init__()
        cfg.check(2)
        self.cfg: NetCfg = cfg
        self.embed: TileEmbed = TileEmbed(cfg)
        self.context: ContextEn = ContextEn(cfg)
        self.global_en: GlobalEn = GlobalEn(cfg)
        self.cross: CrossAttn = CrossAttn(cfg)
        self.dustbin: nn.Parameter = nn.Parameter(torch.zeros(2))           # one per direction (§7.1)
        self.match_mlp: nn.Sequential = nn.Sequential(nn.Linear(4 * cfg.d, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d_u))
        self.agg_tiles: TileAgg = TileAgg(cfg.evidence_agg, cfg.heads, 6)
        self.agg_match: TileAgg = TileAgg(cfg.evidence_agg, 1, cfg.d_u)
        width = 1 + cfg.heads * N_HEAD_EVIDENCE + cfg.d_u
        self.width: int = width
        self.norm: nn.LayerNorm = nn.LayerNorm(width)
        self.up: nn.Linear = nn.Linear(1, 1)
        self.score_mlp: nn.Sequential = nn.Sequential(nn.Linear(width + 1, cfg.score_hidden), nn.GELU(),
                                                      nn.Dropout(cfg.dropout), nn.Linear(cfg.score_hidden, 1))

    # ── the match, by the no-match option ────────────────────────────────────
    def match(self, s_fw: torch.Tensor, s_wf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''`(A_fw, A_wf)` `[B, h, N, N+1]`: for each tile of one set the share it gives each tile of the other,
        the dustbin (the share it gives to "no match") last. Without a dustbin (`sigmoid`) the last column is the
        share left over, `1 - max`, so the rate means the same.'''
        mode = self.cfg.no_match
        if mode == 'dustbin':
            z = self.dustbin.view(2, 1, 1, 1, 1)
            return (torch.softmax(torch.cat([s_fw, z[0].expand(*s_fw.shape[:-1], 1)], dim=-1), dim=-1),
                    torch.softmax(torch.cat([s_wf, z[1].expand(*s_wf.shape[:-1], 1)], dim=-1), dim=-1))
        if mode == 'sinkhorn':
            plan = sinkhorn_plan(s_fw, self.dustbin[0], self.cfg.sinkhorn_iters)        # [B, h, N+1, N+1]
            a_fw = plan[..., :-1, :]
            a_wf = plan[..., :, :-1].transpose(-1, -2)
            return (a_fw / a_fw.sum(dim=-1, keepdim=True).clamp(min=1e-9),
                    a_wf / a_wf.sum(dim=-1, keepdim=True).clamp(min=1e-9))
        p_fw, p_wf = torch.sigmoid(s_fw), torch.sigmoid(s_wf)
        return (torch.cat([p_fw, 1.0 - p_fw.amax(dim=-1, keepdim=True)], dim=-1),
                torch.cat([p_wf, 1.0 - p_wf.amax(dim=-1, keepdim=True)], dim=-1))

    # ── forward, in the three parts it is made of ────────────────────────────
    # What depends on the window alone (`window_side`), on the query alone (`query_side`) and on the pair (`pair_side`):
    # `forward` is the three in a row. Scoring one level against several queries makes `window_side` once and gives it to
    # each query's `pair_side` (the same operations on the same tensors as `forward`, so the same numbers).
    def query_side(self, fq: torch.Tensor, shape: Tuple[int, int], b: Optional[int] = None) -> QuerySide:
        '''The query tiles `fq` `[1 or B, N, F]` as far as they go without a window. With `b`, a query of one row is expanded to
        `b` rows first, as `forward` always did (the same shapes for the layers); without, it stays one row and `pair_side`
        broadcasts it.'''
        xq, gq = self.context(self.embed(fq), shape, None)
        if b is not None and xq.shape[0] != b:
            xq = xq.expand(b, -1, -1)
            gq = gq.expand(b, -1) if gq is not None else None
        g_q = self.global_en(xq, shape, None, gq)
        assert g_q is not None
        k_q, _ = self.cross.answer(xq, values=False)            # (the values of the query are never read)
        return QuerySide(xq, g_q, self.cross.ask(xq), k_q)

    def window_side(self, fw: torch.Tensor, shape: Tuple[int, int], valid: Optional[torch.Tensor] = None) -> WindowSide:
        '''The window tiles `fw` `[B, N, F]` as far as they go without a query.'''
        xw, gw = self.context(self.embed(fw), shape, valid)
        g_w = self.global_en(xw, shape, valid, gw)
        assert g_w is not None
        k_w, v_w = self.cross.answer(xw)
        assert v_w is not None
        return WindowSide(xw, g_w, self.cross.ask(xw), k_w, v_w)

    def pair_side(self, qs: QuerySide, ws: WindowSide, shape: Tuple[int, int], s_up: Optional[torch.Tensor] = None,
                  valid: Optional[torch.Tensor] = None, maps: bool = False) -> Dict[str, torch.Tensor]:
        '''`forward`'s output from the two sides: `score` `[B]` (and `A_fw`, `A_wf` with `maps`).'''
        xw = ws.x
        b, n = xw.shape[0], xw.shape[1]
        xq, g_q = qs.x, qs.g
        if xq.shape[0] != b:                                    # a query kept as one row
            xq, g_q = xq.expand(b, -1, -1), g_q.expand(b, -1)
        s_global = F.cosine_similarity(g_q, ws.g, dim=-1)                               # [B]
        s0_fw, s_fw = self.cross.score(qs.q, ws.k, shape)
        s0_wf, s_wf = self.cross.score(ws.q, qs.k, shape)
        a_fw, a_wf = self.match(s_fw, s_wf)                                             # [B, h, N, N+1]
        c = self.cross.read(a_fw[..., :n], ws.v)                                        # [B, N, d]
        diag = torch.arange(n, device=xw.device)
        base = math.log(n + 1) if self.cfg.no_match != 'sigmoid' else math.log(n)
        log_n = base if self.cfg.log_n else 0.0
        # evidence of each tile, per head: aligned logit, margin, agreement, dustbin each way, entropy
        s_ii = s0_fw[..., diag, diag]                                                   # [B, h, N]
        off = s0_fw.masked_fill(torch.eye(n, dtype=torch.bool, device=xw.device), float('-inf'))
        margin = s_ii - torch.logsumexp(off, dim=-1)
        bi = 0.5 * (a_fw[..., diag, diag].clamp(min=1e-9).log() + a_wf[..., diag, diag].clamp(min=1e-9).log()) + log_n
        rho_fw, rho_wf = a_fw[..., n], a_wf[..., n]
        tile_a = a_fw[..., :n] / a_fw[..., :n].sum(dim=-1, keepdim=True).clamp(min=1e-9)
        entropy = -(tile_a * tile_a.clamp(min=1e-9).log()).sum(dim=-1) / math.log(max(n, 2))
        per_tile = torch.stack([s_ii, margin, bi, rho_fw, rho_wf, entropy], dim=-1).permute(0, 2, 1, 3)   # [B, N, h, 6]
        agg = self.agg_tiles(per_tile, valid)                                           # [B, h, 6]
        # evidence of the window: how spread the best matches are, and whether one tile takes them all
        best = a_fw[..., :n].argmax(dim=-1)                                             # [B, h, N]
        taken = torch.zeros(b, best.shape[1], n, device=xw.device, dtype=agg.dtype).scatter_(-1, best, 1.0)
        cover = taken.sum(dim=-1) / n                                                   # [B, h] distinct best tiles
        collapse = a_fw[..., :n].sum(dim=-2).amax(dim=-1) / n                           # [B, h]
        head_ev = torch.stack([agg[..., 0], agg[..., 1], agg[..., 2], cover, collapse, agg[..., 3], agg[..., 4],
                               agg[..., 5]], dim=-1)                                    # [B, h, 8]
        u = self.match_mlp(torch.cat([xq, c, xq * c, (xq - c).abs()], dim=-1))          # [B, N, d_u]
        u_win = self.agg_match(u.unsqueeze(2), valid).squeeze(1)                        # [B, d_u]
        e_pair = self.norm(torch.cat([s_global.unsqueeze(-1), head_ev.reshape(b, -1), u_win], dim=-1))
        up = self.up(s_up.reshape(b, 1).to(e_pair.dtype)) if s_up is not None else torch.zeros(b, 1, device=xw.device,
                                                                                               dtype=e_pair.dtype)
        out: Dict[str, torch.Tensor] = {'score': self.score_mlp(torch.cat([e_pair, up], dim=-1)).squeeze(-1),
                                        's_global': s_global}
        if maps:
            out.update(A_fw=a_fw, A_wf=a_wf)
        return out

    def forward(self, fq: torch.Tensor, fw: torch.Tensor, shape: Tuple[int, int],
                s_up: Optional[torch.Tensor] = None, valid: Optional[torch.Tensor] = None,
                maps: bool = False) -> Dict[str, torch.Tensor]:
        qs = self.query_side(fq, shape, fw.shape[0])            # the query first: the random draws (dropout) are in this order
        ws = self.window_side(fw, shape, valid)
        return self.pair_side(qs, ws, shape, s_up, valid, maps)


def correspondence_loss(a_fw: torch.Tensor, a_wf: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    '''§9.2, on the positive pairs: `-mean log A_{F->W}[i, i]`, both ways. `[P, h, N, N+1]` -> scalar.'''
    n = a_fw.shape[-2]
    d = torch.arange(n, device=a_fw.device)
    return -0.5 * (a_fw[..., d, d].clamp(min=eps).log().mean() + a_wf[..., d, d].clamp(min=eps).log().mean())


def cycle_loss(a_fw: torch.Tensor, a_wf: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    '''§9.2: `-mean log (A_{F->W} A_{W->F})[i, i]`: a tile should come back to itself.'''
    n = a_fw.shape[-2]
    d = torch.arange(n, device=a_fw.device)
    cyc = a_fw[..., :n] @ a_wf[..., :n]
    return -cyc[..., d, d].clamp(min=eps).log().mean()


def build_gloret(cfg: Dict) -> GloRetNet:
    return GloRetNet(NetCfg(**cfg))


def build_logloret(cfg: Dict) -> LogLoRetNet:
    return LogLoRetNet(NetCfg(**cfg))
