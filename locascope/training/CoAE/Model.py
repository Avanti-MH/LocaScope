'''The CoAE network (spec.md §2).

    net = CoaeNet(cfg, device, n_sources)
    out = net(tiles_uint8)                  # tiles [B, 256, 256, 3] uint8 -> CoaeOut

    tile -> the FM's own transform (resize, centre crop: `net.field_px` px of the tile are seen, gigapath 224, uni2 256; normalise)
      -> the FM's blocks (frozen, or LoRA on every block's qkv) -> the patch tokens of `fm.layers` (resolved to blocks by the encoder's depth), each LayerNorm'd, mixed by softmax
    weights (start `fm.init`, share capped by `fm.cap`) -> tokens g x g x D (the FM's own grid, not resampled: gigapath 14, uni2 16);
    and the FM's own CLS and patch-token mean
      -> Swin stage on g x g (window g / 2, every other block shifted) -> 2 x 2 patch merging + the tokens pooled s x s (linear)
         -> latent s x s x latent_dim (s = g / 2)            (returned as it is)
           -> z = latent + noise (training only; `noisy`): the bottleneck, a regulariser of where the latent's power goes
           -> ConvNeXt decoder s -> 2s -> 4s on z (no skip connection) -> the Co-figure head (point N / dist channels), on the
              FM's field
           -> cross-attention on z: Q = attn_pool(tokens), K, V = the s x s cells of z; no residual from Q -> LN -> MLP -> LN -> cls'
              (returned as it is) -> + noise (training only) -> one linear map -> the place descriptor, 4 x 4 cells (always)
    The source discriminators (domain confusion: `Discriminator`) on every latent cell and on cls' exist only when head.w_adv > 0.
    Nothing of the view's slot, source or perturbation enters the network.
'''
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.checkpoint import checkpoint

from locascope.core.backbones.TileEncoder import encoder_config
from locascope.core.networks.Heads import resolve_encoder_layers
from locascope.core.stage2_retrieval.Retriever import EncoderPool
from locascope.training.CoAE.CoFigure import n_channels, n_dist_channels, n_moments
from locascope.training.CoAE.Config import CoaeConfig, FmConfig, ModelConfig


def noisy(x: torch.Tensor, sigma: float, training: bool) -> torch.Tensor:
    """The bottleneck: in training, x + sigma x rms x N(0, 1), the same sd on every dimension of a vector, rms = the vector's own
    uncentred root mean square over the last dimension (no gradient through it); x itself otherwise (eval, sigma 0). Isotropic and
    additive, not multiplicative and not dropout; not rescaled after. It limits where the power of x goes, not its information."""
    if not (training and sigma > 0):
        return x
    rms: torch.Tensor = x.detach().float().pow(2).mean(-1, keepdim=True).sqrt().to(x.dtype)
    return x + sigma * rms * torch.randn_like(x)


# ── the FM: blocks, LoRA, the layer mix ──────────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """base(x) + (alpha / r) B A x, B starting at zero (the FM unchanged at the start)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base: nn.Linear = base
        self.a: nn.Parameter = nn.Parameter(torch.empty(rank, base.in_features))
        self.b: nn.Parameter = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale: float = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.a.t().to(x.dtype) @ self.b.t().to(x.dtype)) * self.scale


class LayerMix(nn.Module):
    """softmax-weighted sum of LayerNorm'd layers, a layer's share at most its cap (a share over its cap is cut and the rest
    renormalised -- the softmax is kept differentiable everywhere below the caps)."""

    def __init__(self, n: int, dim: int, init: Tuple[float, ...], cap: Tuple[float, ...]) -> None:
        super().__init__()
        start: torch.Tensor = torch.tensor(init, dtype=torch.float32) if init else torch.ones(n)
        self.logits: nn.Parameter = nn.Parameter(torch.log(start / start.sum()))
        self.register_buffer('cap', torch.tensor(cap, dtype=torch.float32) if cap else torch.ones(n))
        self.norms: nn.ModuleList = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n)])
        self.gamma: nn.Parameter = nn.Parameter(torch.ones(()))

    def weights(self) -> torch.Tensor:
        """softmax, then every share over its cap set to the cap and the excess given to the layers under theirs in proportion
        (repeated until none is over): the shares sum to 1 and none exceeds its cap (the caps must sum to 1 or more)."""
        w: torch.Tensor = torch.softmax(self.logits, 0)
        for _ in range(len(w)):
            over: torch.Tensor = w > self.cap
            if not bool(over.any()):
                break
            excess: torch.Tensor = (w - self.cap).clamp(min=0).sum()
            w = torch.where(over, self.cap, w)
            free: torch.Tensor = w < self.cap
            w = torch.where(free, w + excess * w / (w * free).sum().clamp(min=1e-12), w)
        return w

    def forward(self, layers: List[torch.Tensor]) -> torch.Tensor:
        w: torch.Tensor = self.weights()
        return self.gamma * sum(w[i] * self.norms[i](x.float()) for i, x in enumerate(layers))


class FmTokens(nn.Module):
    """The FM's patch tokens at the blocks `cfg.layers` names (`blocks`: absolute 0-based indices, resolved against this encoder's depth), as a grid [B, g, g, D]; and the FM's own CLS and patch-token mean.
    `field_px`: the side in px of the `tile` px input that the FM sees (the transform's centre crop); `grid`: the side of its patch-token grid (crop_size // patch_size)."""

    def __init__(self, cfg: FmConfig, device: torch.device, grad_ckpt: bool, tile: int) -> None:
        super().__init__()
        enc: Any = EncoderPool(device).get(encoder_config(cfg.encoder, pooling='cls'))
        self.vit: nn.Module = enc.model
        self.cfg: FmConfig = cfg
        self.grad_ckpt: bool = grad_ckpt
        self.n_prefix: int = int(getattr(self.vit, 'num_prefix_tokens', 1))
        self.prep: Callable[[torch.Tensor], torch.Tensor] = enc.cfg.transform.build_tensor()   # the FM's own recipe: uint8 [N, H, W, 3] -> normalised [N, 3, S, S]
        tr: Any = enc.cfg.transform
        if tr.crop_size > tr.scale_size or (tr.crop_size * tile) % tr.scale_size != 0:
            raise ValueError(f'the transform (scale {tr.scale_size}, crop {tr.crop_size}) does not see a whole number of px of a {tile} px tile')
        self.field_px: int = int(tr.crop_size * tile // tr.scale_size)
        for p in self.vit.parameters():
            p.requires_grad_(False)
        if cfg.lora:
            for blk in self.vit.blocks:
                blk.attn.qkv = LoRALinear(blk.attn.qkv, cfg.lora_rank, cfg.lora_alpha)
        if cfg.weights_dtype == 'bf16':                                                       # the experiment arms: bfloat16 weights, no autocast in forward
            self.vit.to(torch.bfloat16)
        self.depth: int = len(self.vit.blocks)
        blocks: Tuple[int, ...] = resolve_encoder_layers(cfg.layers, self.depth)          # sorted, 0-based; refuses a bool, an index outside, two entries on one block
        each: Tuple[int, ...] = tuple(resolve_encoder_layers((x,), self.depth)[0] for x in cfg.layers)
        if each != blocks:
            raise ValueError(f'fm.layers {cfg.layers} name blocks {each} of {self.depth}, not in ascending order: fm.init and fm.cap pair '
                             f'with the layers by position')
        self.blocks: Tuple[int, ...] = blocks                                             # the blocks whose patch tokens are mixed (0-based)
        self.last: int = blocks[-1] + 1                                                    # the blocks run
        self.dim: int = int(self.vit.embed_dim)
        patch: int = int(self.vit.patch_embed.patch_size[0])                                  # as TileEncoder.py:738 reads it (not grid_size)
        if tr.crop_size % patch:
            raise ValueError(f'the transform\'s crop {tr.crop_size} px is not a whole number of {patch} px patches')
        self.grid: int = int(tr.crop_size // patch)                                           # the patch-token grid side on the FM's own input

    def forward(self, tiles: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """(the patch-token grids of the `blocks`, the FM's own CLS: the last block run's class token through the final norm, and the
        mean of the final norm's patch tokens), all float32. With `cfg.weights_dtype` 'bf16' the blocks run on bfloat16 weights with the
        caller's autocast switched off; with 'fp32' on float32 weights under the caller's autocast."""
        bf16: bool = self.cfg.weights_dtype == 'bf16'
        with torch.autocast('cuda', enabled=False) if bf16 else contextlib.nullcontext():
            out, cls_fm, avg_fm = self._run(tiles)
        if bf16:
            out = [t.float() for t in out]
        return out, cls_fm.float(), avg_fm.float()

    def _run(self, tiles: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """`forward`'s blocks, in the FM's weight dtype (the input is cast to it)."""
        x: torch.Tensor = self.prep(tiles).to(next(self.vit.parameters()).dtype)
        v: Any = self.vit
        x = v.norm_pre(v.patch_drop(v._pos_embed(v.patch_embed(x))))
        out: List[torch.Tensor] = []
        train_fm: bool = self.cfg.lora and self.training
        for i, blk in enumerate(v.blocks[:self.last]):
            if train_fm and self.grad_ckpt:
                x = checkpoint(blk, x, use_reentrant=False)
            elif train_fm:
                x = blk(x)
            else:
                with torch.no_grad():
                    x = blk(x)
            if i in self.blocks:
                p: torch.Tensor = x[:, self.n_prefix:]
                g: int = int(round(math.sqrt(p.shape[1])))
                out.append(p.reshape(p.shape[0], g, g, p.shape[2]))
        if train_fm:
            tok: torch.Tensor = v.norm(x)
        else:
            with torch.no_grad():
                tok = v.norm(x)
        cls_fm: torch.Tensor = tok[:, 0]
        avg_fm: torch.Tensor = tok[:, self.n_prefix:].mean(1)
        return out, cls_fm, avg_fm


# ── the Swin encoder ─────────────────────────────────────────────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window: int, heads: int) -> None:
        super().__init__()
        self.window: int = window
        self.heads: int = heads
        self.qkv: nn.Linear = nn.Linear(dim, 3 * dim)
        self.proj: nn.Linear = nn.Linear(dim, dim)
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(heads, (2 * window - 1) ** 2))
        c: torch.Tensor = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing='ij')).flatten(1)
        rel: torch.Tensor = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + window - 1
        self.register_buffer('rel_index', rel[..., 0] * (2 * window - 1) + rel[..., 1], persistent=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:          # x [nW*B, w*w, C]
        b, n, c = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        bias: torch.Tensor = self.bias[:, self.rel_index]                                        # [heads, n, n]
        if mask is not None:                                                                    # [nW, n, n]
            nw: int = mask.shape[0]
            bias = bias[None] + mask[:, None]
            bias = bias.repeat(b // nw, 1, 1, 1)
        else:
            bias = bias[None].expand(b, -1, -1, -1)
        y: torch.Tensor = TF.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype))
        return self.proj(y.transpose(1, 2).reshape(b, n, c))


class SwinBlock(nn.Module):
    def __init__(self, dim: int, window: int, heads: int, shift: bool) -> None:
        super().__init__()
        self.window: int = window
        self.shift: bool = shift
        self.n1: nn.LayerNorm = nn.LayerNorm(dim)
        self.attn: WindowAttention = WindowAttention(dim, window, heads)
        self.n2: nn.LayerNorm = nn.LayerNorm(dim)
        self.mlp: nn.Sequential = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def _mask(self, h: int, w: int, device: torch.device) -> Optional[torch.Tensor]:
        if not self.shift or h <= self.window:
            return None
        s: int = self.window // 2
        img: torch.Tensor = torch.zeros(h, w, device=device)
        n: int = 0
        for ys in (slice(0, -self.window), slice(-self.window, -s), slice(-s, None)):
            for xs in (slice(0, -self.window), slice(-self.window, -s), slice(-s, None)):
                img[ys, xs] = n
                n += 1
        win: torch.Tensor = _windows(img[None, :, :, None], self.window).squeeze(-1)              # [nW, n]
        diff: torch.Tensor = win[:, None, :] - win[:, :, None]
        return torch.where(diff != 0, torch.tensor(-100.0, device=device), torch.tensor(0.0, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:                                          # x [B, H, W, C]
        b, h, w, c = x.shape
        win: int = min(self.window, h)
        y: torch.Tensor = self.n1(x)
        shifted: bool = self.shift and h > win
        s: int = win // 2                                                                        # the shift: the same for the roll out, the roll back and _mask
        if shifted:
            y = torch.roll(y, (-s, -s), (1, 2))
        yw: torch.Tensor = _windows(y, win)                                                      # [B*nW, win*win, C]
        self.attn.window = win
        yw = self.attn(yw, self._mask(h, w, x.device) if shifted else None)
        y = _unwindows(yw, win, b, h, w)
        if shifted:
            y = torch.roll(y, (s, s), (1, 2))
        x = x + y
        return x + self.mlp(self.n2(x))


def _windows(x: torch.Tensor, w: int) -> torch.Tensor:
    b, h, wd, c = x.shape
    return x.view(b, h // w, w, wd // w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, c)


def _unwindows(x: torch.Tensor, w: int, b: int, h: int, wd: int) -> torch.Tensor:
    c: int = x.shape[-1]
    return x.view(b, h // w, wd // w, w, w, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, wd, c)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out: int) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(4 * dim)
        self.lin: nn.Linear = nn.Linear(4 * dim, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], -1)
        return self.lin(self.norm(x))


class SwinEncoder(nn.Module):
    """g x g x D_in -> linear to swin_dim -> swin_depth Swin blocks (window g / 2: four windows; odd blocks shifted by half a window, so
    the windows exchange) -> 2 x 2 patch merging -> s x s x latent_dim (s = g / 2); + the tokens averaged 2 x 2 (linear) as a residual;
    LayerNorm. A latent cell is 2 x 2 tokens."""

    def __init__(self, d_in: int, cfg: ModelConfig, grid: int) -> None:
        super().__init__()
        if grid % 2:
            raise ValueError(f'the token grid must be even: the latent is its 2 x 2 merging, got {grid}')
        self.grid: int = grid
        self.inp: nn.Linear = nn.Linear(d_in, cfg.swin_dim)
        self.stage: nn.Sequential = nn.Sequential(*[SwinBlock(cfg.swin_dim, grid // 2, cfg.swin_heads, shift=bool(i % 2))
                                                    for i in range(cfg.swin_depth)])
        self.merge: PatchMerging = PatchMerging(cfg.swin_dim, cfg.latent_dim)
        self.res: nn.Linear = nn.Linear(d_in, cfg.latent_dim)
        self.norm: nn.LayerNorm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:                                    # [B, g, g, D] -> [B, s, s, latent]
        if tokens.shape[1] != self.grid or tokens.shape[2] != self.grid:
            raise ValueError(f'tokens {tuple(tokens.shape[1:3])}, built for {self.grid} x {self.grid}')
        x: torch.Tensor = self.merge(self.stage(self.inp(tokens)))
        pooled: torch.Tensor = TF.avg_pool2d(tokens.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        return self.norm(x + self.res(pooled))


# ── the ConvNeXt decoder ─────────────────────────────────────────────────────────────────────────────────────────────────

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dw: nn.Conv2d = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm: nn.LayerNorm = nn.LayerNorm(dim)
        self.pw1: nn.Linear = nn.Linear(dim, 4 * dim)
        self.pw2: nn.Linear = nn.Linear(4 * dim, dim)
        self.gamma: nn.Parameter = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:                                          # [B, C, H, W]
        y: torch.Tensor = self.dw(x).permute(0, 2, 3, 1)
        y = self.pw2(TF.gelu(self.pw1(self.norm(y)))) * self.gamma
        return x + y.permute(0, 3, 1, 2)


class ConvNeXtDecoder(nn.Module):
    """s x s x latent -> 1x1 to decoder_dim[0], blocks -> (x2 up nearest, 1x1 to the next width, blocks) for each further width -> 4s x 4s
    -> 1x1 to the output channels. Every step is a whole x2: a latent cell becomes 4 x 4 Co-figure cells."""

    def __init__(self, cfg: ModelConfig, out_ch: int) -> None:
        super().__init__()
        self.stem: nn.Sequential = nn.Sequential(nn.Conv2d(cfg.latent_dim, cfg.decoder_dim[0], 1),
                                                 *[ConvNeXtBlock(cfg.decoder_dim[0]) for _ in range(cfg.decoder_depth)])
        ups: List[nn.Module] = []
        d: int = cfg.decoder_dim[0]
        for d_out in cfg.decoder_dim[1:]:
            ups.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(d, d_out, 1),
                                     *[ConvNeXtBlock(d_out) for _ in range(cfg.decoder_depth)]))
            d = d_out
        self.ups: nn.Sequential = nn.Sequential(*ups)
        self.head: nn.Conv2d = nn.Conv2d(d, out_ch, 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:                                     # [B, s, s, C] -> [B, out, 4s, 4s]
        x: torch.Tensor = self.ups(self.stem(latent.permute(0, 3, 1, 2).contiguous()))
        return self.head(x.contiguous())             # standard strides: DDP's gradient buckets expect them (no stride warning)


# ── attn_pool, the cross-attention, the discriminator ────────────────────────────────────────────────────────────────────

class AttnPool(nn.Module):
    """One learned query attending the tokens: q [B, d]."""

    def __init__(self, d_in: int, d: int, heads: int) -> None:
        super().__init__()
        self.query: nn.Parameter = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.norm: nn.LayerNorm = nn.LayerNorm(d_in)
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(d, heads, kdim=d_in, vdim=d_in, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:                                     # [B, g, g, D]
        kv: torch.Tensor = self.norm(tokens.flatten(1, 2))
        return self.attn(self.query.expand(kv.shape[0], -1, -1), kv, kv, need_weights=False)[0][:, 0]


class CrossToCls(nn.Module):
    """cls' = MLP(LN(Attn(Q = q, K = V = latent cells))) with a residual inside the MLP only (nothing of q passes but its choice)."""

    def __init__(self, d_q: int, d_kv: int, d: int, heads: int) -> None:
        super().__init__()
        self.nq: nn.LayerNorm = nn.LayerNorm(d_q)
        self.nkv: nn.LayerNorm = nn.LayerNorm(d_kv)
        self.attn: nn.MultiheadAttention = nn.MultiheadAttention(d, heads, kdim=d_kv, vdim=d_kv, batch_first=True)
        self.q_in: nn.Linear = nn.Linear(d_q, d)
        self.n2: nn.LayerNorm = nn.LayerNorm(d)
        self.mlp: nn.Sequential = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:                    # q [B, d_q], latent [B, s, s, d_kv]
        kv: torch.Tensor = self.nkv(latent.flatten(1, 2))
        a: torch.Tensor = self.attn(self.q_in(self.nq(q))[:, None], kv, kv, need_weights=False)[0][:, 0]
        return a + self.mlp(self.n2(a))


class Discriminator(nn.Module):
    """The source classifier of the adversary (domain confusion, Tzeng et al. 2015): `fit` sees the features detached (it learns to
    tell the sources apart, only its own weights move); `judge` sees them live through FROZEN copies of its weights (only the network
    moves, toward the classifier being unable to tell: its output uniform). No gradient reversal: a network that maximised the
    classifier's cross-entropy could swap the sources' places instead of mixing them (accuracy 0, every source still apart)."""

    def __init__(self, d: int, hidden: int, n_sources: int) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_sources))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.detach())

    def judge(self, x: torch.Tensor) -> torch.Tensor:
        frozen = {k: v.detach() for k, v in self.net.named_parameters()}
        return torch.func.functional_call(self.net, frozen, (x,))


# ── the whole ────────────────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class CoaeOut:
    """latent [B, s, s, L] and cls [B, d] (cls'), both WITHOUT the training noise; cofig [B, C, t, t] (t = cofig.out_cells = 4s; point N /
    dist channels, from the noisy latent); pooled [B, M, c, c] (the linear head on the noisy cls': the place descriptor); cls_fm [B, D] and avg_fm [B, D]
    (the FM's own CLS and patch-token mean). The source classifier's logits, only when the discriminators exist (HeadConfig.w_adv > 0,
    else None): `src_*` on the detached features (its own training, and its accuracy), `conf_*` on the live features through its
    frozen weights (the network's confusion loss): cells [B, s x s, S], cls' [B, S] (None without a classifier on cls')."""
    latent: torch.Tensor
    cls: torch.Tensor
    cofig: torch.Tensor
    pooled: torch.Tensor
    cls_fm: torch.Tensor
    avg_fm: torch.Tensor
    src_cells: Optional[torch.Tensor] = None
    src_cls: Optional[torch.Tensor] = None
    conf_cells: Optional[torch.Tensor] = None
    conf_cls: Optional[torch.Tensor] = None
    cls_fm_part: Optional[torch.Tensor] = None          # with the residual: the FM's own CLS, the FM's part of cls'
    cls_att_part: Optional[torch.Tensor] = None         # with the residual: gain x the attention branch, the AE's part of cls'


class CoaeNet(nn.Module):
    def __init__(self, cfg: CoaeConfig, device: torch.device, n_sources: int) -> None:
        super().__init__()
        self.cfg: CoaeConfig = cfg
        self.fm: FmTokens = FmTokens(cfg.fm, device, cfg.train.grad_ckpt, cfg.cofig.tile)
        d: int = self.fm.dim
        m: ModelConfig = cfg.model
        self.mix: LayerMix = LayerMix(len(self.fm.blocks), d, cfg.fm.init, cfg.fm.cap)
        if self.fm.grid != cfg.token_grid:
            raise ValueError(f'{cfg.fm.encoder}: the FM\'s token grid is {self.fm.grid}, Config.FM_GRIDS says {cfg.token_grid}')
        self.encoder: SwinEncoder = SwinEncoder(d, m, self.fm.grid)
        self.latent_side: int = cfg.latent_side
        out_ch: int = n_channels(cfg.cofig) if cfg.head.output == 'point' else n_dist_channels(cfg.cofig)
        self.decoder: ConvNeXtDecoder = ConvNeXtDecoder(m, out_ch)
        self.cls_dim: int = m.cls_dim or d                                                    # 0: the FM's own size
        self.pool: AttnPool = AttnPool(d, self.cls_dim, m.pool_heads)
        self.cross: CrossToCls = CrossToCls(self.cls_dim, m.latent_dim, self.cls_dim, m.cross_heads)
        self.cls_norm: nn.LayerNorm = nn.LayerNorm(self.cls_dim)
        self.noise: float = m.noise
        # the source discriminators exist only when the confusion loss is on (no parameter DDP would find unused)
        self.disc_cells: Optional[Discriminator] = (Discriminator(m.latent_dim, m.disc_hidden, n_sources)
                                                    if cfg.head.w_adv > 0 else None)
        self.disc_cls: Optional[Discriminator] = (Discriminator(self.cls_dim, m.disc_hidden, n_sources)
                                                  if cfg.head.w_adv > 0 and m.disc_on_cls else None)
        self.adv_lambda: float = 0.0                                                          # the confusion loss's weight (ramped by train.py)
        # the residual from the FM's own CLS (ModelConfig.cls_residual): cls' = cls_fm + gain x the attention branch, cls_fm UNTOUCHED
        # (no projection, no scale: frozen, it cannot change), gain 0 at start -- so cls' starts as exactly the sliding window's own
        # feature, and whatever cls' gains or loses comes from the attention branch (the latent and q), never from a re-mapped FM CLS
        self.residual: bool = m.cls_residual
        if self.residual:
            if self.cls_dim != d:
                raise ValueError(f'cls_residual needs cls_dim the FM\'s size ({d}), got {self.cls_dim}: no projection of the FM\'s CLS')
            self.cls_gain: nn.Parameter = nn.Parameter(torch.zeros(()))
        # cls' always has its content loss: one linear map from (the noisy) cls' to the place descriptor (CoFigure.shared_moments) --
        # the reconstruction itself gives cls' no gradient (it is a branch off the latent); this gives it one, and linear, so that
        # what it must hold lies in cls' itself, not in a head behind it
        self.pool_shape: Tuple[int, int] = (n_moments(cfg.cofig), cfg.head.pool_cells)
        self.cls_to_pooled: nn.Linear = nn.Linear(self.cls_dim, self.pool_shape[0] * self.pool_shape[1] ** 2)
        # the linear head predicts in standardised units (the descriptor's running mean and sd a channel); pooled = head x sd + mean
        self.register_buffer('pool_mu', torch.zeros(self.pool_shape[0]))
        self.register_buffer('pool_sd', torch.ones(self.pool_shape[0]))
        self.register_buffer('pool_seen', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def see_places(self, desc: torch.Tensor, momentum: float = 0.01) -> None:
        """Update the running mean and sd a channel of the place descriptor `desc` [B, M, c, c]: the units the linear head predicts in."""
        v: torch.Tensor = desc.permute(1, 0, 2, 3).flatten(1)                                    # [M, B c c]
        mu: torch.Tensor = v.mean(dim=1)
        sd: torch.Tensor = v.std(dim=1).clamp(min=1e-4)
        if int(self.pool_seen) == 0:
            self.pool_mu.copy_(mu)
            self.pool_sd.copy_(sd)
        else:
            self.pool_mu.mul_(1 - momentum).add_(momentum * mu)
            self.pool_sd.mul_(1 - momentum).add_(momentum * sd)
        self.pool_seen += 1

    @property
    def field_px(self) -> int:
        """The side in px of the 256 px tile that the FM sees: the field the Co-figure targets and the decoder's output cover."""
        return self.fm.field_px

    def trainable(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """(the head's parameters, the FM's LoRA parameters)."""
        lora: List[nn.Parameter] = [p for n, p in self.fm.named_parameters() if p.requires_grad]
        head: List[nn.Parameter] = [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith('fm.')]
        return head, lora

    def forward(self, tiles: torch.Tensor) -> CoaeOut:
        layers, cls_fm, avg_fm = self.fm(tiles)
        tokens: torch.Tensor = self.mix(layers)                                                   # [B, g, g, D] float32, the FM's own grid
        latent: torch.Tensor = self.encoder(tokens)
        z: torch.Tensor = noisy(latent, self.noise, self.training)                                # the bottleneck: the decoder and cross see z
        cofig: torch.Tensor = self.decoder(z)
        branch: torch.Tensor = self.cls_norm(self.cross(self.pool(tokens), z))
        fm_part: Optional[torch.Tensor] = None
        att_part: Optional[torch.Tensor] = None
        cls: torch.Tensor
        head_in: torch.Tensor
        if self.residual:                                     # the FM's CLS as it is, the attention branch added (gain 0 at start)
            fm_part = cls_fm.to(branch.dtype)
            att_part = self.cls_gain * branch
            cls = fm_part + att_part
            head_in = fm_part + self.cls_gain * noisy(branch, self.noise, self.training)
        else:
            cls = branch
            head_in = noisy(branch, self.noise, self.training)
        head: torch.Tensor = self.cls_to_pooled(head_in).view(-1, self.pool_shape[0], self.pool_shape[1], self.pool_shape[1])   # standardised
        pooled: torch.Tensor = head * self.pool_sd.view(1, -1, 1, 1) + self.pool_mu.view(1, -1, 1, 1)                          # [B, M, c, c]
        dc: Optional[Discriminator] = self.disc_cls
        out: CoaeOut = CoaeOut(latent=latent, cls=cls, cofig=cofig, pooled=pooled, cls_fm=cls_fm, avg_fm=avg_fm,
                               cls_fm_part=fm_part, cls_att_part=att_part)
        if self.disc_cells is not None:                       # the discriminators' four outputs, only when they exist
            cells: torch.Tensor = latent.flatten(1, 2)
            out.src_cells = self.disc_cells.fit(cells)
            out.conf_cells = self.disc_cells.judge(cells)
            if dc is not None:
                out.src_cls = dc.fit(cls)
                out.conf_cls = dc.judge(cls)
        return out
