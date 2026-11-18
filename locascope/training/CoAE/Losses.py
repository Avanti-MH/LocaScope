'''The CoAE losses (spec.md §5).

    scale  = ChannelScale(n)                     # a running sd a Co-figure channel: the channels brought to one scale before the loss
    l_rec  = recon_point(pred, tg, scale)        # variants vref, vavg: masked MSE to the shared target (CoFigure.view_targets)
    l_rec  = recon_dist(pred, tg, scale, cfg)    # variant vdist: NLL of every sharer's own map (vMF a direction, Gaussian a channel)
    point_terms / dist_terms                     # the same a signal (the log's columns)
    l_pool = recon_pooled(o.pooled, mean, w, net.pool_sd)   # cls' linear prediction of the place descriptor (CoFigure.shared_moments)
    agreement(cofig_point(pred, ...), target, valid, scale, cfg)   # cos and scaled MSE a signal (the probe's and the real photos' check)
    l_nce  = target_nce(head, place, w, keys, bank, bank_ok, tau)   # the linear head against the place descriptors (own = positive); only when head.w_nce > 0
    l_vnce = view_nce(cls, place, w, tau)        # view-to-view contrast on cls' (experiment arms only); only when head.w_vnce > 0
    l_hnce = view_nce(head, place, w, tau)       # view-to-view contrast on the linear head's output (experiment arms only); only when head.w_hnce > 0
    l_dnce = twin_nce(decoder_blocks(o.cofig, side), sources, w, cells, tau)   # twin contrast on the decoder's blocks (experiment arms only); only when head.w_dnce > 0
    l_lnce = twin_nce(o.latent.flatten(1, 2), sources, w, cells, tau)          # the same on the latent (experiment arms only); only when head.w_lnce > 0
    l_kol  = koleo_by_source(cls, source)      # KoLeo within each source; train.py calls it only when head.w_koleo > 0
    l_fit, l_conf = adversary(out, source)       # the source classifier's CE, the network's confusion; only when head.w_adv > 0

Every loss of the training step is a class (`Loss`: key, output, always, column, grad_name, floor, __call__, grad_part, after), registered in
`LOSSES` in the order the trainer sums them; `step_losses(StepContext)` walks the list (train.py calls it, nothing else). A new loss = a class
here + a row in Config.LOSS_TERMS (when it has a letter).
'''
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as TF

from locascope.training.CoAE.CoFigure import Part, Targets, layout, part_names
from locascope.training.CoAE.Config import CoaeConfig, CoFigureConfig
from locascope.training.CoAE.Model import CoaeNet, CoaeOut

if TYPE_CHECKING:
    from locascope.training.CoAE.Supply import Batch


class ChannelScale(nn.Module):
    """A running sd a channel of the targets (valid cells only), so that every signal weighs alike in the loss."""

    def __init__(self, n: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.register_buffer('sd', torch.ones(n))
        self.register_buffer('seen', torch.zeros((), dtype=torch.long))
        self.momentum: float = momentum

    @torch.no_grad()
    def update(self, mean: torch.Tensor, valid: torch.Tensor) -> None:
        v: torch.Tensor = mean.permute(1, 0, 2, 3)[:, valid]                                     # [N, cells]
        sd: torch.Tensor = v.std(dim=1).clamp(min=1e-4)
        if int(self.seen) == 0:
            self.sd.copy_(sd)
        else:
            self.sd.mul_(1 - self.momentum).add_(self.momentum * sd)
        self.seen += 1


def _masked_mean(x: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """x [B, ..., t, t] averaged over the cells [B, t, t] (and every other axis), a cell counted by its weight (Targets.cells:
    0 = invalid, else the version's loss weight)."""
    m: torch.Tensor = cells.to(x.dtype)
    while m.dim() < x.dim():
        m = m.unsqueeze(1)
    m = m.expand_as(x)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def recon_point(pred: torch.Tensor, tg: Targets, scale: ChannelScale) -> torch.Tensor:
    """Masked MSE of the predicted Co-figure [B, N, t, t] to the place mean, a channel divided by its running sd."""
    s: torch.Tensor = scale.sd.view(1, -1, 1, 1)
    return _masked_mean(((pred - tg.mean) / s) ** 2, tg.cells)


def point_terms(pred: torch.Tensor, tg: Targets, scale: ChannelScale, cfg: CoFigureConfig) -> List[torch.Tensor]:
    """`recon_point` a part of `layout` (its channels only): what each signal contributes; their channel-weighted mean is `recon_point`."""
    s: torch.Tensor = scale.sd.view(1, -1, 1, 1)
    err: torch.Tensor = ((pred - tg.mean) / s) ** 2
    return [_masked_mean(err[:, p.start:p.start + p.size], tg.cells) for p in layout(cfg)]


def recon_pooled(pred: torch.Tensor, mean: torch.Tensor, weight: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """Masked MSE of cls' linear prediction [B, M, c, c] to the place descriptor (`CoFigure.shared_moments`: mean; weight [B, c, c] the
    view's rung weight), a channel divided by its running sd (`sd` [M]: `CoaeNet.pool_sd`, the descriptor's own)."""
    return _masked_mean(((pred - mean) / sd.view(1, -1, 1, 1)) ** 2, weight)


def cofig_point(pred: torch.Tensor, cfg: CoFigureConfig, scale: ChannelScale, output: str) -> torch.Tensor:
    """The decoder's output as a Co-figure [B, N, t, t]: 'point' as it is; 'dist' its locations (a direction block's unit mean
    direction, a Gaussian channel's mean brought back from its running sd)."""
    if output == 'point':
        return pred
    out: List[torch.Tensor] = []
    at: int = 0
    for p in layout(cfg):
        if p.unit:
            out.append(TF.normalize(pred[:, at:at + 3], dim=1))
            at += 4
        else:
            out.append(pred[:, at:at + p.size] * scale.sd[p.start:p.start + p.size].view(1, -1, 1, 1))
            at += 2 * p.size
    return torch.cat(out, 1)


@torch.no_grad()
def agreement(point: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, scale: ChannelScale,
              cfg: CoFigureConfig) -> Dict[str, Dict[str, float]]:
    """How well a predicted Co-figure [B, N, t, t] (`cofig_point`) matches a target on its valid cells [B, t, t], a part of `layout`:
    cos  the cosine over the part's channels and the tile's valid cells, the mean over the tiles
    mse  the masked MSE with a channel divided by its running sd (the unit of `recon_point`: 1 = as far as the channel's own sd)."""
    m: torch.Tensor = valid.float()[:, None]
    keep: torch.Tensor = valid.flatten(1).any(1)
    out: Dict[str, Dict[str, float]] = {}
    for name, p in zip(part_names(cfg), layout(cfg)):
        a: torch.Tensor = (point[:, p.start:p.start + p.size].float() * m).flatten(1)
        b: torch.Tensor = (target[:, p.start:p.start + p.size].float() * m).flatten(1)
        cos: torch.Tensor = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp(min=1e-12)
        s: torch.Tensor = scale.sd[p.start:p.start + p.size].repeat_interleave(valid[0].numel())
        mse: torch.Tensor = (((a - b) / s) ** 2).sum() / (m.sum() * p.size).clamp(min=1.0)
        out[name] = {'cos': float(cos[keep].mean()) if bool(keep.any()) else float('nan'), 'mse': float(mse)}
    return out


def _vmf3_nll(mu: torch.Tensor, log_kappa: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """-log vMF(x | mu, kappa) on the 2-sphere: mu, x unit [..., 3, t, t]; log_kappa [..., t, t]."""
    k: torch.Tensor = log_kappa.clamp(-5.0, 7.0).exp()
    log_norm: torch.Tensor = -log_kappa.clamp(-5.0, 7.0) + math.log(2 * math.pi) + k + torch.log1p(-torch.exp(-2 * k) + 1e-12)
    return log_norm - k * (mu * x).sum(-3)


def dist_terms(pred: torch.Tensor, tg: Targets, scale: ChannelScale, cfg: CoFigureConfig, beta: float) -> List[torch.Tensor]:
    """NLL of every version's aligned map tg.obs [B, V, N, t, t] under the predicted distribution (pred [B, dist channels, t, t]), a
    part of `layout`: a direction block (local phase a frequency) vMF on its unit observations; a Gaussian channel (scaled by its
    running sd) with beta-NLL (the term weighted by the detached variance ** beta)."""
    terms: List[torch.Tensor] = []
    at: int = 0
    obs: torch.Tensor = tg.obs
    for p in layout(cfg):
        x: torch.Tensor = obs[:, :, p.start:p.start + p.size]                                   # [B, V, size, t, t]
        if p.unit:
            mu: torch.Tensor = TF.normalize(pred[:, at:at + 3], dim=1)[:, None]
            lk: torch.Tensor = pred[:, at + 3][:, None]
            nll: torch.Tensor = _vmf3_nll(mu, lk, TF.normalize(x, dim=2))                       # [B, V, t, t]
            terms.append(_masked_mean(nll, tg.cells))
            at += 4
        else:
            s: torch.Tensor = scale.sd[p.start:p.start + p.size].view(1, 1, -1, 1, 1)
            m: torch.Tensor = pred[:, at:at + p.size][:, None]
            log_sd: torch.Tensor = pred[:, at + p.size:at + 2 * p.size][:, None].clamp(-7.0, 5.0)
            var: torch.Tensor = torch.exp(2 * log_sd)
            nll = log_sd + 0.5 * (x / s - m) ** 2 / var
            if beta > 0:
                nll = nll * var.detach() ** beta
            terms.append(_masked_mean(nll, tg.cells))
            at += 2 * p.size
    return terms


def recon_dist(pred: torch.Tensor, tg: Targets, scale: ChannelScale, cfg: CoFigureConfig, beta: float) -> torch.Tensor:
    """Arm D's reconstruction: the mean of `dist_terms` (a part each, whatever its channels)."""
    return torch.stack(dist_terms(pred, tg, scale, cfg, beta)).mean()


def target_nce(head: torch.Tensor, place: torch.Tensor, weight: torch.Tensor, keys: torch.Tensor, bank: torch.Tensor,
               bank_ok: torch.Tensor, tau: float) -> torch.Tensor:
    """Target contrast. head [B, D]: the views' linear heads in standardised units; place [B] int64: a view's row of `keys`;
    weight [B]: the view's rung weight; keys [P, D]: the batch's places' descriptors, standardised, no gradient; bank [K, D]: more
    descriptors (the same units), bank_ok [B, K] bool: which of them a view may take as negatives (those of its place's cell).
    A view's loss = cross entropy of cos(head, .) / tau over keys and its allowed bank rows, its own key the label; the weighted mean."""
    h: torch.Tensor = TF.normalize(head.float(), dim=-1)
    k: torch.Tensor = TF.normalize(torch.cat([keys, bank]).float(), dim=-1)
    logits: torch.Tensor = h @ k.t() / tau                                              # [B, P + K]
    allowed: torch.Tensor = torch.cat([torch.ones(len(h), len(keys), dtype=torch.bool, device=h.device), bank_ok], 1)
    ce: torch.Tensor = TF.cross_entropy(logits.masked_fill(~allowed, float('-inf')), place, reduction='none')
    return (ce * weight).sum() / weight.sum().clamp(min=1e-9)


def view_nce(cls: torch.Tensor, place: torch.Tensor, weight: torch.Tensor, tau: float) -> torch.Tensor:
    """View-to-view contrast (several positives a view). cls [B, D]; place [B]: a view's place; weight [B].
    u = unit cls; s = u u^T / tau with the diagonal left out of every sum; a view i's loss = - the mean over the other views p of its
    place of (s[i, p] - logsumexp over a != i of s[i, a]); the weighted mean over the views. A place of one view alone: ValueError."""
    u: torch.Tensor = TF.normalize(cls.float(), dim=-1)
    s: torch.Tensor = u @ u.t() / tau
    eye: torch.Tensor = torch.eye(len(u), dtype=torch.bool, device=u.device)
    logp: torch.Tensor = s - torch.logsumexp(s.masked_fill(eye, float('-inf')), dim=1, keepdim=True)
    pos: torch.Tensor = (place[:, None] == place[None, :]) & ~eye
    n_pos: torch.Tensor = pos.sum(1)
    if bool((n_pos == 0).any()):
        raise ValueError('view_nce: a place with a single view has no positive')
    per: torch.Tensor = -(logp * pos).sum(1) / n_pos
    return (per * weight).sum() / weight.sum().clamp(min=1e-9)


def twin_nce(vec: torch.Tensor, sources: int, weight: torch.Tensor, cells: int, tau: float) -> torch.Tensor:
    """Twin contrast a cell position. vec [B, C, D]: C local vectors a view (C = the latent's s x s cells), the views in the batch's
    order (every `sources` consecutive views = one (place, slot), its sources: twins, one geometry; Supply.Batch); weight [B].
    `cells` of the C positions are drawn (no repeat), THE SAME for the whole batch. At a drawn position c: the vectors vec[:, c] [B, D],
    a group = a (place, slot); view_nce of them (the group's other sources = the positives, the other groups' vectors AT THE SAME
    POSITION = the negatives: a code of the position alone cannot tell them apart). The mean over the drawn positions."""
    b: int = vec.shape[0]
    c: int = vec.shape[1]
    if sources < 2 or b % sources or not 1 <= cells <= c:
        raise ValueError(f'twin_nce: needs sources >= 2, a batch of {b} views a multiple of sources ({sources}) and 1 <= cells <= {c}, got cells {cells}')
    group: torch.Tensor = torch.arange(b, device=vec.device) // sources
    pick: torch.Tensor = torch.randperm(c, device=vec.device)[:cells]
    return torch.stack([view_nce(vec[:, int(i)], group, weight, tau) for i in pick]).mean()


def decoder_blocks(cofig: torch.Tensor, side: int) -> torch.Tensor:
    """The decoder's output [B, N, t, t] as a latent cell's block each: [B, side x side, N x (t / side)^2] (t a multiple of side; gigapath
    28 / 7: 49 blocks of 10 x 4 x 4 = 160)."""
    b: int = cofig.shape[0]
    n: int = cofig.shape[1]
    t: int = cofig.shape[2]
    if t % side:
        raise ValueError(f'decoder_blocks: the output side {t} is not a multiple of the latent side {side}')
    k: int = t // side
    return cofig.float().reshape(b, n, side, k, side, k).permute(0, 2, 4, 1, 3, 5).reshape(b, side * side, n * k * k)


def koleo(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KoLeo (DINOv2): -mean log of each unit vector's distance to its nearest other."""
    if x.shape[0] < 2:
        return x.new_zeros(())
    u: torch.Tensor = TF.normalize(x.float(), dim=1)
    d: torch.Tensor = torch.cdist(u, u)
    d = d.masked_fill(torch.eye(len(u), dtype=torch.bool, device=u.device), float('inf'))     # not in place: cdist's output is saved for backward
    return -torch.log(d.min(dim=1).values + eps).mean()


def koleo_by_source(cls: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """KoLeo within each source: the batch's places are distinct, so every pair is two places."""
    terms: List[torch.Tensor] = [koleo(cls[source == s]) for s in source.unique()]
    return torch.stack(terms).mean()


def koleo_global(cls: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """KoLeo within each source over the WHOLE job's batch: every process's cls' gathered (differentiably: the gradient goes back to
    each process's own samples) and its sources (the places of a step never overlap across processes, Supply). Without a process
    group it is `koleo_by_source`."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return koleo_by_source(cls, source)
    from torch.distributed.nn.functional import all_gather
    every: torch.Tensor = torch.cat(all_gather(cls.contiguous()))
    src: List[torch.Tensor] = [torch.empty_like(source) for _ in range(dist.get_world_size())]
    dist.all_gather(src, source.contiguous())
    return koleo_by_source(every, torch.cat(src))


def _confusion(logits: torch.Tensor) -> torch.Tensor:
    """KL(uniform || the classifier's output), averaged: the cross-entropy against the uniform distribution minus log(sources); 0 when
    the classifier gives every source the same probability, large when it is sure of ANY source (right or wrong)."""
    k: int = logits.shape[-1]
    return (-TF.log_softmax(logits.float(), -1).mean(-1) - math.log(k)).mean()


def adversary(out: CoaeOut, source: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(the classifier's loss, the network's loss), on every latent cell and on cls':
    fit        the source classifier's cross-entropy on the DETACHED features (it learns to tell the sources apart; only its weights)
    confusion  KL(uniform || the FROZEN classifier's output) on the live features (only the network: toward the classifier unable to tell)."""
    n_cells: int = out.src_cells.shape[1]
    fit: torch.Tensor = TF.cross_entropy(out.src_cells.flatten(0, 1).float(), source.repeat_interleave(n_cells))
    conf: torch.Tensor = _confusion(out.conf_cells)
    if out.src_cls is not None and out.conf_cls is not None:
        fit = fit + TF.cross_entropy(out.src_cls.float(), source)
        conf = conf + _confusion(out.conf_cls)
    return fit, conf


def source_accuracy(logits: Optional[torch.Tensor], source: torch.Tensor) -> float:
    """C5: the share the discriminator gets right (chance = 1 / sources)."""
    if logits is None:
        return float('nan')
    if logits.dim() == 3:
        return float((logits.argmax(-1) == source[:, None]).float().mean())
    return float((logits.argmax(-1) == source).float().mean())


# ── the loss registry (spec.md §5): every loss a class, the trainer walks the list ──────────────────────────────────────────

class TargetBank:
    """The target contrast's bank: a place's descriptor (raw units) as last computed, over the whole pool. Filled as the places are
    seen; on the GPU, not in the checkpoint.
    desc  float32 [places, M, c, c]
    ok    bool [places]: filled
    cell  int64 [places]: the place's (slide, rung) cell (Supply.cell_of)"""

    def __init__(self, cell_of: List[int], pool_shape: Tuple[int, int], device: torch.device) -> None:
        self.desc: torch.Tensor = torch.zeros((len(cell_of), pool_shape[0], pool_shape[1], pool_shape[1]), device=device)
        self.ok: torch.Tensor = torch.zeros(len(cell_of), dtype=torch.bool, device=device)
        self.cell: torch.Tensor = torch.as_tensor(cell_of, device=device)

    def rows_for(self, idx: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
        """The filled rows [K] a step may draw negatives from: not the step's own places (idx [P]), in one of the step's cells ([P])."""
        in_batch: torch.Tensor = torch.zeros_like(self.ok)
        in_batch[idx] = True
        return (self.ok & ~in_batch & torch.isin(self.cell, cells)).nonzero().flatten()

    def write(self, idx: torch.Tensor, desc: torch.Tensor) -> None:
        """Put the step's places' descriptors (desc [P, M, c, c]) in the bank; after the loss: a place is never its own bank negative."""
        self.desc[idx] = desc
        self.ok[idx] = True


@dataclass
class StepContext:
    """What a loss may read of one training step (no loss reads anything else)."""
    cfg: CoaeConfig
    net: CoaeNet                      # the buffers (pool_mu, pool_sd, latent_side, adv_lambda)
    out: CoaeOut
    targets: Targets
    p_mean: torch.Tensor              # the place descriptor of every view [B, M, c, c]
    p_w: torch.Tensor                 # its cell weights [B, c, c]
    scale: ChannelScale
    batch: Batch
    place_v: torch.Tensor             # [B] a view's place
    w_v: torch.Tensor                 # [B] a view's rung weight
    source_v: torch.Tensor            # [B] a view's source
    shares_equal: bool                # every process read its whole share
    bank: Optional[TargetBank]        # the target contrast's bank, None when it is off

    @property
    def views(self) -> int:
        """Views a place (V)."""
        return len(self.batch.place) // len(self.batch.slides)

    @property
    def head_std(self) -> torch.Tensor:
        """The linear head's own output [B, D], in standardised units."""
        mu: torch.Tensor = self.net.pool_mu.view(1, -1, 1, 1)
        sd: torch.Tensor = self.net.pool_sd.view(1, -1, 1, 1)
        return ((self.out.pooled.float() - mu) / sd).flatten(1)


class Loss:
    """One loss. A class attribute names it; `__call__` gives its value and its extra log columns; the weight is `HeadConfig.w_<key>`.
    key       the weight's key
    output    what it acts on: 'cofig' | 'head' | 'cls' | 'latent' | 'source-classifier'
    always    computed and logged (times its weight) even at weight 0
    column    the log column of its value (None: it logs only its own columns)
    grad_name its name on the gradient line"""
    key: str
    output: str
    always: bool = False
    column: Optional[str] = None
    grad_name: str

    def weight(self, cfg: CoaeConfig) -> float:
        return float(getattr(cfg.head, f'w_{self.key}'))

    def floor(self, ctx: StepContext) -> Optional[float]:
        """Its lower bound (None: it has none)."""
        return None

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        """(the loss, more log columns)."""
        raise NotImplementedError

    def grad_part(self, ctx: StepContext, part: torch.Tensor) -> torch.Tensor:
        """The term the gradient line shows, given the weighted loss `part` (the loss itself unless the loss says otherwise)."""
        return part

    def after(self, ctx: StepContext) -> None:
        """State to update once every loss is computed (called when the weight is > 0)."""


#: the losses in the ORDER the trainer sums them (floating-point addition and the random draws keep this order)
LOSSES: Dict[str, Loss] = {}


def register(cls: Type[Loss]) -> Type[Loss]:
    """Class decorator: an instance of the loss goes into `LOSSES` under its key (in the order the classes are defined)."""
    term: Loss = cls()
    if term.key in LOSSES:
        raise ValueError(f'the loss {term.key!r} is registered twice')
    LOSSES[term.key] = term
    return cls


@register
class ReconLoss(Loss):
    """The decoder's reconstruction (point: masked MSE to the shared target; dist: NLL of every sharer's own map); the signals' own
    values are the log's columns `r_<signal>`."""
    key = 'recon'
    output = 'cofig'
    always = True
    column = 'recon'
    grad_name = 'rec'

    def floor(self, ctx: StepContext) -> Optional[float]:
        return 0.0

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg: CoaeConfig = ctx.cfg
        pred: torch.Tensor = ctx.out.cofig.float()
        if cfg.head.output == 'point':
            value: torch.Tensor = recon_point(pred, ctx.targets, ctx.scale)
            with torch.no_grad():
                terms: List[torch.Tensor] = point_terms(pred, ctx.targets, ctx.scale, cfg.cofig)
        else:
            terms = dist_terms(pred, ctx.targets, ctx.scale, cfg.cofig, cfg.head.beta_nll)
            value = torch.stack(terms).mean()
        return value, {f'r_{n}': float(t) for n, t in zip(part_names(cfg.cofig), terms)}


@register
class DescriptorLoss(Loss):
    """cls' linear prediction of the place descriptor."""
    key = 'pool'
    output = 'head'
    always = True
    column = 'pool'
    grad_name = 'pool'

    def floor(self, ctx: StepContext) -> Optional[float]:
        return 0.0

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        return recon_pooled(ctx.out.pooled.float(), ctx.p_mean, ctx.p_w, ctx.net.pool_sd), {}


@register
class KoleoLoss(Loss):
    """KoLeo of cls' within each source: over the whole job's batch (the gather needs equal shares), else each process's own."""
    key = 'koleo'
    output = 'cls'
    column = 'koleo'
    grad_name = 'kol'

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        cls: torch.Tensor = ctx.out.cls.float()
        value: torch.Tensor = koleo_global(cls, ctx.source_v) if ctx.shares_equal else koleo_by_source(cls, ctx.source_v)
        return value, {}


@register
class AdversaryLoss(Loss):
    """The source adversary: the classifier's CE (its weights) + lambda x the network's confusion. Its value has no log column; the
    gradient line shows the confusion alone (w x lambda x confusion), which `__call__` keeps for `grad_part`."""
    key = 'adv'
    output = 'source-classifier'
    grad_name = 'conf'

    def __init__(self) -> None:
        self.conf: Optional[torch.Tensor] = None

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_fit: torch.Tensor
        l_conf: torch.Tensor
        l_fit, l_conf = adversary(ctx.out, ctx.source_v)
        self.conf = l_conf
        lam: float = ctx.net.adv_lambda
        return l_fit + lam * l_conf, dict(fit=float(l_fit), conf=float(l_conf), acc_cells=source_accuracy(ctx.out.src_cells, ctx.source_v),
                                          acc_cls=source_accuracy(ctx.out.src_cls, ctx.source_v), lam=lam)

    def grad_part(self, ctx: StepContext, part: torch.Tensor) -> torch.Tensor:
        return ctx.cfg.head.w_adv * ctx.net.adv_lambda * self.conf


@register
class TargetContrast(Loss):
    """Target contrast on the linear head: a view's head against the place descriptors (own = the positive; the batch's other places
    and the bank's places of its (slide, rung) cell = the negatives). `after` writes the step's descriptors to the bank."""
    key = 'nce'
    output = 'head'
    column = 'nce'
    grad_name = 'nce'

    def floor(self, ctx: StepContext) -> Optional[float]:
        return 0.0

    @staticmethod
    def _places(ctx: StepContext) -> Tuple[torch.Tensor, torch.Tensor]:
        """(the step's places' rows of the pool [P], their descriptors [P, M, c, c]: one a place, the views share it)."""
        desc: torch.Tensor = ctx.p_mean.view(len(ctx.batch.slides), ctx.views, *ctx.p_mean.shape[1:])[:, 0]
        return torch.tensor(ctx.batch.index, device=ctx.p_mean.device), desc

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        bank: Optional[TargetBank] = ctx.bank
        if bank is None:
            raise ValueError('TargetContrast: the step has no target bank (HeadConfig.w_nce is 0)')
        mu: torch.Tensor = ctx.net.pool_mu.view(1, -1, 1, 1)
        sd: torch.Tensor = ctx.net.pool_sd.view(1, -1, 1, 1)
        idx: torch.Tensor
        desc_p: torch.Tensor
        idx, desc_p = self._places(ctx)
        cells: torch.Tensor = bank.cell[idx]                                                   # [P]
        rows: torch.Tensor = bank.rows_for(idx, cells)                                         # [K]
        ok: torch.Tensor = bank.cell[rows][None, :] == cells[ctx.place_v][:, None]             # [B, K]
        value: torch.Tensor = target_nce(ctx.head_std, ctx.place_v, ctx.w_v, ((desc_p - mu) / sd).flatten(1),
                                         ((bank.desc[rows] - mu) / sd).flatten(1), ok, ctx.cfg.head.tau)
        return value, dict(nce_k=float(ok.sum(1).float().mean()))                              # the bank negatives a view, the mean

    def after(self, ctx: StepContext) -> None:
        idx: torch.Tensor
        desc_p: torch.Tensor
        idx, desc_p = self._places(ctx)
        ctx.bank.write(idx, desc_p)


@register
class ViewContrast(Loss):
    """View-to-view contrast on cls' (the place's other views = the positives; experiment arms only)."""
    key = 'vnce'
    output = 'cls'
    column = 'vnce'
    grad_name = 'vnce'

    def floor(self, ctx: StepContext) -> Optional[float]:
        return math.log(ctx.views - 1)

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        return view_nce(ctx.out.cls.float(), ctx.place_v, ctx.w_v, ctx.cfg.head.tau), {}


@register
class HeadContrast(Loss):
    """View-to-view contrast on the linear head's output (the descriptor loss's place; experiment arms only)."""
    key = 'hnce'
    output = 'head'
    column = 'hnce'
    grad_name = 'hnce'

    def floor(self, ctx: StepContext) -> Optional[float]:
        return math.log(ctx.views - 1)

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        return view_nce(ctx.head_std, ctx.place_v, ctx.w_v, ctx.cfg.head.tau), {}


@register
class DecoderTwinContrast(Loss):
    """Twin contrast on the decoder's output, a latent cell's block of it (the reconstruction's place; experiment arms only)."""
    key = 'dnce'
    output = 'cofig'
    column = 'dnce'
    grad_name = 'dnce'

    def floor(self, ctx: StepContext) -> Optional[float]:
        return math.log(len(ctx.batch.sources) - 1)

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        return twin_nce(decoder_blocks(ctx.out.cofig, ctx.net.latent_side), len(ctx.batch.sources), ctx.w_v, ctx.cfg.head.twin_cells,
                        ctx.cfg.head.tau), {}


@register
class LatentTwinContrast(Loss):
    """The same twin contrast on the latent itself (the control of `DecoderTwinContrast`; experiment arms only)."""
    key = 'lnce'
    output = 'latent'
    column = 'lnce'
    grad_name = 'lnce'

    def floor(self, ctx: StepContext) -> Optional[float]:
        return math.log(len(ctx.batch.sources) - 1)

    def __call__(self, ctx: StepContext) -> Tuple[torch.Tensor, Dict[str, float]]:
        return twin_nce(ctx.out.latent.float().flatten(1, 2), len(ctx.batch.sources), ctx.w_v, ctx.cfg.head.twin_cells,
                        ctx.cfg.head.tau), {}


@dataclass
class StepLosses:
    """What `step_losses` gives: the summed loss, each weighted term by its gradient-line name, the log columns."""
    loss: torch.Tensor
    weighted: List[Tuple[str, torch.Tensor]]
    logged: Dict[str, float]


def step_losses(ctx: StepContext) -> StepLosses:
    """Walk `LOSSES` in order: a loss with weight 0 is skipped unless it is `always`; the weighted terms are added in that order; then
    the `after` of every loss with a weight > 0."""
    loss: Optional[torch.Tensor] = None
    weighted: List[Tuple[str, torch.Tensor]] = []
    logged: Dict[str, float] = {}
    for term in LOSSES.values():
        w: float = term.weight(ctx.cfg)
        if w <= 0 and not term.always:
            continue
        value: torch.Tensor
        more: Dict[str, float]
        value, more = term(ctx)
        part: torch.Tensor = w * value
        loss = part if loss is None else loss + part
        weighted.append((term.grad_name, term.grad_part(ctx, part)))
        if term.column is not None:
            logged[term.column] = float(value)
        logged.update(more)
    for term in LOSSES.values():
        if term.weight(ctx.cfg) > 0:
            term.after(ctx)
    if loss is None:
        raise ValueError('no loss is on')
    return StepLosses(loss, weighted, logged)
