'''Training and scoring steps of the three heads that rank a candidate list
(`tilegt`, `tilegt_ctx`, `corrnet`). One FoV's subset is one list; a step is
several lists (`FovData`), their losses averaged. The embedding (`winembed`) ranks a
whole pool of windows instead and is in `Embed.py`.'''
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from RerankChoices import build_rerank
from stage2_retrieval.RerankerNeural import N_GLOBAL, N_PAIR, N_SCALAR, NeuralInputs

from training.RerankHead.Losses import rerank_loss
from training.RerankHead.Metrics import first_hit_rank
from training.RerankHead.Visits import FovData, FovResult, full_inputs

NEURAL_ARCHS: Tuple[str, ...] = ('tilegt', 'tilegt_ctx', 'corrnet')


def build_head(arch: str, in_dim: int, n_tiles: int, dim: int, layers: int, attn_heads: int,
               dropout: float, device: Any) -> nn.Module:
    """A fresh head for features of width `in_dim` and windows of `n_tiles` tiles."""
    trunk: Dict[str, Any] = dict(in_dim=in_dim, n_tiles=n_tiles, proj=dim, dim=dim, layers=layers,
                                 heads=attn_heads, ff_mult=4, dropout=dropout, max_side=8,
                                 n_scalar=N_SCALAR, n_global=N_GLOBAL)
    if arch == 'tilegt':
        return build_rerank('tilegt', trunk).to(device)
    if arch == 'tilegt_ctx':
        return build_rerank('tilegt_ctx', {'trunk': trunk, 'context': dict(
            dim=dim, layers=2, heads=attn_heads, dropout=dropout, n_pair=N_PAIR)}).to(device)
    if arch == 'corrnet':
        return build_rerank('corrnet', dict(n_tiles=n_tiles, width=64, dropout=dropout)).to(device)
    raise KeyError(f'no ranking head {arch!r}; they are {NEURAL_ARCHS}')


def forward_list(arch: str, model: nn.Module, ni: NeuralInputs, pair: Optional[torch.Tensor]
                 ) -> Dict[str, torch.Tensor]:
    """The network on one list (all of `ni`)."""
    if arch == 'tilegt':
        return model(ni.q, ni.w, ni.pos, ni.scalars, ni.corr, ni.glob)
    if arch == 'tilegt_ctx':
        if pair is None:
            raise ValueError('tilegt_ctx needs the pair features of its candidates')
        return model(ni.q, ni.w, ni.pos, ni.scalars, ni.corr, ni.glob, pair)
    logit: torch.Tensor = torch.zeros(len(ni), device=ni.q.device, dtype=ni.corr.dtype)
    for shape in sorted(set(ni.shapes)):
        idx: torch.Tensor = torch.tensor([i for i, s in enumerate(ni.shapes) if s == shape], device=ni.q.device)
        logit = logit.index_put((idx,), model(ni.corr[idx], shape[0], shape[1]).to(logit.dtype))
    return {'logit': logit}


def train_step(arch: str, model: nn.Module, opt: torch.optim.Optimizer, lists: Sequence[FovData],
               weights: Dict[str, float], temperature: float, k: int, margin: float, pos_weight: float,
               clip: float) -> Dict[str, float]:
    """One optimiser step on `lists`; the mean of each loss term over them."""
    model.train()
    opt.zero_grad(set_to_none=True)
    sums: Dict[str, float] = {}
    for d in lists:
        with torch.autocast(device_type=d.ni.q.device.type, dtype=torch.bfloat16):
            out: Dict[str, torch.Tensor] = forward_list(arch, model, d.ni, d.pair)
        out = {key: v.float() for key, v in out.items()}
        terms: Dict[str, torch.Tensor] = rerank_loss(
            out, d.pos, d.neg, d.covers, d.rot_ok, d.loc, d.loc_mask, weights, temperature, k, margin, pos_weight)
        (terms['total'] / len(lists)).backward()
        for name, t in terms.items():
            sums[name] = sums.get(name, 0.0) + float(t.detach()) / len(lists)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return sums


@torch.no_grad()
def score_all(arch: str, model: nn.Module, res: FovResult, pooling: str, chunk: int = 256
              ) -> np.ndarray:
    """The head's logit for every candidate of `res`, `[K]` numpy."""
    model.eval()
    ni, pair = full_inputs(res, pooling)
    with torch.autocast(device_type=ni.q.device.type, dtype=torch.bfloat16):
        if arch == 'tilegt':
            parts: List[torch.Tensor] = []
            for s in range(0, len(ni), chunk):
                sl: slice = slice(s, s + chunk)
                parts.append(model(ni.q[sl], ni.w[sl], ni.pos[sl], ni.scalars[sl], ni.corr[sl], ni.glob[sl])['logit'])
            logit: torch.Tensor = torch.cat(parts)
        else:
            logit = forward_list(arch, model, ni, pair)['logit']
    return logit.float().cpu().numpy()


def rank_of(arch: str, model: nn.Module, res: FovResult, pooling: str) -> float:
    return first_hit_rank(score_all(arch, model, res, pooling), res.labels.hit)


def first_phase_rank(res: FovResult) -> float:
    """The same figure for the order the retriever gave (the baseline)."""
    return first_hit_rank(-np.arange(len(res.cs), dtype=np.float64), res.labels.hit)
