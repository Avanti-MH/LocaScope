'''What both architectures share of a training step (the optimiser step, the idle step of a process with no data, the
mean of the logs), and the ranks validation reads. A GloRet step is `Full.glo_step`, a LogLoRet step `Rerank.log_step`;
which candidates they score is `Pool.py`'s and `Episodes.py`'s.
'''
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from BenchCommon import first_place, listed_windows
from stage2_retrieval.Context import WindowBatch, top_candidates
from stage2_retrieval.RetrieverGloRet import WindowEncodings, pair_scores_gloret, pair_scores_logloret
from stage2_retrieval.StageInterface import Candidate

from training.GloRet.Supply import FovResult

KS: Tuple[int, ...] = (1, 5, 10, 30, 100, 500)


#: the highest GPU memory (GB) of the visit so far, over the phases that `mark_peak` closed (the card's own counter is reset at each)
_VISIT_PEAK: Dict[str, float] = {'gb': 0.0}


def _on_card(device: Any) -> bool:
    return torch.device(device).type == 'cuda'


def mark_peak(device: Any) -> None:
    """Begin a phase: its peak is read by `peak_gb`. The peak so far is kept for `visit_peak_gb`."""
    if _on_card(device):
        _VISIT_PEAK['gb'] = max(_VISIT_PEAK['gb'], torch.cuda.max_memory_allocated(device) / 1e9)
        torch.cuda.reset_peak_memory_stats(device)


def peak_gb(device: Any) -> float:
    """The highest GPU memory (GB) since `mark_peak`; of the whole process, so what its other threads hold counts too."""
    return torch.cuda.max_memory_allocated(device) / 1e9 if _on_card(device) else 0.0


def held_gb(device: Any) -> float:
    """The GPU memory (GB) allocated now."""
    return torch.cuda.memory_allocated(device) / 1e9 if _on_card(device) else 0.0


def begin_visit_peak(device: Any) -> None:
    _VISIT_PEAK['gb'] = 0.0
    if _on_card(device):
        torch.cuda.reset_peak_memory_stats(device)


def visit_peak_gb(device: Any) -> float:
    return max(_VISIT_PEAK['gb'], peak_gb(device))


@dataclass(frozen=True)
class StepCfg:
    sigma: float = 1.0
    w_geo: float = 0.0       # the weight of the soft targets of the near windows (Losses.py): 0, they are ignored and the loss is the one-hot cross-entropy
    near_alpha: float = 0.3  # the soft target of a near window at distance 0 (gaussian)
    soft_target: str = 'gaussian'   # Losses.SOFT_TARGETS
    w_corr: float = 0.1
    w_cycle: float = 0.1
    chunk: int = 4096        # windows (GloRet) or pairs (LogLoRet) encoded at a time
    w_recon: float = 0.0     # the decoders' reconstruction loss of the narrowing projections (GloRet.Decoded); 0, or no narrowing: none
    w_kl: float = 0.0        # and, with recon `vae`, the KL of their draws


# ══════════════════════════════════════════════════════════════════════════════
#  recall
# ══════════════════════════════════════════════════════════════════════════════

def recall_at(ranks: Sequence[float], ks: Sequence[int] = KS) -> Dict[str, float]:
    """`{'R@k': share of FoVs whose first hit is within the top k}`, `ceiling` (the share with any hit in the
    list) and `n`."""
    r: np.ndarray = np.asarray(ranks, dtype=np.float64)
    if r.size == 0:
        return {**{f'R@{k}': float('nan') for k in ks}, 'ceiling': float('nan'), 'n': 0}
    out: Dict[str, float] = {f'R@{k}': float((r <= k).mean()) for k in ks}
    out['ceiling'] = float(np.isfinite(r).mean())
    out['n'] = int(r.size)
    return out


def place_of(order_listed: Sequence[Any], res: FovResult) -> float:
    """Where the first answer stands in a list (`BenchCommon.first_place`: the benches' definition), `inf` when the
    list holds none."""
    place: Optional[int] = first_place(order_listed, res.found, res.tol_px)
    return float('inf') if place is None else float(place)


def upstream_rank(res: FovResult) -> float:
    """The rank in the first phase's own list."""
    return place_of(listed_windows(res.cs), res)


def own_rank(res: FovResult) -> float:
    """The rank in the list of the retriever that made the FoV's features, when another retriever made the list
    (`FovResult.own`); `inf` when there is no such list."""
    return place_of(listed_windows(res.own), res) if res.own is not None and len(res.own) else float('inf')


@torch.no_grad()
def gloret_rank(encodings: WindowEncodings, res: FovResult, k: int) -> float:
    """The rank when the network alone lists the level's windows (its top `k`, the first phase's separation rule)
    -- what `RetrieverGloRet.run_wsi` does."""
    maps = encodings.window_maps(res.source, len(res.cs.grids), res.level)
    frame = replace(res.cs, candidates=())
    cs = top_candidates(maps, frame, k, res.tol_px / res.tile_l0, Candidate)
    return place_of(listed_windows(cs), res) if len(cs) else float('inf')


@torch.no_grad()
def logloret_rank(net: torch.nn.Module, res: FovResult, pooling: str, keep: int, batch: int = 64) -> float:
    """The rank when the network re-orders the first `keep` windows of the first phase's list."""
    cs = replace(res.cs, candidates=res.cs.candidates[:keep])
    if not len(cs):
        return float('inf')
    device: Any = next(net.parameters()).device
    scores: np.ndarray = pair_scores_logloret(net, WindowBatch.build(cs, res.source, pooling), batch, device).cpu().numpy()
    listed = listed_windows(cs)
    return place_of([listed[i] for i in np.argsort(-scores, kind='stable')], res)


@torch.no_grad()
def gloret_rerank_rank(net: torch.nn.Module, res: FovResult, pooling: str, keep: int, batch: int = 64) -> float:
    """The rank when a bi-encoder (GloRet) re-orders the first `keep` windows of the first phase's list: by the score of
    the query against each window's vector (`@items=list`, a reranker)."""
    cs = replace(res.cs, candidates=res.cs.candidates[:keep])
    if not len(cs):
        return float('inf')
    device: Any = next(net.parameters()).device
    scores: np.ndarray = pair_scores_gloret(net, WindowBatch.build(cs, res.source, pooling), batch, device).cpu().numpy()
    listed = listed_windows(cs)
    return place_of([listed[i] for i in np.argsort(-scores, kind='stable')], res)


# ══════════════════════════════════════════════════════════════════════════════
#  the common part of a step
# ══════════════════════════════════════════════════════════════════════════════

def apply_step(net: torch.nn.Module, opt: torch.optim.Optimizer, clip: float,
               sync: Optional[Callable[[torch.nn.Module], None]]) -> None:
    """Average the gradients over the processes, clip and step (the backward is done)."""
    if sync is not None:
        sync(net)
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()


def finish_step(net: torch.nn.Module, opt: torch.optim.Optimizer, loss: torch.Tensor, clip: float,
                sync: Optional[Callable[[torch.nn.Module], None]]) -> None:
    """Backward, average over the processes, clip and step."""
    opt.zero_grad(set_to_none=True)
    loss.float().backward()
    apply_step(net, opt, clip, sync)


def net_input(x: torch.Tensor) -> torch.Tensor:
    """What a network is given of window features: on a card as they are (fp16 -- under autocast the network casts
    what it needs, and a float32 copy of tokens would be twice the size of the tokens), on the host in float32."""
    return x if x.is_cuda else x.float()


def idle_step(net: torch.nn.Module, opt: torch.optim.Optimizer, clip: float,
              sync: Callable[[torch.nn.Module], None]) -> None:
    """The step a process takes when it has no data and the others do: it adds nothing to the averaged gradient
    and takes the same optimiser step, so the models stay one model."""
    net.train()
    opt.zero_grad(set_to_none=True)
    sync(net)
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()


def mean_logs(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """The mean of each term over the rows that have it."""
    names: List[str] = list(dict.fromkeys(k for r in rows for k in r))
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in names}


__all__ = ['begin_visit_peak', 'held_gb', 'mark_peak', 'peak_gb', 'visit_peak_gb', 'KS', 'StepCfg', 'apply_step', 'finish_step', 'gloret_rank', 'gloret_rerank_rank', 'idle_step', 'logloret_rank', 'mean_logs', 'net_input',
           'own_rank', 'place_of', 'recall_at', 'upstream_rank']
