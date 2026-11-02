'''The steps of a LogLoRet head (Arch 2, a pair scorer) over a POOL of the items of a FoV (`Items.py`: the entries of the
list another retriever gave, `--raw-topk` of them, or the windows of the level).

The items are windows and rotations; the labels are `Supply.mark` (on a list: an answer that is not in it is stood for by
the window that kept its place, and when no window does, the answer is added to the items). A pool (`Pool.py`) is some of
the items, chosen as a bi-encoder's is; its softmax is over the scores of the pairs (query, window).

The score of an entry is one number of the pair (the query, the window), so the softmax over a pool can have a
GRADIENT CACHE (Gao et al. 2021, arXiv 2101.06983) as the retriever's has (`Full.py`):

    1  the pairs are scored without a graph, in chunks                      s_i
    2  the loss is made from the scores and goes back to them only          dL/ds_i
    3  every chunk is scored AGAIN, with a graph, and goes back with its part of dL/ds_i; the correspondence and cycle
       losses (on the answers' pairs) are made in this pass; one optimiser step
    `--window-split G`: each of G processes scores 1/G of a pool, the scores are gathered (a number each), every process
    makes the same loss, each goes back through its own pairs, and the parameter gradients are summed over the group.
    `--no-gradcache`: the pairs are scored once, with their graph (the activations of a pool in memory).
'''
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from locascope.cli.bench.BenchCommon import listed_windows
from locascope.core.networks.GloRet import correspondence_loss, cycle_loss, take_aux
from locascope.core.stage2_retrieval.Context import top_candidates
from locascope.core.stage2_retrieval.RetrieverGloRet import amp_context
from locascope.core.stage2_retrieval.StageInterface import Candidate

from locascope.training.GloRet.Full import Split, seeded
from locascope.training.GloRet.Items import Items, level_items
from locascope.training.GloRet.Losses import fov_loss
from locascope.training.GloRet.Profile import note
from locascope.training.GloRet.Steps import StepCfg, apply_step, held_gb, mark_peak, mean_logs, net_input, peak_gb, place_of
from locascope.training.GloRet.Supply import ROTATIONS, Cands, Fov, FovResult, Level, Marks, fov_of, level_of


@dataclass(frozen=True)
class RerankCfg:
    chunk: int = 256           # pairs scored at a time
    seed: int = 0
    gradcache: bool = True



Blocks = List[Tuple[int, np.ndarray, torch.Tensor]]      # per rotation present: (rotation, positions in `idx`, window features)


def window_blocks(level: Level, f: Fov, cands: Cands, idx: np.ndarray) -> Blocks:
    """The tile features of the windows `idx` of `cands`, a block per rotation (`f` only through its shapes), so that FoVs
    whose shapes are the same can score the same windows without gathering them again."""
    out: Blocks = []
    for rot in ROTATIONS:
        where: np.ndarray = np.flatnonzero(cands.rot[idx] == rot)
        if not len(where):
            continue
        sub: np.ndarray = idx[where]
        out.append((rot, where, level.window_features([cands.part[j] for j in sub], cands.row[sub], cands.col[sub], f.shape(rot))))
    return out


def level_blocks(level: Level, f: Fov, items: Items, idx: np.ndarray) -> Blocks:
    """`window_blocks` for items that are the windows of the level (`items.cands is None`): from the windows' ids in the
    universe of each rotation's shape, so that no candidate list (a Python object per window) is made."""
    assert items.cands is None
    rot: np.ndarray = items.rot[idx]
    wid: np.ndarray = items.window_ids()[idx]
    out: Blocks = []
    for r in ROTATIONS:
        where: np.ndarray = np.flatnonzero(rot == r)
        if not len(where):
            continue
        uni = level.universe(f.shape(r))
        part, row, col = uni.locate(wid[where])
        out.append((r, where, level.window_features_at(uni.parts, part, row, col, f.shape(r))))
    return out


def window_sides(net: torch.nn.Module, f: Fov, blocks: Blocks, device: Any) -> List[Any]:
    """What the network makes of each block's windows alone (`LogLoRetNet.window_side`), the layers' work that does not depend
    on the query: made once for a chunk, it serves every FoV that scores the chunk (`score_pairs(sides=...)`)."""
    out: List[Any] = []
    for rot, _, feats in blocks:
        with amp_context(device, True):
            out.append(net.window_side(net_input(feats), f.shape(rot)))
    return out


def score_pairs(net: torch.nn.Module, level: Level, f: Fov, cands: Optional[Cands], idx: np.ndarray, seed: int, device: Any,
                maps: Sequence[bool] = (), blocks: Optional[Blocks] = None, sides: Optional[List[Any]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """The scores of the entries `idx` of `cands` (in that order) and, for those of them flagged in `maps` (a bool per
    entry of `idx`), the match maps `(A_fw, A_wf)` `[P, h, N, N+1]` of those pairs, in order. Under the caller's
    grad mode; the chunk's own dropout seed (none is drawn without a graph in evaluation mode, so the generator is left
    alone then). `blocks`: the windows' features when they were gathered already (`window_blocks`, same `cands`, `idx`);
    `sides`: what the network makes of those windows alone (`window_sides`, one for each block), so that a window is not
    encoded again for each FoV."""
    scores: List[Tuple[np.ndarray, torch.Tensor]] = []
    a_fw: List[Tuple[np.ndarray, torch.Tensor]] = []
    a_wf: List[Tuple[np.ndarray, torch.Tensor]] = []
    flagged: np.ndarray = np.asarray(maps, dtype=bool) if len(maps) else np.zeros(len(idx), dtype=bool)
    if blocks is None:
        blocks = window_blocks(level, f, cands, idx)
    random_draws: bool = net.training or torch.is_grad_enabled()
    with seeded(device, seed) if random_draws else contextlib.nullcontext():
        for i, (rot, where, feats) in enumerate(blocks):
            shape = f.shape(rot)
            fq: torch.Tensor = f.queries[rot].reshape(1, shape[0] * shape[1], -1).float()
            with amp_context(device, True):
                if sides is None:
                    out: Dict[str, torch.Tensor] = net(fq, net_input(feats), shape, None, None, maps=bool(flagged[where].any()))
                else:                                      # the same operations as `net(...)`, the window's part made before
                    out = net.pair_side(net.query_side(fq, shape, len(where)), sides[i], shape, None, None,
                                        maps=bool(flagged[where].any()))
            scores.append((where, out['score'].float()))
            if 'A_fw' in out and flagged[where].any():
                pick: torch.Tensor = torch.as_tensor(flagged[where], device=out['score'].device)
                a_fw.append((where[flagged[where]], out['A_fw'][pick].float()))
                a_wf.append((where[flagged[where]], out['A_wf'][pick].float()))
    if len(scores) == 1 and len(scores[0][0]) == len(idx):         # one rotation: the positions are 0..n-1 already, no reordering
        ordered: torch.Tensor = scores[0][1]
    else:
        order: np.ndarray = np.concatenate([w for w, _ in scores])
        flat: torch.Tensor = torch.cat([s for _, s in scores])
        ordered = flat[torch.as_tensor(np.argsort(order, kind='stable'), device=flat.device)]
    result: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if a_fw:
        pos_order: np.ndarray = np.concatenate([w for w, _ in a_fw])
        sort: torch.Tensor = torch.as_tensor(np.argsort(pos_order, kind='stable'), device=ordered.device)
        result = (torch.cat([t for _, t in a_fw])[sort], torch.cat([t for _, t in a_wf])[sort])
    return ordered, result


@torch.no_grad()
def rerank_scores(net: torch.nn.Module, level: Level, fovs: Sequence[Fov], items: Sequence[Optional[Items]],
                  step_no: int, cfg: RerankCfg, split: Optional[Split] = None) -> List[Optional[np.ndarray]]:
    """The network's score of every item of each FoV, without a graph (evaluation mode: no dropout)."""
    device: Any = next(net.parameters()).device
    was_training: bool = net.training
    net.eval()
    out: List[Optional[np.ndarray]] = []
    # FoVs whose items are the same windows at the same rotations and shapes (every window of the level) share the gathered
    # features of a chunk: gathered once, scored by each (the same inputs to the network in the same order as one by one)
    groups: List[List[int]] = []
    for n, it in enumerate(items):
        if it is None:
            continue
        for g in groups:
            first = items[g[0]]
            assert first is not None
            if (it.cands is None and first.cands is None and it.n == first.n
                    and all(fovs[n].shape(r) == fovs[g[0]].shape(r) for r in ROTATIONS)
                    and np.array_equal(it.rot, first.rot) and np.array_equal(it.window_ids(), first.window_ids())):
                g.append(n)
                break
        else:
            groups.append([n])
    note(f'rerank_scores: {sum(len(g) for g in groups)} FoVs in {len(groups)} groups of the same windows, sizes '
         f'{[len(g) for g in groups]}; query shape per rotation of each group\'s first FoV '
         f'{[[fovs[g[0]].shape(r) for r in ROTATIONS] for g in groups]}; items per FoV {[items[g[0]].n for g in groups]}'    # type: ignore[union-attr]
         f'; chunk {cfg.chunk}')
    local: Dict[int, List[torch.Tensor]] = {n: [] for g in groups for n in g}
    for g in groups:
        lead = items[g[0]]
        assert lead is not None
        cut: List[int] = split.bounds(lead.n) if split is not None else [0, lead.n]
        me: int = split.index if split is not None else 0
        for c in range(cut[me], cut[me + 1], cfg.chunk):
            idx: np.ndarray = np.arange(c, min(c + cfg.chunk, cut[me + 1]))
            at: np.ndarray = np.arange(len(idx))
            cands: Optional[Cands] = None if lead.cands is None else lead.cands_of(idx)
            blocks: Blocks = (level_blocks(level, fovs[g[0]], lead, idx) if cands is None
                              else window_blocks(level, fovs[g[0]], cands, at))
            sides: List[Any] = window_sides(net, fovs[g[0]], blocks, device)    # the windows alone: once for all the FoVs
            for n in g:
                local[n].append(score_pairs(net, level, fovs[n], cands, at, 0, device, blocks=blocks, sides=sides)[0].float())
            del blocks, sides
    for n, (f, it) in enumerate(zip(fovs, items)):
        if it is None:
            out.append(None)
            continue
        cut = split.bounds(it.n) if split is not None else [0, it.n]
        mine: torch.Tensor = torch.cat(local[n]) if local[n] else torch.zeros(0, device=device)
        if split is not None:
            import torch.distributed as dist                          # noqa: PLC0415
            widest: int = max(b - a for a, b in zip(cut[:-1], cut[1:]))
            pad: torch.Tensor = torch.zeros(widest, device=device)
            pad[:len(mine)] = mine
            gathered: List[torch.Tensor] = [torch.empty_like(pad) for _ in range(split.size)]
            dist.all_gather(gathered, pad, group=split.group)
            mine = torch.cat([g[:b - a] for g, a, b in zip(gathered, cut[:-1], cut[1:])])
        out.append(mine.cpu().numpy())
    net.train(was_training)
    return out


@torch.no_grad()
def logloret_level_rank(net: torch.nn.Module, res: FovResult, pooling: str, chunk: int, k: int = 500) -> float:
    """The rank when the pair scorer alone lists the level: EVERY window at the four rotations scored (a pair each,
    `rerank_scores`), then the top `k` under the first phase's separation rule -- what `Steps.gloret_rank` does with a
    bi-encoder's score maps. `inf` when the level has no window of the FoV's shape or the answer is not in the top `k`."""
    level: Level = level_of(res, pooling)
    f: Fov = fov_of(res, with_list=False)
    items: Optional[Items] = level_items(level, f, {})
    if items is None:
        return float('inf')
    scores: np.ndarray = rerank_scores(net, level, [f], [items], 0, RerankCfg(chunk=chunk))[0]
    maps: Dict[Tuple[int, int, str], torch.Tensor] = {}
    at: int = 0
    for rot in ROTATIONS:                                  # the items are the universes' windows, rotation after rotation
        uni = level.universe(f.shape(rot))
        block: np.ndarray = scores[at:at + uni.total]
        at += uni.total
        for p, (region, lattice) in enumerate(uni.parts):
            maps[(rot, region, lattice)] = torch.as_tensor(block[int(uni.starts[p]):int(uni.starts[p + 1])]).reshape(-1, uni.wide[p])
    cs = top_candidates(maps, replace(res.cs, candidates=()), k, res.tol_px / res.tile_l0, Candidate)
    return place_of(listed_windows(cs), res) if len(cs) else float('inf')


def log_step(net: torch.nn.Module, opt: torch.optim.Optimizer, level: Level, fovs: Sequence[Fov],
             items: Sequence[Items], pools: Sequence[np.ndarray], step_no: int, cfg: RerankCfg, step: StepCfg,
             clip: float, sync: Optional[Callable[[torch.nn.Module], None]], split: Optional[Split] = None
             ) -> Optional[Dict[str, float]]:
    """One optimiser step of a LogLoRet head on `fovs`, each with the `pool` (sorted indices into its list) its softmax
    is over; the mean of each loss term and the seconds of each part, or None when there is no FoV. With `split` every
    process of the group calls it with the same FoVs, pools and `step_no`."""
    if not len(fovs):
        return None
    device: Any = next(net.parameters()).device
    params: List[torch.nn.Parameter] = [p for p in net.parameters() if p.requires_grad]
    timing: Dict[str, float] = {'t_scores': 0.0, 't_loss': 0.0, 't_back': 0.0}
    share: int = split.size if split is not None else 1
    me: int = split.index if split is not None else 0
    net.train()
    opt.zero_grad(set_to_none=True)
    base: int = cfg.seed + step_no * 100003
    # this process's share of each pool, as [lo, hi) into it
    cuts: List[List[int]] = [split.bounds(len(p)) if split is not None else [0, len(p)] for p in pools]
    # 1  the scores of the pairs (this process's share), with a graph or without
    mark_peak(device)
    timing['m_held'] = held_gb(device)
    t0: float = time.perf_counter()
    local: List[torch.Tensor] = []
    kept: Dict[Tuple[int, int], Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]] = {}
    for i, (f, it, pool) in enumerate(zip(fovs, items, pools)):
        lo, hi = cuts[i][me], cuts[i][me + 1]
        mine: List[torch.Tensor] = []
        for c in range(lo, hi, cfg.chunk):
            sel: np.ndarray = pool[c:min(c + cfg.chunk, hi)]
            flag: np.ndarray = it.pos[sel]
            here: Cands = it.cands_of(sel)
            local_idx: np.ndarray = np.arange(len(sel))
            if cfg.gradcache:
                with torch.no_grad():
                    s, _ = score_pairs(net, level, f, here, local_idx, base + 1000 * i + c // cfg.chunk, device)
            else:
                s, a = score_pairs(net, level, f, here, local_idx, base + 1000 * i + c // cfg.chunk, device, flag)
                kept[(i, c)] = (s, a)
            mine.append(s)
        local.append(torch.cat(mine) if mine else torch.zeros(0, device=device))
    # the scores of every share, gathered (a number each)
    scores: List[torch.Tensor] = []
    if split is not None:
        import torch.distributed as dist                              # noqa: PLC0415
        widest: int = max(max(b - a for a, b in zip(c[:-1], c[1:])) for c in cuts)
        pad: torch.Tensor = torch.zeros(len(fovs), widest, device=device)
        for i, s in enumerate(local):
            pad[i, :len(s)] = s.detach()
        gathered: List[torch.Tensor] = [torch.empty_like(pad) for _ in range(split.size)]
        dist.all_gather(gathered, pad, group=split.group)
        for i in range(len(fovs)):
            scores.append(torch.cat([gathered[r][i, :cuts[i][r + 1] - cuts[i][r]] for r in range(split.size)]))
    else:
        scores = [s.detach() for s in local]
    leaves: List[torch.Tensor] = [s.float().clone().requires_grad_(True) for s in scores]
    timing['t_scores'] = time.perf_counter() - t0
    timing['m_scores'] = peak_gb(device)
    mark_peak(device)
    # 2  the loss from the scores
    t0 = time.perf_counter()
    total: Optional[torch.Tensor] = None
    logs: List[Dict[str, float]] = []
    for i, (it, pool, leaf) in enumerate(zip(items, pools, leaves)):
        m: Marks = it.take(pool)
        terms = fov_loss(leaf, torch.as_tensor(m.pos, device=device), torch.as_tensor(m.ignore, device=device),
                         torch.as_tensor(m.right, device=device), torch.as_tensor(m.dist, device=device),
                         step.sigma, step.w_geo, step.near_alpha, step.soft_target)
        total = terms['loss'] if total is None else total + terms['loss']
        logs.append({k: float(v.detach()) for k, v in terms.items()})
    assert total is not None
    (total / len(fovs)).backward()                   # to the scores: dL/ds
    timing['t_loss'] = time.perf_counter() - t0
    mark_peak(device)
    # 3  back through the pairs, each chunk with its part of dL/ds, and the losses on the answers' pairs
    t0 = time.perf_counter()
    seen: Dict[str, List[float]] = {'corr': [], 'cycle': [], 'recon': [], 'kl': []}
    for i, (f, it, pool, leaf) in enumerate(zip(fovs, items, pools, leaves)):
        lo, hi = cuts[i][me], cuts[i][me + 1]
        grad: torch.Tensor = leaf.grad
        n_pos: int = int(it.pos[pool].sum())
        for c in range(lo, hi, cfg.chunk):
            top: int = min(c + cfg.chunk, hi)
            sel = pool[c:top]
            flag = it.pos[sel]
            if cfg.gradcache:
                s, a = score_pairs(net, level, f, it.cands_of(sel), np.arange(len(sel)), base + 1000 * i + c // cfg.chunk, device, flag)
            else:
                s, a = kept[(i, c)]
            loss: torch.Tensor = (s * grad[c:top]).sum()
            if a is not None and n_pos:
                weight: float = float(flag.sum()) / n_pos / len(fovs)
                corr, cyc = correspondence_loss(*a), cycle_loss(*a)
                loss = loss + weight * (step.w_corr * corr + step.w_cycle * cyc)
                seen['corr'].append(float(corr.detach()))
                seen['cycle'].append(float(cyc.detach()))
            recon, kl = take_aux(net)
            part: float = (top - c) / max(len(pool), 1) / len(fovs)
            if recon is not None:
                loss = loss + step.w_recon * recon * part / share
                seen['recon'].append(float(recon.detach()))
            if kl is not None:
                loss = loss + step.w_kl * kl * part / share
                seen['kl'].append(float(kl.detach()))
            loss.backward(retain_graph=not cfg.gradcache)
    if split is not None:
        import torch.distributed as dist                              # noqa: PLC0415
        from locascope.training.GloRet.Full import flat_grads, set_grads        # noqa: PLC0415
        flat: torch.Tensor = flat_grads(params)
        dist.all_reduce(flat, group=split.group)     # the shares' parts add up
        set_grads(params, flat)
    timing['t_back'] = time.perf_counter() - t0
    timing['m_back'] = peak_gb(device)
    apply_step(net, opt, clip, sync)
    extra: Dict[str, float] = {k: float(np.mean(v)) for k, v in seen.items() if v}
    return {**mean_logs(logs), **extra, 'candidates': float(sum(len(p) for p in pools) / len(pools)), **timing}


__all__ = ['RerankCfg', 'log_step', 'rerank_scores', 'score_pairs']
