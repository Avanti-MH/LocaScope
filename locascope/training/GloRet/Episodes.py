'''A visit's steps: which items each FoV's softmax is over, and when they are chosen (`Pool.py` has the regimes and the
choosing; `Items.py` the items; `Full.py` and `Rerank.py` the steps of a bi-encoder and of a pair scorer).

A visit is a level of a slide and its FoVs. A FoV's items are every window of the level at the four rotations
(`--items level`, with the first phase's score of each as the teacher's) or the entries of the list another retriever gave
(`--items list`, in its order, the order as the teacher's score). `--top-p` says how many of them go into the softmax
(`Pool.TopP`):

    all         every item                     the steps over the visit's FoVs, a few at a time
    each step   most of them (share above 0.5)  the network scores every item at each step; the pool is its best
    episodes    few of them                    the visit's `--steps-per-visit` steps are shared out between
                                               `--episodes` episodes; at the start of each, every FoV's pool is chosen
                                               (and its memory kept): from the teacher's scores in the first
                                               `--warmup-epochs`, from the network's own after; the FoVs are the same in
                                               every episode (one visit's)

The scorer (`gloret`: a vector per window, `logloret`: one pair at a time) changes how the items are scored and stepped on,
not how they are chosen. A pair scorer cannot rank a LEVEL's windows at each step (a pair each, ~5 * 10^5 of them): over a
level it takes `episodes` only. Its pools are chosen as a bi-encoder's are: from the teacher's scores in the warmup
epochs, from its own after (every window of the level scored, once per FoV and episode).

The processes of a `--window-split` group make the same choices (same FoVs, same random numbers, the scores of the same
gathered vectors), so they take the same steps.
'''
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from locascope.training.GloRet.Full import FullCfg, Split, glo_step, rank_windows
from locascope.training.GloRet.Items import Items, items_of
from locascope.training.GloRet.Pool import Memory, TopP, make_pool
from locascope.training.GloRet.Rerank import RerankCfg, log_step, rerank_scores
from locascope.training.GloRet.Profile import profiled
from locascope.training.GloRet.Steps import StepCfg, mark_peak, mean_logs, peak_gb
from locascope.training.GloRet.Supply import Fov, Level

#: a pair scorer over a level's items without a ranking first: the most items it takes in one step (a list is 10^4)
PAIR_ALL_LIMIT: int = 20000
#: on a lazy level (tokens read when asked for) the most items a step may hold the tokens of: 12 MB a window
LAZY_ALL_LIMIT: int = 1000


@dataclass(frozen=True)
class PoolCfg:
    top: TopP
    min_candidates: int = 256     # a share is raised to this many negatives for a small level
    episodes: int = 3
    warmup_epochs: int = 2
    steps: int = 16               # the optimiser steps of a visit
    fovs_per_step: int = 8
    max_fov_uses: int = 2         # a FoV is trained on at most about this many times in a visit (0: no limit), see `fovs_each_step`


def marks_for_pool(pos: np.ndarray, ignore: np.ndarray, right: np.ndarray, step: StepCfg) -> Dict[str, np.ndarray]:
    """What a pool may choose among and what it holds whatever: `eligible` (the negatives that may be chosen), `always`
    (the answers and, with the soft target on, the near windows: the ignored ones at the right rotation)."""
    zone: np.ndarray = (ignore & right & ~pos) if step.w_geo else np.zeros_like(pos)
    return {'eligible': ~pos & ~ignore, 'always': pos | zone}


def fovs_each_step(pc: PoolCfg, live: int) -> int:
    """FoVs a step takes: `pc.fovs_per_step`, fewer when the visit has so few FoVs that the `pc.steps` steps would train on each
    one more than `pc.max_fov_uses` times. The steps stay as many (the processes take their steps together)."""
    if not pc.max_fov_uses:
        return pc.fovs_per_step
    return max(1, min(pc.fovs_per_step, -(-pc.max_fov_uses * live // max(pc.steps, 1))))


def take_in_turn(order: List[int], live: Sequence[int], n: int, rng: np.random.Generator) -> List[int]:
    """The next `n` of the FoVs `live` in a fresh order for each lap (`order` is the rest of the lap)."""
    while len(order) < n:
        order.extend(rng.permutation(np.asarray(live)).tolist())
    pick: List[int] = list(dict.fromkeys(order[:n]))
    del order[:n]
    return pick


def choose_counted(scores: np.ndarray, pos: np.ndarray, ignore: np.ndarray, right: np.ndarray, k: int, step: StepCfg,
                   memory: Optional[Memory]) -> Tuple[np.ndarray, int]:
    """One FoV's pool (sorted flat indices) and N', the negatives that beat the answer, from the scores of its items."""
    m = marks_for_pool(pos, ignore, right, step)
    return make_pool(scores, pos, m['eligible'], m['always'], k, memory)


def choose(scores: np.ndarray, pos: np.ndarray, ignore: np.ndarray, right: np.ndarray, k: int, step: StepCfg,
           memory: Optional[Memory]) -> np.ndarray:
    """One FoV's pool (sorted flat indices) from the scores of its items."""
    return choose_counted(scores, pos, ignore, right, k, step, memory)[0]


def hard_negative_stats(seen: Sequence[Tuple[int, int, int, int, float]], uses: Sequence[int]) -> Dict[str, float]:
    """What the pools of a visit say about the hard negatives, the keys `hn/...` of the visit's result. `seen` has a row for
    each pool that was chosen: `(items, K, N', pool size, share of the pool that was in the FoV's last pool or -1 for the first)`;
    `uses` the steps each FoV was trained in. N' is the negatives that beat the answer: the answer's rank is N' + 1."""
    if not seen:
        return {'hn/uses': float(np.mean(uses)) if len(uses) else 0.0}
    n, k, ahead, size, kept = (np.array(c, dtype=np.float64) for c in zip(*seen))
    ratio: np.ndarray = ahead / np.maximum(k, 1.0)
    again: np.ndarray = kept[kept >= 0]
    out: Dict[str, float] = {'hn/items': float(n.mean()), 'hn/k': float(k.mean()), 'hn/ahead': float(ahead.mean()),
                             'hn/ratio_med': float(np.median(ratio)), 'hn/ratio_p90': float(np.percentile(ratio, 90)),
                             'hn/first': float((ahead == 0).mean()), 'hn/full': float((ahead >= k).mean()),
                             'hn/padded': float(np.maximum(k - ahead, 0.0).sum() / max(k.sum(), 1.0)),
                             'hn/pool': float(size.mean()), 'hn/uses': float(np.mean(uses)) if len(uses) else 0.0}
    if len(again):
        out['hn/kept'] = float(again.mean())
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  a visit
# ══════════════════════════════════════════════════════════════════════════════

def run_visit(arch: str, source: str, net: torch.nn.Module, opt: torch.optim.Optimizer, level: Level, fovs: Sequence[Fov],
              epoch: int, pc: PoolCfg, gcfg: FullCfg, rcfg: RerankCfg, step: StepCfg, rng: np.random.Generator,
              cache: Dict[Any, Any], clip: float, sync: Optional[Callable[[torch.nn.Module], None]],
              split: Optional[Split], memories: Dict[int, Memory], limit: Optional[int] = None
              ) -> Optional[Dict[str, float]]:
    """`pc.steps` steps of a head (`arch`: `gloret` | `logloret`) on the items (`source`: `level` | `list`) of the visit's
    FoVs; the mean of each logged term, or None when no FoV has an answer among its items (the caller takes the idle
    steps). `memories`: the FoVs' hard-negative memories, by position. On a LAZY level (`Lazy.py`, stage B: tokens read when
    asked for) the tiles of the pool are read once and kept for the steps, the tiles of a ranking are read and dropped, and
    a list takes no added answer (its tokens are not there to be read)."""
    lazy: bool = bool(getattr(level, 'lazy', False))
    items: List[Optional[Items]] = [items_of(source, level, f, cache, limit, inject=not lazy) for f in fovs]
    live: List[int] = [i for i, it in enumerate(items) if it is not None]
    if not live:
        return None
    logs: List[Dict[str, float]] = []
    order: List[int] = []
    step_no: int = int(rng.integers(2 ** 31))
    pair_on_level: bool = arch == 'logloret' and source == 'level'

    def k_of(i: int) -> int:
        it = items[i]
        assert it is not None
        return pc.top.size(it.n, pc.min_candidates)

    per_step: int = fovs_each_step(pc, len(live))
    card: Any = next(net.parameters()).device
    pool_s: float = 0.0                                  # seconds and highest GPU memory (GB) of choosing the pools (scoring the items)
    pool_gb: float = 0.0
    first = items[live[0]]
    assert first is not None
    regime: str = pc.top.regime(first.n, k_of(live[0]))
    if lazy and regime == 'all' and first.n > LAZY_ALL_LIMIT:
        raise ValueError(f'a lazy level cannot hold the tokens of all {first.n} items for a step: use a --top-p of at most '
                         f'{LAZY_ALL_LIMIT} items (a count) or of 0.5 or less')
    if pair_on_level:
        if regime == 'each step':        # a count of a small level can be more than half of it: the pair scorer cannot rank the level at each step,
            regime = 'episodes'           # its pool is chosen once per episode (below), whatever the share
        if regime == 'all' and first.n > PAIR_ALL_LIMIT:
            raise ValueError(f'a pair scorer over a level of {first.n} items cannot score them all in a step: use --top-p '
                             'of 0.5 or less or --items list')

    score_s: List[float] = []                            # the seconds of each scoring of the items, to split `pool_s` into it and the rest

    def score_all(idx: Sequence[int], no: int) -> List[Optional[np.ndarray]]:
        began: float = time.perf_counter()
        with profiled(f'{arch}-{source}-pool', net):          # with `--profile`: the first pass profiled, the second timed by part
            ranked: List[Optional[np.ndarray]] = score_all_plain(idx, no)
        score_s.append(time.perf_counter() - began)
        return ranked

    def score_all_plain(idx: Sequence[int], no: int) -> List[Optional[np.ndarray]]:
        chosen = [items[i] for i in idx]
        if lazy:
            level.release()                                  # the steps' tokens go; a ranking keeps none (`streaming`)
            level.streaming = True                           # type: ignore[attr-defined]
        try:
            if arch == 'gloret':
                return rank_windows(net, level, [fovs[i] for i in idx], chosen, no, gcfg, split)
            return rerank_scores(net, level, [fovs[i] for i in idx], chosen, no, rcfg, split)
        finally:
            if lazy:
                level.streaming = False                      # type: ignore[attr-defined]

    seen_pools: List[Tuple[int, int, int, int, float]] = []
    last_pool: Dict[int, np.ndarray] = {}
    use_of: Dict[int, int] = {}

    def do_step(idx: Sequence[int], flat: Sequence[np.ndarray], no: int) -> None:
        with profiled(f'{arch}-{source}-step'):
            do_step_plain(idx, flat, no)

    def do_step_plain(idx: Sequence[int], flat: Sequence[np.ndarray], no: int) -> None:
        for i in idx:
            use_of[i] = use_of.get(i, 0) + 1
        chosen = [items[i] for i in idx]
        if arch == 'gloret':
            out = glo_step(net, opt, level, [fovs[i] for i in idx], chosen, [it.pool_of(p) for it, p in zip(chosen, flat)],
                           no, gcfg, step, clip, sync, split)                               # type: ignore[union-attr]
        else:
            out = log_step(net, opt, level, [fovs[i] for i in idx], chosen, list(flat), no, rcfg, step, clip, sync,
                           split)                                                           # type: ignore[arg-type]
        if out is not None:
            logs.append(out)

    if regime == 'all':
        everything: Dict[int, np.ndarray] = {i: np.arange(items[i].n, dtype=np.int64) for i in live}   # type: ignore[union-attr]
        for s in range(pc.steps):
            pick = take_in_turn(order, live, per_step, rng)
            do_step(pick, [everything[i] for i in pick], step_no + s)
    elif regime == 'each step':
        for s in range(pc.steps):
            pick = take_in_turn(order, live, per_step, rng)
            began: float = time.perf_counter()
            mark_peak(card)
            ranked = score_all(pick, step_no + s)
            flat: List[np.ndarray] = []
            for i, sc in zip(pick, ranked):
                it = items[i]
                assert it is not None and sc is not None
                pool_i, ahead_i = choose_counted(sc, it.pos, it.ignore, it.right, k_of(i), step, None)
                seen_pools.append((it.n, k_of(i), ahead_i, len(pool_i), -1.0))
                flat.append(pool_i)
            pool_s += time.perf_counter() - began
            pool_gb = max(pool_gb, peak_gb(card))
            do_step(pick, flat, step_no + s)
    else:
        per_episode: int = -(-pc.steps // max(pc.episodes, 1))
        done: int = 0
        seen_beat: List[float] = []
        pools_of: Dict[int, np.ndarray] = {}
        have_teacher: bool = all(items[i].teacher is not None for i in live)                # type: ignore[union-attr]
        if pair_on_level and epoch < pc.warmup_epochs and not have_teacher:
            raise ValueError('a pair scorer over a level chooses its pools in the warmup epochs from the first phase\'s scores, '
                             'and it made none')
        for e in range(max(pc.episodes, 1)):
            if lazy:
                level.release()                                  # type: ignore[attr-defined]  # the last episode's pool tokens
            began = time.perf_counter()
            mark_peak(card)
            if epoch < pc.warmup_epochs and have_teacher:
                scores: List[Optional[np.ndarray]] = [items[i].teacher if i in live else None            # type: ignore[union-attr]
                                                      for i in range(len(fovs))]
            else:
                ranked = score_all(live, step_no + 7919 * (e + 1))
                scores = [None] * len(fovs)
                for i, sc in zip(live, ranked):
                    scores[i] = sc
            for i in live:
                it = items[i]
                assert it is not None and scores[i] is not None
                mem = memories.setdefault(i, Memory())
                pools_of[i], ahead_i = choose_counted(scores[i], it.pos, it.ignore, it.right, k_of(i), step, mem)
                before = last_pool.get(i)
                seen_pools.append((it.n, k_of(i), ahead_i, len(pools_of[i]),
                                   float(np.isin(pools_of[i], before).mean()) if before is not None else -1.0))
                last_pool[i] = pools_of[i]
                seen_beat.append(float(len(mem)))
            pool_s += time.perf_counter() - began
            pool_gb = max(pool_gb, peak_gb(card))
            for _ in range(min(per_episode, pc.steps - done)):
                pick = take_in_turn(order, live, per_step, rng)
                do_step(pick, [pools_of[i] for i in pick], step_no + done)
                done += 1
        if seen_beat:
            logs.append({'beat': float(np.mean(seen_beat))})
    if not logs:
        return None
    logs.append(hard_negative_stats(seen_pools, list(use_of.values())))
    if pool_s:
        pool_log: Dict[str, float] = {'t_pool': pool_s, 'm_pool': pool_gb}          # a visit's seconds in all (not a step's), and its highest memory
        if score_s:                                          # of those, the scoring of the items (all, the first, the last) and the rest is choosing
            pool_log.update({'t_pool_score': float(sum(score_s)), 't_pool_first': score_s[0], 't_pool_last': score_s[-1]})
        logs.append(pool_log)
    result: Dict[str, float] = mean_logs(logs)
    if source == 'list':
        result['injected'] = float(np.mean([items[i].injected > 0 for i in live]))          # type: ignore[union-attr]
    return result


__all__ = ['LAZY_ALL_LIMIT', 'PAIR_ALL_LIMIT', 'PoolCfg', 'choose', 'choose_counted', 'fovs_each_step', 'hard_negative_stats', 'marks_for_pool', 'run_visit', 'take_in_turn']
