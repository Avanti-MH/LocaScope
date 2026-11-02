'''The steps of a GloRet head (Arch 1, a bi-encoder) over a POOL of windows of a level, up to every window of it.

The items of a FoV (`Items.py`) are the windows of the level at the four rotations (`--items level`) or the entries of
another retriever's list (`--items list`). A pool (`Pool.py`) is some of them, handed here as the window ids of each
rotation; with `--top-p 1.0` it is all of them, as the retrieval is. The labels are `Supply.mark`: the answers at the
right rotation are the positives, the zone around them (and, by `--wrong-rotation`, their places at the wrong
rotations) is ignored, the rest are negatives.

The windows of a pool are too many to keep the activations of (a level has ~5 * 10^5 of 20 tiles: hundreds of GB). Two
ways to give the gradient of one pass over all of them, which can be used together:

    A GRADIENT CACHE (Gao et al. 2021, arXiv 2101.06983)   the default
        1  the windows are encoded without a graph, in chunks, and only their vectors are kept     K  [W, d_r]
        2  the loss is made from K (and the FoVs' queries, with their graph) and goes back to K only: dL/dK
        3  every chunk is encoded AGAIN, with a graph, and goes back with its chunk of dL/dK; the gradients of
           the parameters add up; one optimiser step
    A WINDOW SPLIT   (`--window-split G`, G processes that work on the same level)
        each process encodes its 1/G of the windows; the vectors are gathered; every process makes the same loss
        and dL/dK; each goes back through its own slice only, and the parameter gradients of the slices are
        summed over the G processes (the queries' are the same in all of them and are counted once).
        With `--no-gradcache` the slice is encoded with its graph at once (a pass less, 1/G of the activations);
        else the slice is itself a gradient cache.

Dropout draws the same mask in the two encodings of a chunk (a seed per chunk), and the queries' in every process of
a group (a seed per step). The windows of a step are encoded once and shared by all the FoVs of the step: a step costs
its windows, whatever the number of FoVs, so take many FoVs (`--fovs-per-step`) and few steps for a visit.

`Align` keeps a window's tokens (each turned to a unit vector, laid side by side: the score is one product over the
number of tiles); TilePool `attnpool` pools every tile once (`pooled_leaf`), not once in each window that holds it.
'''
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from GloRet import take_aux
from stage2_retrieval.RetrieverGloRet import amp_context

from training.GloRet.Items import Items, Pool
from training.GloRet.Losses import fov_loss
from training.GloRet.Steps import StepCfg, apply_step, held_gb, mark_peak, mean_logs, net_input, peak_gb
from training.GloRet.Supply import ROTATIONS, Fov, Level, Marks, Shape, Universe


@dataclass(frozen=True)
class FullCfg:
    chunk: int = 4096          # windows encoded at a time
    seed: int = 0              # the chunks' dropout seeds are this plus the step's and their number
    gradcache: bool = True     # a slice (all the windows, or a process's share) is encoded twice, the first time without a graph


@dataclass
class Split:
    """The processes that work on the same level: this one is number `index` of `ranks` (global ranks, in order), and
    `group` is their process group."""
    group: Any
    ranks: List[int]
    index: int
    every: Any = None          # every group that was made (all processes make all), to be destroyed at the end

    @property
    def size(self) -> int:
        return len(self.ranks)

    def bounds(self, total: int) -> List[int]:
        """The first window of each process's share of `total`, and `total`."""
        return [(total * r) // self.size for r in range(self.size + 1)]


def make_split(world: int, rank: int, size: int) -> Optional[Split]:
    """The groups of `size` consecutive ranks of `world`; this process's, or None when `size` is 1. Every process must
    call it (making a group is a collective)."""
    if size <= 1:
        return None
    if world % size:
        raise ValueError(f'--window-split {size}: the {world} processes do not make groups of {size}')
    import torch.distributed as dist                                  # noqa: PLC0415
    mine: Optional[Split] = None
    made: List[Any] = []
    for g in range(world // size):
        ranks: List[int] = list(range(g * size, (g + 1) * size))
        group = dist.new_group(ranks)
        made.append(group)
        if rank in ranks:
            mine = Split(group=group, ranks=ranks, index=rank - g * size)
    if mine is not None:
        mine.every = made
    return mine


# ══════════════════════════════════════════════════════════════════════════════
#  encoding
# ══════════════════════════════════════════════════════════════════════════════

def seeded(device: Any, seed: int) -> Any:
    """A context in which the random numbers start from `seed` and the generator is put back after."""
    on_card: bool = torch.device(device).type == 'cuda'
    ctx = torch.random.fork_rng(devices=[torch.device(device).index or 0] if on_card else [])

    class _Seeded:
        def __enter__(self) -> None:
            ctx.__enter__()
            torch.manual_seed(seed)

        def __exit__(self, *exc: Any) -> Any:
            return ctx.__exit__(*exc)

    return _Seeded()


def flat_vectors(net: torch.nn.Module, encoded: torch.Tensor) -> torch.Tensor:
    """What a window (or the query) is compared by, `[B, width]` in float32: its vector; for `Align` its tokens, each
    turned to a unit vector and laid side by side (the score is then one product, over the number of tiles)."""
    if getattr(net, 'aligned', False):
        return torch.nn.functional.normalize(encoded.float(), dim=-1).reshape(encoded.shape[0], -1)
    return encoded.float()


def encode_ids(net: torch.nn.Module, uni: Universe, ids: np.ndarray, seed: int, device: Any) -> torch.Tensor:
    """The vectors of the windows `ids` of the universe, under the chunk's own dropout seed."""
    with seeded(device, seed):
        feats: torch.Tensor = uni.features(ids)
        with amp_context(device, True):
            return flat_vectors(net, net.encode(net_input(feats), uni.shape, 'window'))


def pooled_leaf(net: torch.nn.Module, level: Level, chunk: int, leaf: bool = True) -> Tuple[Level, Callable[..., None]]:
    """For TilePool `attnpool`: the level with every tile pooled by the network's attention pool once (`[R, C, P * F]`
    patch tokens to `[R, C, F]`) and, with `leaf`, the grids as leaves that will get the gradient of the windows made from
    them, and the function that takes that gradient back through the pool (a tile's tokens pooled again, with a graph, a
    chunk at a time; the gradient of the leaves is first summed over the processes of a group). Without this a tile
    would be pooled in each of the ~20 windows that hold it. Any other pooling: the level itself."""
    cfg = net.cfg
    if cfg.tile_pool != 'attnpool' or getattr(level, 'lazy', False):      # a lazy level has no grid to pool: its windows are pooled as they come
        return level, lambda group=None: None
    device: Any = next(net.parameters()).device
    raw: Dict[Any, torch.Tensor] = {}
    leaves: Dict[Any, torch.Tensor] = {}
    with torch.no_grad():
        for part, g in level.grids.items():
            rows, cols, width = (int(s) for s in g.shape)
            raw[part] = g.reshape(rows * cols, width)
            pooled: List[torch.Tensor] = []
            for s in range(0, rows * cols, chunk):
                tokens: torch.Tensor = raw[part][s:s + chunk].to(device).reshape(-1, width // cfg.in_dim, cfg.in_dim).float()
                pooled.append(net.embed.pooled(tokens))
            leaves[part] = torch.cat(pooled).reshape(rows, cols, cfg.in_dim).requires_grad_(leaf)

    def back(group: Any = None) -> None:
        for part, grid in leaves.items():
            if group is not None:                     # every process of the group holds a part of the windows' gradient
                import torch.distributed as dist                          # noqa: PLC0415
                if grid.grad is None:
                    grid.grad = torch.zeros_like(grid)
                dist.all_reduce(grid.grad, group=group)
            if grid.grad is None:
                continue
            grad: torch.Tensor = grid.grad.reshape(-1, cfg.in_dim)
            for s in range(0, grad.shape[0], chunk):
                tokens = raw[part][s:s + chunk].to(device).reshape(-1, raw[part].shape[-1] // cfg.in_dim, cfg.in_dim).float()
                (net.embed.pooled(tokens) * grad[s:s + chunk]).sum().backward()

    return Level(slide=level.slide, level=level.level, grids=leaves, xs=level.xs, ys=level.ys,
                 tile_l0=level.tile_l0), back


def flat_grads(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])


def set_grads(params: Sequence[torch.nn.Parameter], flat: torch.Tensor) -> None:
    at: int = 0
    for p in params:
        p.grad = flat[at:at + p.numel()].reshape(p.shape).to(p.dtype).clone()
        at += p.numel()


def check_same_fovs(split: Optional[Split], fovs: Sequence[Fov], level: Level, device: Any) -> None:
    """The processes of a group must hold the same FoVs: else they would wait for each other."""
    if split is None:
        return
    import torch.distributed as dist                                  # noqa: PLC0415
    lo_n, hi_n = (torch.tensor([float(len(fovs))], device=device) for _ in range(2))
    dist.all_reduce(lo_n, op=dist.ReduceOp.MIN, group=split.group)
    dist.all_reduce(hi_n, op=dist.ReduceOp.MAX, group=split.group)
    if float(lo_n) != float(hi_n):
        raise RuntimeError(f'--window-split: the processes of a group hold {int(lo_n)}..{int(hi_n)} FoVs of this '
                           f'level, not the same ({level.slide} L{level.level})')


def shape_members(fovs: Sequence[Fov]) -> Dict[Shape, List[Tuple[int, int]]]:
    """shape -> [(fov index, rotation)]: the queries that share a shape."""
    out: Dict[Shape, List[Tuple[int, int]]] = {}
    for i, f in enumerate(fovs):
        for rot in ROTATIONS:
            out.setdefault(f.shape(rot), []).append((i, rot))
    return out


def window_ids(work: Level, shape: Shape, members: Sequence[Tuple[int, int]], pools: Optional[Sequence[Pool]]) -> np.ndarray:
    """The ids of the windows of `shape` a step needs: all of them, or those of the pools of its queries."""
    total: int = work.universe(shape).total
    if pools is None:
        return np.arange(total, dtype=np.int64)
    ids: List[np.ndarray] = [pools[i][rot] for i, rot in members if len(pools[i][rot])]
    return np.unique(np.concatenate(ids)) if ids else np.zeros(0, dtype=np.int64)


def encode_keys(net: torch.nn.Module, work: Level, shape: Shape, ids: np.ndarray, base: int, cfg: FullCfg,
                split: Optional[Split], device: Any, graph: bool) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """The vectors of the windows `ids` of `shape`: `(all of them [len(ids), width], this process's share of them with
    its graph when `graph` and no cache, the share's `[lo, hi)` in `ids`)`. With a window split each process encodes
    a part of the ids and the parts are gathered."""
    uni: Universe = work.universe(shape)
    me: int = split.index if split is not None else 0
    cut: List[int] = split.bounds(len(ids)) if split is not None else [0, len(ids)]
    lo, hi = cut[me], cut[me + 1]
    parts: List[torch.Tensor] = []
    for c in range(lo, hi, cfg.chunk):
        top: int = min(c + cfg.chunk, hi)
        if graph and not cfg.gradcache:
            parts.append(encode_ids(net, uni, ids[c:top], base + c // cfg.chunk, device))
        else:
            with torch.no_grad():
                parts.append(encode_ids(net, uni, ids[c:top], base + c // cfg.chunk, device))
    width: int = int(net.cfg.d) * shape[0] * shape[1] if getattr(net, 'aligned', False) else int(net.cfg.d_r)
    local: torch.Tensor = torch.cat(parts) if parts else torch.zeros(0, width, device=device)
    if split is None:
        return local.detach(), local, (lo, hi)
    import torch.distributed as dist                                  # noqa: PLC0415
    widest: int = max(b - a for a, b in zip(cut[:-1], cut[1:]))
    pad: torch.Tensor = torch.zeros(widest, width, device=device)
    pad[:hi - lo] = local.detach()
    gathered: List[torch.Tensor] = [torch.empty_like(pad) for _ in range(split.size)]
    dist.all_gather(gathered, pad, group=split.group)
    return torch.cat([g[:b - a] for g, a, b in zip(gathered, cut[:-1], cut[1:])]), local, (lo, hi)


def encode_queries(net: torch.nn.Module, fovs: Sequence[Fov], shapes: Dict[Shape, List[Tuple[int, int]]], base: int,
                   split: Optional[Split], device: Any) -> Dict[Tuple[int, int], torch.Tensor]:
    """The vector of each (FoV, rotation) query, with its graph (the caller decides about no_grad)."""
    queries: Dict[Tuple[int, int], torch.Tensor] = {}
    for shape, members in shapes.items():
        stack: torch.Tensor = torch.stack([fovs[i].queries[rot].reshape(-1, fovs[i].queries[rot].shape[-1])
                                           for i, rot in members]).float().to(device)
        if split is not None:                        # the group's queries are one: its first process's
            import torch.distributed as dist                              # noqa: PLC0415
            dist.broadcast(stack, src=split.ranks[0], group=split.group)
        with seeded(device, base + 99991), amp_context(device, True):
            q: torch.Tensor = flat_vectors(net, net.encode(stack, shape, 'query'))
        for n, member in enumerate(members):
            queries[member] = q[n]
    return queries


def per_tile_scale(net: torch.nn.Module, shape: Shape) -> float:
    """`Align`: the mean over the tiles of the cosine; else 1."""
    return 1.0 / (shape[0] * shape[1]) if getattr(net, 'aligned', False) else 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  the ranking of every window by the network as it is
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def rank_windows(net: torch.nn.Module, level: Level, fovs: Sequence[Fov], items: Sequence[Optional[Items]],
                 step_no: int, cfg: FullCfg, split: Optional[Split] = None) -> List[Optional[np.ndarray]]:
    """The network's score of every item of each FoV (as `Items`), without a graph; None for a FoV without items. The
    windows of all the FoVs' items are encoded once. In evaluation mode: no dropout."""
    device: Any = next(net.parameters()).device
    was_training: bool = net.training
    net.eval()
    work, _ = pooled_leaf(net, level, cfg.chunk, leaf=False)
    live: List[Tuple[int, Fov, Items]] = [(i, f, it) for i, (f, it) in enumerate(zip(fovs, items)) if it is not None]
    out: List[Optional[np.ndarray]] = [None] * len(fovs)
    if live:
        shapes = shape_members([f for _, f, _ in live])
        base: int = cfg.seed + step_no * 100003
        queries = encode_queries(net, [f for _, f, _ in live], shapes, base, split, device)
        pools: List[Pool] = [it.everything() for _, _, it in live]
        keys: Dict[Shape, torch.Tensor] = {}
        ids: Dict[Shape, np.ndarray] = {}
        for shape, members in shapes.items():
            found: np.ndarray = window_ids(work, shape, members, pools)
            if len(found):
                ids[shape] = found
                keys[shape], _, _ = encode_keys(net, work, shape, found, base, cfg, split, device, graph=False)
        for n, (i, f, it) in enumerate(live):
            scores: np.ndarray = np.full(it.n, -np.inf, dtype=np.float32)
            wid: np.ndarray = it.window_ids()
            for rot in ROTATIONS:
                shape = f.shape(rot)
                at: np.ndarray = np.flatnonzero((it.rot == rot) & (wid >= 0))
                if shape not in keys or not len(at):
                    continue
                rows: torch.Tensor = torch.as_tensor(np.searchsorted(ids[shape], wid[at]), device=device)
                scale: float = per_tile_scale(net, shape)
                scores[at] = ((keys[shape][rows] @ queries[(n, rot)]) * scale / net.tau).float().cpu().numpy()
            out[i] = scores
    net.train(was_training)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  a step
# ══════════════════════════════════════════════════════════════════════════════

def glo_step(net: torch.nn.Module, opt: torch.optim.Optimizer, level: Level, fovs: Sequence[Fov],
             items: Sequence[Items], pools: Optional[Sequence[Pool]], step_no: int, cfg: FullCfg, step: StepCfg,
             clip: float, sync: Optional[Callable[[torch.nn.Module], None]], split: Optional[Split] = None
             ) -> Optional[Dict[str, float]]:
    """One optimiser step of a GloRet head on `fovs`, each with its `items` and the `pool` of them its softmax is over
    (None: every item, as the retrieval is); the mean of each loss term and the seconds of each part, or None when no FoV has an answer
    window. With `split` every process of the group calls it with the same FoVs, pools and `step_no`."""
    device: Any = next(net.parameters()).device
    params: List[torch.nn.Parameter] = [p for p in net.parameters() if p.requires_grad]
    work, pool_back = pooled_leaf(net, level, cfg.chunk)     # attnpool: the tiles pooled once; else the level itself
    timing: Dict[str, float] = {'t_keys': 0.0, 't_loss': 0.0, 't_back': 0.0}
    check_same_fovs(split, fovs, level, device)
    share: int = split.size if split is not None else 1
    net.train()
    opt.zero_grad(set_to_none=True)
    shapes = shape_members(fovs)
    if pools is None:
        pools = [it.everything() for it in items]
    base: int = cfg.seed + step_no * 100003
    # 1  the vectors of the windows of each shape (this process's share), with a graph or without
    mark_peak(device)
    timing['m_held'] = held_gb(device)
    t0: float = time.perf_counter()
    ids: Dict[Shape, np.ndarray] = {}
    mine: Dict[Shape, Tuple[int, int]] = {}                            # shape -> the slice [lo, hi) of its ids this process encodes
    local: Dict[Shape, torch.Tensor] = {}
    keys: Dict[Shape, torch.Tensor] = {}
    for shape, members in shapes.items():
        found: np.ndarray = window_ids(work, shape, members, pools)
        if not len(found):
            continue
        ids[shape] = found
        full, local[shape], mine[shape] = encode_keys(net, work, shape, found, base, cfg, split, device, graph=True)
        keys[shape] = full.detach().requires_grad_(True)
    recon_w, kl_w = take_aux(net)                    # the window path's decoder losses (with a graph only without a cache)
    timing['t_keys'] = time.perf_counter() - t0
    timing['m_keys'] = peak_gb(device)
    mark_peak(device)
    # 2  the loss from the vectors and the queries (these with their graph)
    t0 = time.perf_counter()
    queries = encode_queries(net, fovs, {s: m for s, m in shapes.items() if s in keys}, base, split, device)
    total: Optional[torch.Tensor] = None
    logs: List[Dict[str, float]] = []
    for i, f in enumerate(fovs):
        logit: List[torch.Tensor] = []
        rows: Dict[str, List[np.ndarray]] = {'pos': [], 'ignore': [], 'right': [], 'dist': []}
        for rot in ROTATIONS:
            shape = f.shape(rot)
            mk: Optional[Marks] = items[i].dense_marks(rot) if shape in keys else None
            if mk is None:
                continue
            which: np.ndarray = pools[i][rot]
            if not len(which):
                continue
            at: np.ndarray = np.searchsorted(ids[shape], which)
            logit.append((keys[shape][torch.as_tensor(at, device=device)] @ queries[(i, rot)])
                         * per_tile_scale(net, shape) / net.tau)
            for name in rows:
                rows[name].append(getattr(mk, name)[which])
        if not logit:
            continue
        cat = {name: torch.as_tensor(np.concatenate(v), device=device) for name, v in rows.items()}
        if not bool(cat['pos'].any()):
            continue
        terms: Dict[str, torch.Tensor] = fov_loss(torch.cat(logit), cat['pos'], cat['ignore'], cat['right'], cat['dist'],
                                                  step.sigma, step.w_geo, step.near_alpha, step.soft_target)
        total = terms['loss'] if total is None else total + terms['loss']
        logs.append({k: float(v.detach()) for k, v in terms.items()})
    recon_q, kl_q = take_aux(net)                    # the queries' decoder losses (made in this pass, with a graph)
    seen_recon: List[float] = [float(recon_q.detach())] if recon_q is not None else []
    seen_kl: List[float] = [float(kl_q.detach())] if kl_q is not None else []
    if total is None:
        return None
    total = total / len(logs)
    if recon_q is not None:
        total = total + step.w_recon * recon_q
    if kl_q is not None:
        total = total + step.w_kl * kl_q
    total.backward()                                 # to the keys (dL/dK) and the queries' graph
    timing['t_loss'] = time.perf_counter() - t0
    mark_peak(device)
    # 3  back through the windows, each share or chunk with its part of dL/dK
    t0 = time.perf_counter()
    queries_grad: Optional[torch.Tensor] = None
    if split is not None:                            # the queries' part is the same in every process: keep it apart
        queries_grad = flat_grads(params)
        for p in params:
            p.grad = None
    for shape, (lo, hi) in mine.items():
        uni: Universe = work.universe(shape)
        grad: Optional[torch.Tensor] = keys[shape].grad
        assert grad is not None
        if cfg.gradcache:
            for c in range(lo, hi, cfg.chunk):
                top: int = min(c + cfg.chunk, hi)
                out: torch.Tensor = encode_ids(net, uni, ids[shape][c:top], base + c // cfg.chunk, device)
                recon, kl = take_aux(net)
                if recon is not None:
                    seen_recon.append(float(recon.detach()))
                if kl is not None:
                    seen_kl.append(float(kl.detach()))
                loss: torch.Tensor = (out * grad[c:top]).sum()
                if recon is not None:
                    loss = loss + step.w_recon * recon / share
                if kl is not None:
                    loss = loss + step.w_kl * kl / share
                loss.backward()
        elif hi > lo:
            (local[shape] * grad[lo:hi]).sum().backward(retain_graph=True)
    if not cfg.gradcache:
        aux: Optional[torch.Tensor] = None
        if recon_w is not None:
            aux = step.w_recon * recon_w / share
        if kl_w is not None:
            aux = step.w_kl * kl_w / share if aux is None else aux + step.w_kl * kl_w / share
        if aux is not None:
            aux.backward()
    if split is not None:
        import torch.distributed as dist                                  # noqa: PLC0415
        flat: torch.Tensor = flat_grads(params)
        dist.all_reduce(flat, group=split.group)     # the shares' parts add up
        assert queries_grad is not None
        set_grads(params, flat + queries_grad)
    pool_back(split.group if split is not None else None)   # attnpool: the windows' gradient through the pool, tile by tile
    timing['t_back'] = time.perf_counter() - t0
    timing['m_back'] = peak_gb(device)
    apply_step(net, opt, clip, sync)
    n_windows: int = sum(len(v) for v in ids.values())
    aux_logs: Dict[str, float] = {}                  # the decoders' losses, as logged (their weights are in the loss)
    for name, vals in (('recon', seen_recon), ('kl', seen_kl)):
        if vals:
            aux_logs[name] = float(np.mean(vals))
    return {**mean_logs(logs), **aux_logs, 'tau': float(net.tau.detach()), 'windows': float(n_windows), **timing}


__all__ = ['FullCfg', 'Items', 'Pool', 'Split', 'encode_ids', 'glo_step', 'make_split', 'rank_windows']
