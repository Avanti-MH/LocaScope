#!/usr/bin/env python3
'''Train the second phase's heads on synthetic FoVs, made as they are used.

    python -m locascope.training.RerankHead.train --heads tilegt corrnet winembed \\
        --encoder uni2 --pooling cls_avg --k 100 --epochs 30 --tag run1

Several heads are trained in ONE process on the SAME data. Encoding a level of a
slide is the expensive step and nothing is kept on disk (the design,
`stage2_retrieval/spec.md`), so a head trained alone would pay for every level
again; trained together, a level is encoded once for all of them.

THE DATA (`Visits.py`): training slides are `--train-datasets` (`@pure-train`), which shares no slide
with the validation and test sets (`@pure-half-val`, `@pure-half-test`). A source dataset's own id
(`ki67_pure`, `bracs/train`) is the whole of it, validation and test slides included. For each slide the
retriever (`RetrieverSlidingWinSimRot` of `--encoder` read through `--pooling`) is built, FoV positions are drawn over every level with a
seed of their own (`--fov-seed`), and a LEVEL AT A TIME `--fov-per-visit` of them
are photographed and put through the retriever. A visit's FoVs are trained on
`--steps-per-visit` times, `--fovs-per-step` at once, and dropped.

THE HEADS:
    tilegt, tilegt_ctx, corrnet   re-rank the retriever's candidates (K of
                                  them); the loss is over lists (Losses.py)
    winembed                      one learned vector per window, over every
                                  window of the level (Embed.py)

VALIDATION: the held-out slides' FoVs (`--val-datasets`: the set the benches use to
choose, `@pure-half-val`), every candidate scored, and the share of FoVs whose first hit is in
the top k (`Metrics.py`), for each head AND for the order the retriever gave, so
the gain is read off the same FoVs. The checkpoint is the epoch that scores best
at `--select-k`. Weights: `RerankHead/weights/<data>--<encoder>--<pooling>--<arch>[--<tag>]--
<best|last>.pt` under the results root (`weights_path`).

NOT YET: crash-resume (a killed job restarts from its `_last` weights only with
`--init-from last`; the optimiser is not restored).
'''
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
from locascope.paths import LOG_DIR, RESULT_DIR   # noqa: E402

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from locascope.core.slides.AccessDatasets import by_source, locate                          # noqa: E402
from locascope.host.cpu import CpuBudget                                        # noqa: E402
from locascope.cli.bench.BenchCommon import require_gpu_if_allocated                      # noqa: E402
from locascope.core.query_sim.FovSupply import add_fov_args, fov_from_args                     # noqa: E402
from locascope.core.networks.RerankChoices import build_rerank                                # noqa: E402
from locascope.core.networks.Checkpoints import data_tag, save_rerank_checkpoint, weights_name   # noqa: E402
from locascope.core.backbones.TileEncoder import encoder_config                           # noqa: E402
from locascope.core.stage2_retrieval.RetrieverSlidingWinSimRot import (RetrieverSlidingWinSimRot,   # noqa: E402
                                                        RetrieverSlidingWinSimRotConfig)

from locascope.training.RerankHead.Embed import EmbedVisit, embed_loss, embed_rank, fov_of, visit_of   # noqa: E402
from locascope.training.RerankHead.Labels import LabelConfig                    # noqa: E402
from locascope.training.RerankHead.Metrics import KS, recall_at                 # noqa: E402
from locascope.training.RerankHead.Runtime import (all_sum, broadcast_parameters, dist_done, dist_init,   # noqa: E402
                                          gather_objects, plain_args, sync_gradients, wandb_finish,
                                          wandb_init, wandb_log)
from locascope.training.RerankHead.Usage import UsageMeter, advice, usage_lines   # noqa: E402
from locascope.training.RerankHead.Steps import (NEURAL_ARCHS, build_head, first_phase_rank,   # noqa: E402
                                        idle_step, rank_of, train_step)
from locascope.training.RerankHead.Visits import (FovData, FovResult, MemoryFeatureStore, fov_stream,   # noqa: E402
                                        prepare_fov, take_timers, tick)

ARCHS: Tuple[str, ...] = (*NEURAL_ARCHS, 'winembed')

#: this process's rank, and the console (Slurm's log) when its own output goes to a file
RANK: int = 0
CONSOLE: Optional[Any] = None


def say(*parts: Any, console: bool = False, **kwargs: Any) -> None:
    """`print` to this process's output; with several processes that is its own file (`split_output`), and a line
    meant for the Slurm log too (`console`) is copied there by rank 0 only."""
    print(*parts, **kwargs)
    if console and RANK == 0 and CONSOLE is not None:
        print(*parts, file=CONSOLE, **kwargs)


def host_memory() -> str:
    """This process's resident memory and the node's shared-memory filesystem (what DataLoader workers pass tiles
    through, and where a `bus error` comes from when it is full), for the log line of a visit."""
    try:
        with open('/proc/self/status') as f:
            rss: float = next(int(line.split()[1]) for line in f if line.startswith('VmRSS:')) / 1e6
        shm = os.statvfs('/dev/shm')
        used: float = (shm.f_blocks - shm.f_bfree) * shm.f_frsize / 1e9
        return f'host RSS {rss:.1f} GB, node /dev/shm {used:.1f} of {shm.f_blocks * shm.f_frsize / 1e9:.0f} GB'
    except (OSError, StopIteration, ValueError):
        return 'host memory unknown'


# ── NOT ACTIVE (comments only) -- the fixes to try once job 592907's retry says which memory it is ──────────────────
# Job 592907 died at its 17th visit when rank 0 could not allocate 34 MB (numpy, in a photo thread) and a DataLoader
# worker said `bus error ... shm`, with MaxRSS only ~360 GB of 1600 GB a node. Three candidates, what shows each, and
# the change for it. Take off the `# ` when the log (host RSS and /dev/shm on every visit line) points at one.
#
# (1) /dev/shm full -- Job 596314 showed it: bus errors / "No space left on device" in 7 of 16 processes, though the
#     visit line's /dev/shm (sampled once a visit) read 11-52 GB; the peaks are while a level is being read. /dev/shm
#     is a node's own, so the first change is fewer processes on a node (4 nodes of 4 GPUs, 11 workers each: 44
#     readers a node, not 88); `--read-workers N` is there for fewer workers, and the 10-minute resource lines now
#     give the peak. If it is not enough: `--prefetch 1`, and blocks that go through a pipe instead (numpy in
#     `SlideReader`'s blocks, not tensors: no /dev/shm).
#
# (2) the job step has less memory than the job -- RSS and /dev/shm both small when it dies. In RerankHead.sh the
#     step is `srun --ntasks=$NODES ...`; give it the node's memory:
#         # srun --mem=0 --ntasks=$NODES ...
#
# (3) too many memory maps (vm.max_map_count): allocations fail with free RAM. Log the count in `host_memory`:
#         # with open('/proc/self/maps') as f:
#         #     maps = sum(1 for _ in f)
#         # with open('/proc/sys/vm/max_map_count') as f:
#         #     limit = int(f.read())
#         # ... f', maps {maps} of {limit}'
#     and, if it is that, fewer threads per process: `OMP_NUM_THREADS=1` is already set; MALLOC_ARENA_MAX=2 in
#     RerankHead.sh (glibc makes one arena, with its own maps, per thread).
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def split_output(job: str, rank: int) -> None:
    """With several processes, each writes its own log, `<log>/<job>.r<rank>`, and rank 0 keeps Slurm's log for
    the lines that say `console` (the settings, an epoch, the validation, the best epoch): the pieces are read
    one by one, or put together after (`RerankHead.sh`). Rank 0's errors stay on the console; the others' go to
    their files (torchrun also reports which process failed)."""
    global CONSOLE
    path: str = os.path.join(LOG_DIR, f'{job}.r{rank}')
    CONSOLE = os.fdopen(os.dup(1), 'w', buffering=1)
    own = open(path, 'w', buffering=1)
    os.dup2(own.fileno(), 1)
    if rank > 0:
        os.dup2(own.fileno(), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  arguments
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--heads', nargs='+', choices=ARCHS, default=['tilegt'])
    ap.add_argument('--encoder', default='uni2', choices=('uni2', 'gigapath'),
                    help='the encoder of the sliding-window retriever whose list the heads rerank, and of the '
                         'features they read')
    ap.add_argument('--pooling', default='cls_avg',
                    help='the pooling the encoder is read through (the heads read the same)')
    ap.add_argument('--k', type=int, default=500,
                    help='the candidates a list is trained on: the first K of the retriever\'s list')
    ap.add_argument('--train-datasets', nargs='+', default=['@pure-train'])
    ap.add_argument('--val-datasets', nargs='+', default=['@pure-half-val'])
    ap.add_argument('--n-train-wsi', type=int, default=0, help='slides of each dataset a training id draws from; 0 = all')
    ap.add_argument('--n-val-wsi', type=int, default=10, help='slides of each dataset a validation id draws from')
    # --fov, --max-ds, --sampler-*, --camera-*
    add_fov_args(ap)
    ap.add_argument('--fov-seed', type=int, default=1000, help='the training draws; the benches use 0')
    ap.add_argument('--new-fovs-each-epoch', action=argparse.BooleanOptionalAction, default=True,
                    help='draw other FoV positions (and photograph them with other camera draws) every epoch: the '
                         'seed is --fov-seed plus the epoch. Off: the same ~30 thousand FoVs every epoch')
    ap.add_argument('--val-fov-seed', type=int, default=0)
    ap.add_argument('--mask-cache-job', default='Masks', help='where the hest masks are')
    # data per visit
    ap.add_argument('--fov-per-visit', type=int, default=64)
    ap.add_argument('--val-fov-per-visit', type=int, default=16)
    ap.add_argument('--steps-per-visit', type=int, default=16)
    ap.add_argument('--fovs-per-step', type=int, default=8,
                    help='FoVs one process takes a step on; with several processes the step is this times '
                         'their number')
    ap.add_argument('--keep-levels', action=argparse.BooleanOptionalAction, default=True,
                    help='keep each slide\'s encoded levels on the host between epochs (a slide is the same '
                         'process\'s every epoch). Off: encode them every epoch, which is most of the time')
    ap.add_argument('--prefetch', type=int, default=2,
                    help='visits made ahead of the training by a thread (0: none); each holds its level\'s '
                         'window grids on the GPU')
    ap.add_argument('--read-workers', type=int, default=0,
                    help='DataLoader workers each process reads a level\'s tiles with (0: all the CPUs it has, minus '
                         'one). They hand their blocks of tiles to the process through /dev/shm, a node\'s own, and '
                         'in job 596314 (8 processes a node, 11 workers each) 7 of 16 processes ran it out (bus '
                         'error / No space left on device, a level\'s read lost). Fewer processes a node '
                         '(--gpus-per-node 4) or fewer workers: fewer blocks in flight on a node')
    ap.add_argument('--n-cand', type=int, default=128, help='candidates of one FoV a step uses')
    ap.add_argument('--hard-share', nargs=2, type=float, default=[0.2, 1.0],
                    help='share of the negatives that are the hardest, first and last epoch')
    # labels
    ap.add_argument('--near-tiles', type=float, default=2.0,
                    help='a window at the right rotation that is not an answer and whose centre is within this '
                         'many tiles of the truth\'s is a near window (Labels.py): not a negative, and the truth '
                         'has to beat it by its soft target')
    ap.add_argument('--near-alpha', type=float, default=0.3, help='the soft target of a near window at distance 0')
    ap.add_argument('--near-sigma', type=float, default=1.0,
                    help='tiles; the soft target falls as exp(-d^2 / 2 sigma^2)')
    ap.add_argument('--loc-min-iou', type=float, default=0.1)
    # model
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--attn-heads', type=int, default=8)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--embed-dim', type=int, default=512)
    # loss
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--embed-temperature', type=float, default=0.05)
    ap.add_argument('--w-topk', type=float, default=0.5)
    ap.add_argument('--w-bce', type=float, default=0.1)
    ap.add_argument('--w-rot', type=float, default=0.5)
    ap.add_argument('--w-loc', type=float, default=0.5)
    ap.add_argument('--topk', type=int, default=30, help='k of the top-k hinge')
    ap.add_argument('--margin', type=float, default=1.0)
    ap.add_argument('--pos-weight', type=float, default=5.0)
    # optimisation
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-4, help='the learning rate for ONE process (see --lr-scale)')
    ap.add_argument('--embed-lr', type=float, default=1e-4)
    ap.add_argument('--lr-scale', choices=('none', 'sqrt', 'linear'), default='sqrt',
                    help='the step averages over processes x --fovs-per-step FoVs: the learning rates are '
                         'multiplied by sqrt(processes) (the usual rule for Adam), by the number of processes, '
                         'or left as given')
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--select-k', type=int, default=30, help='the checkpoint is the epoch best at R@k')
    # run
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto', help="'auto': the GPU, or stop if the job was given one that does not work")
    ap.add_argument('--tag', default='', help='added to the weights\' names')
    ap.add_argument('--out', default=None, help='default <results>/RerankHead')
    ap.add_argument('--max-visits', type=int, default=0, help='stop an epoch after N visits (a smoke run)')
    ap.add_argument('--dry-run', action='store_true', help='print the plan; open and write nothing')
    ap.add_argument('--usage-interval', type=float, default=20.0,
                    help='seconds between the samples of the GPU / CPU / memory / /dev/shm use each node reports '
                         'every epoch (0: not measured)')
    ap.add_argument('--wandb-project', default='rerank-head')
    ap.add_argument('--wandb-mode', default=os.environ.get('WANDB_MODE', 'online'))
    ap.add_argument('--run-name', default='')
    args: argparse.Namespace = ap.parse_args(argv)
    if args.select_k not in KS:
        ap.error(f'--select-k must be one of {KS}')
    return args


def slides_of(datasets: Sequence[str], n: int) -> List[Tuple[str, str, str]]:
    """`(dataset, name, path)` of the slides each AccessDatasets id holds, by the
    dataset each comes from, the first `n` of each (`by_source`; 0 = all)."""
    out: List[Tuple[str, str, str]] = []
    for dataset_id in datasets:
        for source, names in by_source(dataset_id, n or None):
            out += [(source, name, str(locate(name, dataset=source).path)) for name in names]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  one head in training
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HeadRun:
    arch: str
    model: Optional[torch.nn.Module] = None
    opt: Optional[torch.optim.Optimizer] = None
    sched: Optional[Any] = None
    best: float = -1.0
    best_epoch: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)
    #: `(in_dim, n_tiles, shapes)` the model was built from, so a process that has not yet seen a sample can build
    #: the same one (`agree_models`)
    spec: Optional[Tuple[int, int, Tuple[Tuple[int, int], ...]]] = None
    #: the parameters are rank 0's (several processes train one model)
    synced: bool = False


def ensure_model(run: HeadRun, args: argparse.Namespace, in_dim: int, n_tiles: int,
                 shapes: Sequence[Tuple[int, int]], device: Any) -> None:
    """Build the head (and its optimiser) the first time a sample of the right
    width is seen: the width is the encoder's, known once a level is encoded."""
    if run.model is not None:
        return
    run.spec = (int(in_dim), int(n_tiles), tuple((int(a), int(b)) for a, b in shapes))
    if run.arch == 'winembed':
        run.model = build_rerank('winembed', dict(in_dim=in_dim, dim=args.embed_dim,
                                                  shapes=tuple(tuple(s) for s in shapes))).to(device)
        lr: float = args.embed_lr
        run.extra['shapes'] = [list(s) for s in shapes]
    else:
        run.model = build_head(run.arch, in_dim, n_tiles, args.dim, args.layers, args.attn_heads,
                               args.dropout, device)
        lr = args.lr
    lr *= {'none': 1.0, 'sqrt': float(args.world) ** 0.5, 'linear': float(args.world)}[args.lr_scale]
    run.opt = torch.optim.AdamW(run.model.parameters(), lr=lr, weight_decay=args.weight_decay)
    run.sched = torch.optim.lr_scheduler.CosineAnnealingLR(run.opt, T_max=max(args.epochs, 1))
    n_params: int = sum(p.numel() for p in run.model.parameters())
    say(f'  [{run.arch}] {n_params / 1e6:.2f}M parameters  lr {lr:.3g}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  several processes, one model (Runtime.py)
# ══════════════════════════════════════════════════════════════════════════════

def agree_models(runs: List[HeadRun], args: argparse.Namespace, device: Any) -> None:
    """Every process has every head's model, and it is rank 0's. A process that has not seen a sample yet
    builds the model from a process that has (`HeadRun.spec`); then all take rank 0's parameters, once. All
    processes call it, at the same points."""
    specs: List[List[Optional[Tuple]]] = gather_objects([r.spec for r in runs], args.world)
    for i, run in enumerate(runs):
        if run.model is None:
            spec: Optional[Tuple] = next((s[i] for s in specs if s[i] is not None), None)
            if spec is not None:
                ensure_model(run, args, spec[0], spec[1], spec[2], device)
        if run.model is not None and not run.synced:
            broadcast_parameters(run.model)
            run.synced = True


def idle_round(runs: List[HeadRun], args: argparse.Namespace) -> None:
    """The steps of one visit for a process that has none: it adds nothing and takes the same steps."""
    for run in runs:
        if run.model is None or run.opt is None:
            continue
        for _ in range(args.steps_per_visit):
            idle_step(run.model, run.opt, args.clip, lambda m: sync_gradients(m, False))


# ══════════════════════════════════════════════════════════════════════════════
#  the visits, made ahead of the training
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Visit:
    """A level of one slide and what its FoVs are for the heads: the lists of the ranking heads, the level's
    window grids and FoVs for the embedding, and the sizes a model is built from (`ensure_model`)."""
    key: Tuple[str, int]
    neural: List[FovData] = field(default_factory=list)
    embed: Optional[EmbedVisit] = None
    neural_spec: Optional[Tuple[int, int]] = None                      # (in_dim, n_tiles)
    embed_spec: Optional[Tuple[int, int, List[Tuple[int, int]]]] = None   # (in_dim, n_tiles, shapes)

    def has_data(self) -> bool:
        return bool(self.neural) or (self.embed is not None and bool(self.embed.fovs))


def make_visits(stream: Iterator[FovResult], pooling: str, n_cand: int, hard_share: float,
                rng: np.random.Generator, need_neural: bool, need_pair: bool, need_embed: bool
                ) -> Iterator[Visit]:
    """The FoVs of `stream` (a level of a slide at a time) as visits that hold something to train on. A
    `FovResult` is only valid until the next is asked for, so each is turned into tensors at once."""
    current: Optional[Visit] = None
    for res in stream:
        key: Tuple[str, int] = (res.slide, res.level)
        if current is None or key != current.key:
            if current is not None and current.has_data():
                yield current
            current = Visit(key=key, embed=visit_of(res, pooling) if need_embed else None)
        if need_neural:
            t0: float = time.perf_counter()
            d: Optional[FovData] = prepare_fov(res, pooling, n_cand, hard_share, rng, need_pair)
            tick('prepare', time.perf_counter() - t0)
            if d is not None:
                current.neural.append(d)
                current.neural_spec = (int(d.ni.q.shape[-1]), int(d.ni.q.shape[1]))
        if need_embed:
            qg: torch.Tensor = res.queries[0]
            current.embed_spec = (int(qg.shape[-1]), int(qg.shape[0] * qg.shape[1]),
                                  [(int(qg.shape[0]), int(qg.shape[1])), (int(qg.shape[1]), int(qg.shape[0]))])
            assert current.embed is not None
            current.embed.fovs.append(fov_of(res))
    if current is not None and current.has_data():
        yield current


def prefetch(items: Iterator[Visit], depth: int, device: Any) -> Iterator[Visit]:
    """`items`, made by a thread that stays up to `depth` ahead of whoever iterates (depth 0: no thread). An
    exception in the thread is raised where the iterating is; stopping early stops the thread."""
    if depth <= 0:
        yield from items
        return
    channel: 'queue.Queue[Any]' = queue.Queue(maxsize=depth)
    stop: threading.Event = threading.Event()
    done: object = object()

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                channel.put(item, timeout=1.0)
                return True
            except queue.Full:
                continue
        return False

    def work() -> None:
        try:
            target: torch.device = torch.device(device)
            if target.type == 'cuda':                  # a new thread starts on card 0: the process's card, indexed or not
                torch.cuda.set_device(target.index or 0)
            for item in items:
                if not put(item):
                    return
            put(done)
        except BaseException as exc:                                          # noqa: BLE001
            put(exc)

    thread: threading.Thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        while True:
            item: Any = channel.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()


# ══════════════════════════════════════════════════════════════════════════════
#  training and validation
# ══════════════════════════════════════════════════════════════════════════════

def train_visit(runs: List[HeadRun], neural: List[FovData], embed: Optional[EmbedVisit],
                args: argparse.Namespace, rng: np.random.Generator, weights: Dict[str, float],
                label_cfg: LabelConfig) -> Dict[str, float]:
    """`--steps-per-visit` steps of every head on one visit's FoVs."""
    logs: Dict[str, List[float]] = {}
    sync: Optional[Callable[[torch.nn.Module], None]] = (
        (lambda m: sync_gradients(m, True)) if args.world > 1 else None)
    for run in runs:
        assert run.model is not None and run.opt is not None
        pool: int = len(embed.fovs) if run.arch == 'winembed' and embed is not None else len(neural)
        if not pool:
            # nothing for this head on this visit: with other processes it still takes its steps, with no vote
            for _ in range(args.steps_per_visit if args.world > 1 else 0):
                idle_step(run.model, run.opt, args.clip, lambda m: sync_gradients(m, False))
            continue
        for _ in range(args.steps_per_visit):
            pick: np.ndarray = rng.choice(pool, size=min(args.fovs_per_step, pool), replace=False)
            if run.arch == 'winembed':
                assert embed is not None
                run.model.train()
                run.opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=args.device.split(':')[0], dtype=torch.bfloat16):
                    loss: torch.Tensor = embed_loss(run.model, embed, [embed.fovs[i] for i in pick],
                                                    label_cfg, args.embed_temperature)
                loss.float().backward()
                if sync is not None:
                    sync(run.model)
                torch.nn.utils.clip_grad_norm_(run.model.parameters(), args.clip)
                run.opt.step()
                logs.setdefault(f'{run.arch}/loss', []).append(float(loss.detach()))
            else:
                terms: Dict[str, float] = train_step(
                    run.arch, run.model, run.opt, [neural[i] for i in pick], weights, args.temperature,
                    args.topk, args.margin, args.pos_weight, args.clip, sync)
                for name, v in terms.items():
                    logs.setdefault(f'{run.arch}/{name}', []).append(v)
    return {k: float(np.mean(v)) for k, v in logs.items()}


def run_epoch(epoch: int, runs: List[HeadRun], slides: List[Tuple[str, str, str]], args: argparse.Namespace,
              make_retriever: Any, fov: Any, label_cfg: LabelConfig, pooling: str, device: Any,
              rng: np.random.Generator, workers: int) -> Dict[str, float]:
    """One pass over the training slides, a visit at a time."""
    share: float = args.hard_share[0] + (args.hard_share[1] - args.hard_share[0]) * (
        epoch / max(args.epochs - 1, 1))
    weights: Dict[str, float] = dict(topk=args.w_topk, bce=args.w_bce, rot=args.w_rot, loc=args.w_loc)
    need_neural: bool = any(r.arch in NEURAL_ARCHS for r in runs)
    need_pair: bool = any(r.arch == 'tilegt_ctx' for r in runs)
    need_embed: bool = any(r.arch == 'winembed' for r in runs)
    sums: Dict[str, List[float]] = {}
    visits_done: int = 0
    # the visits are made by a thread that runs ahead of the training (encoding a level is minutes, a visit's
    # steps are seconds), so with several processes a step waits for the slowest one's SUM of encodings, not
    # for the slowest encoding of every visit
    prep_rng: np.random.Generator = np.random.default_rng(int(rng.integers(2 ** 32)))
    stream = fov_stream(slides, make_retriever, fov, args.mask_cache_job, device, label_cfg,
                        args.fov_per_visit, args.fov_seed, epoch, workers=workers, topk=args.k,
                        say=lambda text: say('  ' + text))
    visits: Iterator[Visit] = iter(prefetch(make_visits(stream, pooling, args.n_cand, share, prep_rng, need_neural,
                                                       need_pair, need_embed), args.prefetch, device))
    while True:
        t0: float = time.perf_counter()
        visit: Optional[Visit] = next(visits, None)
        tick('wait_for_a_visit', time.perf_counter() - t0)
        if visit is None:
            break
        for run in runs:                         # the model is built from the first sample it sees
            if run.arch in NEURAL_ARCHS and visit.neural_spec is not None:
                ensure_model(run, args, visit.neural_spec[0], visit.neural_spec[1], [], device)
            if run.arch == 'winembed' and visit.embed_spec is not None:
                ensure_model(run, args, visit.embed_spec[0], visit.embed_spec[1], visit.embed_spec[2], device)
        t0 = time.perf_counter()
        if args.world > 1:
            agree_models(runs, args, device)
            all_sum(1.0, device)                 # this process has a visit: the rounds are counted by all of them
        tick('wait_for_the_others', time.perf_counter() - t0)
        t0 = time.perf_counter()
        logs: Dict[str, float] = train_visit(runs, visit.neural, visit.embed, args, rng, weights, label_cfg)
        tick('train_steps', time.perf_counter() - t0)
        for k, v in logs.items():
            sums.setdefault(k, []).append(v)
        visits_done += 1
        say(f'  visit {visits_done:4d}  {visit.key[0]} L{visit.key[1]}  {len(visit.neural)} lists'
              f'  {("  ".join(f"{k} {v:.3f}" for k, v in logs.items() if k.endswith(("/list", "/loss"))))}'
              f'  ({time.perf_counter() - t0:.1f} s'
              + (f', GPU peak so far {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB'
                 if torch.device(device).type == 'cuda' else '') + f', {host_memory()})', flush=True)
        if args.max_visits and visits_done >= args.max_visits:
            break
    while args.world > 1:
        # out of slides: take the others' steps, with no vote, until every process is
        t0 = time.perf_counter()
        agree_models(runs, args, device)
        if all_sum(0.0, device) == 0.0:
            break
        idle_round(runs, args)
        tick('wait_for_the_others', time.perf_counter() - t0)
    return {k: float(np.mean(v)) for k, v in sums.items()}


def timing_line(who: str, wall: float, timers: Dict[str, float]) -> str:
    """Where `wall` seconds went (this process), from the timers: what making the visits took (the thread that
    makes them runs beside the training), and what the training waited for."""
    t = lambda name: timers.get(name, 0.0)                                           # noqa: E731
    tiles: float = t('level_tiles')
    encode: str = (f'{t("level_encode"):.0f} s for {tiles / 1e6:.2f}M tiles ({tiles / max(t("level_encode"), 1e-9):.0f}/s)'
                   if tiles else 'none')
    return (f'  time {who}: {wall:.0f} s | making visits: levels encoded/read back {encode}; first photo waits '
            f'{t("photo_wait"):.0f} s; search {t("retrieve") - t("level_encode"):.0f} s; slides opened+masks '
            f'{t("open_and_mask"):.0f} s; FoV draw {t("draw_positions"):.0f} s; tensors {t("prepare"):.0f} s | '
            f'training: waited for a visit {t("wait_for_a_visit"):.0f} s, for the others {t("wait_for_the_others"):.0f} s, '
            f'steps {t("train_steps"):.0f} s')


@torch.no_grad()
def validate(runs: List[HeadRun], slides: List[Tuple[str, str, str]], args: argparse.Namespace,
             make_retriever: Any, fov: Any, label_cfg: LabelConfig, pooling: str, device: Any,
             workers: int) -> Dict[str, Dict[str, float]]:
    """`{name: recall figures}` for the retriever's own order (`first_phase`) and
    each head, on the held-out FoVs: `R@k` for every k, `ceiling`, `n`."""
    ranks: Dict[str, List[float]] = {'first_phase': []}
    for run in runs:
        ranks[run.arch] = []
    embed_state: Dict[str, Any] = {}
    stream = fov_stream(slides, make_retriever, fov, args.mask_cache_job, device, label_cfg,
                        args.val_fov_per_visit, args.val_fov_seed, 0, workers=workers, topk=args.k,
                        say=lambda text: say('  val ' + text))
    for res in stream:
        ranks['first_phase'].append(first_phase_rank(res))
        for run in runs:
            if run.model is None:
                continue
            if run.arch == 'winembed':
                ranks[run.arch].append(embed_rank(run.model, res, pooling, label_cfg, embed_state))
            else:
                ranks[run.arch].append(rank_of(run.arch, run.model, res, pooling))
    if args.world > 1:                                    # the slides were shared out: every FoV counts once
        parts: List[Dict[str, List[float]]] = gather_objects(ranks, args.world)
        ranks = {name: [r for part in parts for r in part[name]] for name in ranks}
    return {name: recall_at(r) for name, r in ranks.items() if r}


def weights_path(args: argparse.Namespace, encoder: str, pooling: str, arch: str, which: str) -> str:
    """`<out>/weights/<data>--<encoder>--<pooling>--<arch>[--<tag>]--<best|last>.pt`
    (`Checkpoints.weights_name`), the data the training slides' ids (`data_tag`)."""
    out: str = args.out or os.path.join(RESULT_DIR, 'RerankHead')
    return os.path.join(out, 'weights', weights_name(data_tag(args.train_datasets), encoder, pooling, arch,
                                                    *([args.tag] if args.tag else []), which))


def main() -> int:
    global RANK
    RANK = int(os.environ.get('RANK', '0'))        # torchrun's, known before the process group is joined
    args: argparse.Namespace = parse_args()
    rank, world, local = dist_init()
    args.rank, args.world = rank, world
    if world > 1:
        split_output(os.environ.get('SLURM_JOB_NAME', 'RerankHead'), rank)
    fov = fov_from_args(args)
    fov = replace(fov, sampler=replace(fov.sampler, seed=args.fov_seed))
    val_fov = replace(fov, sampler=replace(fov.sampler, seed=args.val_fov_seed))
    label_cfg: LabelConfig = LabelConfig(near_tiles=args.near_tiles, near_alpha=args.near_alpha,
                                         near_sigma=args.near_sigma, loc_min_iou=args.loc_min_iou)
    encoder: str = args.encoder
    pooling: str = args.pooling
    # the retriever whose list is reranked: the sliding window's baseline settings on this encoder and pooling,
    # every field written (no recipe to register for an encoder and pooling that is asked for)
    retr_cfg: RetrieverSlidingWinSimRotConfig = RetrieverSlidingWinSimRotConfig(
        encoder=encoder_config(encoder, pooling=pooling).with_model(dtype='fp16'),
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, score='mean')
    train_slides: List[Tuple[str, str, str]] = slides_of(args.train_datasets, args.n_train_wsi)
    val_slides: List[Tuple[str, str, str]] = slides_of(args.val_datasets, args.n_val_wsi)
    say(f'heads      : {" ".join(args.heads)}   retriever {encoder} (K={args.k}, pooling {pooling})', console=True)
    say(f'train      : {len(train_slides)} slides of {" ".join(args.train_datasets)}   '
        f'{args.fov_per_visit} FoVs a visit, {args.steps_per_visit} steps of {args.fovs_per_step}', console=True)
    say(f'val        : {len(val_slides)} slides of {" ".join(args.val_datasets)}   '
        f'{args.val_fov_per_visit} FoVs a visit', console=True)
    if args.dry_run:
        say('dry run: nothing opened, nothing written', console=True)
        return 0

    torch.manual_seed(args.seed)                  # the same on every process: the same first weights
    rng: np.random.Generator = np.random.default_rng(args.seed + 1000 * rank)   # this process's choices
    cuda: str = f'cuda:{local}' if world > 1 else 'cuda'
    device: torch.device = torch.device((cuda if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    require_gpu_if_allocated(device, args.device)
    args.device = str(device)                     # the rest reads the resolved name
    # the CPUs are a node's: shared by the processes ON the node (torchrun's LOCAL_WORLD_SIZE)
    budget = CpuBudget.for_job(processes=int(os.environ.get('LOCAL_WORLD_SIZE', world))).apply()
    say(f'device     : {device}   rank {rank} of {world}   {budget.line()}', console=True, flush=True)
    # one meter per node (its first process): the counters of the node's GPUs, the CPUs of the job and its memory
    # and, every 10 minutes, the figures of those minutes in this process's log (a long first epoch, or a crash in
    # it, is seen without waiting for the epoch's end)
    def show_usage(s: Dict[str, Any]) -> None:
        say('  resource use, the last 10 min:')
        for line in usage_lines(s):
            say(line, flush=True)

    meter: Optional[UsageMeter] = (UsageMeter(args.usage_interval, 600.0, show_usage).start()
                                   if args.usage_interval > 0 and local == 0 else None)
    # A slide's levels are encoded the first time it is visited and kept on the host (`MemoryFeatureStore`), so
    # every later epoch finds them: the slides are the same process's every epoch (below), each process holding
    # only its own. `--no-keep-levels` encodes them again each time.
    stores: Dict[str, MemoryFeatureStore] = {}

    def make_retriever(slide: str) -> RetrieverSlidingWinSimRot:
        store: Optional[MemoryFeatureStore] = stores.setdefault(slide, MemoryFeatureStore()) if args.keep_levels else None
        return RetrieverSlidingWinSimRot(retr_cfg, device, read_workers=args.read_workers or budget.workers,
                                         feature_store=store)

    # the slides this process trains on, every epoch: every `world`-th of the list by name
    my_slides: List[Tuple[str, str, str]] = sorted(train_slides, key=lambda s: s[1])[rank::world]
    runs: List[HeadRun] = [HeadRun(arch=a) for a in args.heads]
    # one wandb run per arm: a formal run first deletes the earlier runs of its name (not one still running); a smoke
    # run (stopped after a few visits) has a name of its own and deletes nothing
    arm_name: str = (args.run_name or '-'.join([encoder, *args.heads] + ([args.tag] if args.tag else []))) \
        + ('-smoke' if args.max_visits else '')
    wb = (wandb_init(args.wandb_project, args.wandb_mode, arm_name, config=plain_args(args),
                     replace_previous=not args.max_visits) if rank == 0 else None)
    started: float = time.perf_counter()
    for epoch in range(args.epochs):
        say(f'\n======== epoch {epoch + 1}/{args.epochs} ========', console=True, flush=True)
        epoch_started: float = time.perf_counter()
        # this process's slides, in a new order each epoch
        mine: List[Tuple[str, str, str]] = [my_slides[i] for i in
                                            np.random.default_rng([args.seed, epoch, rank]).permutation(len(my_slides))]
        # new FoV positions and photos every epoch (the draw's and the camera's seed move with it), so a long run
        # sees many more than the ~30 thousand one draw holds
        fov_epoch = replace(fov, sampler=replace(fov.sampler, seed=args.fov_seed + epoch)) if args.new_fovs_each_epoch else fov
        losses: Dict[str, float] = run_epoch(epoch, runs, mine, args, make_retriever,
                                             fov_epoch, label_cfg, pooling, device, rng, budget.workers)
        say(timing_line(f'rank {rank}, training', time.perf_counter() - epoch_started, take_timers()), console=True)
        for run in runs:
            if run.sched is not None:
                run.sched.step()
        val_started: float = time.perf_counter()
        figures: Dict[str, Dict[str, float]] = validate(runs, val_slides[rank::world], args, make_retriever,
                                                        val_fov, label_cfg, pooling, device, budget.workers)
        say(timing_line(f'rank {rank}, validation', time.perf_counter() - val_started, take_timers()), console=True)
        say(f'  levels kept on the host: {sum(s.nbytes() for s in stores.values()) / 1e9:.1f} GB '
            f'({sum(s.hits for s in stores.values())} read back, {sum(s.misses for s in stores.values())} encoded)')
        metrics: Dict[str, float] = dict(losses)
        # what the GPUs, CPUs and memory of every node were used for since the last epoch's report
        usage: List[Optional[Dict[str, Any]]] = gather_objects(meter.summary() if meter is not None else None, world) \
            if world > 1 else [meter.summary() if meter is not None else None]
        say('  resource use this epoch (training + validation)', console=True)
        for node in (u for u in usage if u is not None):
            for line in usage_lines(node):
                say(line, console=True)
            for hint in advice(node):
                say(f'      -> {hint}', console=True)
            if node['gpus']:
                metrics[f'usage/{node["host"]}/gpu_busy'] = sum(v['util_mean'] for v in node['gpus'].values()) / len(node['gpus'])
            metrics[f'usage/{node["host"]}/cpu_busy'] = node['cpu_mean']
            metrics[f'usage/{node["host"]}/mem_peak_gb'] = node['mem_peak']
            metrics[f'usage/{node["host"]}/shm_peak_gb'] = node['shm_peak']
        say('  validation (share of FoVs whose first hit is in the top k)', console=True)
        for name, f in figures.items():
            say(f'    {name:12s} ' + '  '.join(f'R@{k} {f[f"R@{k}"]:.3f}' for k in KS)
                + f'   ceiling {f["ceiling"]:.3f}  n {f["n"]}', console=True)
            metrics.update({f'val/{name}/{key}': v for key, v in f.items()})
        for run in runs:
            if run.arch not in figures or run.model is None:
                continue
            score: float = figures[run.arch][f'R@{args.select_k}']
            for which in (('last',) + (('best',) if score > run.best else ())) if rank == 0 else ():
                save_rerank_checkpoint(
                    weights_path(args, encoder, pooling, run.arch, which), model=run.model, arch=run.arch,
                    pooling=pooling, encoder=encoder, epoch=epoch + 1, val=figures[run.arch],
                    run_args=plain_args(args), extra=run.extra,
                    optimizer_state=run.opt.state_dict() if run.opt is not None else None)
            if score > run.best:
                run.best, run.best_epoch = score, epoch + 1
                say(f'    [{run.arch}] new best R@{args.select_k} {score:.3f} -> '
                    f'{weights_path(args, encoder, pooling, run.arch, "best")}', console=True, flush=True)
        wandb_log(wb, epoch + 1, metrics)
    wandb_finish(wb)
    if meter is not None:
        meter.stop()
    say(f'\ndone in {(time.perf_counter() - started) / 60:.1f} min   '
        + '  '.join(f'{r.arch} best R@{args.select_k} {r.best:.3f} (epoch {r.best_epoch})' for r in runs),
        console=True)
    dist_done()
    return 0


def run() -> int:
    """`main`, and with several processes an error ends THIS process at once. A process that raises is not over
    until its interpreter has shut down, and that waits on the process group the others are inside of: the
    others, held up in a collective for the dead one, are found out by the NCCL watchdog ten minutes later
    (job 592907 lost the real error to a timeout that hid it). `os._exit` is what torchrun can see at once, and
    it then ends the others."""
    try:
        return main()
    except Exception:                                                          # noqa: BLE001
        if int(os.environ.get('WORLD_SIZE', '1')) <= 1:
            raise
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == '__main__':
    sys.exit(run())
