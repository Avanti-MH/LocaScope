#!/usr/bin/env python3
'''Train GloRet (Arch 1) and LogLoRet (Arch 2) of `stage2_retrieval/proposal.md` on synthetic FoVs, made as they are
used.

    python training/GloRet/cli/train.py --heads GloRet-SelfAttn-AttnPool LogLoRet-SelfAttn-AttnPool+dustbin \\
        --encoder gigapath --pooling cls_avg --epochs 20

A head is a method of the proposal's names -- `GloRet-{ContextEn}-{GlobalEn}`, `LogLoRet-{ContextEn}-{GlobalEn}+{no-match}`
(`dustbin`, `sinkhorn`, `sigmoid`; trained in, so a different option is a different head) -- and several are trained
in ONE process on the SAME FoVs: a level of a slide is encoded once, whatever the number of heads. GloRet's search
(`exact`, `hnsw`, `ivf`) changes no weights and is not trained.

THE DATA (`Supply.py`): training slides are `--train-datasets` (`@pure-train`), which shares no slide with the
validation set (`@pure-half-val`). For each slide the first phase (`RetrieverSlidingWinSimRot` of `--encoder` read
through `--pooling`) is built, FoV positions are drawn over every level with a seed of their own, and a LEVEL AT A
TIME `--fov-per-visit` of them are photographed and put through the first phase. A visit's FoVs are trained on
`--steps-per-visit` times, `--fovs-per-step` at once, and dropped. The labels are in `Supply.py`.

THE ITEMS of a FoV's softmax (`--top-p`, `Items.py`, `Pool.py`, `Episodes.py`): the windows of the level at the four rotations
(`@items=level`, a GloRet head's default) or the entries of the list the first phase made (`@items=list`, a LogLoRet head's;
`--raw-topk` of them; when none of them is a positive the answer is ADDED, a FoV is not left out). How many go into the
softmax is `--top-p`: a share (`0.05`; 1.0 is all, as the retrieval is) or a count (`2000`), whichever the head is given
(`Name@top=0.05`; else the job's: the share for a GloRet head, the count for a LogLoRet head). With a share of 0.5 or less they
are chosen at the start of each of `--episodes` episodes, from the first phase's scores in the first `--warmup-epochs`, from
the network's after, with a memory of each FoV's hard negatives. The softmax is over all of them, by a gradient cache
(`Full.py`, `Rerank.py`); `--window-split` shares them between the processes of a group, `--no-gradcache` skips the cache.
A pair scorer (LogLoRet) cannot rank a level's windows by itself: over `@items=level` it takes episodes from the first
phase's scores only.

THE PROCESSES: with `torchrun`, one per GPU. Each takes every Nth slide of every epoch (a `--window-split` group takes one
together), encodes their levels (kept on the host after the first epoch: `--keep-levels`) and trains on them; the gradients
are averaged after every step, so there is one model per head.

VALIDATION (the benches' hit: `BenchCommon.first_place`) is by the head's ROLE, which is how it is trained (`Name@items=level|list`), on the
held-out FoVs:
    retriever  (trained over the level: `@items=level`, a GloRet head's default) the network lists the level's windows alone (top 500, the first phase's
               separation) -- what it does as a first phase -- and R@k is the share of FoVs with a hit in its top k; the
               checkpoint is the epoch best at R@500
    reranker   (trained over the list: `@items=list`, a LogLoRet head's default; either may be either) it re-orders the first `--val-keep` windows of
               the sliding window's list; the checkpoint is the epoch best at R@10, and the conditional recall
               R@10 / R@100 of that list is printed. A pair scorer as a retriever scores every window of the level (a pair each).
A head given something other than its defaults (`@items=`, `@top=`) is another head: its name and its weights'.
The sliding window's own figures on the same FoVs are printed beside them.

Weights: `GloRet/weights/<data>--<encoder>--<pooling>--<method>[--<no-match>]--<best|last>.pt` under the results root
(`ArmSpec.learned_weights`), where an arm reads them.
'''
from __future__ import annotations

import argparse
import copy
import os
import queue
import sys
import threading
import time
import zlib
import traceback
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'utilities'))
from _paths import LOG_DIR, RESULT_DIR, setup_import_paths              # noqa: E402
setup_import_paths()
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'utilities', 'bench_modules'))   # BenchCommon

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from AccessDatasets import by_source, locate                          # noqa: E402
from CpuBudget import CpuBudget                                        # noqa: E402
from BenchCommon import require_gpu_if_allocated                      # noqa: E402
from FovSupply import FOV_RECIPES, add_fov_args, fov_from_args                     # noqa: E402
from RerankChoices import build_rerank                                # noqa: E402
from Checkpoints import data_tag, save_rerank_checkpoint              # noqa: E402
from GloRet import NO_MATCH, TILE_POOLS, NetCfg, combo_problem   # noqa: E402
from TileEncoderFunc import encoder_config                           # noqa: E402
from stage2_retrieval.ArmSpec import SEARCHES, learned_weights       # noqa: E402
from stage2_retrieval.Retriever import EncoderPool                    # noqa: E402
from stage2_retrieval.RetrieverGloRet import WindowEncodings          # noqa: E402
from stage2_retrieval.RetrieverSlidingWinSimRot import (RetrieverSlidingWinSimRot,   # noqa: E402
                                                        RetrieverSlidingWinSimRotConfig)

from training.GloRet.Runtime import (all_sum, broadcast_parameters, dist_done, dist_init,   # noqa: E402
                                      gather_objects, plain_args, sync_gradients, wandb_finish,
                                      wandb_init, wandb_log)
from training.GloRet.Steps import (KS, StepCfg, begin_visit_peak, visit_peak_gb, gloret_rank, gloret_rerank_rank, idle_step,   # noqa: E402
                                   logloret_rank, own_rank, recall_at, upstream_rank)
from training.GloRet.Items import SOURCES   # noqa: E402
from training.GloRet.Patches import (FROZEN_POOLING, POOLING as PATCH_POOLING, PatchSource,   # noqa: E402
                                     load_pool)
from training.GloRet.Lazy import gloret_rerank_rank_lazy, lazy_level_of, logloret_rank_lazy   # noqa: E402
from training.GloRet.Episodes import PoolCfg, run_visit   # noqa: E402
from training.GloRet.Full import FullCfg, make_split   # noqa: E402
from training.GloRet.Losses import SOFT_TARGETS   # noqa: E402
from training.GloRet.RealPhotos import RealPhoto, choose_photos, real_stream, write_list   # noqa: E402
from training.GloRet.Pool import Memory, TopP, parse_top_p   # noqa: E402
from training.GloRet.Rerank import RerankCfg, logloret_level_rank   # noqa: E402
from training.GloRet.Profile import start as profile_start   # noqa: E402
from training.GloRet.Supply import (CardBudget, Fov, FovResult, Level, MemoryFeatureStore, fov_of, fov_stream,   # noqa: E402
                                    level_of, take_timers, tick)
from training.GloRet.LogLayout import hard_line, memory_line, hard_negatives_block, labels, losses_block, num, recall_block, rule, steps_block, table, timing_lines   # noqa: E402
from training.GloRet.Usage import UsageMeter, advice, usage_line, usage_lines   # noqa: E402

#: The width of one patch token of each encoder (what `NetCfg.in_dim` is for TilePool `attnpool`); a tile's grid
#: entry is `P * TOKEN_DIM` numbers.
TOKEN_DIM: Dict[str, int] = {'gigapath': 1536, 'uni2': 1536}

#: this process's rank, and the console (Slurm's log) when its own output goes to a file
RANK: int = 0
WINDOW_SPLIT: Optional[Any] = None      # the group of processes that share a level's windows (--window-split)
CONSOLE: Optional[Any] = None

#: what a head is selected on, by its role: the share of FoVs with a hit in the top k. A retriever lists the level's windows
#: itself (R@500); a reranker re-orders the first phase's list (R@10 of it)
SELECT_K: Dict[str, int] = {'retriever': 500, 'reranker': 10}


def say(*parts: Any, console: bool = False, **kwargs: Any) -> None:
    """`print` to this process's output; with several processes that is its own file (`split_output`), and a line
    meant for the Slurm log too (`console`) is copied there by rank 0 only."""
    print(*parts, **kwargs)
    if console and RANK == 0 and CONSOLE is not None:
        print(*parts, file=CONSOLE, **kwargs)


#: the highest GPU memory (GB) a visit of this process reached so far (`visit_report`)
GPU_PEAK: Dict[str, float] = {'so_far': 0.0}


def host_memory() -> str:
    """This process's resident memory and the node's shared-memory filesystem (what DataLoader workers pass tiles
    through, and where a `bus error` comes from when it is full), for the log line of a visit."""
    try:
        with open('/proc/self/status') as f:
            rss: float = next(int(line.split()[1]) for line in f if line.startswith('VmRSS:')) / 1e6
        shm = os.statvfs('/dev/shm')
        used: float = (shm.f_blocks - shm.f_bfree) * shm.f_frsize / 1e9
        return (f'host RSS {rss:.1f} GB, node /dev/shm {used:.1f} of {shm.f_blocks * shm.f_frsize / 1e9:.0f} GB, '
                f'{process_facts()}')
    except (OSError, StopIteration, ValueError):
        return 'host memory unknown'


def process_facts() -> str:
    """What bounds a process's allocations other than the node's memory, for the log line of a visit: the memory maps it holds
    against `vm.max_map_count` (a failed `mmap` is a `MemoryError` of a 17 MiB array on a node with 1.3 TB free; job 635195),
    its threads, open files and virtual size."""
    try:
        with open('/proc/self/maps') as f:
            maps: int = sum(1 for _ in f)
        with open('/proc/sys/vm/max_map_count') as f:
            limit: int = int(f.read())
        with open('/proc/self/status') as f:
            status: Dict[str, int] = {line.split(':')[0]: int(line.split()[1]) for line in f
                                      if line.split(':')[0] in ('VmSize', 'Threads') and len(line.split()) > 1}
        return (f'maps {maps} of {limit}, threads {status.get("Threads", 0)}, fds {len(os.listdir("/proc/self/fd"))}, '
                f'virtual {status.get("VmSize", 0) / 1e6:.0f} GB')
    except (OSError, ValueError):
        return 'process facts unknown'


def split_output(job: str, rank: int) -> None:
    """With several processes, each writes its own log, `<log>/<job>.r<rank>`, and rank 0 keeps Slurm's log for the
    lines that say `console`. Rank 0's errors stay on the console; the others' go to their files."""
    global CONSOLE
    element: str = os.environ.get('SLURM_ARRAY_TASK_ID', '')     # an array element: its own files
    path: str = os.path.join(LOG_DIR, f'{job}{"." + element if element else ""}.r{rank}')
    CONSOLE = os.fdopen(os.dup(1), 'w', buffering=1)
    own = open(path, 'w', buffering=1)
    os.dup2(own.fileno(), 1)
    if rank > 0:
        os.dup2(own.fileno(), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  the heads
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HeadRun:
    """One head in training. `name` is what the proposal calls it (and the log), `method` the weights' method,
    `op` the no-match option of a LogLoRet ('' for a GloRet)."""
    name: str
    arch: str                                  # 'gloret' | 'logloret'
    method: str
    op: str
    model: Optional[torch.nn.Module] = None
    opt: Optional[torch.optim.Optimizer] = None
    sched: Optional[Any] = None
    best: float = -1.0
    best_epoch: int = 0
    #: the width of the tile features the model was built from, so a process that has not yet seen a FoV can build
    #: the same one (`agree_models`)
    in_dim: Optional[int] = None
    #: the parameters are rank 0's (several processes train one model)
    synced: bool = False
    #: validation: the window encodings of the slide being scored (GloRet as a retriever)
    encodings: Optional[WindowEncodings] = None
    #: where the items of its softmax come from (`Items.SOURCES`); that is what it is validated and selected as (`role`)
    items: str = 'level'
    #: how many of the items its softmax is over (`--top-p`, or its own `@top=`)
    top: Optional[TopP] = None
    #: what was asked of this head beyond the defaults, as part of its weights' name ('' for the defaults)
    variant: str = ''
    #: the flags it was given after its name (`HEAD_FLAGS`), as text; `args` is the job's with them applied, `step` its loss
    #: settings, `relabel` its own (ignore zone, wrong rotations ignored) when the job's are not what it was given
    flags: Dict[str, str] = field(default_factory=dict)
    args: Optional[argparse.Namespace] = None
    step: Optional[StepCfg] = None
    relabel: Optional[Tuple[float, bool]] = None

    @property
    def shown(self) -> str:
        """The name for the log and wandb: always with its items (`name` leaves out the default ones, which tells the two architectures
        apart by what they are given, not what they are trained over). The name itself, and so the weights, are not changed."""
        return self.name if '@items=' in self.name else f'{self.name}@items={self.items}'

    @property
    def role(self) -> str:
        """`retriever` (trained over the level's windows: validated listing the level, selected on R@500) or `reranker`
        (trained over the first phase's list: validated re-ordering it, selected on R@10), by its items."""
        return ROLE_OF_ITEMS[self.items]


#: what a head may be given after its name, `Name@key=value`: where its items come from (which makes it a retriever or a
#: reranker) and its own `--top-p`
HEAD_OPTIONS: Tuple[str, ...] = ('items', 'top')
#: and any of these flags of the job (`Name@w-geo=1.0`): what a head can have its own of, so that settings are compared on the
#: same FoVs in one job. A setting of the levels (the data, the optimiser's schedule) is the job's
HEAD_FLAGS: Tuple[str, ...] = (
    'ignore-expand', 'wrong-rotation', 'min-candidates', 'episodes', 'warmup-epochs', 'steps-per-visit', 'fovs-per-step', 'max-fov-uses',
    'soft-target', 'sigma', 'near-alpha', 'w-geo', 'w-corr', 'w-cycle', 'recon', 'w-recon', 'w-kl',
    'dim', 'embed-dim', 'layers', 'attn-heads', 'dropout', 'tau-init', 'lr', 'weight-decay', 'chunk', 'pair-chunk')
#: the items a head is given when it is not told (@items=): a GloRet head's are the level's, a LogLoRet head's the list's. Either
#: architecture can be trained over either
DEFAULT_ITEMS: Dict[str, str] = {'gloret': 'level', 'logloret': 'list'}
#: what a head is, by the items it is trained over: it lists the level (a retriever) or re-orders the first phase's list (a reranker)
ROLE_OF_ITEMS: Dict[str, str] = {'level': 'retriever', 'list': 'reranker'}


def parse_head(text: str) -> HeadRun:
    """`GloRet-SelfAttn-AttnPool` or `LogLoRet-SelfAttn-AttnPool[+dustbin]`, each with `@items=level|list` and
    `@top=<--top-p value>` as wanted -> a `HeadRun`. What a head is (a retriever or a reranker) is how it is trained, its items:
    the level's windows or the list's, whatever its architecture."""
    base, *options = text.split('@')
    given: Dict[str, str] = {}
    for option in options:
        key, equals, value = option.partition('=')
        if key == 'role':
            raise ValueError(f'head {text!r}: there is no @role=: a head trained over the level (@items=level) is a retriever, '
                             'over the list (@items=list) a reranker')
        if not equals or key not in HEAD_OPTIONS + HEAD_FLAGS or not value:
            raise ValueError(f'head {text!r}: after the name only {", ".join("@" + k + "=<value>" for k in HEAD_OPTIONS + HEAD_FLAGS)}')
        given[key] = value
    method, _, op = base.partition('+')
    kind, _, rest = method.partition('-')
    context, _, global_en = rest.partition('-')
    if kind not in ('GloRet', 'LogLoRet'):
        raise ValueError(f'head {text!r}: the method must start with GloRet- or LogLoRet-')
    arch: int = 1 if kind == 'GloRet' else 2
    problem: Optional[str] = combo_problem(context, global_en, arch)
    if problem is not None:
        raise ValueError(f'head {text!r}: {problem}')
    name: str = method
    if arch == 1:
        if op:
            raise ValueError(f'head {text!r}: GloRet\'s search ({", ".join(SEARCHES)}) changes no weights and is '
                             'not trained: name the method only')
        kind_name: str = 'gloret'
    else:
        op = op or 'dustbin'
        if op not in NO_MATCH:
            raise ValueError(f'head {text!r}: the no-match option must be one of {NO_MATCH}')
        name, kind_name = f'{method}+{op}', 'logloret'
    items: str = given.get('items', DEFAULT_ITEMS[kind_name])
    if items not in SOURCES:
        raise ValueError(f'head {text!r}: @items= is one of {SOURCES}')
    top: Optional[TopP] = None
    if 'top' in given:
        share, count = parse_top_p([given['top']])
        top = share or count
    # the options that are not the defaults make the head another one: its name and its weights'
    changed: List[Tuple[str, str]] = [(k, v) for k, v in given.items()
                                      if (k, v) != ('items', DEFAULT_ITEMS[kind_name])]
    return HeadRun(name=name + ''.join(f'@{k}={v}' for k, v in changed), arch=kind_name, method=method, op=op if arch == 2 else '',
                   items=items, top=top, variant='_'.join(f'{k}-{v}' for k, v in changed),
                   flags={k: v for k, v in given.items() if k in HEAD_FLAGS})


def weights_file(args: argparse.Namespace, run: HeadRun, which: str) -> str:
    """Where an arm reads the head's weights (`ArmSpec.learned_weights`), `which` = best | last, under `--out`."""
    data: str = data_tag(args.train_datasets) + (f'_{args.tag}' if args.tag else '') + (f'_{run.variant}' if run.variant else '')
    path: str = learned_weights(args.encoder, args.pooling, run.method, run.op, data)
    if which == 'last':
        path = path[:-len('--best.pt')] + '--last.pt'
    return os.path.join(args.out or RESULT_DIR, path)


def ensure_model(run: HeadRun, args: argparse.Namespace, in_dim: int, device: Any) -> None:
    """Build the head (and its optimiser) the first time a FoV of the right width is seen: the width is the
    encoder's, known once a level is encoded."""
    if run.model is not None:
        return
    args = run.args if run.args is not None else args          # the head's own flags (`HEAD_FLAGS`) applied
    run.in_dim = int(in_dim)
    context, global_en = run.method.split('-', 2)[1:]
    cfg: NetCfg = NetCfg(in_dim=int(in_dim), tile_pool=args.pooling, context=context, global_en=global_en, d=args.dim,
                         d_r=args.embed_dim, layers=args.layers, heads=args.attn_heads, dropout=args.dropout,
                         tau_init=args.tau_init, no_match=run.op or 'dustbin',
                         recon=(args.recon or ('vae' if args.w_kl > 0 else 'ae')) if (args.w_recon > 0 or args.w_kl > 0) else '')
    run.model = build_rerank(run.arch, asdict(cfg)).to(device)
    if args.init_tag:                                     # stage B: start from the weights stage A trained
        start: str = weights_file(argparse.Namespace(**{**vars(args), 'tag': args.init_tag}), replace(run, variant=''), 'best')   # the head, not its variant
        ckpt: Dict[str, Any] = torch.load(start, map_location='cpu')
        same = {k: v for k, v in ckpt['hparams'].items() if k != 'recon'} == {k: v for k, v in asdict(cfg).items() if k != 'recon'}
        if not same:                                    # (the decoders of `recon` are no part of what the weights are)
            raise SystemExit(f'--init-tag {args.init_tag}: {start} was built with other settings than this run '
                             f'({ckpt["hparams"]} against {asdict(cfg)})')
        missing = run.model.load_state_dict(ckpt['state'], strict=False)
        if [k for k in (*missing.missing_keys, *missing.unexpected_keys) if '.dec.' not in k and '.logvar.' not in k]:
            raise SystemExit(f'--init-tag {args.init_tag}: {start} does not fit this network: {missing}')
        say(f'  [{run.name}] starts from {start} (epoch {ckpt["epoch"]})', flush=True)
    lr: float = args.lr * {'none': 1.0, 'sqrt': float(args.dp) ** 0.5, 'linear': float(args.dp)}[args.lr_scale]
    run.opt = torch.optim.AdamW(run.model.parameters(), lr=lr, weight_decay=args.weight_decay)
    run.sched = torch.optim.lr_scheduler.CosineAnnealingLR(run.opt, T_max=max(args.epochs, 1))
    say(f'  [{run.name}] {sum(p.numel() for p in run.model.parameters()) / 1e6:.2f}M parameters  lr {lr:.3g}',
        flush=True)


def agree_models(runs: List[HeadRun], args: argparse.Namespace, device: Any) -> None:
    """Every process has every head's model, and it is rank 0's. A process that has not seen a FoV yet builds the
    model from a process that has; then all take rank 0's parameters, once. All processes call it, at the same
    points."""
    widths: List[List[Optional[int]]] = gather_objects([r.in_dim for r in runs], args.world)
    for i, run in enumerate(runs):
        if run.model is None:
            width: Optional[int] = next((w[i] for w in widths if w[i] is not None), None)
            if width is not None:
                ensure_model(run, args, width, device)
        if run.model is not None and not run.synced:
            broadcast_parameters(run.model)
            run.synced = True


def idle_round(runs: List[HeadRun], args: argparse.Namespace) -> None:
    """The steps of one visit for a process that has none: it adds nothing and takes the same steps."""
    for run in runs:
        if run.model is None or run.opt is None:
            continue
        for _ in range((run.args or args).steps_per_visit):
            idle_step(run.model, run.opt, args.clip, lambda m: sync_gradients(m, False))


# ══════════════════════════════════════════════════════════════════════════════
#  arguments
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--heads', nargs='+', default=['GloRet-SelfAttn-AttnPool'],
                    help='GloRet-{ContextEn}-{GlobalEn}, LogLoRet-{ContextEn}-{GlobalEn}[+{no-match}]')
    ap.add_argument('--encoder', default='gigapath', choices=('uni2', 'gigapath'),
                    help='the encoder of the sliding-window first phase and of the features the heads read')
    ap.add_argument('--pooling', default='cls_avg', choices=list(TILE_POOLS),
                    help='the TilePool. `attnpool` reads each tile\'s patch tokens (0.6 MB a tile, 100x `cls_avg`): '
                         'use it with --min-ds, and the list comes from another sliding window (--list-pooling, '
                         'default `cls`)')
    ap.add_argument('--pool-weights', default='',
                    help='for --pooling attnpool_frozen (stage C): the checkpoint of a network trained with TilePool '
                         '`attnpool` (stage A or B) whose attention pool pools every tile\'s patch tokens, frozen')
    ap.add_argument('--on-demand', action='store_true',
                    help='stage B (--pooling attnpool, LogLoRet heads): the level is NOT encoded; each FoV\'s '
                         'candidates are drawn once for the visit and only the tiles under them are read and encoded '
                         '(Lazy.py), so levels whose patch tokens do not fit the card can be trained on. Use a small '
                         '--fov-per-visit (16) and --n-cand (128)')
    ap.add_argument('--init-tag', default='',
                    help='start each head from the best weights of the run with this --tag (the weights file '
                         'of the same head, encoder and pooling): stage B starts from stage A\'s')
    ap.add_argument('--min-ds', type=float, default=0.0,
                    help='only the levels whose downsample is at least this, for training and validation (the '
                         'coarse levels, whose tiles are few: 4 for `attnpool`, stage A)')
    ap.add_argument('--usage-interval', type=float, default=20.0,
                    help='seconds between the samples of the GPU / CPU / memory use each node reports every epoch '
                         '(0: not measured)')
    ap.add_argument('--list-encoder', default='', choices=('', 'uni2', 'gigapath'),
                    help='the encoder of the sliding window that makes the LIST a LogLoRet is trained on (default: '
                         '--encoder). A list is places and rotations, so it need not come from the features the '
                         'network reads; a different one is a second retriever, whose levels are encoded too')
    ap.add_argument('--list-pooling', default='',
                    help='the pooling of that sliding window (default: --pooling)')
    ap.add_argument('--raw-topk', type=int, default=10000,
                    help='the length of the list the sliding window makes (after its separation): the training '
                         'data of a LogLoRet. 10000 is a list the pooling cannot tell from the answer, at the '
                         'size of a retrieval')
    ap.add_argument('--val-keep', type=int, default=100,
                    help='LogLoRet validation re-orders the first K windows of the list (a list of 10000 '
                         'scored every epoch is an hour); R@10 is read against the list\'s R@K')
    ap.add_argument('--train-datasets', nargs='+', default=['@pure-train'])
    ap.add_argument('--val-datasets', nargs='+', default=['@pure-half-val'])
    ap.add_argument('--n-train-wsi', type=int, default=0, help='slides of each dataset a training id draws from; 0 = all')
    ap.add_argument('--n-val-wsi', type=int, default=10, help='slides of each dataset a validation id draws from')
    ap.add_argument('--real-photos', type=int, default=0,
                    help='also read every epoch\'s recall on this many real photos whose place a person confirmed '
                         '(`RealPhotos.py`; a seeded draw from the checklists); 0 = off. Monitoring only: no head is selected on them')
    ap.add_argument('--real-dataset', default='ki67_with_photo', help='the dataset of the real photos')
    ap.add_argument('--real-job', default='RealTest', help='the cache job of the real-photo runs (the confirmed places\' windows)')
    ap.add_argument('--real-checklists', default='/work/u26130998/datasets/Ki67_with_photo/checklists')
    ap.add_argument('--real-stage1', default='prototype-uni2-attn-set-transformer-attn-score-cross-on',
                    help='the stage 1 and stage 2 arm the checklist rows were labelled from')
    ap.add_argument('--real-stage2', default='slidewin-gigapath-cls-mean')
    ap.add_argument('--real-seed', type=int, default=0)
    ap.add_argument('--val-logloret-share', type=float, default=1.0,
                    help='a LogLoRet head validated as a retriever scores every window of the level (a pair each): the share of the '
                         'FoVs (validation and real photos) it is ranked on, a fixed pick; 1.0 = all')
    add_fov_args(ap)
    ap.add_argument('--val-fov', choices=sorted(FOV_RECIPES), default='',
                    help='the FoV recipe validation draws from (the recipe as it is, no --sampler-/--camera- flags); default: --fov with its flags')
    ap.add_argument('--fov-seed', type=int, default=1000, help='the training draws; the benches use 0')
    ap.add_argument('--new-fovs-each-epoch', action=argparse.BooleanOptionalAction, default=True,
                    help='draw other FoV positions (and photograph them with other camera draws) every epoch: the '
                         'seed is --fov-seed plus the epoch')
    ap.add_argument('--val-fov-seed', type=int, default=0)
    ap.add_argument('--mask-cache-job', default='Masks', help='where the hest masks are')
    # data per visit
    ap.add_argument('--fov-per-visit', type=int, default=64)
    ap.add_argument('--val-fov-per-visit', type=int, default=16)
    ap.add_argument('--steps-per-visit', type=int, default=16)
    ap.add_argument('--fovs-per-step', type=int, default=8,
                    help='FoVs one process takes a step on; with several processes the step is this times their number')
    ap.add_argument('--max-fov-uses', type=int, default=2,
                    help='a FoV is trained on at most about this many times in a visit (0: no limit): when a visit has few FoVs the '
                         'steps take fewer FoVs each (--steps-per-visit stays), so that a FoV is not seen again and again')
    ap.add_argument('--keep-levels', action=argparse.BooleanOptionalAction, default=True,
                    help='keep each slide\'s encoded levels on the host between epochs. Off: encode them every epoch')
    ap.add_argument('--gpu-levels-gb', type=float, default=40.0,
                    help='GB of the card the encoded levels may stay on between epochs (no copy back to the host, none to the card '
                         'again); a level that does not fit stays on the host (0: all on the host). Per process')
    ap.add_argument('--profile', action=argparse.BooleanOptionalAction, default=False,
                    help='run the first pool choice and the first step of each head under torch.profiler (rank 0) and write '
                         '<out>/profile/<label>.txt: how much of the time the card ran kernels, the copies, the slowest operations')
    ap.add_argument('--prefetch', type=int, default=2,
                    help='visits made ahead of the training by a thread (0: none); each holds its level\'s grids on the GPU')
    # samples (Steps.py)
    ap.add_argument('--ignore-expand', type=float, default=0.5,
                    help='the windows around an answer whose origin is within this share of the FoV\'s own window (half its '
                         'height, half its width) are ignored: neither a hit nor a miss (Supply.py)')
    ap.add_argument('--wrong-rotation', choices=('ignore', 'negative'), default='ignore',
                    help='the answer\'s own place and the windows of its zone found at the WRONG rotations of the query '
                         'are ignored, or are negatives')
    # which candidates a head's softmax is over (Pool.py, Episodes.py, Full.py, Rerank.py)
    ap.add_argument('--top-p', nargs='+', required=True, metavar='P',
                    help='how many of a FoV\'s items its softmax is over (the answers come on top). A number with a point '
                         '(0.05, 1.0) is a SHARE of the items (1.0 is all of them, as the retrieval is); an integer (2000) '
                         'is a COUNT of them. The items are the windows of the level at the four rotations (a GloRet head\'s) '
                         'or the entries of the list (--raw-topk; a LogLoRet head\'s). A job gives the share for its GloRet '
                         'heads and the count for its LogLoRet heads (one of each when it has both); a head may have its '
                         'own with Name@top=<value>, and the items it is over with Name@items=level|list. A share above 0.5 is made again at every step from the network\'s '
                         'own scores; 0.5 or less is chosen at the start of each episode (--episodes), from the first '
                         'phase\'s scores in the first --warmup-epochs and the network\'s after, with a memory of each '
                         'FoV\'s hard negatives')
    ap.add_argument('--min-candidates', type=int, default=256,
                    help='a share (--top-p) is raised to at least this many negatives for a small level')
    ap.add_argument('--episodes', type=int, default=3,
                    help='episodes a visit\'s steps are shared out between (--top-p of 0.5 or less); the pools are '
                         'chosen again at the start of each')
    ap.add_argument('--warmup-epochs', type=int, default=2,
                    help='epochs in which the first phase\'s scores choose the pools (--top-p of 0.5 or less); after '
                         'them the network\'s own do')
    ap.add_argument('--chunk', type=int, default=4096,
                    help='windows (GloRet) encoded at a time; for a LogLoRet head pairs scored at a time (--pair-chunk). '
                         'Memory against speed, not the result')
    ap.add_argument('--pair-chunk', type=int, default=4096, help='LogLoRet: pairs scored at a time (memory against speed, not the result)')
    ap.add_argument('--encoder-batch', type=int, default=2048,
                    help='tiles the tile encoder runs at a time (its batch size, not part of the vectors\' identity: memory against speed)')
    ap.add_argument('--window-split', type=int, default=1,
                    help='processes in a group that share ONE level\'s candidates (each takes 1/G of them; the groups see '
                         'other slides). 1: none, a process has them all. The processes must make groups of this many')
    ap.add_argument('--no-gradcache', action='store_true',
                    help='score a process\'s candidates once, with their graph, instead of twice (a gradient cache): one '
                         'pass less, and the activations of its share of the candidates in memory')
    # model
    ap.add_argument('--dim', type=int, default=512, help='d: the width after W_in')
    ap.add_argument('--embed-dim', type=int, default=512, help='d_r: the width of GloRet\'s retrieval vector')
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--attn-heads', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--tau-init', type=float, default=0.05)
    # loss
    ap.add_argument('--soft-target', choices=SOFT_TARGETS, default='gaussian',
                    help='the soft target of the near windows (--w-geo, Losses.py). gaussian: alpha * exp(-d^2 / 2 sigma^2) '
                         'against the answers\' 1, not a distribution (the form RerankHead\'s WinEmbed trains with). '
                         'exponential: a distribution proportional to exp(-d / sigma), as a cross-entropy added to the loss')
    ap.add_argument('--sigma', type=float, default=1.0, help='tiles; the soft target falls over this distance')
    ap.add_argument('--near-alpha', type=float, default=0.3,
                    help='gaussian: the soft target of a near window at distance 0 (against the answers\' 1)')
    ap.add_argument('--w-geo', type=float, default=0.0,
                    help='the weight of the soft targets: the windows of the ignore zone at the right rotation are not '
                         'left out of the softmax but get a target that falls with their distance from the truth '
                         '(--soft-target). 0: off, they are ignored and the loss is the one-hot cross-entropy')
    ap.add_argument('--w-corr', type=float, default=0.1)
    ap.add_argument('--w-cycle', type=float, default=0.1)
    ap.add_argument('--recon', choices=('', 'ae', 'vae'), default='',
                    help='a decoder of the same shape (a linear map back) on each projection that NARROWS (W_in: the '
                         'feature width -> d; GloRet\'s W_q, W_k: d -> d_r), with a loss that makes it keep what it '
                         'dropped: \'ae\' the cosine of the rebuilt input to the input, \'vae\' also a KL. Where nothing '
                         'narrows (d_r = d) the loss does not exist. Off unless --w-recon or --w-kl is above 0')
    ap.add_argument('--w-recon', type=float, default=0.0, help='the weight of the reconstruction loss (about 0.1)')
    ap.add_argument('--w-kl', type=float, default=0.0, help='the weight of the KL with --recon vae (small, about 1e-3)')
    # optimisation
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=3e-4, help='the learning rate for ONE process (see --lr-scale)')
    ap.add_argument('--lr-scale', choices=('none', 'sqrt', 'linear'), default='sqrt')
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--clip', type=float, default=1.0)
    # run
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto', help="'auto': the GPU, or stop if the job was given one that does not work")
    ap.add_argument('--tag', default='', help='added to the weights\' data segment (a smoke run\'s, so it overwrites nothing)')
    ap.add_argument('--out', default=None, help='the results root the weights go under (default: the results root)')
    ap.add_argument('--max-visits', type=int, default=0, help='stop an epoch after N visits (a smoke run)')
    ap.add_argument('--dry-run', action='store_true', help='print the plan; open and write nothing')
    ap.add_argument('--wandb-project', default='gloret')
    ap.add_argument('--wandb-mode', default=os.environ.get('WANDB_MODE', 'online'))
    ap.add_argument('--run-name', default='')
    return ap


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def slides_of(datasets: Sequence[str], n: int) -> List[Tuple[str, str, str]]:
    """`(dataset, name, path)` of the slides each AccessDatasets id holds, by the dataset each comes from, the first
    `n` of each (`by_source`; 0 = all)."""
    out: List[Tuple[str, str, str]] = []
    for dataset_id in datasets:
        for source, names in by_source(dataset_id, n or None):
            out += [(source, name, str(locate(name, dataset=source).path)) for name in names]
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  the visits, made ahead of the training
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Visit:
    """A level of one slide and the FoVs placed on it."""
    key: Tuple[str, int]
    level: Level
    in_dim: int = 0
    cache: Dict[Any, Any] = field(default_factory=dict)      # what the steps compute once per FoV of the visit (marks, items)


def make_visits(stream: Iterator[FovResult], pooling: str, token_dim: int = 0, expand: float = 0.5,
                wrong_rotation_ignored: bool = True, with_list: bool = True) -> Iterator[Visit]:
    """The FoVs of `stream` (a level of a slide at a time) as visits. A `FovResult` is only valid until the next is
    asked for, so each is turned into a `Fov` at once. `token_dim`: the width of a patch token, for the grids that
    carry a tile's tokens flattened (`attnpool`); the network is built for it."""
    current: Optional[Visit] = None
    for res in stream:
        key: Tuple[str, int] = (res.slide, res.level)
        if current is None or key != current.key:
            if current is not None:
                yield current
            current = Visit(key=key, level=level_of(res, pooling))
        t0: float = time.perf_counter()
        fov = fov_of(res, expand, wrong_rotation_ignored, current.level, with_list)
        tick('prepare', time.perf_counter() - t0)
        current.level.fovs.append(fov)
        width: int = int(fov.queries[0].shape[-1])
        if token_dim and width % token_dim:
            raise ValueError(f'a tile\'s grid entry has {width} numbers, not a whole number of tokens of {token_dim}')
        current.in_dim = token_dim or width
    if current is not None:
        yield current


def make_lazy_visits(stream: Iterator[FovResult], token_dim: int, device: Any, expand: float = 0.5,
                     wrong_rotation_ignored: bool = True) -> Iterator[Visit]:
    """Stage B's visits: a level's geometry (no features) and its FoVs. The tokens of the tiles under a window are read when
    the window is asked for (`Lazy.LazyLevel`), so the steps, the ranking and the choosing of a visit are those of any
    level (`Episodes.run_visit`). Each FoV carries the first phase's scores of the level's windows (the teacher's) when its
    first phase made them."""
    current: Optional[Visit] = None
    for res in stream:
        key: Tuple[str, int] = (res.slide, res.level)
        if current is None or key != current.key:
            if current is not None:
                yield current
            current = Visit(key=key, level=lazy_level_of(res, device), in_dim=token_dim)
        current.level.fovs.append(fov_of(res, expand, wrong_rotation_ignored, current.level))
    if current is not None:
        yield current


def prefetch(items: Iterator[Visit], depth: int, device: Any) -> Iterator[Visit]:
    """`items`, made by a thread that stays up to `depth` ahead of whoever iterates (depth 0: no thread). An exception
    in the thread is raised where the iterating is; stopping early stops the thread."""
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

def relabelled(visit: Visit, expand: float, wrong_ignored: bool) -> List[Fov]:
    """The visit's FoVs with another ignore zone (`--ignore-expand`) and wrong-rotation rule (`--wrong-rotation`): the same
    photos, lists and teacher scores, other labels. Made once and kept in the visit's cache (the steps' items are kept
    there by the FoV, which must stay the same object)."""
    key: Tuple[str, float, bool] = ('relabelled', float(expand), bool(wrong_ignored))
    if key not in visit.cache:
        out: List[Fov] = []
        for f in visit.level.fovs:
            rows, cols = f.shape(f.placed.rot)
            tile: float = visit.level.tile_l0
            out.append(replace(f, placed=replace(f.placed, zone=(expand * cols * tile, expand * rows * tile),
                                                 wrong_rotation_ignored=bool(wrong_ignored))))
        visit.cache[key] = out
    return visit.cache[key]


def train_visit(runs: List[HeadRun], visit: Visit, epoch: int, args: argparse.Namespace, rng: np.random.Generator,
                cfg: StepCfg) -> Dict[str, float]:
    """`--steps-per-visit` steps of every head on one visit's FoVs, each FoV's softmax over its pool of items
    (`--top-p`): the windows of the level or the entries of the first phase's list. See `Episodes.py`; the pools are
    scored by a gradient cache, and shared between the processes of a `--window-split` group. A head's own flags
    (`HEAD_FLAGS`) are its `args`; a head with other labels trains on the visit's FoVs relabelled."""
    logs: Dict[str, List[float]] = {}
    sync = (lambda m: sync_gradients(m, True)) if args.world > 1 else None
    for run in runs:
        assert run.model is not None and run.opt is not None and run.top is not None
        a: argparse.Namespace = run.args if run.args is not None else args
        fovs = visit.level.fovs if run.relabel is None else relabelled(visit, *run.relabel)
        memories: Dict[int, Memory] = {}                     # a head's hard negatives of each FoV, for this visit
        pc: PoolCfg = PoolCfg(top=run.top, min_candidates=a.min_candidates, episodes=a.episodes,
                              warmup_epochs=a.warmup_epochs, steps=a.steps_per_visit, fovs_per_step=a.fovs_per_step, max_fov_uses=a.max_fov_uses)
        terms = run_visit(run.arch, run.items, run.model, run.opt, visit.level, fovs, epoch, pc,
                          FullCfg(chunk=a.chunk, gradcache=not a.no_gradcache),
                          RerankCfg(chunk=a.pair_chunk, gradcache=not a.no_gradcache), run.step or cfg, rng, visit.cache,
                          a.clip, sync, WINDOW_SPLIT, memories)
        if terms is None:                      # no FoV with an answer: with others, the steps with no vote
            for _ in range(a.steps_per_visit if args.world > 1 else 0):
                idle_step(run.model, run.opt, a.clip, lambda m: sync_gradients(m, False))
            continue
        for name, v in terms.items():
            logs.setdefault(f'{run.name}/{name}', []).append(v)
    if visit.level.lazy:
        visit.level.release()                              # type: ignore[attr-defined]  # the tokens kept for the heads' steps
    return {k: float(np.mean(v)) for k, v in logs.items()}


def run_epoch(epoch: int, runs: List[HeadRun], slides: List[Tuple[str, str, str]], args: argparse.Namespace,
              make_retriever: Any, make_lister: Optional[Any], fov: Any, device: Any, rng: np.random.Generator,
              workers: int, cfg: StepCfg) -> Dict[str, float]:
    """One pass over the training slides, a visit at a time."""
    sums: Dict[str, List[float]] = {}
    visits_done: int = 0
    # the visits are made by a thread that runs ahead of the training (encoding a level is minutes, a visit's steps
    # are seconds), so with several processes a step waits for the slowest one's SUM of encodings
    listed: bool = any(r.items == 'list' for r in runs)     # only these read the first phase's list
    stream = fov_stream(slides, make_retriever, fov, args.mask_cache_job, device, args.fov_per_visit, args.fov_seed,
                        epoch, workers=workers, topk=args.raw_topk if listed else 1, say=lambda text: say('  ' + text),
                        make_lister=make_lister, min_ds=args.min_ds)
    if args.on_demand:         # stage B: only the tiles under the windows a visit scores are encoded, once, for its steps
        made: Iterator[Visit] = make_lazy_visits(stream, args.token_dim, device, args.ignore_expand,
                                                 args.wrong_rotation == "ignore")
    else:
        made = make_visits(stream, args.pooling, args.token_dim, args.ignore_expand,
                           args.wrong_rotation == "ignore", listed)
    visits: Iterator[Visit] = iter(prefetch(made, args.prefetch, device))
    while True:
        t0: float = time.perf_counter()
        visit: Optional[Visit] = next(visits, None)
        tick('wait_for_a_visit', time.perf_counter() - t0)
        if visit is None:
            break
        for run in runs:                         # the model is built from the first FoV it sees
            ensure_model(run, args, visit.in_dim, device)
        t0 = time.perf_counter()
        if args.world > 1:
            agree_models(runs, args, device)
            all_sum(1.0, device)                 # this process has a visit: the rounds are counted by all of them
        tick('wait_for_the_others', time.perf_counter() - t0)
        on_gpu: bool = torch.device(device).type == 'cuda'
        if on_gpu:
            begin_visit_peak(device)       # the peak of THIS visit (the next visit's levels are encoded meanwhile, on this card)
        t0 = time.perf_counter()
        logs: Dict[str, float] = train_visit(runs, visit, epoch, args, rng, cfg)
        tick('train_steps', time.perf_counter() - t0)
        for k, v in logs.items():
            sums.setdefault(k, []).append(v)
        visits_done += 1
        visit_peak: Optional[float] = visit_peak_gb(device) if on_gpu else None
        if visit_peak is not None:
            GPU_PEAK['so_far'] = max(GPU_PEAK['so_far'], visit_peak)
        say(visit_report(visits_done, visit, logs, [r.name for r in runs], time.perf_counter() - t0, visit_peak),
            flush=True)
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


def visit_report(n: int, visit: Any, logs: Dict[str, float], heads: Sequence[str], seconds: float,
                 gpu_peak: Optional[float]) -> str:
    """The log of one visit, a few short lines: the slide, the level and the FoVs, then one line a head (its loss on the list and
    the seconds its steps took), then what the process holds. A head is named by its family (`GloRet`, `LogLoRet`) when
    that tells the heads apart, else by its whole name."""
    family: List[str] = [h.split('-')[0] for h in heads]
    label: Dict[str, str] = {h: (f if family.count(f) == 1 else h) for h, f in zip(heads, family)}
    width: int = max(len(v) for v in label.values())
    out: List[str] = [f'  visit {n:4d}  {visit.key[0]}  level {visit.key[1]}  {len(visit.level.fovs)} FoVs   '
                      f'{seconds:.0f} s for the steps']
    for h in heads:
        got: Dict[str, float] = {k.rsplit('/', 1)[1]: v for k, v in logs.items() if k.startswith(h + '/')}
        if not got:
            continue
        parts: List[str] = []
        if 'list' in got:
            parts.append(f'list loss {got["list"]:.3f}')
        if 'windows' in got:
            parts.append(f'softmax over {got["windows"]:.0f} windows')
        seconds: List[str] = []
        for k in ('t_pool', 't_scores', 't_keys', 't_loss', 't_back'):
            if k not in got:
                continue
            seconds.append(f'{k[2:]} {got[k]:.2f}')
            if k == 't_pool' and 't_pool_score' in got:         # the pool split: scoring the items (all, first, last), the rest is choosing
                seconds[-1] += (f' [scoring {got["t_pool_score"]:.2f}: first {got["t_pool_first"]:.2f}, last {got["t_pool_last"]:.2f};'
                                f' rest {got["t_pool"] - got["t_pool_score"]:.2f}]')
        parts.append('seconds: ' + ' '.join(seconds))
        out.append(f'      {label[h]:<{width}}  ' + ',  '.join(parts))
        if 'ahead' in got:
            out.append(f'      {"":<{width}}  ' + hard_line(got))
        if any(k.startswith('m_') for k in got):
            out.append(f'      {"":<{width}}  ' + memory_line(got))
    out.append('      process  ' + (f'GPU peak this visit {gpu_peak:.1f} GB (so far {GPU_PEAK["so_far"]:.1f}),  '
                                    if gpu_peak is not None else '') + host_memory())
    return '\n'.join(out)


def timing_line(who: str, wall: float, timers: Dict[str, float]) -> str:
    """Where `wall` seconds went (this process), from the timers: what making the visits took (the thread that makes
    them runs beside the training), and what the training waited for."""
    t = lambda name: timers.get(name, 0.0)                                           # noqa: E731
    tiles: float = t('level_tiles')
    encode: str = (f'{t("level_encode"):.0f} s for {tiles / 1e6:.2f}M tiles ({tiles / max(t("level_encode"), 1e-9):.0f}/s)'
                   if tiles else 'none')
    return (f'  time {who}: {wall:.0f} s | making visits: levels encoded/read back {encode}; first photo waits '
            f'{t("photo_wait"):.0f} s; search {t("retrieve") - t("level_encode"):.0f} s; slides opened+masks '
            f'{t("open_and_mask"):.0f} s; FoV draw {t("draw_positions"):.0f} s; tensors {t("prepare"):.0f} s | '
            f'training: waited for a visit {t("wait_for_a_visit"):.0f} s, for the others {t("wait_for_the_others"):.0f} s, '
            f'steps {t("train_steps"):.0f} s') + "".join(f" | {k} {v:.0f} s" for k, v in timers.items() if k.startswith(("rank ", "save")))


@torch.no_grad()
def validate(runs: List[HeadRun], slides: List[Tuple[str, str, str]], args: argparse.Namespace,
             make_retriever: Any, make_lister: Optional[Any], fov: Any, device: Any,
             workers: int, stream: Optional[Iterator[FovResult]] = None) -> Dict[str, Dict[str, float]]:
    """`{name: recall figures}` for the list (`SlidingWindow`; with another retriever's features, also that
    retriever's own, `SlidingWindow-features`) and each head, on the held-out FoVs (or on the `FovResult`s of `stream`:
    the real photos, `RealPhotos.real_stream`)."""
    ranks: Dict[str, List[float]] = {'SlidingWindow': []}
    own_list: bool = make_lister is not None and args.pooling not in (PATCH_POOLING, FROZEN_POOLING)   # tokens rank nothing
    if own_list:
        ranks['SlidingWindow-features'] = []
    for run in runs:
        ranks[run.name] = []
        if run.model is not None:
            run.model.eval()
        run.encodings = None
    last_slide: Optional[str] = None
    # the list is cut at 500: GloRet's R@500, and the first phase's own figures to the same k
    if stream is None:
        stream = fov_stream(slides, make_retriever, fov, args.mask_cache_job, device, args.val_fov_per_visit,
                            args.val_fov_seed, 0, workers=workers, topk=max(500, args.val_keep),
                            say=lambda text: say('  val ' + text), make_lister=make_lister, own_topk=500,
                            min_ds=args.min_ds)
    for res in stream:
        ranks['SlidingWindow'].append(upstream_rank(res))
        if own_list:
            ranks['SlidingWindow-features'].append(own_rank(res))
        for run in runs:
            if run.model is None:
                continue
            ranked_at: float = time.perf_counter()
            # the ROLE (how it was trained) says what is ranked; the architecture only how its scores are made
            if run.role == 'retriever':                    # the network lists the level's windows alone
                if run.arch == 'logloret':
                    # a pair each, every window of the level (a level's windows at four rotations), for the share of the FoVs
                    # `--val-logloret-share` asks (a fixed pick by the FoV's name, the same at every epoch)
                    if zlib.crc32(f'{res.slide}|{res.level}|{res.index}'.encode()) % 1000 < args.val_logloret_share * 1000:
                        ranks[run.name].append(logloret_level_rank(run.model, res, args.pooling, args.pair_chunk, SELECT_K['retriever']))
                else:
                    if run.encodings is None:
                        # a band of rows of windows at a time: a tile's tokens are 0.6 MB, a window of 20 tiles 12 MB
                        run.encodings = WindowEncodings(run.model, device, 1 if args.pooling == PATCH_POOLING else 16, True)
                    if res.slide != last_slide:
                        run.encodings.reset()          # a new slide: the windows of the last are not its
                    ranks[run.name].append(gloret_rank(run.encodings, res, SELECT_K['retriever']))
            elif run.arch == 'gloret':                     # a reranker re-orders the list
                ranks[run.name].append(gloret_rerank_rank_lazy(run.model, res, res.tiles, args.val_keep) if args.on_demand
                                       else gloret_rerank_rank(run.model, res, args.pooling, args.val_keep))
            else:
                ranks[run.name].append(logloret_rank_lazy(run.model, res, res.tiles, args.val_keep) if args.on_demand
                                       else logloret_rank(run.model, res, args.pooling, args.val_keep))
            tick(f'rank {run.name}', time.perf_counter() - ranked_at)
        last_slide = res.slide
    if args.world > 1:                                    # the slides were shared out: every FoV counts once
        parts: List[Dict[str, List[float]]] = gather_objects(ranks, args.world)
        ranks = {name: [r for part in parts for r in part[name]] for name in ranks}
    ks = tuple(sorted(set(KS) | {args.val_keep}))
    return {name: recall_at(r, ks) for name, r in ranks.items() if r}


def head_settings(runs: List[HeadRun], args: argparse.Namespace) -> None:
    """Give each head its own `args` (the job's with the flags it was given after its name applied, each read as the flag
    reads its value), its `step` (loss settings) and its `relabel` (its own ignore zone and wrong-rotation rule, when they
    are not the job's)."""
    actions: Dict[str, argparse.Action] = {a.dest: a for a in build_parser()._actions}
    for r in runs:
        own: argparse.Namespace = copy.copy(args)
        for key, text in r.flags.items():
            action = actions[key.replace('-', '_')]
            try:
                value: Any = action.type(text) if action.type is not None else text
            except (TypeError, ValueError):
                raise SystemExit(f'{r.name}: @{key}={text} is not a value of --{key}')
            if action.choices is not None and value not in action.choices:
                raise SystemExit(f'{r.name}: @{key}={text} is not one of {list(action.choices)}')
            setattr(own, key.replace('-', '_'), value)
        r.args = own
        r.step = StepCfg(sigma=own.sigma, w_geo=own.w_geo, near_alpha=own.near_alpha, soft_target=own.soft_target,
                         w_corr=own.w_corr, w_cycle=own.w_cycle, chunk=own.chunk, w_recon=own.w_recon, w_kl=own.w_kl)
        mine: Tuple[float, bool] = (float(own.ignore_expand), own.wrong_rotation == 'ignore')
        r.relabel = None if mine == (float(args.ignore_expand), args.wrong_rotation == 'ignore') else mine
        if args.on_demand and r.relabel is not None:
            raise SystemExit(f'{r.name}: --on-demand fixes the labels with the tokens it reads: no @ignore-expand / @wrong-rotation')


def main() -> int:
    global RANK
    RANK = int(os.environ.get('RANK', '0'))        # torchrun's, known before the process group is joined
    args: argparse.Namespace = parse_args()
    runs: List[HeadRun] = [parse_head(h) for h in args.heads]
    rank, world, local = dist_init()
    args.rank, args.world = rank, world
    G: int = max(args.window_split, 1)
    if world % G:
        raise SystemExit(f'--window-split {G}: the {world} processes do not make groups of {G}')
    args.dp = world // G                        # processes that see other slides (a group sees one)
    gid: int = rank // G
    global WINDOW_SPLIT
    WINDOW_SPLIT = make_split(world, rank, G)
    if world > 1:
        split_output(os.environ.get('SLURM_JOB_NAME', 'GloRet'), rank)
    fov = fov_from_args(args)
    fov = replace(fov, sampler=replace(fov.sampler, seed=args.fov_seed, n_per_rung=args.fov_per_visit))   # the recipe draws as many FoVs a level as a visit takes
    val_base = FOV_RECIPES[args.val_fov] if args.val_fov else fov
    val_fov = replace(val_base, sampler=replace(val_base.sampler, seed=args.val_fov_seed, n_per_rung=args.val_fov_per_visit))
    cfg: StepCfg = StepCfg(sigma=args.sigma, w_geo=args.w_geo, near_alpha=args.near_alpha, soft_target=args.soft_target,
                           w_corr=args.w_corr, w_cycle=args.w_cycle, chunk=args.chunk, w_recon=args.w_recon, w_kl=args.w_kl)
    try:
        share_top, count_top = parse_top_p(args.top_p)
    except ValueError as exc:
        raise SystemExit(str(exc))
    for r in runs:                                # a head's own @top=, else the job's: a share for a head over the level, a count for one over the list
        if r.top is None:
            r.top = share_top if r.items == 'level' else count_top
        if r.top is None:
            raise SystemExit(f'--top-p: {r.name} needs a '
                             + ('share (a number with a point: 0.05, 1.0)' if r.items == 'level' else 'count (an integer: 2000)')
                             + f', or its own @top= ({r.name}@top=...)')
    head_settings(runs, args)
    if args.chunk < 1 or args.pair_chunk < 1 or args.episodes < 1 or args.min_candidates < 0:
        raise SystemExit('--chunk, --pair-chunk and --episodes are at least 1, --min-candidates 0 or more')
    for r in runs:
        if r.items == 'list' and r.top is not None and r.top.count is not None and r.top.count > args.raw_topk:
            raise SystemExit(f'--top-p {r.top.count} of {r.name} is more than the list: --raw-topk is {args.raw_topk}')
        if args.on_demand and r.items != 'list':
            raise SystemExit('--on-demand is for heads that re-order the list (items list)')
    if args.val_keep > args.raw_topk:
        raise SystemExit(f'--val-keep {args.val_keep} reads the first phase\'s list, which --raw-topk {args.raw_topk} makes: '
                         'raise it')
    # the sliding window's baseline settings on an encoder and pooling, every field written: the one that makes the
    # LIST (`--list-encoder`, `--list-pooling`: by default the network's own features) and, when another, the one
    # whose features the network reads
    def sliding(encoder: str, pooling: str, raw_topk: int) -> RetrieverSlidingWinSimRotConfig:
        return RetrieverSlidingWinSimRotConfig(
            encoder=replace(encoder_config(encoder, pooling=pooling).with_model(dtype='fp16'), batch_size=args.encoder_batch),
            tile_size=256, overlap=True, raw_topk=raw_topk, min_sep_tiles=1.0, score='mean')

    list_encoder: str = args.list_encoder or args.encoder
    raw_tokens: bool = args.pooling == PATCH_POOLING        # the grids hold a tile's patch tokens (stages A, B)
    patches: bool = raw_tokens or args.pooling == FROZEN_POOLING   # the features come from patch tokens: no sliding window reads them
    list_pooling: str = args.list_pooling or ('cls' if patches else args.pooling)
    if list_pooling == PATCH_POOLING:
        raise SystemExit('--list-pooling attnpool: a sliding window ranks windows by a pooled vector, not by patch tokens')
    separate: bool = patches or (list_encoder, list_pooling) != (args.encoder, args.pooling)
    args.token_dim = TOKEN_DIM[args.encoder] if raw_tokens else 0
    if args.on_demand and not raw_tokens:
        raise SystemExit('--on-demand is stage B: --pooling attnpool (the tokens of a tile are what does not fit a level)')
    if args.pooling == FROZEN_POOLING and not args.pool_weights:
        raise SystemExit('--pooling attnpool_frozen pools with the attention pool of a trained network: give --pool-weights')
    list_cfg: RetrieverSlidingWinSimRotConfig = sliding(list_encoder, list_pooling, args.raw_topk)
    retr_cfg: Optional[RetrieverSlidingWinSimRotConfig] = (
        None if patches else sliding(args.encoder, args.pooling, 1000) if separate else list_cfg)
    train_slides: List[Tuple[str, str, str]] = slides_of(args.train_datasets, args.n_train_wsi)
    val_slides: List[Tuple[str, str, str]] = slides_of(args.val_datasets, args.n_val_wsi)
    say(f'heads      : {" ".join(r.shown for r in runs)}   features {args.encoder}/{args.pooling}   list: sliding window '
        f'{list_encoder}/{list_pooling}, {args.raw_topk} windows; --top-p {" ".join(args.top_p)}', console=True)
    for r in runs:
        assert r.top is not None
        say(f'  {r.name}: items {r.items}, role {r.role} (selected on R@{SELECT_K[r.role]}), top-p '
            f'{r.top.share if r.top.share is not None else r.top.count}', console=True)
    say(f'train      : {len(train_slides)} slides of {" ".join(args.train_datasets)}   '
        f'{args.fov_per_visit} FoVs a visit, {args.steps_per_visit} steps of {args.fovs_per_step}', console=True)
    say(f'val        : {len(val_slides)} slides of {" ".join(args.val_datasets)}   '
        f'{args.val_fov_per_visit} FoVs a visit', console=True)
    if args.dry_run:
        for r in runs:
            say(f'  {r.name}: best -> {weights_file(args, r, "best")}', console=True)
        say('dry run: nothing opened, nothing written', console=True)
        return 0

    torch.manual_seed(args.seed)                  # the same on every process: the same first weights
    rng: np.random.Generator = np.random.default_rng(args.seed + 1000 * gid)   # this process's choices (a group makes the same)
    cuda: str = f'cuda:{local}' if world > 1 else 'cuda'
    device: torch.device = torch.device((cuda if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    require_gpu_if_allocated(device, args.device)
    args.device = str(device)                     # the rest reads the resolved name
    budget = CpuBudget.for_job(processes=int(os.environ.get('LOCAL_WORLD_SIZE', world))).apply()
    say(f'device     : {device}   rank {rank} of {world}   {budget.line()}', console=True, flush=True)
    # one meter per node (its first process): the counters of the node's GPUs, the CPUs of the job and its memory
    # and, every 10 minutes, the figures of those minutes in this process's log (a long first epoch, or a crash in
    # it, is seen without waiting for the epoch's end)
    def show_usage(s: Dict[str, Any]) -> None:
        say('  ⏱ last 10 min · ' + usage_line(s), flush=True)

    meter: Optional[UsageMeter] = (UsageMeter(args.usage_interval, 600.0, show_usage).start()
                                   if args.usage_interval > 0 and local == 0 else None)
    stores: Dict[str, MemoryFeatureStore] = {}
    list_stores: Dict[str, MemoryFeatureStore] = {}
    card_budget: Optional[CardBudget] = CardBudget(int(args.gpu_levels_gb * 1e9)) if args.gpu_levels_gb > 0 else None
    profile_start(os.path.join(args.out or RESULT_DIR) if args.profile and rank == 0 else None)

    encoders: EncoderPool = EncoderPool(device)        # one copy of each encoder, shared by the retrievers of a process
    # stage C: the attention pool of a trained network, frozen; a tile is pooled as it is encoded and the level kept
    frozen_pool: Optional[torch.nn.Module] = load_pool(args.pool_weights, device) if args.pooling == FROZEN_POOLING else None

    def make_retriever(slide: str) -> Any:
        store: Optional[MemoryFeatureStore] = stores.setdefault(slide, MemoryFeatureStore(card_budget)) if args.keep_levels else None
        if patches:                                    # the tile encoder read as patch tokens (Patches.py)
            tile_encoder = encoder_config(args.encoder, pooling='cls').with_model(dtype='fp16')
            return PatchSource(lambda: encoders.get(tile_encoder), device, read_workers=budget.workers,
                               feature_store=store, pool=frozen_pool, lazy=args.on_demand)
        return RetrieverSlidingWinSimRot(retr_cfg, device, encoders=encoders, read_workers=budget.workers,
                                         feature_store=store)

    def make_lister(slide: str) -> RetrieverSlidingWinSimRot:
        store: Optional[MemoryFeatureStore] = (list_stores.setdefault(slide, MemoryFeatureStore(card_budget))
                                               if args.keep_levels else None)
        return RetrieverSlidingWinSimRot(list_cfg, device, encoders=encoders, read_workers=budget.workers,
                                         feature_store=store)

    lister: Optional[Any] = make_lister if separate else None

    # the real photos read every epoch (monitoring only): the confirmed photos of the real-photo slides, a seeded draw; a
    # process takes the slides whose position in the list is its own
    real_photos: List[RealPhoto] = []
    real_paths: Dict[str, str] = {}
    if args.real_photos:
        real_slides: List[Tuple[str, str, str]] = slides_of([args.real_dataset], 0)
        real_paths = {name: path for _, name, path in real_slides}
        real_photos = choose_photos([name for _, name, _ in real_slides], args.real_photos, args.real_seed, job=args.real_job,
                                    checklists=args.real_checklists, stage1=args.real_stage1, stage2=args.real_stage2,
                                    dataset=args.real_dataset)
        per_slide: Dict[str, int] = {}
        for p in real_photos:
            per_slide[p.slide] = per_slide.get(p.slide, 0) + 1
        say(f'real photos: {len(real_photos)} confirmed photos of {len(per_slide)} slides (monitoring only)   '
            + ' '.join(f'{s.split(",")[0]}:{n}' for s, n in sorted(per_slide.items())), console=True)
        if rank == 0:
            write_list(real_photos, os.path.join(args.out or RESULT_DIR, 'RealPhotos', f'seed{args.real_seed}_n{args.real_photos}.tsv'))
    real_slide_of: Dict[str, int] = {s: i % world for i, s in enumerate(sorted({p.slide for p in real_photos}))}

    # the slides this process trains on, every epoch: every `world`-th of the list by name
    my_slides: List[Tuple[str, str, str]] = sorted(train_slides, key=lambda s: s[1])[gid::args.dp]
    # one wandb run per arm: a formal run first deletes the earlier runs of its name (not one still running); a smoke
    # run (stopped after a few visits) has a name of its own and deletes nothing
    # (the name holds the pooling and the tag: the stages of the `attnpool` arm train the same heads on other grids, and
    # one stage must not delete another's run)
    arm_name: str = (args.run_name or '-'.join([args.encoder, args.pooling, *(r.shown for r in runs)]
                                                + ([args.tag] if args.tag else []))) \
        + ('-smoke' if args.max_visits else '')
    wb = (wandb_init(args.wandb_project, args.wandb_mode, arm_name, config=plain_args(args),
                     replace_previous=not args.max_visits) if rank == 0 else None)
    started: float = time.perf_counter()
    for epoch in range(args.epochs):
        say('\n' + rule(f'EPOCH {epoch + 1}/{args.epochs}'), console=True, flush=True)
        epoch_started: float = time.perf_counter()
        mine: List[Tuple[str, str, str]] = [my_slides[i] for i in
                                            np.random.default_rng([args.seed, epoch, gid]).permutation(len(my_slides))]
        fov_epoch = replace(fov, sampler=replace(fov.sampler, seed=args.fov_seed + epoch)) if args.new_fovs_each_epoch else fov
        losses: Dict[str, float] = run_epoch(epoch, runs, mine, args, make_retriever, lister, fov_epoch, device, rng,
                                             budget.workers, cfg)
        names: List[str] = [r.name for r in runs]
        lab: Dict[str, str] = labels(names)
        train_s: float = time.perf_counter() - epoch_started
        say(rule('TRAINING', f'{train_s:.0f} s'), console=True)
        for line in timing_lines(train_s, take_timers(), lab):
            say(line, console=True)
        for run in runs:
            if run.sched is not None:
                run.sched.step()
        metrics: Dict[str, float] = dict(losses)
        # every term of a head's loss as it is BEFORE its weight (`raw`), the weight, and weight x raw (`weighted`), so that the
        # weights can be set to make the terms of one size: `geo` is the near windows' cross-entropy (gaussian: per unit of
        # target), `loss` the combined loss the step minimises (not the sum of the weighted terms)
        loss_weights: Dict[str, Dict[str, float]] = {}
        for run in runs:
            own_args: argparse.Namespace = run.args if run.args is not None else args
            loss_weights[run.name] = {'geo': own_args.w_geo, 'corr': own_args.w_corr, 'cycle': own_args.w_cycle,
                                      'recon': own_args.w_recon, 'kl': own_args.w_kl}
            for term in ('list', 'geo', 'corr', 'cycle', 'recon', 'kl', 'loss'):
                raw: Optional[float] = losses.get(f'{run.name}/{term}')
                if raw is None:
                    continue
                metrics[f'loss_raw/{run.name}/{term}'] = raw
                if term in loss_weights[run.name]:
                    metrics[f'loss_weighted/{run.name}/{term}'] = loss_weights[run.name][term] * raw
        for line in losses_block(names, losses, loss_weights) + steps_block(names, losses):
            say(line, console=True)
        for line in hard_negatives_block(names, losses):
            say(line, console=True)
        val_started: float = time.perf_counter()
        figures: Dict[str, Dict[str, float]] = validate(runs, val_slides[rank::world], args, make_retriever, lister,
                                                        val_fov, device, budget.workers)
        val_s: float = time.perf_counter() - val_started
        say('\n' + rule(f'VALIDATION   FoVs "{args.val_fov or args.fov}" of the held-out slides', f'{val_s:.0f} s'), console=True)
        for line in timing_lines(val_s, take_timers(), lab):
            say(line, console=True)
        kept: List[MemoryFeatureStore] = [*stores.values(), *list_stores.values()]
        say(f'    levels kept           {sum(s.nbytes() for s in kept) / 1e9:.1f} GB, {sum(s.card_bytes() for s in kept) / 1e9:.1f} GB of it on the card   '
            f'({sum(s.misses for s in stores.values())} encoded, {sum(s.hits for s in stores.values())} read back)')
        selected: Dict[str, int] = {r.name: SELECT_K[r.role] for r in runs}
        for line in recall_block('share of the FoVs whose first hit is in the top k', figures, KS, selected):
            say(line, console=True)
        for name, f in figures.items():
            metrics.update({f'val/{name}/{key}': v for key, v in f.items()})
        real_started_at: float = time.perf_counter()
        real_s: float = 0.0
        if real_photos:
            # the real photos: the same ranks, on the confirmed photos; printed and logged, never selected on (the SlidingWindow
            # row is the baseline: the confirmed places were found by a sliding window, so it is favoured)
            real_started: float = time.perf_counter()
            mine_real: List[RealPhoto] = [p for p in real_photos if real_slide_of[p.slide] == rank]
            real_figures: Dict[str, Dict[str, float]] = validate(
                runs, [], args, make_retriever, lister, val_fov, device, budget.workers,
                stream=real_stream(mine_real, real_paths, make_retriever, args.mask_cache_job, device, budget.workers,
                                   say=lambda text: say('  real ' + text), topk=max(500, args.val_keep),
                                   make_lister=lister, own_topk=500))
            real_s = time.perf_counter() - real_started
            say('\n' + rule(f'REAL PHOTOS   {len(real_photos)} confirmed photos (monitoring only, never selected on)',
                            f'{real_s:.0f} s'), console=True)
            for line in timing_lines(real_s, take_timers(), lab):
                say(line, console=True)
            for line in recall_block('share of the confirmed photos whose place is in the top k', real_figures, KS, {}):
                say(line, console=True)
            for name, f in real_figures.items():
                metrics.update({f'val_real/{name}/{key}': v for key, v in f.items()})
        select_started: float = time.perf_counter()
        ceiling100: float = figures.get("SlidingWindow", {}).get(f"R@{args.val_keep}", float("nan"))
        picked: List[List[str]] = []
        notes: List[str] = []
        for run in runs:
            if run.name not in figures or run.model is None:
                continue
            k: int = SELECT_K[run.role]
            score: float = figures[run.name][f'R@{k}']
            if run.role == 'reranker':
                notes.append(f'    {lab[run.name]}: conditional R@10 = R@10 / R@{args.val_keep} of the list = '
                             f'{score:.3f} / {ceiling100:.3f} = {score / ceiling100 if ceiling100 else float("nan"):.3f}')
            better: bool = score > run.best
            for which in (('last',) + (('best',) if better else ())) if rank == 0 else ():
                save_rerank_checkpoint(
                    weights_file(args, run, which), model=run.model, arch=run.arch, pooling=args.pooling,
                    encoder=args.encoder, epoch=epoch + 1, val=figures[run.name], run_args=plain_args(args),
                    extra={'method': run.method, 'op': run.op, 'name': run.name, 'pool_weights': args.pool_weights,
                           'items': run.items, 'role': run.role,
                           'top_p': run.top.share if run.top is not None and run.top.share is not None
                           else (run.top.count if run.top is not None else None)},
                    optimizer_state=run.opt.state_dict() if run.opt is not None else None)
            if better:
                run.best, run.best_epoch = score, epoch + 1
            picked.append([lab[run.name], f'R@{k}', num(score), num(run.best), str(run.best_epoch),
                           'best + last' if better else 'last'])
        say('\n' + rule('SELECTION'), console=True)
        for line in table(['head', 'selected on', 'this epoch', 'best', 'at epoch', 'saved'], picked):
            say(line, console=True)
        for note in notes:
            say(note, console=True)
        if epoch == 0 and runs:
            say(f'    weights in {os.path.dirname(weights_file(args, runs[0], "best"))}', console=True, flush=True)
        select_s: float = time.perf_counter() - select_started
        # what the GPUs, CPUs and memory of every node were used for since the last epoch's report
        usage: List[Optional[Dict[str, Any]]] = gather_objects(meter.summary() if meter is not None else None, world) \
            if world > 1 else [meter.summary() if meter is not None else None]
        say('\n' + rule('RESOURCES   this epoch (training, validation and real photos)'), console=True)
        for node in (u for u in usage if u is not None):
            say('    ' + usage_line(node), console=True)
            for hint in advice(node):
                say(f'      -> {hint}', console=True)
            if node['gpus']:
                metrics[f'usage/{node["host"]}/gpu_busy'] = sum(v['util_mean'] for v in node['gpus'].values()) / len(node['gpus'])
            metrics[f'usage/{node["host"]}/cpu_busy'] = node['cpu_mean']
            metrics[f'usage/{node["host"]}/mem_peak_gb'] = node['mem_peak']
        metrics.update({'time/training_s': val_started - epoch_started, 'time/validation_s': real_started_at - val_started,
                        'time/real_photos_s': real_s, 'time/selection_save_s': select_s, 'time/epoch_s': time.perf_counter() - epoch_started})
        say('\n' + rule(f'epoch {epoch + 1} took {time.perf_counter() - epoch_started:.0f} s',
                        f'job so far {time.perf_counter() - started:.0f} s'), console=True)
        say(f'    training {val_started - epoch_started:.0f} s · validation {real_started_at - val_started:.0f} s · '
            f'real photos {real_s:.0f} s · selection and save {select_s:.0f} s', console=True)
        wandb_log(wb, epoch + 1, metrics)
    wandb_finish(wb)
    if meter is not None:
        meter.stop()
    say(f'\ndone in {(time.perf_counter() - started) / 60:.1f} min   '
        + '  '.join(f'{r.name} best R@{SELECT_K[r.role]} {r.best:.3f} (epoch {r.best_epoch})' for r in runs),
        console=True)
    if WINDOW_SPLIT is not None:                      # the groups first: a group left to the end aborts the process (SIGABRT, "terminate called")
        import torch.distributed as dist                          # noqa: PLC0415
        dist.barrier()
        for group in (WINDOW_SPLIT.every or [WINDOW_SPLIT.group]):
            dist.destroy_process_group(group)
    dist_done()
    return 0


def run() -> int:
    """`main`, and with several processes an error ends THIS process at once. A process that raises is not over
    until its interpreter has shut down, and that waits on the process group the others are inside of: the others,
    held up in a collective for the dead one, are found out by the NCCL watchdog ten minutes later, with the real
    error buried (RerankHead's job 592907). `os._exit` is what torchrun can see at once, and it then ends the
    others."""
    try:
        return main()
    except Exception:                                                          # noqa: BLE001
        if int(os.environ.get('WORLD_SIZE', '1')) <= 1:
            raise
        traceback.print_exc()
        print(f'  [rank {os.environ.get("RANK", "0")}] at the error: {process_facts()}', file=sys.stderr)
        try:
            with open('/proc/self/limits') as f:
                print(f.read(), file=sys.stderr)
        except OSError:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == '__main__':
    sys.exit(run())
