#!/usr/bin/env python3
'''Train the second phase's heads on synthetic FoVs, made as they are used.

    python training/RerankHead/cli/train.py --heads tilegt corrnet winembed \\
        --retriever uni2-avg --k 500 --epochs 30 --tag run1

Several heads are trained in ONE process on the SAME data. Encoding a level of a
slide is the expensive step and nothing is kept on disk (the design,
`stage2_retrieval/spec.md`), so a head trained alone would pay for every level
again; trained together, a level is encoded once for all of them.

THE DATA (`Visits.py`): training slides are the corpora of the stage-1 routing
heads (`bracs/train`, `ki67_pure`), whose slides the evaluation sets (`bracs/test`,
`ki67_with_photo`) do not share. For each slide the retriever (a recipe of
`RetrieverSlidingWinSimRot`) is built, FoV positions are drawn over every level with a
seed of their own (`--fov-seed`), and a LEVEL AT A TIME `--fov-per-visit` of them
are photographed and put through the retriever. A visit's FoVs are trained on
`--steps-per-visit` times, `--fovs-per-step` at once, and dropped.

THE HEADS:
    tilegt, tilegt_ctx, corrnet   re-rank the retriever's candidates (K of
                                  them); the loss is over lists (Losses.py)
    winembed                      one learned vector per window, over every
                                  window of the level (Embed.py)

VALIDATION: the held-out slides' FoVs (`--val-datasets`: the set the benches use to
choose, `@legacy-val`), every candidate scored, and the share of FoVs whose first hit is in
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
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'utilities'))
from _paths import RESULT_DIR, setup_import_paths                      # noqa: E402
setup_import_paths()
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'utilities', 'bench_modules'))   # BenchCommon

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from AccessDatasets import by_source, locate                          # noqa: E402
from CpuBudget import CpuBudget                                        # noqa: E402
from BenchCommon import require_gpu_if_allocated                      # noqa: E402
from FovSupply import add_fov_args, fov_from_args                     # noqa: E402
from RerankChoices import build_rerank                                # noqa: E402
from Checkpoints import data_tag, save_rerank_checkpoint, weights_name   # noqa: E402
from stage2_retrieval.RetrieverSlidingWinSimRot import SLIDEWIN_RECIPES, RetrieverSlidingWinSimRot   # noqa: E402

from training.RerankHead.Embed import EmbedVisit, embed_loss, embed_rank, fov_of, visit_of   # noqa: E402
from training.RerankHead.Labels import LabelConfig                    # noqa: E402
from training.RerankHead.Metrics import KS, recall_at                 # noqa: E402
from training.RerankHead.Runtime import plain_args, wandb_finish, wandb_init, wandb_log   # noqa: E402
from training.RerankHead.Steps import (NEURAL_ARCHS, build_head, first_phase_rank,   # noqa: E402
                                        rank_of, train_step)
from training.RerankHead.Visits import FovData, fov_stream, prepare_fov   # noqa: E402

ARCHS: Tuple[str, ...] = (*NEURAL_ARCHS, 'winembed')


# ══════════════════════════════════════════════════════════════════════════════
#  arguments
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--heads', nargs='+', choices=ARCHS, default=['tilegt'])
    ap.add_argument('--retriever', default='uni2-avg', choices=sorted(SLIDEWIN_RECIPES),
                    help='sliding-window recipe (RetrieverSlidingWinSimRot): the encoder and the pooling the '
                         'heads read')
    ap.add_argument('--k', type=int, default=500,
                    help='the candidates a list is trained on: the first K of the retriever\'s list')
    ap.add_argument('--train-datasets', nargs='+', default=['bracs/train', 'ki67_pure'])
    ap.add_argument('--val-datasets', nargs='+', default=['@legacy-val'])
    ap.add_argument('--n-train-wsi', type=int, default=0, help='slides of each dataset a training id draws from; 0 = all')
    ap.add_argument('--n-val-wsi', type=int, default=10, help='slides of each dataset a validation id draws from')
    # --fov, --max-ds, --sampler-*, --camera-*
    add_fov_args(ap)
    ap.add_argument('--fov-seed', type=int, default=1000, help='the training draws; the benches use 0')
    ap.add_argument('--val-fov-seed', type=int, default=0)
    ap.add_argument('--mask-cache-job', default='Masks', help='where the hest masks are')
    # data per visit
    ap.add_argument('--fov-per-visit', type=int, default=64)
    ap.add_argument('--val-fov-per-visit', type=int, default=16)
    ap.add_argument('--steps-per-visit', type=int, default=16)
    ap.add_argument('--fovs-per-step', type=int, default=8)
    ap.add_argument('--n-cand', type=int, default=128, help='candidates of one FoV a step uses')
    ap.add_argument('--hard-share', nargs=2, type=float, default=[0.2, 1.0],
                    help='share of the negatives that are the hardest, first and last epoch')
    # labels
    ap.add_argument('--pos-iou', type=float, default=0.5)
    ap.add_argument('--hit-iou', type=float, default=0.5)
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
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--embed-lr', type=float, default=1e-4)
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


def ensure_model(run: HeadRun, args: argparse.Namespace, in_dim: int, n_tiles: int,
                 shapes: Sequence[Tuple[int, int]], device: Any) -> None:
    """Build the head (and its optimiser) the first time a sample of the right
    width is seen: the width is the encoder's, known once a level is encoded."""
    if run.model is not None:
        return
    if run.arch == 'winembed':
        run.model = build_rerank('winembed', dict(in_dim=in_dim, dim=args.embed_dim,
                                                  shapes=tuple(tuple(s) for s in shapes))).to(device)
        lr: float = args.embed_lr
        run.extra['shapes'] = [list(s) for s in shapes]
    else:
        run.model = build_head(run.arch, in_dim, n_tiles, args.dim, args.layers, args.attn_heads,
                               args.dropout, device)
        lr = args.lr
    run.opt = torch.optim.AdamW(run.model.parameters(), lr=lr, weight_decay=args.weight_decay)
    run.sched = torch.optim.lr_scheduler.CosineAnnealingLR(run.opt, T_max=max(args.epochs, 1))
    n_params: int = sum(p.numel() for p in run.model.parameters())
    print(f'  [{run.arch}] {n_params / 1e6:.2f}M parameters  lr {lr}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  training and validation
# ══════════════════════════════════════════════════════════════════════════════

def train_visit(runs: List[HeadRun], neural: List[FovData], embed: Optional[EmbedVisit],
                args: argparse.Namespace, rng: np.random.Generator, weights: Dict[str, float]
                ) -> Dict[str, float]:
    """`--steps-per-visit` steps of every head on one visit's FoVs."""
    logs: Dict[str, List[float]] = {}
    for run in runs:
        assert run.model is not None and run.opt is not None
        pool: int = len(embed.fovs) if run.arch == 'winembed' and embed is not None else len(neural)
        if not pool:
            continue
        for _ in range(args.steps_per_visit):
            pick: np.ndarray = rng.choice(pool, size=min(args.fovs_per_step, pool), replace=False)
            if run.arch == 'winembed':
                assert embed is not None
                run.model.train()
                run.opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=args.device.split(':')[0], dtype=torch.bfloat16):
                    loss: torch.Tensor = embed_loss(run.model, embed, [embed.fovs[i] for i in pick],
                                                    args.pos_iou, args.embed_temperature)
                loss.float().backward()
                torch.nn.utils.clip_grad_norm_(run.model.parameters(), args.clip)
                run.opt.step()
                logs.setdefault(f'{run.arch}/loss', []).append(float(loss.detach()))
            else:
                terms: Dict[str, float] = train_step(
                    run.arch, run.model, run.opt, [neural[i] for i in pick], weights, args.temperature,
                    args.topk, args.margin, args.pos_weight, args.clip)
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
    state: Dict[str, Any] = dict(key=None, neural=[], embed=None, visits=0, fovs=0)
    sums: Dict[str, List[float]] = {}

    def flush() -> None:
        neural: List[FovData] = state['neural']
        embed: Optional[EmbedVisit] = state['embed']
        if not (neural or (embed is not None and embed.fovs)):
            return
        t0: float = time.perf_counter()
        logs: Dict[str, float] = train_visit(runs, neural, embed, args, rng, weights)
        for k, v in logs.items():
            sums.setdefault(k, []).append(v)
        state['visits'] += 1
        print(f'  visit {state["visits"]:4d}  {state["key"][0]} L{state["key"][1]}  {len(neural)} lists'
              f'  {("  ".join(f"{k} {v:.3f}" for k, v in logs.items() if k.endswith(("/list", "/loss"))))}'
              f'  ({time.perf_counter() - t0:.1f} s)', flush=True)

    stream = fov_stream(slides, make_retriever, fov, args.mask_cache_job, device, label_cfg,
                        args.fov_per_visit, args.fov_seed, epoch, workers=workers, topk=args.k)
    for res in stream:
        key: Tuple[str, int] = (res.slide, res.level)
        if key != state['key']:
            flush()
            if args.max_visits and state['visits'] >= args.max_visits:
                break
            state.update(key=key, neural=[], embed=visit_of(res, pooling) if need_embed else None)
        state['fovs'] += 1
        if need_neural:
            d: Optional[FovData] = prepare_fov(res, pooling, label_cfg, args.n_cand, share, rng, need_pair)
            if d is not None:
                for run in runs:
                    if run.arch in NEURAL_ARCHS:
                        ensure_model(run, args, int(d.ni.q.shape[-1]), int(d.ni.q.shape[1]), [], device)
                state['neural'].append(d)
        if need_embed:
            for run in runs:
                if run.arch == 'winembed':
                    qg: torch.Tensor = res.queries[0]
                    shapes: List[Tuple[int, int]] = [(int(qg.shape[0]), int(qg.shape[1])),
                                                     (int(qg.shape[1]), int(qg.shape[0]))]
                    ensure_model(run, args, int(qg.shape[-1]), int(qg.shape[0] * qg.shape[1]), shapes, device)
            state['embed'].fovs.append(fov_of(res))
    else:
        flush()
    return {k: float(np.mean(v)) for k, v in sums.items()}


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
                        args.val_fov_per_visit, args.val_fov_seed, 0, workers=workers, topk=args.k)
    for res in stream:
        ranks['first_phase'].append(first_phase_rank(res))
        for run in runs:
            if run.model is None:
                continue
            if run.arch == 'winembed':
                ranks[run.arch].append(embed_rank(run.model, res, pooling, label_cfg, embed_state))
            else:
                ranks[run.arch].append(rank_of(run.arch, run.model, res, pooling))
    return {name: recall_at(r) for name, r in ranks.items() if r}


def weights_path(args: argparse.Namespace, encoder: str, pooling: str, arch: str, which: str) -> str:
    """`<out>/weights/<data>--<encoder>--<pooling>--<arch>[--<tag>]--<best|last>.pt`
    (`Checkpoints.weights_name`), the data the training slides' ids (`data_tag`)."""
    out: str = args.out or os.path.join(RESULT_DIR, 'RerankHead')
    return os.path.join(out, 'weights', weights_name(data_tag(args.train_datasets), encoder, pooling, arch,
                                                    *([args.tag] if args.tag else []), which))


def main() -> int:
    args: argparse.Namespace = parse_args()
    fov = fov_from_args(args)
    fov = replace(fov, sampler=replace(fov.sampler, seed=args.fov_seed))
    val_fov = replace(fov, sampler=replace(fov.sampler, seed=args.val_fov_seed))
    label_cfg: LabelConfig = LabelConfig(pos_iou=args.pos_iou, hit_iou=args.hit_iou, loc_min_iou=args.loc_min_iou)
    retr_cfg = SLIDEWIN_RECIPES[args.retriever]
    pooling: str = retr_cfg.encoder.pooling
    encoder: str = args.retriever.split('-')[0]
    train_slides: List[Tuple[str, str, str]] = slides_of(args.train_datasets, args.n_train_wsi)
    val_slides: List[Tuple[str, str, str]] = slides_of(args.val_datasets, args.n_val_wsi)
    print(f'heads      : {" ".join(args.heads)}   retriever {args.retriever} (K={args.k}, pooling {pooling})')
    print(f'train      : {len(train_slides)} slides of {" ".join(args.train_datasets)}   '
          f'{args.fov_per_visit} FoVs a visit, {args.steps_per_visit} steps of {args.fovs_per_step}')
    print(f'val        : {len(val_slides)} slides of {" ".join(args.val_datasets)}   '
          f'{args.val_fov_per_visit} FoVs a visit')
    if args.dry_run:
        print('dry run: nothing opened, nothing written')
        return 0

    torch.manual_seed(args.seed)
    rng: np.random.Generator = np.random.default_rng(args.seed)
    device: torch.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    require_gpu_if_allocated(device, args.device)
    args.device = str(device)                     # the rest reads the resolved name
    budget = CpuBudget.for_job(processes=1).apply()
    print(f'device     : {device}   {budget.line()}', flush=True)
    make_retriever = lambda: RetrieverSlidingWinSimRot(retr_cfg, device, read_workers=budget.workers)   # noqa: E731
    runs: List[HeadRun] = [HeadRun(arch=a) for a in args.heads]
    wb = wandb_init(args.wandb_project, args.wandb_mode, args.run_name or '-'.join([encoder, *args.heads] + ([args.tag] if args.tag else [])),
                    config=plain_args(args))
    started: float = time.perf_counter()
    for epoch in range(args.epochs):
        order: np.ndarray = rng.permutation(len(train_slides))
        print(f'\n======== epoch {epoch + 1}/{args.epochs} ========', flush=True)
        losses: Dict[str, float] = run_epoch(epoch, runs, [train_slides[i] for i in order], args, make_retriever,
                                             fov, label_cfg, pooling, device, rng, budget.workers)
        for run in runs:
            if run.sched is not None:
                run.sched.step()
        figures: Dict[str, Dict[str, float]] = validate(runs, val_slides, args, make_retriever, val_fov,
                                                        label_cfg, pooling, device, budget.workers)
        metrics: Dict[str, float] = dict(losses)
        print('  validation (share of FoVs whose first hit is in the top k)')
        for name, f in figures.items():
            print(f'    {name:12s} ' + '  '.join(f'R@{k} {f[f"R@{k}"]:.3f}' for k in KS)
                  + f'   ceiling {f["ceiling"]:.3f}  n {f["n"]}')
            metrics.update({f'val/{name}/{key}': v for key, v in f.items()})
        for run in runs:
            if run.arch not in figures or run.model is None:
                continue
            score: float = figures[run.arch][f'R@{args.select_k}']
            for which in ('last',) + (('best',) if score > run.best else ()):
                save_rerank_checkpoint(
                    weights_path(args, encoder, pooling, run.arch, which), model=run.model, arch=run.arch,
                    pooling=pooling, encoder=encoder, epoch=epoch + 1, val=figures[run.arch],
                    run_args=plain_args(args), extra=run.extra,
                    optimizer_state=run.opt.state_dict() if run.opt is not None else None)
            if score > run.best:
                run.best, run.best_epoch = score, epoch + 1
                print(f'    [{run.arch}] new best R@{args.select_k} {score:.3f} -> '
                      f'{weights_path(args, encoder, pooling, run.arch, "best")}', flush=True)
        wandb_log(wb, epoch + 1, metrics)
    wandb_finish(wb)
    print(f'\ndone in {(time.perf_counter() - started) / 60:.1f} min   '
          + '  '.join(f'{r.arch} best R@{args.select_k} {r.best:.3f} (epoch {r.best_epoch})' for r in runs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
