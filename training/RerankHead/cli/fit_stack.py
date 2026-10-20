#!/usr/bin/env python3
'''Fit the stacker (S7): one score per candidate from the columns the other
scorers wrote.

    python training/RerankHead/cli/fit_stack.py \\
        --arm 'uni2:SlidingWindow::cls_avg+mean@500::Dual::cls_avg+dual::CSLS::cls_avg+csls::Corr::cls_avg+sinkhorn' \\
        --columns first dual csls sinkhorn contrast mnn

The arm (`ArmSpec.config_of_arm`, the chain a pipeline runs as `--stage2 arm:<arm>`) is
run on FoVs; its last phase's list -- every candidate, with every column the
rerankers of the chain wrote for it (their `columns` extras, matched by window) --
is the stacker's input and the labels (`Labels.py`) its target. Each column enters
as its rank within the query's list and as a z-score (`stack_features`), so the
stacker only ever sees how a candidate compares with the others of its own query.

WHICH SLIDES. The learned scorers in the recipe were trained on the training slides
(`train.py`), where their scores are better than they will be anywhere else; a
stacker fitted there would trust them too much. So it is fitted on the held-out
slides (`--fit-datasets @legacy-val`, which the benches use to choose), a third of
them (by slide) kept to pick the epoch. Confirm on `@legacy-test`, which nothing
here reads.

Loss: the listwise loss and the top-k hinge of `Losses.py`. Weights:
`RerankHead/weights/<data>--<encoder>--<pooling>--stack[--<tag>]--<best|last>.pt` (the data the fit
slides' ids), with the columns in the file (`extra['columns']`), which the scorer refuses to run without.
'''
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'utilities'))
from _paths import RESULT_DIR, setup_import_paths                      # noqa: E402
setup_import_paths()
sys.path.insert(0, os.path.join(_HERE, '..', '..', '..', 'utilities', 'bench_modules'))   # BenchCommon

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from CpuBudget import CpuBudget                                        # noqa: E402
from BenchCommon import require_gpu_if_allocated                      # noqa: E402
from FovSupply import add_fov_args, fov_from_args                     # noqa: E402
from RerankChoices import build_rerank                                # noqa: E402
from Checkpoints import data_tag, save_rerank_checkpoint, weights_name   # noqa: E402
from stage2_retrieval.ArmSpec import config_of_arm, parse_arm         # noqa: E402
from stage2_retrieval.Context import ScoreTable, incoming_table, window_boxes   # noqa: E402
from stage2_retrieval.MultiPhases import MultiPhases                  # noqa: E402
from stage2_retrieval.RerankerNeural import stack_features            # noqa: E402

from training.RerankHead.Labels import LabelConfig, label_windows     # noqa: E402
from training.RerankHead.Losses import listwise_loss, topk_hinge_loss   # noqa: E402
from training.RerankHead.Metrics import KS, first_hit_rank, recall_at   # noqa: E402
from training.RerankHead.Runtime import plain_args                    # noqa: E402
from training.RerankHead.Visits import FovResult, fov_stream, shape_fn   # noqa: E402
from training.RerankHead.cli.train import slides_of                   # noqa: E402


@dataclass
class StackSample:
    feats: torch.Tensor          # [K, 2 * n_columns]
    pos: torch.Tensor
    neg: torch.Tensor
    hit: np.ndarray
    raw: Dict[str, np.ndarray]   # column -> [K], for the single-column baselines
    slide: str


def chain_table(res: FovResult, columns: Sequence[str]) -> ScoreTable:
    """The table of the chain's last list: the incoming columns (`in`, `first`, `first_rank`) and, for each of
    `columns`, the values the nearest reranker that wrote it gave each window."""
    cands = res.cs
    table: ScoreTable = incoming_table(cands)
    keys = [c.key() for c in cands.candidates]
    for extra in res.out.find('columns'):
        for name in columns:
            if name in table:
                continue
            values = extra.value.column(name, keys)
            if values is not None:
                table.put(name, torch.tensor(values, dtype=torch.float32))
    missing: List[str] = [c for c in columns if c not in table]
    if missing:
        raise KeyError(f'the chain wrote no column {missing} for its last list; it has {table.names()}')
    return table


def collect(res: FovResult, columns: Sequence[str], label_cfg: LabelConfig,
            device: Any) -> Optional[StackSample]:
    """The stacker's sample from one FoV, or None when the list holds no positive."""
    cands = res.cs
    table: ScoreTable = chain_table(res, columns)
    boxes: np.ndarray = window_boxes(cands, shape_fn(res.source))
    lab = label_windows(boxes, np.array([c.rotation for c in cands.candidates]), res.where,
                        float(res.cs.grids[0].tile_size) * float(cands.ds), label_cfg)
    if not lab.pos.any():
        return None
    return StackSample(feats=stack_features(table, list(columns)).to(device),
                       pos=torch.as_tensor(lab.pos, device=device), neg=torch.as_tensor(lab.neg, device=device),
                       hit=lab.hit, raw={c: table[c].cpu().numpy() for c in columns}, slide=res.slide)


@torch.no_grad()
def evaluate(model: torch.nn.Module, samples: Sequence[StackSample]) -> Dict[str, float]:
    model.eval()
    ranks: List[float] = [first_hit_rank(model(s.feats).float().cpu().numpy(), s.hit) for s in samples]
    return recall_at(ranks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arm', required=True, help='the chain whose rerankers write the columns (ArmSpec grammar)')
    ap.add_argument('--columns', nargs='+', required=True)
    ap.add_argument('--fit-datasets', nargs='+', default=['@legacy-val'])
    ap.add_argument('--n-wsi', type=int, default=10, help='slides of each dataset a --fit-datasets id draws from')
    ap.add_argument('--fov-per-visit', type=int, default=32)
    add_fov_args(ap)
    ap.add_argument('--fov-seed', type=int, default=2000)
    ap.add_argument('--mask-cache-job', default='Masks')
    ap.add_argument('--pos-iou', type=float, default=0.5)
    ap.add_argument('--hit-iou', type=float, default=0.5)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--batch', type=int, default=32, help='FoVs a step')
    ap.add_argument('--topk', type=int, default=30)
    ap.add_argument('--w-topk', type=float, default=0.5)
    ap.add_argument('--select-k', type=int, default=30)
    ap.add_argument('--valid-every', type=int, default=3, help='every Nth slide is kept to pick the epoch')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto', help="'auto': the GPU, or stop if the job was given one that does not work")
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=None)
    args: argparse.Namespace = ap.parse_args()
    if args.select_k not in KS:
        ap.error(f'--select-k must be one of {KS}')

    fov = fov_from_args(args)
    fov = replace(fov, sampler=replace(fov.sampler, seed=args.fov_seed))
    label_cfg: LabelConfig = LabelConfig(pos_iou=args.pos_iou, hit_iou=args.hit_iou, loc_min_iou=0.1, fov_margin=1.2)
    arm = parse_arm(args.arm)
    cfg = config_of_arm(arm)
    pooling: str = cfg.phases[0].encoder.pooling
    encoder: str = arm.retriever.encoder
    device: torch.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    require_gpu_if_allocated(device, args.device)
    args.device = str(device)                     # the rest reads the resolved name
    budget = CpuBudget.for_job(processes=1).apply()
    slides: List[Tuple[str, str, str]] = slides_of(args.fit_datasets, args.n_wsi)
    print(f'arm {arm.text()}  columns {" ".join(args.columns)}')
    print(f'fit on {len(slides)} slides of {" ".join(args.fit_datasets)}', flush=True)

    samples: List[StackSample] = []
    t0: float = time.perf_counter()
    make = lambda: MultiPhases(cfg, device, read_workers=budget.workers)         # noqa: E731
    for res in fov_stream(slides, make, fov, args.mask_cache_job, device, label_cfg, args.fov_per_visit,
                          args.fov_seed, 0, workers=budget.workers):
        s: Optional[StackSample] = collect(res, args.columns, label_cfg, device)
        if s is not None:
            samples.append(s)
    print(f'{len(samples)} FoVs with a positive in the list  ({(time.perf_counter() - t0) / 60:.1f} min)', flush=True)
    if not samples:
        return 1

    names: List[str] = sorted({s.slide for s in samples}, key=lambda n: zlib.crc32(n.encode()))
    held: set = {n for i, n in enumerate(names) if i % max(args.valid_every, 1) == 0}
    train = [s for s in samples if s.slide not in held]
    valid = [s for s in samples if s.slide in held] or train
    print(f'fit {len(train)} FoVs ({len(names) - len(held)} slides), pick the epoch on {len(valid)} ({len(held)} slides)')
    print('single columns on the picking FoVs:')
    base: List[Tuple[str, Dict[str, float]]] = [('first_phase', recall_at(
        [first_hit_rank(-np.arange(len(s.hit), dtype=np.float64), s.hit) for s in valid]))]
    for col in args.columns:
        base.append((col, recall_at([first_hit_rank(s.raw[col], s.hit) for s in valid])))
    for name, f in base:
        print(f'  {name:14s} ' + '  '.join(f'R@{k} {f[f"R@{k}"]:.3f}' for k in KS))

    torch.manual_seed(args.seed)
    rng: np.random.Generator = np.random.default_rng(args.seed)
    model = build_rerank('stack', dict(n_features=2 * len(args.columns), hidden=args.hidden,
                                       dropout=args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_epoch = -1.0, 0
    out: str = args.out or os.path.join(RESULT_DIR, 'RerankHead')
    tags: List[str] = [args.tag] if args.tag else []
    for epoch in range(args.epochs):
        model.train()
        order: np.ndarray = rng.permutation(len(train))
        for s0 in range(0, len(order), args.batch):
            opt.zero_grad(set_to_none=True)
            part = order[s0:s0 + args.batch]
            loss: torch.Tensor = sum(
                listwise_loss(model(train[i].feats), train[i].pos, train[i].neg)
                + args.w_topk * topk_hinge_loss(model(train[i].feats), train[i].pos, train[i].neg, args.topk)
                for i in part) / len(part)
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 and epoch + 1 != args.epochs:
            continue
        f: Dict[str, float] = evaluate(model, valid)
        print(f'  epoch {epoch + 1:4d}  ' + '  '.join(f'R@{k} {f[f"R@{k}"]:.3f}' for k in KS), flush=True)
        for which in ('last',) + (('best',) if f[f'R@{args.select_k}'] > best else ()):
            save_rerank_checkpoint(os.path.join(out, 'weights', weights_name(data_tag(args.fit_datasets), encoder,
                                                                    pooling, 'stack', *tags, which)),
                                   model=model, arch='stack', pooling=pooling, encoder=encoder, epoch=epoch + 1,
                                   val=f, run_args=plain_args(args), extra=dict(columns=list(args.columns)))
        if f[f'R@{args.select_k}'] > best:
            best, best_epoch = f[f'R@{args.select_k}'], epoch + 1
    print(f'best R@{args.select_k} {best:.3f} at epoch {best_epoch}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
