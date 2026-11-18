#!/usr/bin/env python3
'''Train CoAE (spec.md): one arm a run.

    torchrun --nproc_per_node 8 -m locascope.training.CoAE.train --out RESULT/coae/<job> [--level large] [--mechanism mix]
        [--variant vref] [--sims none] [--noise 0.5] [--pool-cells 4] [--lora] [--epochs 3] [--warmup-epochs 1.0 ...]

An arm has a code (`CoaeConfig.arm`: `RP3`, `RN3`, `RP3.pool512.epochs12`, ...) and a directory name (`CoaeConfig.dir_name`:
`large-mix-vref`, `large-mix-vref-nce`, ...); `--print-arm` / `--print-dir` print one and stop (`CoaeTrain.sh` takes the output directory's
name from `--print-dir`). A step: each process makes its share of the batch (`Supply`: places x views, a view a
(slot, source): slot 0 canonical, slot 1 perturbed at `--level`; the workers read the places and draw the views' geometry and
photometric parameters, `views_of` frames and simulates them on the GPU), the network runs on the views' input tiles, the targets are
made from the views' photo canvases (no gradient: each view's own Co-figure on the FM's field, shared among the views the mechanism
says; the place descriptor of the canonical ref for cls'), and the losses (spec.md §5) are summed:
    w_recon x reconstruction (point: masked MSE to the shared target; dist (vdist): NLL of every sharer's own map)
  + w_pool  x cls' linear prediction of the place descriptor (raw first and second moments of the Co-figure over 4 x 4 cells): MSE
  + w_nce   x target contrast on the linear head: a view's head against the place descriptors (own = the positive; the batch's other
              places and the bank's places of its (slide, rung) cell = the negatives)   (only when > 0, default 0)
  + w_vnce  x view-to-view contrast on cls' (the place's other views = the positives; experiment arms only)   (only when > 0, default 0)
  + w_hnce  x view-to-view contrast on the linear head's output (the descriptor loss's place; experiment arms only)   (only when > 0, default 0)
  + w_dnce  x twin contrast on the decoder's output, a latent cell's block of it (the reconstruction's place; experiment arms only)   (only when > 0, default 0)
  + w_lnce  x the same twin contrast on the latent itself (the control of w_dnce; experiment arms only)   (only when > 0, default 0)
  + w_koleo x KoLeo of cls' within each source          (only when > 0, default 0)
  + w_adv   x the source adversary (domain confusion)  (only when > 0, default 0)
The pool is screened at start (no blank place is listed). A process's batch is the places it read, `places / world` unless a screened
place came back blank (then fewer, and the processes' gradients are weighed by their places); a step in which a process read none
updates nothing. Gradients are averaged over the processes (DDP). The probe (`Validate.probe_measures`, places of slides never trained on, no real
photo anywhere in the flow; not limited by `--pool-cap`) is measured once before the training and after every `--probe-every`-th
epoch (and the last); each epoch writes a checkpoint (`checkpoint`: the weights, the optimiser, the targets' scale, the Supply's seen
state and its pool's checksum), before the probe is measured, whole or not at all.
`--epochs 0`: build the probe, measure once, stop (no weights).
`--resume <k>`: continue after epoch k from weights/epoch<k>.pt: epoch k+1 on, the step count, lr schedule, optimiser and seen state
as they were; log.tsv and val.tsv are continued (their rows after epoch k dropped first), not overwritten. The contract: the same
`--out`, the same flags (`--epochs` too) and the same process count: any config field that differs is refused before anything is
written. A resume cannot add epochs (the cosine schedule is set by the total steps). Resume after the last epoch (k = epochs - 1): only the
probe of that epoch is measured, when it is missing. Without `--resume` an arm sent again starts over.
Out: <out>/log.tsv (a row a logged step), <out>/val.tsv (a `probe` row a measure), <out>/weights/{last,epoch<k>}.pt, <out>/config.json.
'''
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from locascope.training.CoAE.CoFigure import Targets, n_channels, part_names, shared_moments, targets_of_views
from locascope.training.CoAE.Config import CELLS_A_TOKEN, FM_GRIDS, LEVELS, MECHANISMS, VARIANTS, CoaeConfig
from locascope.training.CoAE.GpuSims import frames, simulate
from locascope.training.CoAE.Losses import ChannelScale, StepContext, StepLosses, TargetBank, step_losses
from locascope.training.CoAE import LogLayout as LL
from locascope.training.CoAE.Model import CoaeNet
from locascope.training.CoAE.Sims import GREY_WEIGHTS, sim_labs
from locascope.training.CoAE.Supply import Batch, Probe, Supply
from locascope.training.CoAE.Validate import probe_measures

#: simulated canvases made at once on the GPU (96 x 3 x 640^2 float32 = 0.47 GB an operand; a few live at a time)
SIM_CHUNK: int = 96


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--print-arm', action='store_true', help="print the arm's code (CoaeConfig.arm) of these flags and stop: nothing is built (--out is still required, any value)")
    ap.add_argument('--print-dir', action='store_true', help="print the run's directory name (CoaeConfig.dir_name) of these flags and stop, as --print-arm")
    ap.add_argument('--lora', action='store_true', help='LoRA on the FM; else frozen')
    ap.add_argument('--level', choices=tuple(LEVELS), default=None, help='the perturbation level of the perturbed group (DataConfig.level; 0 = no perturbed group)')
    ap.add_argument('--mechanism', choices=MECHANISMS, default=None, help='how the views of a place share their targets (DataConfig.mechanism)')
    ap.add_argument('--variant', choices=VARIANTS, default=None, help='what the decoder\'s target is made of (HeadConfig.variant)')
    ap.add_argument('--sims', default=None, help='the simulated sources, comma separated (Sims.sim_labs names), or none (DataConfig.sims)')
    ap.add_argument('--noise', type=float, default=None, help='the bottleneck noise on the latent and on cls\' (ModelConfig.noise; 0 = off)')
    ap.add_argument('--pool-cells', type=int, default=None, help='the side of cls\' place descriptor in cells (HeadConfig.pool_cells)')
    ap.add_argument('--w-pool', type=float, default=None, help='the weight of cls\' linear prediction of the place descriptor (HeadConfig.w_pool; >= 0; 0 = the ablation without cls\' descriptor loss)')
    ap.add_argument('--w-recon', type=float, default=None, help='the weight of the decoder\'s reconstruction (HeadConfig.w_recon; 0 only with a contrast: --w-vnce, --w-dnce or --w-lnce > 0, an experiment arm)')
    ap.add_argument('--w-nce', type=float, default=None, help='the weight of the target contrast on the linear head (HeadConfig.w_nce; >= 0)')
    ap.add_argument('--w-vnce', type=float, default=None, help='the weight of the view-to-view contrast on cls\' (HeadConfig.w_vnce; >= 0; experiment arms only)')
    ap.add_argument('--w-hnce', type=float, default=None, help='the weight of the view-to-view contrast on the linear head\'s output (HeadConfig.w_hnce; >= 0; experiment arms only)')
    ap.add_argument('--w-dnce', type=float, default=None, help='the weight of the twin contrast on the decoder\'s output blocks (HeadConfig.w_dnce; >= 0; experiment arms only)')
    ap.add_argument('--w-lnce', type=float, default=None, help='the weight of the twin contrast on the latent (HeadConfig.w_lnce; >= 0; experiment arms only)')
    ap.add_argument('--tau', type=float, default=None, help='the temperature of the contrasts (HeadConfig.tau)')
    ap.add_argument('--seed', type=int, default=None, help='TrainConfig.seed')
    ap.add_argument('--cls-residual', action='store_true', help='cls\' = the FM\'s CLS as it is + gain x the attention branch (no projection)')
    ap.add_argument('--encoder', default=None)
    ap.add_argument('--fm-dtype', choices=('fp32', 'bf16'), default=None, help="the FM's weight dtype (FmConfig.weights_dtype; bf16 = the experiment arms, fp32 = the formal arms and LoRA)")
    ap.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True, help="torch.compile on the FM's forward (mode default, static shapes); not part of the config: the result does not depend on it")
    ap.add_argument('--places', type=int, default=None, help="places a step over all processes (128 a GPU; x views = the tiles)")
    ap.add_argument('--pool-cap', type=int, default=None, help='places a (slide, rung) cell keeps for the whole training (DataConfig.pool_cap; 0 = all drawn)')
    ap.add_argument('--cell-parts', type=int, default=None, help='an epoch lists 1 / this of each cell\'s pool, the least seen first (DataConfig.cell_parts)')
    ap.add_argument('--mask', default=None, help='the mask recipe the places are drawn in (DataConfig.mask; MASK_RECIPES)')
    ap.add_argument('--draw-cache', default=None, help='whose cache the place draws are in (DataConfig.draw_cache)')
    ap.add_argument('--slides', type=int, default=None, help='the first this many training slides (DataConfig.slides; 0 = all)')
    ap.add_argument('--epochs', type=int, default=None, help='TrainConfig.epochs (an epoch is the supply\'s steps: at most 448 at pool_cap 128, cell_parts 1; at most 359 at pool_cap 512, cell_parts 4: fewer where the screen drops places; config 3); 0 = the probe once, then stop')
    ap.add_argument('--warmup-epochs', type=float, default=None, help='linear warm-up in epochs, a fraction is fine (TrainConfig.warmup_epochs, config 1.0)')
    ap.add_argument('--cell-epoch', type=int, default=None, help='places of a (slide, rung) cell an epoch lists at most (DataConfig.cell_epoch; a smoke run lowers it)')
    ap.add_argument('--probe-slides', type=int, default=None, help='the first this many slides of the probe dataset (ProbeConfig.slides; 0 = all)')
    ap.add_argument('--probe-places', type=int, default=None, help='places a (slide, rung) cell of the probe (ProbeConfig.places)')
    ap.add_argument('--log-every', type=int, default=20, help='a table row every N steps (their means)')
    ap.add_argument('--mix-every', type=int, default=200, help='the layer mix on its own line every N steps')
    ap.add_argument('--probe-every', type=int, default=1, help='measure the probe after every N-th epoch (and after the last one); a checkpoint is written every epoch all the same')
    ap.add_argument('--resume', type=int, default=None, help='continue after this epoch: load weights/epoch<N>.pt and train epoch N+1 on (N = epochs - 1: resume after the last epoch, only the probe of that epoch is measured, when it is missing); log.tsv and val.tsv are continued, not overwritten. Without it an arm sent again starts over')
    ap.add_argument('--workers', type=int, default=4, help='reader threads of the probe\'s Supply (at start only; the training\'s places are read by the worker processes, DataConfig.processes)')
    return ap.parse_args()


def given(**fields: Any) -> Dict[str, Any]:
    """The fields a flag was given for (None = not given): what `replace` should change, and nothing else."""
    return {k: v for k, v in fields.items() if v is not None}


def make_cfg(a: argparse.Namespace) -> CoaeConfig:
    c: CoaeConfig = CoaeConfig()
    sims: Optional[Tuple[str, ...]] = None
    if a.sims is not None:
        sims = () if a.sims == 'none' else tuple(a.sims.split(','))
        unknown: List[str] = [s for s in sims if s not in sim_labs()]
        if unknown:
            raise ValueError(f'--sims: unknown simulator(s) {unknown}; known {tuple(sim_labs())}')
    fm = replace(c.fm, lora=a.lora, **given(encoder=a.encoder, weights_dtype=a.fm_dtype))
    data = replace(c.data, **given(places=a.places, mask=a.mask, draw_cache=a.draw_cache, slides=a.slides, cell_epoch=a.cell_epoch,
                                   pool_cap=a.pool_cap, cell_parts=a.cell_parts, level=a.level, mechanism=a.mechanism, sims=sims))
    train = replace(c.train, **given(epochs=a.epochs, warmup_epochs=a.warmup_epochs, seed=a.seed))
    head = replace(c.head, **given(variant=a.variant, w_pool=a.w_pool, pool_cells=a.pool_cells, w_recon=a.w_recon, w_nce=a.w_nce,
                                   w_vnce=a.w_vnce, w_hnce=a.w_hnce, w_dnce=a.w_dnce, w_lnce=a.w_lnce, tau=a.tau))
    model = replace(c.model, cls_residual=a.cls_residual, **given(noise=a.noise))
    probe = replace(c.probe, **given(slides=a.probe_slides, places=a.probe_places))
    cofig = replace(c.cofig, out_cells=CELLS_A_TOKEN * FM_GRIDS.get(fm.encoder, 0))        # an unknown encoder: CoaeConfig.__post_init__ names it
    return replace(c, fm=fm, data=data, train=train, head=head, model=model, cofig=cofig, probe=probe)


def own_state(net: CoaeNet) -> Dict[str, torch.Tensor]:
    """The checkpoint: every parameter and buffer but the frozen FM's weights (the LoRA adapters kept; the FM is reloaded from its registry)."""
    return {k: v for k, v in net.state_dict().items() if not k.startswith('fm.vit.') or k.endswith(('.qkv.a', '.qkv.b'))}


def lr_at(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@dataclass
class Views:
    """A batch's views on the device: `tiles` uint8 [B, tile, tile, 3], the network's inputs (each view's canvas's centre);
    `grey` float32 [places x slots x kept, photo, photo], the canvases the targets read (each slot's first `kept` sources:
    CoaeConfig.canvases), slot outside, source inside."""
    tiles: torch.Tensor
    grey: torch.Tensor


def views_of(b: Batch, cfg: CoaeConfig, device: torch.device) -> Views:
    """The views of a batch: the frames are `GpuSims.frames` of the reference reads; a ref view is its frame, a simulated view is
    `GpuSims.simulate` of its frame and its shot, SIM_CHUNK at a time. Float32, no gradient, outside autocast."""
    with torch.no_grad():
        refs: torch.Tensor = b.refs.to(device)
        bases: torch.Tensor = frames(refs, [r for r, _ in b.frames], [g for _, g in b.frames], cfg.data.photo_px)     # uint8 [F, H, W, 3]
        del refs
        base: torch.Tensor = b.base.to(device)
        k: torch.Tensor = b.source.to(device)
        n: int = len(b.shots)
        s_n: int = len(b.sources)
        kept: int = cfg.canvases
        tile: int = cfg.cofig.tile
        h: int = bases.shape[1]
        a: int = h // 2 - tile // 2
        w: torch.Tensor = torch.tensor(GREY_WEIGHTS, device=device)
        tiles: torch.Tensor = torch.empty((n, tile, tile, 3), dtype=torch.uint8, device=device)
        grey: torch.Tensor = torch.empty((n // s_n * kept, h, h), dtype=torch.float32, device=device)
        row: torch.Tensor = torch.arange(n, device=device) // s_n * kept + k            # a view's row of grey (used when k < kept)

        def put(idx: torch.Tensor, canv: torch.Tensor) -> None:                          # canv uint8 [m, h, h, 3]
            tiles[idx] = canv[:, a:a + tile, a:a + tile]
            r: torch.Tensor = k[idx] < kept
            if bool(r.any()):
                grey[row[idx[r]]] = canv[r].float() @ w

        for sel in ([j for j, s in enumerate(b.shots) if s is None], [j for j, s in enumerate(b.shots) if s is not None]):
            for c in range(0, len(sel), SIM_CHUNK):
                part: List[int] = sel[c:c + SIM_CHUNK]
                idx: torch.Tensor = torch.tensor(part, device=device)
                canv: torch.Tensor = bases[base[idx]]
                put(idx, canv if b.shots[part[0]] is None else simulate(canv, [b.shots[j] for j in part]))
        return Views(tiles, grey)


def targets_of(b: Batch, grey: torch.Tensor, cfg: CoaeConfig, device: torch.device, field_px: int) -> Tuple[Targets, torch.Tensor, torch.Tensor]:
    """The targets of a batch (no gradient), from the canvases `views_of` made (`Views.grey`, `cfg.canvases` a slot): (the decoder's, `targets_of_views` (`view_targets` of each
    view's own Co-figure on the FM's field; vref makes only the ref canvases' figures), weighed as its place's rung; cls''s place descriptor of every view [B, M, c, c], `shared_moments` of the ref
    canvases; its cell weights [B, c, c], the view's rung weight)."""
    with torch.no_grad():
        sources: int = len(b.sources)
        tg: Targets = targets_of_views(grey, cfg.cofig, field_px, b.slots, sources, cfg.data.twins, cfg.head.variant,
                                       cfg.canvases)
        tg = replace(tg, weight=b.weight.to(device))                      # each view weighs as its place's rung
        p_mean: torch.Tensor = shared_moments(grey, b.slots, sources, cfg.data.place_shared, cfg.cofig, field_px, cfg.head.pool_cells,
                                              cfg.data.dilate_px, cfg.canvases)
        c: int = cfg.head.pool_cells
        p_w: torch.Tensor = b.weight.to(device).view(-1, 1, 1).expand(-1, c, c)
        return tg, p_mean, p_w


def epoch_seed(seed: int, rank: int, epoch: int) -> int:
    """The torch seed of a process's epoch (the bottleneck noise): a function of the three, so a resumed epoch draws what the
    uninterrupted run would."""
    return (seed * 1000 + rank) * 1000 + epoch


def places_read(n: int, world: int, device: torch.device) -> List[int]:
    """The places each process read for this step, in rank order (a gather over the processes; [n] alone)."""
    if world == 1:
        return [n]
    mine: torch.Tensor = torch.tensor([n], dtype=torch.long, device=device)
    every: List[torch.Tensor] = [torch.zeros_like(mine) for _ in range(world)]
    dist.all_gather(every, mine)
    return [int(t) for t in every]


def checkpoint(cfg: CoaeConfig, net: CoaeNet, scale: ChannelScale, opt: torch.optim.Optimizer, supply: Supply, world: int,
               epoch: int, step: int) -> Dict[str, Any]:
    """All a run needs to go on after `epoch`: the weights and buffers (`own_state`: pool_mu / pool_sd / pool_seen, the layer mix,
    LoRA), the targets' running scale, the optimiser, the Supply's seen state and its pool's checksum; `step` = the steps done."""
    return {'cfg': asdict(cfg), 'state': own_state(net), 'scale': scale.state_dict(), 'opt': opt.state_dict(),
            'supply': supply.state(), 'world': world, 'epoch': epoch, 'step': step}


def save_checkpoint(ckpt: Dict[str, Any], path: Path) -> None:
    """Written whole or not at all: to `<path>.tmp`, then os.replace (a job killed while saving leaves the older file)."""
    tmp: Path = path.with_name(path.name + '.tmp')
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, cfg: CoaeConfig, world: int) -> Dict[str, Any]:
    """The checkpoint a run resumes from; raises unless it is of this very run: the same config (every field) and process count."""
    if not path.is_file():
        raise FileNotFoundError(f'--resume: no {path}')
    ck: Dict[str, Any] = torch.load(path, map_location='cpu', weights_only=False)
    now: Dict[str, Any] = asdict(cfg)
    if ck['cfg'] != now:
        diff: List[str] = [f'{sec}.{k}: {ck["cfg"].get(sec, {}).get(k)!r} -> {now[sec][k]!r}' for sec in now for k in now[sec]
                           if ck['cfg'].get(sec, {}).get(k) != now[sec][k]]
        raise ValueError(f'--resume: the config differs from the checkpoint\'s ({"; ".join(diff)}); a resumed run changes nothing')
    if ck['world'] != world:
        raise ValueError(f'--resume: the checkpoint is of {ck["world"]} process(es), this run has {world}; a resumed run changes nothing')
    return ck


def keep_rows(path: Path, keep: Callable[[str], bool]) -> None:
    """Rewrite a .tsv with only the lines `keep` accepts (to `<path>.tmp`, then os.replace); a missing file becomes an empty one."""
    lines: List[str] = path.read_text().splitlines(keepends=True) if path.is_file() else []
    tmp: Path = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(l for l in lines if keep(l)))
    os.replace(tmp, path)


def log_row_kept(line: str, step: int) -> bool:
    """Whether a log.tsv line is a whole row (ends with a newline, parses as JSON) of a step before `step`; a row a killed job left
    half written is dropped."""
    if not line.endswith('\n'):
        return False
    try:
        return int(json.loads(line)['step']) < step
    except (ValueError, KeyError, TypeError):
        return False


def val_row_kept(line: str, epoch: int) -> bool:
    """Whether a val.tsv line is a whole `probe` row (ends with a newline, third column parses as JSON) of an epoch <= `epoch`; a row a
    killed job left half written is dropped."""
    if not line.endswith('\n'):
        return False
    cols: List[str] = line.rstrip('\n').split('\t', 2)
    if len(cols) < 3:
        return False
    try:
        json.loads(cols[2])
        return int(cols[1]) <= epoch
    except ValueError:
        return False


def end_epoch(cfg: CoaeConfig, net: CoaeNet, scale: ChannelScale, opt: torch.optim.Optimizer, supply: Supply, world: int,
              probe: Probe, device: torch.device, out: Path, val_f: Any, say: Any, epoch: int, step: int, seconds: float,
              means: Dict[str, float], measure: bool) -> None:
    """Rank 0, an epoch's end: the checkpoints (`last.pt`, `epoch<k>.pt`, written before the probe is measured), the epoch's means,
    and when `measure` the probe (the same one every epoch)."""
    ckpt: Dict[str, Any] = checkpoint(cfg, net, scale, opt, supply, world, epoch, step)
    save_checkpoint(ckpt, out / 'weights' / 'last.pt')
    save_checkpoint(ckpt, out / 'weights' / f'epoch{epoch}.pt')
    say(LL.epoch_line(epoch, seconds, means))
    if measure:
        pm: Dict[str, Any] = probe_measures(net, probe, cfg, scale, device)
        say(LL.probe_table(pm))
        val_f.write(f'probe\t{epoch}\t{json.dumps(pm)}\n')
        val_f.flush()


def main() -> None:
    a = parse()
    cfg: CoaeConfig = make_cfg(a)
    if a.print_arm or a.print_dir:
        print(cfg.arm if a.print_arm else cfg.dir_name)
        return
    if a.probe_every < 1:
        raise ValueError(f'--probe-every must be >= 1, got {a.probe_every}')
    distributed: bool = 'RANK' in os.environ
    if distributed:
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))   # rank 0 validates while the others wait
    rank: int = dist.get_rank() if distributed else 0
    world: int = dist.get_world_size() if distributed else 1
    local: int = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device: torch.device = torch.device('cuda', local)
    out: Path = Path(a.out)
    say = (lambda s: print(s, flush=True)) if rank == 0 else (lambda s: None)
    ck: Optional[Dict[str, Any]] = None
    if a.resume is not None:                                           # every process, before anything is written
        if not 0 <= a.resume < cfg.train.epochs:
            raise ValueError(f'--resume {a.resume}: must be an epoch of the run, 0 <= k < epochs ({cfg.train.epochs})')
        ck = load_checkpoint(out / 'weights' / f'epoch{a.resume}.pt', cfg, world)
    if rank == 0:
        (out / 'weights').mkdir(parents=True, exist_ok=True)
        (out / 'config.json').write_text(json.dumps(asdict(cfg), indent=1))
    say(LL.__doc__)                                                    # the columns, once
    ablation: str = " (ablation: cls' has no content loss)" if cfg.head.w_pool == 0 else ''
    view_only: str = ' (experiment arm: no reconstruction loss; the decoder gets a contrast at most)' if cfg.head.w_recon == 0 else ''
    say(f'CoAE arm {cfg.arm}, level {cfg.data.level}, mechanism {cfg.data.mechanism}, variant {cfg.head.variant}, sims {cfg.data.sims}, '
        f'noise {cfg.model.noise:g}, encoder {cfg.fm.encoder} ({cfg.fm.weights_dtype} weights, compile {a.compile}), layers {cfg.fm.layers}, {world} process(es), places {cfg.data.places} x '
        f'{cfg.data.slots * (1 + len(cfg.data.sims))} views, weights: recon {cfg.head.w_recon:g}{view_only} pool {cfg.head.w_pool:g}'
        f'{ablation} '
        f'nce {cfg.head.w_nce:g} vnce {cfg.head.w_vnce:g} hnce {cfg.head.w_hnce:g} dnce {cfg.head.w_dnce:g} lnce {cfg.head.w_lnce:g} '
        f'({cfg.head.twin_cells} cell positions) tau {cfg.head.tau:g} '
        f'koleo {cfg.head.w_koleo:g} adv {cfg.head.w_adv:g}, cls residual {cfg.model.cls_residual}, seed {cfg.train.seed}, out {out}')
    parts: List[str] = part_names(cfg.cofig)
    cols: List[LL.Col] = LL.columns(parts)

    supply: Supply = Supply(cfg, rank, world, device, workers=a.workers, say=say) if rank == 0 else None   # rank 0 makes the masks first
    if distributed:
        dist.barrier()
    if supply is None:
        supply = Supply(cfg, rank, world, device, workers=a.workers, say=say)                              # the others read them cached
    if distributed:                                                   # every process screened its own pool: they must agree
        mine_crc: torch.Tensor = torch.tensor([supply.pool_crc], dtype=torch.long, device=device)
        every_crc: List[torch.Tensor] = [torch.zeros_like(mine_crc) for _ in range(world)]
        dist.all_gather(every_crc, mine_crc)
        if len({int(t) for t in every_crc}) != 1:
            raise RuntimeError(f'the processes screened different pools ({[int(t) for t in every_crc]}): the listing would not be the same on all')
    E: int = supply.steps_an_epoch
    total: int = cfg.train.epochs * E
    warmup: int = max(1, round(cfg.train.warmup_epochs * E))
    adv_ramp: int = max(1, round(cfg.head.adv_ramp_epochs * E))
    first: int = 0 if ck is None else a.resume + 1
    say(f'an epoch: {E} steps; {cfg.train.epochs} epochs = {total} steps; warm-up {warmup} steps, adversary ramp {adv_ramp} steps; '
        f'probe every {a.probe_every} epoch(s)' + ('' if ck is None else f'; resuming at epoch {first}'))
    net: CoaeNet = CoaeNet(cfg, device, n_sources=len(supply.sources)).to(device)
    say(f'fm {cfg.fm.encoder}: {net.fm.depth} blocks; the layer mix reads blocks {[b + 1 for b in net.fm.blocks]} (1-based) of fm.layers {cfg.fm.layers}; '
        f'tokens {net.fm.grid} x {net.fm.grid} -> latent {net.latent_side} x {net.latent_side} x {cfg.model.latent_dim}, Co-figure '
        f'{cfg.cofig.out_cells} x {cfg.cofig.out_cells} ({net.field_px // cfg.cofig.out_cells} px a cell)')
    if a.compile:                                                      # only the FM's forward; each batch shape (the probe's, the probe's last, the training's) compiles once, ~40 s each
        net.fm.forward = torch.compile(net.fm.forward, mode='default', dynamic=False)
        say('compile: the FM forward is compiled at its first call of each batch shape (the probe before the training, then the first training step, ~40 s each)')
    if ck is not None:
        missing, unexpected = net.load_state_dict(ck['state'], strict=False)
        if unexpected or [k for k in missing if not k.startswith('fm.vit.')]:            # only the frozen FM's own weights may be absent
            raise ValueError(f'--resume: the checkpoint does not fit the network (missing {missing[:3]}, unexpected {unexpected[:3]})')
    field: int = net.field_px                                          # the px of the tile the FM sees: the targets' field
    head_p, lora_p = net.trainable()
    groups: List[Dict[str, Any]] = [{'params': head_p, 'lr': cfg.train.lr}]
    if lora_p:
        groups.append({'params': lora_p, 'lr': cfg.train.lr_lora})
    opt: torch.optim.Optimizer = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    if ck is not None:
        opt.load_state_dict(ck['opt'])
    # the schedule's base lrs come from the config, never from opt.param_groups (a loaded optimizer state carries the lr of the step it was saved at)
    base_lr: List[float] = [cfg.train.lr] + ([cfg.train.lr_lora] if lora_p else [])
    model: torch.nn.Module = DDP(net, device_ids=[local]) if distributed else net
    scale: ChannelScale = ChannelScale(n_channels(cfg.cofig)).to(device)
    if ck is not None:
        scale.load_state_dict(ck['scale'])
        supply.load_state(ck['supply'])
    bank: Optional[TargetBank] = None             # the target contrast's bank (the whole pool's descriptors, as last computed)
    if cfg.head.w_nce > 0:
        bank = TargetBank(supply.cell_of, net.pool_shape, device)
        say(f'target bank: {len(supply.places)} places x {bank.desc[0].numel()} values = {bank.desc.numel() * 4 / 1e6:.0f} MB on the GPU; filled as the places are seen; not in the checkpoint')
    step: int = 0 if ck is None else int(ck['step'])
    log_f: Any = None
    val_f: Any = None
    if rank == 0:
        if ck is None:
            log_f = open(out / 'log.tsv', 'w')                         # an arm sent again starts over
            val_f = open(out / 'val.tsv', 'w')
        else:
            keep_rows(out / 'log.tsv', lambda l: log_row_kept(l, step))                   # the whole rows of the epochs kept
            keep_rows(out / 'val.tsv', lambda l: val_row_kept(l, a.resume))
            log_f = open(out / 'log.tsv', 'a')
            val_f = open(out / 'val.tsv', 'a')
            say(f'resumed after epoch {a.resume} at step {step} ({out / "weights" / f"epoch{a.resume}.pt"})')

    def due(e: int) -> bool:
        """Whether the probe is measured after epoch e."""
        return (e + 1) % a.probe_every == 0 or e == cfg.train.epochs - 1

    probe: Optional[Probe] = None
    if rank == 0:                                                      # the probe: slides never trained on, one fixed set for every measure
        val: Supply = Supply(replace(cfg, data=replace(cfg.data, dataset=cfg.probe.dataset, slides=cfg.probe.slides)),
                             0, 1, device, workers=a.workers, say=say, pool=False)
        both = set(val.names) & set(supply.names)
        if both:
            raise ValueError(f'{sorted(both)[:3]} ... are probe slides AND training slides')
        probe = val.probe_set(cfg, field, device)
        val.close()
        say(f'probe: {len(probe.cell)} places in {len(probe.cells)} cells, views {probe.kinds}')
        row: Optional[int] = None
        if ck is None:
            row = -1                                                   # the measure before the training
        elif due(a.resume) and not any(l.startswith(f'probe\t{a.resume}\t') for l in (out / 'val.tsv').read_text().splitlines()):
            row = a.resume                                             # the run died between that checkpoint and its measure
        if row is not None:
            pm: Dict[str, Any] = probe_measures(net, probe, cfg, scale, device)
            say(LL.probe_table(pm))
            val_f.write(f'probe\t{row}\t{json.dumps(pm)}\n')
            val_f.flush()
    if distributed:
        dist.barrier()

    def clock() -> float:
        torch.cuda.synchronize(device)
        return time.perf_counter()

    for epoch in range(first, cfg.train.epochs):
        torch.manual_seed(epoch_seed(cfg.train.seed, rank, epoch))     # CPU and CUDA generators: every process, every epoch its own noise
        model.train()
        t0: float = time.time()
        win: LL.Window = LL.Window()
        ep: LL.Window = LL.Window()
        say('\n' + LL.header(cols))
        it = iter(supply.batches(epoch))                       # at most E steps; fewer when the epoch could not fill a step
        while True:
            t_a: float = clock()
            try:
                b: Batch = next(it)
            except StopIteration:
                break
            t_b: float = clock()
            if distributed:                                    # every process has its batch: the wait for the slowest one, apart
                dist.barrier()
            ns: List[int] = places_read(len(b.slides), world, device)      # every process, every step (the same list on all of them)
            t_s: float = clock()
            if min(ns) == 0:                                   # a process has nothing to run: no process updates (DDP needs them all)
                say(f'step {step} (epoch {epoch}): places read a process {ns}: no update')
                step += 1                                      # the schedule's step is used up all the same
                continue
            # 1.0 whenever every process read its whole share -- the rule, since the pool is screened at start. It differs only in a step
            # where a place came back blank or off the slide after all (Batch.blank > 0: the screen and the full read disagreed at the
            # threshold, or the slide failed in between) on some process and not on the others
            share: float = len(b.slides) * world / sum(ns)     # this process's places against an even share
            f: float = lr_at(step, total, warmup)
            for g, lr in zip(opt.param_groups, base_lr):
                g['lr'] = lr * f
            net.adv_lambda = cfg.head.adv_max * min(1.0, step / adv_ramp)
            v: Views = views_of(b, cfg, device)
            t_v: float = clock()
            tg, p_mean, p_w = targets_of(b, v.grey, cfg, device, field)
            tiles: torch.Tensor = v.tiles
            del v                                              # the grey canvases are not held through the network's step
            scale.update(tg.mean, tg.valid)
            net.see_places(p_mean)
            src: torch.Tensor = b.source.to(device)
            t_c: float = clock()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model(tiles)
            ctx: StepContext = StepContext(cfg, net, o, tg, p_mean, p_w, scale, b, b.place.to(device), b.weight.to(device), src,
                                           len(set(ns)) == 1, bank)       # each source's KoLeo over the whole job's batch needs equal shares
            sl: StepLosses = step_losses(ctx)                  # every loss that is on, summed in LOSSES' order; the target bank is written after them
            loss: torch.Tensor = sl.loss
            weighted: List[Tuple[str, torch.Tensor]] = sl.weighted
            t_d: float = clock()
            gnorm: Dict[str, Tuple[float, float]] = {}
            if (step + 1) % a.mix_every == 0:                  # what each loss pulls on the latent and on cls' (every process: KoLeo gathers)
                for name_, term in weighted:
                    if not term.requires_grad:
                        continue
                    gl, gc = torch.autograd.grad(term, [o.latent, o.cls], retain_graph=True, allow_unused=True)
                    gnorm[name_] = (float(gl.norm()) if gl is not None else 0.0, float(gc.norm()) if gc is not None else 0.0)
            opt.zero_grad(set_to_none=True)
            (loss * share).backward()                          # DDP averages the processes' gradients evenly: x share makes it the mean over the job's places
            t_e: float = clock()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.0)
            opt.step()
            t_f: float = clock()
            if rank == 0:
                vals: Dict[str, float] = dict(loss=float(loss), **sl.logged,
                                              t_wait=t_b - t_a, t_sync=t_s - t_b, t_sim=t_v - t_s, t_targ=t_c - t_v, t_fwd=t_d - t_c, t_bwd=t_e - t_d, t_opt=t_f - t_e,
                                              s_step=t_f - t_a, blank=float(b.blank), n=float(len(b.slides)), **{f'm_{k}': v for k, v in b.timing.items()})
                win.add(**vals)
                ep.add(**vals)
                if (step + 1) % a.log_every == 0:
                    m: Dict[str, float] = win.means()
                    m.update(step=step, epoch=epoch, tiles_s=m['n'] * world * supply.versions / max(m['s_step'], 1e-9),
                             mem=torch.cuda.max_memory_allocated(device) / 1e9)
                    say(LL.row(m, cols))
                    if (step + 1) % a.mix_every == 0:
                        say(LL.mix_line(step, [b_ + 1 for b_ in net.fm.blocks],net.mix.weights().detach().cpu().tolist()))
                        if gnorm:
                            say(LL.grad_line(step, gnorm))
                            m['grad'] = gnorm
                    log_f.write(json.dumps(m | {'mix': net.mix.weights().detach().cpu().tolist()}) + '\n')
                    log_f.flush()
                    win.clear()
                    torch.cuda.reset_peak_memory_stats(device)
            step += 1
        if rank == 0:
            end_epoch(cfg, net, scale, opt, supply, world, probe, device, out, val_f, say, epoch, step, time.time() - t0, ep.means(),
                      measure=due(epoch))
        if distributed:
            dist.barrier()
    supply.close()                                             # the worker processes, and the shared memory of the boards
    if distributed:
        dist.destroy_process_group()
    say('done')


if __name__ == '__main__':
    try:
        main()
    except Exception:                                          # e.g. BrokenProcessPool: a Supply worker died. Leave at once, non-zero: the
        traceback.print_exc()                                  # interpreter's own shutdown can wait on the dead pool, and torchrun ends the
        sys.stdout.flush()                                     # other processes only when this one has exited
        sys.stderr.flush()
        os._exit(1)
