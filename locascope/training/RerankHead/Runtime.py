'''`cli/train.py`'s wandb wrapper -- a DUPLICATE of `PrototypicalRoutingHead/
Runtime.py`'s own three functions (all generic), not an import across the
package boundary: two independent training packages, and a change to one's copy
should not move the other's runs. And the few functions that let several
processes (one per GPU) train one model together.'''
from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple


def previous_runs(api: Any, entity: str, project: str, name: str) -> List[Any]:
    '''The runs of `project` called `name` that are not running: the earlier versions of this arm. A run still running
    (another job of the same name) is never one of them.'''
    return [r for r in api.runs(f'{entity}/{project}', filters={'display_name': name}) if r.state != 'running']


def delete_previous(wandb: Any, project: str, name: str) -> int:
    '''Delete the earlier versions of the arm `name` of `project` on the wandb server (no undo; their artifacts are
    kept), so that an arm has one run. A failure only prints: a training is not stopped for it. Returns how many.'''
    try:
        api = wandb.Api()
        gone = previous_runs(api, api.default_entity, project, name)
        for r in gone:
            print(f'wandb: deleting the earlier run {r.id} of {name} ({r.state})', flush=True)
            r.delete(delete_artifacts=False)
        return len(gone)
    except Exception as exc:                                            # noqa: BLE001
        print(f'wandb: could not delete the earlier runs of {name}: {type(exc).__name__}: {exc}', flush=True)
        return 0


def wandb_init(project: str, mode: str, name: str, config: Dict, run_id: Optional[str] = None,
               replace_previous: bool = False):
    '''A run, or `None` if wandb is not installed -- the other two no-op on `None`.
    `config` goes in through `config.update(..., allow_val_change=True)`. `replace_previous`: the earlier runs of the
    same name in the project are deleted first (online only; a smoke run must not ask for it).'''
    try:
        import wandb                                                # noqa: PLC0415
    except ImportError:
        print('wandb not installed; logging to stdout only', flush=True)
        return None
    if replace_previous and mode == 'online' and name:
        delete_previous(wandb, project, name)
    run = wandb.init(project=project, mode=mode, name=name or None,
                     id=run_id, resume='allow' if run_id else None)
    run.config.update(config, allow_val_change=True)
    return run


def wandb_log(run, step: int, metrics: Dict[str, float]) -> None:
    '''NaN values are dropped, not sent.'''
    if run is None:
        return
    run.log({k: v for k, v in metrics.items() if v == v}, step=step)


def wandb_finish(run) -> None:
    if run is not None:
        run.finish()


def plain_args(args) -> Dict:
    '''`vars(args)` without the flags nothing set: `ConfigArgs` leaves an unset flag at a
    bare `object()` sentinel, which neither wandb nor a checkpoint's metadata can hold
    (and which would mean nothing to whoever read it). What is left is JSON-friendly.'''
    return {k: v for k, v in vars(args).items()
            if v is None or isinstance(v, (bool, int, float, str, list, tuple))}


# ══════════════════════════════════════════════════════════════════════════════
#  several processes, one per GPU, training one model together
#
#  Launched by `torchrun` (one node): each process takes its own slides, encodes their levels and trains on
#  them; after every step the gradients are averaged over the processes (`sync_gradients`) and every process
#  takes the same optimiser step, so the models stay one model. A process that has run out of slides takes the
#  step with a zero gradient and no vote (`has_data=False`) until all have.
# ══════════════════════════════════════════════════════════════════════════════

def dist_init() -> Tuple[int, int, int]:
    '''`(rank, world, local_rank)`; joins the process group when launched by `torchrun` (NCCL on GPUs, gloo
    without), `(0, 1, 0)` otherwise.'''
    world: int = int(os.environ.get('WORLD_SIZE', '1'))
    if world == 1:
        return 0, 1, 0
    import torch                                                    # noqa: PLC0415
    import torch.distributed as dist                                # noqa: PLC0415
    rank, local = int(os.environ['RANK']), int(os.environ.get('LOCAL_RANK', '0'))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    # the default timeout (10 minutes, for the first collective as for every later one) ends the job when a rank is
    # slow to arrive (a level encoding, a node busy reading); a rank that FAILS leaves at once (`cli/train.py: run`),
    # so a long timeout only lets a slow one be slow
    minutes: int = int(os.environ.get('DIST_TIMEOUT_MIN', '180'))
    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo', timeout=timedelta(minutes=minutes))
    # NCCL connects its processes at the FIRST collective, which would be minutes into the first epoch (job 614353: "No route
    # to host" to one node, found only after the first visit was made). A barrier now finds a bad link at the start.
    dist.barrier(device_ids=[local] if torch.cuda.is_available() else None)
    return rank, world, local


def dist_done() -> None:
    '''Leave the process group, if one was joined.'''
    import torch.distributed as dist                                # noqa: PLC0415
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_sum(value: float, device: Any) -> float:
    '''The sum of `value` over the processes.'''
    import torch                                                    # noqa: PLC0415
    import torch.distributed as dist                                # noqa: PLC0415
    t = torch.tensor([float(value)], device=device)
    dist.all_reduce(t)
    return float(t[0])


def gather_objects(obj: Any, world: int) -> List[Any]:
    '''`obj` of every process, in rank order.'''
    import torch.distributed as dist                                # noqa: PLC0415
    out: List[Any] = [None] * world
    dist.all_gather_object(out, obj)
    return out


def broadcast_parameters(model: Any) -> None:
    '''Every process takes rank 0's parameters (and buffers).'''
    import torch                                                    # noqa: PLC0415
    import torch.distributed as dist                                # noqa: PLC0415
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor, src=0)


def sync_gradients(model: Any, has_data: bool) -> None:
    '''Replace each `p.grad` by the mean over the processes that had data (`has_data`); a process without
    contributes nothing and takes the same mean. All processes must call it, in the same order.'''
    import torch                                                    # noqa: PLC0415
    import torch.distributed as dist                                # noqa: PLC0415
    params = [p for p in model.parameters() if p.requires_grad]
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float()
                      for p in params] + [torch.zeros(1, device=params[0].device)])
    if has_data:
        flat[-1] = 1.0
    else:
        flat.zero_()
    dist.all_reduce(flat)
    flat = flat[:-1] / max(float(flat[-1]), 1.0)
    offset: int = 0
    for p in params:
        p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
        offset += p.numel()
