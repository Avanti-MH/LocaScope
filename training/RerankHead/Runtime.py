'''`cli/train.py`'s wandb wrapper -- a DUPLICATE of `PrototypicalRoutingHead/
Runtime.py`'s own three functions (all generic), not an import across the
package boundary: two independent training packages, and a change to one's copy
should not move the other's runs.'''
from __future__ import annotations

from typing import Dict, Optional


def wandb_init(project: str, mode: str, name: str, config: Dict, run_id: Optional[str] = None):
    '''A run, or `None` if wandb is not installed -- the other two no-op on `None`.
    `config` goes in through `config.update(..., allow_val_change=True)`.'''
    try:
        import wandb                                                # noqa: PLC0415
    except ImportError:
        print('wandb not installed; logging to stdout only', flush=True)
        return None
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
