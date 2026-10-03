'''`cli/train.py`'s wandb wrapper and held-out-val scoring
(`score`/`rescore`/`rescore_by_rung`) -- both DUPLICATES of `MppRoutingHead/
Runtime.py`'s own same-named functions, not imports from there; see each
section's own comment for why.

The Stage 2 (G/F/Collapse)/Stage 3 REGISTRIES themselves (`SUPPORT_
CONTEXT_CHOICES`/`QUERY_CONTEXT_CHOICES`/`COLLAPSE_CHOICES`/`ROUTING_
HEAD_CHOICES`) moved to `aiNNModel/models/PrototypeChoices.py` (2026-09-22)
-- see that module's own docstring for the full picture (the pipeline
shape, why `COLLAPSE_CHOICES` is not called `GENERATOR_CHOICES` any more,
why "Matching Net" is a composition rather than a registry entry). Only
RE-EXPORTED here so `cli/train.py`'s existing `from Runtime import (...)`
did not need to change: the move was so the GENERIC checkpoint layer
(`aiNNModel/models/common/Checkpoints.py`) could read the same registries
`cli/train.py` does, without importing this training package into
`aiNNModel/models/` -- a training package importing FROM the generic
layer (this file always did that) is fine; the generic layer importing
FROM a training package is the direction that is not.

`cli/train.py` imports this file as `training.PrototypicalRoutingHead.
Runtime`: `training/MppRoutingHead/` and `training/PrototypicalRoutingHead/`
are real Python packages (each with its own `__init__.py`, under
`training/__init__.py`), so a same-named `Runtime.py` in each package is
disambiguated by the QUALIFIED import path itself, not by which directory
`sys.path` happens to favour. `setup_import_paths()` puts no training
package's directory on `sys.path` at all, so no caller here needs
`add_training_package` -- a qualified import needs no ordering rule to
resolve.

(`cli/train_baseline.py`, Stage 4's Baseline arm, existed 2026-09-20
through 2026-09-22 and was DELETED -- its own definition of what a
"Baseline" checkpoint should do at real inference kept changing across
that window and never settled; it will be redesigned from scratch once
that positioning is actually decided, not resumed from this version.
`LearnedPrototypeClassifier` (`aiNNModel/models/PrototypeRoutingHeads.py`)
was built for it and was deleted alongside it, same day -- see that
module's own docstring for how to rebuild the same shape if a redesign
wants it: any `ROUTING_HEAD_CHOICES` entry already accepts a learned
`[K, D]` tensor as `prototypes` with no separate contract of its own.)
'''
from __future__ import annotations

from typing import Dict, List

import torch

from training.MppRoutingHead.Datasets import RUNGS
from PrototypeChoices import (SUPPORT_CONTEXT_CHOICES, QUERY_CONTEXT_CHOICES,  # noqa: E501,F401
                              COLLAPSE_CHOICES, ROUTING_HEAD_CHOICES)


# ══════════════════════════════════════════════════════════════════════════
#  wandb -- a DUPLICATE of MppRoutingHead/Runtime.py's own wandb_init/
#  wandb_log/wandb_finish (all three are 100% generic -- neither reads
#  HEAD_CHOICES nor anything else specific to that package), not an import
#  from there -- these are two INDEPENDENT training packages (this file's
#  own module docstring), and importing one's generic plumbing from the
#  other would couple them for no reason now that each is its own real
#  Python package (2026-09-22): a change to MppRoutingHead's copy should
#  not be able to move PrototypicalRoutingHead's runs. `wandb_epoch_metrics`
#  is NOT duplicated --
#  that one IS shaped around MppRoutingHead's own val_report row schema
#  (`row['head']`/`row['val_dataset']`), so this project's own per-epoch
#  metrics (episode loss, redraws, per-rung TRAIN accuracy -- see
#  `cli/train.py`'s epoch loop) are flattened directly at the call site
#  instead.
# ══════════════════════════════════════════════════════════════════════════

def wandb_init(project: str, mode: str, name: str, config: Dict,
               run_id: str = None):
    '''Returns a run, or `None` if wandb is not installed -- every other
    function here takes that `None` and no-ops.

    `run_id` makes the run CONTINUABLE: with an id, `resume='allow'` appends to
    the run of that id when it exists and starts it when it does not, so a model
    resumed from its checkpoint keeps drawing on the same curves. None starts a
    fresh run each time.

    `config` goes in through `config.update(..., allow_val_change=True)` rather
    than `init(config=...)`: a continued run already holds a config, and a value
    that differs from last time (`slurm_job_id` always does) is not an error
    here, it is the new job.'''
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
    '''NaN values are dropped, not sent -- e.g. a rung with zero query
    examples this epoch (`--n-choices` skipped it every draw), whose
    accuracy is NaN by construction, not a real zero to chart.'''
    if run is None:
        return
    run.log({k: v for k, v in metrics.items() if v == v}, step=step)


def wandb_finish(run) -> None:
    if run is not None:
        run.finish()


# ══════════════════════════════════════════════════════════════════════════
#  scoring -- a DUPLICATE of MppRoutingHead/Runtime.py's own score/rescore/
#  rescore_by_rung (same reasoning as the wandb section above: generic,
#  duplicated rather than imported across the package boundary). Added
#  2026-09-21 for `cli/train.py`'s own held-out val loop.
# ══════════════════════════════════════════════════════════════════════════

def score(pred_class: torch.Tensor, true_class: torch.Tensor,
         native: torch.Tensor) -> Dict:
    rungs = torch.tensor(RUNGS, dtype=torch.float64)
    pred_r, true_r = rungs[pred_class], rungs[true_class]
    correct = pred_class == true_class
    wrong = ~correct
    out = dict(
        n=int(len(pred_class)),
        level_accuracy=float(correct.double().mean()),
        mpp_error_relative_p50=float(((pred_r - true_r).abs() / true_r).median()),
        coarse_share_of_errors=(float((pred_r[wrong] > true_r[wrong]).double().mean())
                                if wrong.any() else float('nan')),
    )
    for label, keep in (('native', native), ('resampled', ~native)):
        n = int(keep.sum())
        out[f'n_{label}'] = n
        out[f'level_accuracy_{label}'] = (float(correct[keep].double().mean())
                                          if n else float('nan'))
    return out


def rescore(detail_rows: List[Dict]) -> Dict:
    if not detail_rows:
        nan = float('nan')
        return dict(n=0, level_accuracy=nan, mpp_error_relative_p50=nan,
                    coarse_share_of_errors=nan, n_native=0,
                    level_accuracy_native=nan, n_resampled=0,
                    level_accuracy_resampled=nan)
    pick = lambda k, dt: torch.tensor([r[k] for r in detail_rows], dtype=dt)  # noqa: E731
    return score(pick('pred_class', torch.int64), pick('gt_class', torch.int64),
                pick('native', torch.bool))


def rescore_by_rung(detail_rows: List[Dict]) -> Dict[float, Dict]:
    return {rung: rescore([r for r in detail_rows if r['gt_rung'] == rung])
           for rung in RUNGS}
