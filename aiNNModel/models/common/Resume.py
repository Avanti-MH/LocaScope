'''Crash-resume state for a training run: one file per model, rewritten every
epoch, holding everything the NEXT epoch needs to be the epoch it would have
been had the job not stopped.

    state = ResumeFile.for_model(resume_dir, name)   # None dir -> disabled
    got = state.load(identity)                       # None -> start fresh
    if got: epoch0 = got['epoch']; restore(...)
    for epoch in range(epoch0 + 1, epochs + 1):
        ...
        state.save(identity, epoch=epoch, modules=..., optimizers=...,
                   schedulers=..., best=..., extra=..., rngs=...)

THE RULE (user, 2026-09-24):
    no --resume-dir        train from scratch, write nothing here
    --resume-dir, no file  train from scratch, write the file every epoch
    --resume-dir, a file   continue from the epoch after the one it records,
                           and keep writing it every epoch

`--epochs` is the TOTAL. A file that already records the last epoch leaves
nothing to run, so resuming a finished model is a no-op rather than twenty
more epochs -- which is what the warm start this replaced did, reading
`_best.pt` and treating `--epochs` as "this many more".

WHAT IS RESTORED, and why each piece is here:
    module / optimizer / scheduler state   the model and how it is moving
    epoch                                  where to pick up
    best                                   so the first resumed epoch is
                                           compared against the real best, not
                                           against -inf
    extra                                  whatever the caller accumulates
                                           across epochs (the val CSV rows),
                                           which would otherwise be lost
    python / numpy / torch / cuda RNG      so the draws after the resume are
                                           the draws that would have come next
    named `random.Random` objects          the caller's own generators (the
                                           episode sampler's), same reason

IDENTITY IS CHECKED, NOT ASSUMED. The file records the run's identity (every
argument that changes what is being trained) and a load under a different one
is refused, naming the fields. Resuming an ord_a file into a bal run would
otherwise produce a model that is neither.

ATOMIC. Written to a temp name and renamed, so a job killed mid-write leaves
the previous epoch's file, never a half one.
'''
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


def capture_rngs(named: Optional[Mapping[str, random.Random]] = None) -> Dict[str, Any]:
    '''Every global generator's state, plus the caller's own named ones.'''
    out = {'python': random.getstate(), 'numpy': np.random.get_state(),
           'torch': torch.get_rng_state(),
           'named': {k: r.getstate() for k, r in (named or {}).items()}}
    if torch.cuda.is_available():
        out['cuda'] = torch.cuda.get_rng_state_all()
    return out


def restore_rngs(state: Dict[str, Any],
                 named: Optional[Mapping[str, random.Random]] = None) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    for key, gen in (named or {}).items():
        if key not in state['named']:
            raise KeyError(f'resume file has no RNG named {key!r}; it has '
                           f'{sorted(state["named"])}')
        gen.setstate(state['named'][key])


def _plain(value):
    '''Identity values as comparable plain data: tuples and lists alike.'''
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class ResumeMismatch(RuntimeError):
    pass


class ResumeFile:
    '''One model's resume file, or a disabled stand-in when `path` is None.'''

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path is not None else None

    @classmethod
    def for_model(cls, resume_dir, name: str) -> 'ResumeFile':
        '''`<resume_dir>/<name>_resume.pt`; disabled when `resume_dir` is
        falsy (no `--resume-dir`).'''
        if not resume_dir:
            return cls(None)
        return cls(Path(resume_dir) / f'{name}_resume.pt')

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def load(self, identity: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        '''The saved state, or None to start fresh (disabled, or no file yet).
        Refuses a file written under a different identity.'''
        if not self.enabled or not self.path.exists():
            return None
        state = torch.load(self.path, map_location='cpu', weights_only=False)
        saved = state['identity']
        diff = sorted(k for k in set(saved) | set(identity)
                      if _plain(saved.get(k)) != _plain(identity.get(k)))
        if diff:
            detail = ', '.join(f'{k}: file={saved.get(k)!r} now={identity.get(k)!r}'
                               for k in diff)
            raise ResumeMismatch(
                f'{self.path} was written by a different run ({detail}). '
                f'Delete it to start this one from scratch, or point '
                f'--resume-dir elsewhere')
        return state

    def save(self, identity: Mapping[str, Any], *, epoch: int,
             modules: Mapping[str, torch.nn.Module],
             optimizers: Mapping[str, Any] = None,
             schedulers: Mapping[str, Any] = None,
             best: Any = None, extra: Any = None,
             named_rngs: Optional[Mapping[str, random.Random]] = None) -> None:
        if not self.enabled:
            return
        state = dict(
            identity=dict(identity), epoch=int(epoch),
            modules={k: m.state_dict() for k, m in modules.items()},
            optimizers={k: o.state_dict() for k, o in (optimizers or {}).items()},
            schedulers={k: s.state_dict() for k, s in (schedulers or {}).items()},
            best=best, extra=extra, rngs=capture_rngs(named_rngs))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f'.{self.path.name}.{os.getpid()}.tmp')
        torch.save(state, tmp)
        os.replace(tmp, self.path)

    @staticmethod
    def restore(state: Dict[str, Any], *, modules: Mapping[str, torch.nn.Module],
                optimizers: Mapping[str, Any] = None,
                schedulers: Mapping[str, Any] = None,
                named_rngs: Optional[Mapping[str, random.Random]] = None) -> None:
        '''Loads every piece `save` wrote back into live objects. A module,
        optimizer or scheduler the file has but the caller did not pass (or the
        reverse) is an error -- the two runs were not built the same way.'''
        for kind, live, saved in (('module', modules, state['modules']),
                                  ('optimizer', optimizers or {}, state['optimizers']),
                                  ('scheduler', schedulers or {}, state['schedulers'])):
            if set(live) != set(saved):
                raise ResumeMismatch(
                    f'{kind}s differ: file has {sorted(saved)}, run has {sorted(live)}')
            for key, obj in live.items():
                obj.load_state_dict(saved[key])
        restore_rngs(state['rngs'], named_rngs)


def resume_identity(args, exclude) -> Dict[str, Any]:
    '''`vars(args)` minus the arguments that do not change what is trained
    (where output goes, how many epochs in total, logging, devices).'''
    return {k: _plain(v) for k, v in sorted(vars(args).items()) if k not in set(exclude)}
