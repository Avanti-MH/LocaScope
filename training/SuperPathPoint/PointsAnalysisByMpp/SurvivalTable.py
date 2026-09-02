"""The store Stage B produces. spec.md 3.2 輸出.

    table = SurvivalBatch.of(rows, rungs)
    path  = SurvivalTable.save(root, table, meta)
    table, meta = SurvivalTable.load(path)

ONE ROW PER KEYPOINT, COLUMNS INDEXED BY RUNG
===============================================
`score`, `dist` and `suppressed_by_*` are `[N, L]`: N points, L rungs, finest
first. Wide and not long, because every question asked of this table -- which
pattern, which cause, is it a band -- reads the whole vector at once, and a long
table would make each of them a group-by.

THERE IS NO `alive` COLUMN, AND THAT IS THE DESIGN
====================================================
`alive` is `score` over a threshold AND `dist` within `tau`. Storing it freezes
both into the data, and re-cutting at another threshold then means re-running
the detector over every stack -- hours, to answer a question that is one
comparison on an array already in memory (ClaudeRules 8). `Patterns.alive_from`
applies the two, and the threshold sweep that decides whether a finding survives
is a re-read of this file.

The same reasoning puts `born_rung` outside the table: it is the first alive
rung, so it MOVES with the threshold. A stored `born_rung` would quietly keep
the answer from whichever threshold happened to be in force when the file was
written, while every other column re-cut around it.

WHAT `stack_kind` IS DOING IN THE METADATA
============================================
'R' and 'F' answer different questions -- whether a point survives losing detail
and whether it survives being shrunk -- so a survival number that does not say
which is meaningless (spec.md 3.2). It is metadata rather than a column because
one file is one axis: the two are produced by different processes and a file
holding both would make every read a filter.

`detector_id` is here for the reason spec.md 3.4 gives: Stage B's table has to
name the detector that produced it, or a re-run with a better detector cannot be
told from the run before it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


class SurvivalMismatch(RuntimeError):
    """A store that does not describe what the caller asked for."""


@dataclass
class SurvivalBatch:
    """`[N, L]` columns plus the per-point ones. Finest rung first."""
    #: level-0 sub-pixel position of the point, taken at the rung it was first
    #: detected at under the store's own permissive cut.
    x0:   np.ndarray                      # float32 [N]
    y0:   np.ndarray                      # float32 [N]
    #: Which chain the point came from. Kept because a chain is the unit that
    #: succeeds or fails as a whole -- a slide whose chains are all in one
    #: corner is a different corpus from one whose chains are spread, and
    #: without this the table cannot say which it is.
    chain: np.ndarray                     # int32 [N]

    score:               np.ndarray       # float32 [N, L]
    dist:                np.ndarray       # float32 [N, L], -1 = no partner
    suppressed_by_score: np.ndarray       # float32 [N, L], -1 = not suppressed
    suppressed_by_dist:  np.ndarray       # float32 [N, L], -1 = not suppressed

    #: THE SAME PROBE AT A PLACE THE POINT IS NOT, `decoy_alpha * ds` away.
    #: A decoy rate computed from these is a real measurement -- "would an
    #: arbitrary nearby location also show something within tau" -- where one
    #: derived as `dist + shift <= tau` is the match rate at a shifted alpha,
    #: which is the same curve under another name.
    decoy_score: np.ndarray               # float32 [N, L]
    decoy_dist:  np.ndarray               # float32 [N, L], -1 = nothing near

    def __len__(self) -> int:
        return int(len(self.x0))

    def check(self) -> 'SurvivalBatch':
        """Shapes agree, and the sentinels are the ones the readers expect."""
        n, length = self.score.shape
        for name in ('dist', 'suppressed_by_score', 'suppressed_by_dist',
                     'decoy_score', 'decoy_dist'):
            got = getattr(self, name).shape
            if got != (n, length):
                raise SurvivalMismatch(
                    f'{name} is {got} and score is {(n, length)}; they index '
                    f'the same (point, rung)')
        for name in ('x0', 'y0', 'chain'):
            if len(getattr(self, name)) != n:
                raise SurvivalMismatch(
                    f'{name} has {len(getattr(self, name))} entries against '
                    f'{n} points')
        return self


@dataclass
class SurvivalMeta:
    wsi_stem: str
    stack_kind: str                       # 'R' or 'F'
    tile: int
    rungs: Tuple[float, ...]              # finest first, and this is the order
                                          # every [N, L] column is indexed by

    #: The checkpoint that detected the points (spec.md 3.4). A table that
    #: cannot name its detector is indistinguishable from the round before it.
    detector_id: str = ''
    detector_path: str = ''

    #: The PERMISSIVE cut the store was written at. Every question asked later
    #: re-cuts at some threshold >= this one, on the same arrays; a question
    #: below it cannot be answered without re-running.
    score_threshold: float = 0.001
    #: `tau = max(tau_floor_um / mpp_0, alpha * shrink)`. Both recorded because
    #: `alive` is derived and a reader has to be able to reproduce it.
    tau_alpha: float = 1.5

    #: How far the decoy probe sits from the anchor, as a multiple of `shrink`.
    #: Far enough that it is a different place -- well outside any tau worth
    #: choosing -- and near enough to be the same tissue, because a decoy on
    #: glass would be trivially beaten and would say nothing.
    decoy_alpha: float = 8.0
    tau_floor_um: float = 0.0
    mpp_0: float = 0.0

    n_chains: int = 0
    created_at: str = ''
    notes: str = ''

    def filename(self) -> str:
        return f'{self.wsi_stem}__{self.stack_kind}__t{self.tile}.safetensors'

    def to_strings(self) -> Dict[str, str]:
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            out[f.name] = (json.dumps(list(value)) if isinstance(value, tuple)
                           else str(value))
        return out

    @classmethod
    def from_strings(cls, md: Dict[str, str]) -> 'SurvivalMeta':
        return cls(
            wsi_stem=md['wsi_stem'], stack_kind=md['stack_kind'],
            tile=int(md['tile']), rungs=tuple(json.loads(md['rungs'])),
            detector_id=md.get('detector_id', ''),
            detector_path=md.get('detector_path', ''),
            score_threshold=float(md.get('score_threshold', 0.001)),
            tau_alpha=float(md.get('tau_alpha', 1.5)),
            tau_floor_um=float(md.get('tau_floor_um', 0.0)),
            mpp_0=float(md.get('mpp_0', 0.0)),
            decoy_alpha=float(md.get('decoy_alpha', 8.0)),
            n_chains=int(md.get('n_chains', 0)),
            created_at=md.get('created_at', ''), notes=md.get('notes', ''))

    def tau(self) -> np.ndarray:
        """`[L]` -- the level-0 tolerance at each rung, from the two terms.

        `alpha * shrink` is the floor a coarse rung imposes on itself: one of
        its pixels IS `shrink` level-0 px, so a point that really exists can
        only be located to that precision there. Writing tau as a fixed level-0
        number instead makes the coarse rungs unable to match BY DEFINITION,
        and the output reads as "keypoints all die at coarse resolution" -- a
        wrong answer that looks like a discovery (spec.md 3.2 配對).
        """
        floor = (self.tau_floor_um / self.mpp_0) if self.mpp_0 else 0.0
        return np.array([max(floor, self.tau_alpha * float(d))
                         for d in self.rungs], dtype=np.float64)


def save(root, batch: SurvivalBatch, meta: SurvivalMeta) -> Path:
    batch.check()
    if batch.score.shape[1] != len(meta.rungs):
        raise SurvivalMismatch(
            f'the columns are {batch.score.shape[1]} wide and the meta names '
            f'{len(meta.rungs)} rungs {meta.rungs}. The [N, L] columns are '
            f'indexed BY that tuple, so a mismatch means every rung label is '
            f'wrong and nothing later would say so')
    if meta.stack_kind not in ('R', 'F'):
        raise SurvivalMismatch(
            f"stack_kind must be 'R' or 'F', got {meta.stack_kind!r}; a "
            f'survival number that does not say which axis is meaningless')

    meta = dataclasses.replace(
        meta, created_at=meta.created_at or time.strftime('%Y-%m-%dT%H:%M:%S'))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / meta.filename()
    tmp = path.with_suffix('.tmp')
    save_file({'x0': np.ascontiguousarray(batch.x0, np.float32),
               'y0': np.ascontiguousarray(batch.y0, np.float32),
               'chain': np.ascontiguousarray(batch.chain, np.int32),
               'score': np.ascontiguousarray(batch.score, np.float32),
               'dist': np.ascontiguousarray(batch.dist, np.float32),
               'suppressed_by_score': np.ascontiguousarray(
                   batch.suppressed_by_score, np.float32),
               'suppressed_by_dist': np.ascontiguousarray(
                   batch.suppressed_by_dist, np.float32),
               'decoy_score': np.ascontiguousarray(batch.decoy_score,
                                                   np.float32),
               'decoy_dist': np.ascontiguousarray(batch.decoy_dist,
                                                  np.float32)},
              str(tmp), metadata=meta.to_strings())
    os.replace(tmp, path)
    return path


def load_meta(path) -> SurvivalMeta:
    with safe_open(str(path), framework='numpy') as handle:
        md = handle.metadata()
    if md is None:
        raise SurvivalMismatch(f'{path} has no metadata -- not a survival store')
    return SurvivalMeta.from_strings(md)


def load(path) -> Tuple[SurvivalBatch, SurvivalMeta]:
    with safe_open(str(path), framework='numpy') as handle:
        md = handle.metadata()
        if md is None:
            raise SurvivalMismatch(f'{path} has no metadata')
        data = {k: handle.get_tensor(k) for k in handle.keys()}
    return SurvivalBatch(**data).check(), SurvivalMeta.from_strings(md)


def find(root, **eq) -> List[Path]:
    """Stores under `root` whose metadata matches every keyword.

    By metadata and not by filename, `PreTileStore.find`'s reason: the name
    carries fields a caller would otherwise have to re-derive, and an identity
    rule recomputed elsewhere is one that drifts.
    """
    hits = []
    for candidate in sorted(Path(root).glob('*.safetensors')):
        try:
            meta = load_meta(candidate)
        except Exception:                                    # noqa: BLE001
            continue
        if all(str(getattr(meta, k, None)) == str(v) for k, v in eq.items()):
            hits.append(candidate)
    return hits
