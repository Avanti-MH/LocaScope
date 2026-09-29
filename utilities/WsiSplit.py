"""The val/test WSI split: written once, read by everyone.

    python utilities/cli/build_cache/make_split.py            # the one writer
    val, test = read_split(split_path(SPLIT_JOB, 'bracs/test'))

A split is a property of a DATASET, not of whichever package trained on it
first. It used to live in `training/MppRoutingHead/Datasets.py`, so
PrototypicalRoutingHead, the stage-1 bench and the WSI diagnostics all reached
sideways into one training package to find out which slides were held out --
and before that, three of them wrote a missing split themselves, each its own
way, so the held-out slides depended on which job ran first.

    result/cache/<made_by>_split/<dataset>/wsi_split.csv      split,wsi_name

`SPLIT_JOB` is the writer's default job name and so every reader's default
`--split-cache-job`: a reader that names no job reads the split `make_split.py`
wrote under its own name.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from AccessDatasets import list_names, locate
from Cache import atomic_file, cache_root
from DsLadder import DsLadder

#: The writer's own job name. Every reader defaults to it.
SPLIT_JOB = 'MakeSplit'

#: Rungs a 4x-per-level pyramid can supply at its OWN native levels -- BRACS
#: SVS steps 4x per level (`DsLadder`'s module docstring: ds 1, 4, 16, 32).
#:
#: NOT EVERY BRACS WSI HAS THIS SHAPE (measured 2026-09-22, `wsi_info.py
#: --dataset bracs/test --val-only` on the held-out ten): 6 end their pyramid at
#: ds 32 as expected, 3 end it at ds 64 (so their rung 32 always resamples from
#: ds 16), and one has only 3 levels, topping out at ds 16.
#: `native_bracs_rung_wsi_names` is how a caller that needs every position here
#: to be genuinely native excludes the ones that would silently resample.
BRACS_RUNGS: frozenset = frozenset((1.0, 4.0, 16.0, 32.0))


def split_path(made_by: str, dataset_id: str) -> Path:
    """`result/cache/<made_by>_split/<dataset>/wsi_split.csv`."""
    return cache_root(made_by, 'split') / dataset_id.replace('/', '_') / 'wsi_split.csv'


def native_bracs_rung_wsi_names(dataset_id: str) -> List[str]:
    """WSI names in `dataset_id` whose own pyramid is native at EVERY rung in
    `BRACS_RUNGS`.

    Opens every WSI to read its `level_downsamples` -- header only, the cost
    `wsi_info.py` pays, not a pixel read. `tile_size` does not affect
    `is_native` (`shrink = rung_ds / level_ds`), so the 256 is arbitrary.
    """
    from SafeSlide import SafeSlide                               # noqa: PLC0415
    ladder = DsLadder(rungs=tuple(sorted(BRACS_RUNGS)))
    out: List[str] = []
    for name in list_names(dataset=dataset_id):
        wsi = SafeSlide(locate(name, dataset=dataset_id).path)
        try:
            if all(p.is_native for p in ladder.plan_for(wsi, tile_size=256)):
                out.append(name)
        finally:
            wsi.close()
    return out


def split_wsi_names(dataset_id: str, n_val: int, *, seed: int = 42,
                    candidate_names: Optional[Sequence[str]] = None
                    ) -> Tuple[List[str], List[str]]:
    """`(val_names, test_names)` -- a WSI-LEVEL split, so no position from one
    slide can appear on both sides. The val half is the first `n_val` of a
    seeded shuffle.

    `sorted()` before the shuffle: `list_names` returns whatever order the
    registry holds, so shuffling it directly would make the split depend on
    something nobody controls. `candidate_names` narrows the pool (the BRACS
    native filter) without this function growing an opinion about what
    belongs in it.
    """
    names = sorted(candidate_names if candidate_names is not None
                   else list_names(dataset=dataset_id))
    if not 0 < n_val < len(names):
        raise ValueError(f'n_val={n_val}: dataset {dataset_id!r} has '
                         f'{len(names)} candidate WSIs, need 0 < n_val < that')
    random.Random(seed).shuffle(names)
    return names[:n_val], names[n_val:]


def _write(val_names: Sequence[str], test_names: Sequence[str], path) -> Path:
    """`split,wsi_name` rows. Row ORDER is part of the record: the names were
    shuffled before splitting, so a prefix of the `test` rows is already a
    random sample and callers take one instead of drawing again."""
    with atomic_file(path) as tmp:
        with open(tmp, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(('split', 'wsi_name'))
            writer.writerows(('val', n) for n in val_names)
            writer.writerows(('test', n) for n in test_names)
    return Path(path)


def _read(path) -> Tuple[List[str], List[str]]:
    with open(Path(path), newline='') as fh:
        rows = list(csv.DictReader(fh))
    return ([r['wsi_name'] for r in rows if r['split'] == 'val'],
            [r['wsi_name'] for r in rows if r['split'] == 'test'])


def make_split(dataset_id: str, n_val: int, path, *, seed: int = 42
               ) -> Tuple[List[str], List[str]]:
    """The recorded split if `path` exists, a fresh one written there if not.
    ONLY `make_split.py` calls this -- see `read_split`.

    EXISTING WINS, always. Re-deriving on every run would mean adding or
    removing one slide silently reshuffles which ten are held out, and
    checkpoints selected on the old val split get scored against a test set
    containing some of it. Deleting the file is how you ask for a new split,
    and doing that invalidates every checkpoint selected on the old one.

    BRACS datasets draw only from slides native at every `BRACS_RUNGS` rung,
    applied here, by the one writer, always.
    """
    path = Path(path)
    if path.exists():
        return _read(path)
    candidates = (native_bracs_rung_wsi_names(dataset_id)
                  if dataset_id.startswith('bracs') else None)
    val_names, test_names = split_wsi_names(dataset_id, n_val, seed=seed,
                                            candidate_names=candidates)
    _write(val_names, test_names, path)
    return val_names, test_names


def read_split(path) -> Tuple[List[str], List[str]]:
    """`(val_names, test_names)` as recorded, in file order -- or an error
    naming the one tool that writes it. Never writes: a missing file is a
    refusal, not an invitation."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} does not exist. The val/test split is written once, by '
            f'utilities/cli/build_cache/make_split.py -- run it first, so '
            f'every package scores on the same held-out slides')
    return _read(path)
