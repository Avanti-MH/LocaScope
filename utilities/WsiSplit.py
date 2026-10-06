"""The WSI split -- val, test and (optionally) train: written once, read by everyone.

    python utilities/cli/build_cache/make_split.py            # the one writer
    val  = list_names(dataset='bracs/test#val')       # AccessDatasets, to READ it

A split is a property of a DATASET, not of whichever package trained on it
first, and it has one writer, so the held-out slides never depend on which job
ran first.

    result/cache/<made_by>_split/<dataset>/wsi_split.csv      split,wsi_name

`SPLIT_JOB` is the writer's default job name and so every reader's default
`--split-cache-job`: a reader that names no job reads the split `make_split.py`
wrote under its own name.

WHAT A SPLIT SAYS, AND HOW IT IS WRITTEN
----------------------------------------
A spec names the sets and how big each is -- an integer is a number of slides, a
decimal is a share of the pool, `rest` is what is left:

    val=10,test=rest                 the split every package already reads
    val=10,test=20,train=rest        the same val, a test that is the first 20
                                     of the old test, and the old test's tail
                                     as train
    train=0.6,val=0.2,test=rest

The pool is sorted, shuffled by a seed, and cut in the order VAL, TEST, TRAIN.
That order is the compatibility rule: the val of a spec with the same seed and
size is the val already recorded, and a new test is a PREFIX of the old one, so
`test_names[:n]` -- which is how the evaluators and the stage-1 bench take their
slides -- keeps returning the same slides. Train, the set that did not exist, is
cut from the TAIL, where nothing has read yet.

EXISTING WINS. A recorded split is never re-derived (see `make_split_spec`): it
is the record of which slides a checkpoint was selected on. `check_split` says
whether a spec would reproduce, extend or contradict a record, and
`upgrade_split` adds `train` to a record only when it extends it, keeping the
old file beside it.
"""
from __future__ import annotations

import csv
import math
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import AccessDatasets
from AccessDatasets import SPLIT_JOB, list_names, locate               # noqa: F401
from Cache import atomic_file
from DsLadder import DsLadder

# `SPLIT_JOB` is defined in AccessDatasets, which reads a recorded split as the
# dataset `<id>#<split>`; it is re-exported here because every writer and older
# reader of this module already imports it from here.

#: Rungs a 4x-per-level pyramid can supply at its OWN native levels -- BRACS
#: SVS steps 4x per level (`DsLadder`'s module docstring: ds 1, 4, 16, 32).
#:
#: NOT EVERY BRACS WSI HAS THIS SHAPE (`wsi_info.py --dataset bracs/test
#: --val-only` on the held-out ten): 6 end their pyramid at
#: ds 32 as expected, 3 end it at ds 64 (so their rung 32 always resamples from
#: ds 16), and one has only 3 levels, topping out at ds 16.
#: `native_bracs_rung_wsi_names` is how a caller that needs every position here
#: to be genuinely native excludes the ones that would silently resample.
BRACS_RUNGS: frozenset = frozenset((1.0, 4.0, 16.0, 32.0))


def split_path(made_by: str, dataset_id: str) -> Path:
    """`result/cache/<made_by>_split/<dataset>/wsi_split.csv`."""
    return AccessDatasets.split_file(dataset_id, made_by)


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


#: The sets a split can hold, in the order they are CUT from the shuffled pool
#: and in which their rows are written. The order is the compatibility rule in
#: the module docstring, so it is written once, here.
SPLIT_SETS = ('val', 'test', 'train')

#: How big a set is: a count, a share of the pool, or the word `rest`.
Size = Union[int, float, str]


def parse_spec(text: str) -> Dict[str, Size]:
    """`'val=10,test=20,train=rest'` -> `{'val': 10, 'test': 20, 'train': 'rest'}`.

    `10` is ten slides; `0.6` and `1.0` are shares (a decimal point is what makes
    a number one); `rest` is what the others leave, and at most one set takes it.
    A set that is not val / test / train, written twice, or of no slides, is an
    error here rather than a split that quietly differs from what was typed."""
    spec: Dict[str, Size] = {}
    for part in (p.strip() for p in text.split(',')):
        if not part:
            continue
        name, sep, value = part.partition('=')
        name, value = name.strip(), value.strip().lower()
        if not sep or name not in SPLIT_SETS:
            raise ValueError(f'{part!r}: want <set>=<size> with set one of '
                             f'{list(SPLIT_SETS)}')
        if name in spec:
            raise ValueError(f'{name} is given twice')
        if value == 'rest':
            spec[name] = 'rest'
        elif re.fullmatch(r'\d+', value):
            if int(value) < 1:
                raise ValueError(f'{name}={value}: a set needs at least 1 slide')
            spec[name] = int(value)
        elif re.fullmatch(r'\d*\.\d+|\d+\.\d*', value):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f'{name}={value}: a share is in (0, 1]')
            spec[name] = float(value)
        else:
            raise ValueError(f'{name}={value!r}: a count (10), a share (0.6) or '
                             f'rest')
    if not spec:
        raise ValueError('an empty spec names no set')
    if list(spec.values()).count('rest') > 1:
        raise ValueError('only one set can be `rest`')
    return spec


def resolve_sizes(spec: Dict[str, Size], n: int) -> Dict[str, int]:
    """Slides per set for a pool of `n`. A share is rounded to the nearest whole
    slide and is at least one; `rest` is what is left and must be at least one.
    Slides a spec without `rest` does not ask for belong to no set."""
    sizes: Dict[str, int] = {}
    for name, size in spec.items():
        if size == 'rest':
            continue
        sizes[name] = (size if isinstance(size, int)
                       else max(1, int(math.floor(size * n + 0.5))))
    used = sum(sizes.values())
    if used > n:
        raise ValueError(f'{sizes} ask for {used} slides of a pool of {n}')
    for name, size in spec.items():
        if size == 'rest':
            if n - used < 1:
                raise ValueError(f'{name}=rest would be empty: {sizes} already '
                                 f'take the whole pool of {n}')
            sizes[name] = n - used
    return sizes


def split_sets(dataset_id: str, spec: Dict[str, Size], *, seed: int = 42,
               candidate_names: Optional[Sequence[str]] = None
               ) -> Dict[str, List[str]]:
    """`{set: names}` -- a WSI-LEVEL split, so no position from one slide can be
    in two sets. The pool is `sorted()` (`list_names` returns whatever order the
    registry holds, so shuffling it directly would make the split depend on
    something nobody controls), shuffled by `seed`, and cut VAL, TEST, TRAIN.
    `candidate_names` narrows the pool (the BRACS native filter) without this
    function growing an opinion about what belongs in it."""
    names = sorted(candidate_names if candidate_names is not None
                   else list_names(dataset=dataset_id))
    sizes = resolve_sizes(spec, len(names))
    random.Random(seed).shuffle(names)
    out: Dict[str, List[str]] = {}
    at = 0
    for name in SPLIT_SETS:
        if name in sizes:
            out[name] = names[at:at + sizes[name]]
            at += sizes[name]
    return out


def split_wsi_names(dataset_id: str, n_val: int, *, seed: int = 42,
                    candidate_names: Optional[Sequence[str]] = None
                    ) -> Tuple[List[str], List[str]]:
    """`(val_names, test_names)`: the spec `val=<n_val>,test=rest`. Kept because
    it is the split every package already reads, and it is one line of
    `split_sets`, so the two cannot disagree."""
    names = sorted(candidate_names if candidate_names is not None
                   else list_names(dataset=dataset_id))
    if not 0 < n_val < len(names):
        raise ValueError(f'n_val={n_val}: dataset {dataset_id!r} has '
                         f'{len(names)} candidate WSIs, need 0 < n_val < that')
    sets = split_sets(dataset_id, {'val': n_val, 'test': 'rest'}, seed=seed,
                      candidate_names=names)
    return sets['val'], sets['test']


def _write(sets: Dict[str, Sequence[str]], path) -> Path:
    """`split,wsi_name` rows, in `SPLIT_SETS` order. Row ORDER is part of the
    record: the names were shuffled before splitting, so a prefix of the `test`
    rows is already a random sample and callers take one instead of drawing
    again. A reader that knows only val and test skips the `train` rows."""
    with atomic_file(path) as tmp:
        with open(tmp, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(('split', 'wsi_name'))
            for name in SPLIT_SETS:
                writer.writerows((name, n) for n in sets.get(name, ()))
    return Path(path)


def read_sets(path) -> Dict[str, List[str]]:
    """Every set a split file records, `{set: names}` in file order."""
    with open(Path(path), newline='') as fh:
        rows = list(csv.DictReader(fh))
    out: Dict[str, List[str]] = {}
    for r in rows:
        out.setdefault(r['split'], []).append(r['wsi_name'])
    return out


def _candidates(dataset_id: str, candidate_names):
    """The pool a split is drawn from: BRACS datasets only from slides native at
    every `BRACS_RUNGS` rung, applied by the one writer, always."""
    if candidate_names is not None:
        return candidate_names
    return (native_bracs_rung_wsi_names(dataset_id)
            if dataset_id.startswith('bracs') else None)


def make_split_spec(dataset_id: str, spec: Dict[str, Size], path, *,
                    seed: int = 42,
                    candidate_names: Optional[Sequence[str]] = None
                    ) -> Dict[str, List[str]]:
    """The recorded split if `path` exists, a fresh one written there if not.
    ONLY `make_split.py` calls this; everyone else reads the split as the
    dataset `<id>#<split>` (AccessDatasets), which never writes.

    EXISTING WINS, always. Re-deriving on every run would mean adding or
    removing one slide silently reshuffles which ten are held out, and
    checkpoints selected on the old val split get scored against a test set
    containing some of it. Deleting the file is how you ask for a new split,
    and doing that invalidates every checkpoint selected on the old one.
    `check_split` and `upgrade_split` are the two ways to touch a record."""
    path = Path(path)
    if path.exists():
        return read_sets(path)
    sets = split_sets(dataset_id, spec, seed=seed,
                      candidate_names=_candidates(dataset_id, candidate_names))
    _write(sets, path)
    return sets


def make_split(dataset_id: str, n_val: int, path, *, seed: int = 42
               ) -> Tuple[List[str], List[str]]:
    """`make_split_spec` for `val=<n_val>,test=rest`, as `(val, test)`."""
    sets = make_split_spec(dataset_id, {'val': n_val, 'test': 'rest'}, path,
                           seed=seed)
    return sets.get('val', []), sets.get('test', [])


def _first_difference(a: Sequence[str], b: Sequence[str]) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f'first difference at position {i}: {x!r} recorded, {y!r} by the spec'
    return 'one is a prefix of the other'


def check_split(dataset_id: str, spec: Dict[str, Size], path, *, seed: int = 42,
                candidate_names: Optional[Sequence[str]] = None
                ) -> Tuple[str, List[str]]:
    """`(status, reasons)` for what `spec` would make against the record at
    `path`, writing nothing:

        missing      no record yet
        same         the spec reproduces the record, set for set
        upgradable   the record lacks `train`, its val is the spec's, and its
                     test is the spec's test followed by the spec's train:
                     adding train takes nothing from val and only the tail from
                     test, so every `test[:n]` read so far still reads the same
        mismatch     anything else, with the first difference of each set
    """
    path = Path(path)
    if not path.exists():
        return 'missing', [f'{path} does not exist']
    recorded = read_sets(path)
    wanted = split_sets(dataset_id, spec, seed=seed,
                        candidate_names=_candidates(dataset_id, candidate_names))
    missing = [n for n in wanted if n not in recorded]
    differ = [n for n in wanted if n in recorded and recorded[n] != wanted[n]]
    extra = [n for n in recorded if n not in wanted]
    if not missing and not differ and not extra:
        return 'same', []
    if (missing == ['train'] and differ == ['test'] and not extra
            and recorded['test'] == wanted['test'] + wanted['train']):
        return 'upgradable', [
            f'train ({len(wanted["train"])}) is the tail of the recorded test '
            f'({len(recorded["test"])}); test keeps its first '
            f'{len(wanted["test"])}']
    reasons = [f'the record has no {n} set' for n in missing]
    reasons += [f'the record holds {n}, which the spec does not name'
                for n in extra]
    for n in differ:
        reasons.append(f'{n}: {len(recorded[n])} recorded, {len(wanted[n])} by '
                       f'the spec; {_first_difference(recorded[n], wanted[n])}')
    return 'mismatch', reasons


def upgrade_split(dataset_id: str, spec: Dict[str, Size], path, *,
                  seed: int = 42,
                  candidate_names: Optional[Sequence[str]] = None) -> str:
    """Add `train` to a record the spec only EXTENDS (`check_split` says
    `upgradable`), keeping the old file as `<name>.orig` first. `'same'` when
    there is nothing to do; a record the spec contradicts raises with the
    differences, and so does an existing backup -- it is the original, and is not
    overwritten by a later upgrade."""
    path = Path(path)
    cands = _candidates(dataset_id, candidate_names)
    status, reasons = check_split(dataset_id, spec, path, seed=seed,
                                  candidate_names=cands)
    if status == 'same':
        return 'same'
    if status != 'upgradable':
        raise ValueError(f'{path}: {status}: ' + '; '.join(reasons))
    backup = path.with_name(path.name + '.orig')
    if backup.exists():
        raise FileExistsError(f'{backup} exists: it is the split as it was before '
                              f'the first upgrade, and is not overwritten')
    shutil.copy2(path, backup)
    _write(split_sets(dataset_id, spec, seed=seed, candidate_names=cands), path)
    return 'upgraded'
