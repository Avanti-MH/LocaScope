#!/usr/bin/env python3
'''Write the WSI split -- the ONE tool that does.

    python utilities/cli/build_cache/make_split.py
    python utilities/cli/build_cache/make_split.py --datasets bracs/test ki67_with_photo

    # val / test / train, by count, by share, or `rest`
    python utilities/cli/build_cache/make_split.py \\
        --spec bracs/test:val=10,test=20,train=rest

    # what a spec would do to the split already recorded -- writes nothing
    python utilities/cli/build_cache/make_split.py --check \\
        --spec bracs/test:val=10,test=20,train=rest ki67_with_photo:val=10,test=rest

    # add train to a record the spec only EXTENDS; the old split is kept as orig
    python utilities/cli/build_cache/make_split.py --upgrade \\
        --spec bracs/test:val=10,test=20,train=rest

Writes `<cache-job>`'s `dataset=<dataset>/split/` entry for each dataset
that has none yet, and leaves every existing one alone (EXISTING WINS -- see
`WsiSplit.make_split_spec`). `<cache-job>` defaults to SLURM_JOB_NAME, else
`MakeSplit` -- which is also every reader's default `--split-cache-job`. Run it
inside another job's sbatch and pass `--cache-job MakeSplit`, or the readers
will not find it.

A SPEC is `<dataset>:<set>=<size>,...` with set one of val / test / train and size
a count (`10`), a share (`0.6`) or `rest`. The pool is sorted, shuffled by
`--seed`, and cut val, then test, then train; see `WsiSplit` for why that order
keeps every split already recorded readable as it was.

`--check` exits 1 if a spec CONTRADICTS a record (`mismatch`); `same`,
`upgradable` and `missing` exit 0.

Every other package READS the split, as the dataset `<id>#<split>` of
AccessDatasets (`list_names(dataset='bracs/test#val')`), and refuses when it is
missing: MppRoutingHead,
PrototypicalRoutingHead, the stage-1 bench, the WSI diagnostics. BRACS datasets
draw only from slides native at every `BRACS_RUNGS` rung, which opens every slide
in the dataset to read its pyramid -- header only, seconds.
'''
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _paths import setup_import_paths                                  # noqa: E402
setup_import_paths()

from Cache import job_name                                          # noqa: E402
from WsiSplit import (SPLIT_JOB, SPLIT_SETS, check_split, make_split,  # noqa: E402
                      make_split_spec, parse_spec, split_path, upgrade_split)


def _specs(items):
    """`[(dataset, spec)]` from `dataset:sets` items."""
    out = []
    for item in items:
        dataset_id, sep, text = item.partition(':')
        if not sep or not dataset_id:
            raise SystemExit(f'--spec {item!r}: want <dataset>:<set>=<size>,... '
                             f'(e.g. bracs/test:val=10,test=rest)')
        try:
            out.append((dataset_id, parse_spec(text)))
        except ValueError as exc:
            raise SystemExit(f'--spec {item!r}: {exc}')
    return out


def _sizes(sets) -> str:
    return '  '.join(f'{name} {len(sets[name])}' for name in SPLIT_SETS
                     if name in sets)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+',
                    default=['bracs/test', 'ki67_with_photo'],
                    help='without --spec: these datasets get val-n-wsi val slides '
                         'and the rest as test')
    ap.add_argument('--val-n-wsi', type=int, default=10,
                    help='WSIs held out per dataset for val; the rest is test')
    ap.add_argument('--spec', nargs='+', metavar='DATASET:SETS', default=None,
                    help='val / test / train per dataset, e.g. '
                         'bracs/test:val=10,test=20,train=rest')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true',
                      help='say what each spec would do to the recorded split; '
                           'write nothing')
    mode.add_argument('--upgrade', action='store_true',
                      help='add train to a recorded split the spec only extends, '
                           'keeping the old split beside it as orig_recorded.csv')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cache-job', default=None,
                    help=f'whose cache, result/cache/<made_by>/, the split is written to. '
                         f'Default: this job (SLURM_JOB_NAME or {SPLIT_JOB})')
    args = ap.parse_args()

    made_by = args.cache_job or job_name(SPLIT_JOB)

    if args.spec is None:
        if args.check or args.upgrade:
            ap.error('--check and --upgrade need --spec')
        for dataset_id in args.datasets:
            path = split_path(made_by, dataset_id)
            existed = path.exists()
            val, test = make_split(dataset_id, args.val_n_wsi, made_by,
                                   seed=args.seed)
            print(f'{dataset_id:17s} {"kept   " if existed else "written"} {path}\n'
                  f'{"":17s} val {len(val)}: {", ".join(val)}\n'
                  f'{"":17s} test {len(test)}', flush=True)
        return 0

    worst = 0
    for dataset_id, spec in _specs(args.spec):
        path = split_path(made_by, dataset_id)
        if args.check:
            status, reasons = check_split(dataset_id, spec, made_by, seed=args.seed)
            print(f'{dataset_id:17s} {status.upper():10s} {path}', flush=True)
            for reason in reasons:
                print(f'{"":17s}   {reason}')
            worst = max(worst, 1 if status == 'mismatch' else 0)
        elif args.upgrade:
            try:
                result = upgrade_split(dataset_id, spec, made_by, seed=args.seed)
            except (ValueError, FileExistsError) as exc:
                print(f'{dataset_id:17s} REFUSED    {exc}', flush=True)
                worst = 1
                continue
            print(f'{dataset_id:17s} {result.upper():10s} {path}', flush=True)
        else:
            existed = path.exists()
            sets = make_split_spec(dataset_id, spec, made_by, seed=args.seed)
            print(f'{dataset_id:17s} {"kept   " if existed else "written"} {path}\n'
                  f'{"":17s} {_sizes(sets)}', flush=True)
            if 'val' in sets:
                print(f'{"":17s} val: {", ".join(sets["val"])}')
            if existed:
                status, reasons = check_split(dataset_id, spec, made_by, seed=args.seed)
                print(f'{"":17s} against this spec: {status}'
                      + (f' ({"; ".join(reasons)})' if reasons else ''))
    return worst


if __name__ == '__main__':
    raise SystemExit(main())
