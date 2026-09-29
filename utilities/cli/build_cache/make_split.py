#!/usr/bin/env python3
'''Write the val/test WSI split -- the ONE tool that does.

    python utilities/cli/build_cache/make_split.py
    python utilities/cli/build_cache/make_split.py --datasets bracs/test ki67_with_photo

Writes `result/cache/<cache-job>_split/<dataset>/wsi_split.csv` for each dataset
that has none yet, and leaves every existing one alone (EXISTING WINS -- see
`WsiSplit.make_split`). `<cache-job>` defaults to SLURM_JOB_NAME, else
`MakeSplit` -- which is also every reader's default `--split-cache-job`. Run it
inside another job's sbatch and pass `--cache-job MakeSplit`, or the readers
will not find it.

Every other package READS the split (`WsiSplit.read_split`) and refuses when it
is missing: MppRoutingHead, PrototypicalRoutingHead, the stage-1 bench, the WSI
diagnostics. BRACS datasets draw only from slides native at every `BRACS_RUNGS`
rung, which opens every slide in the dataset to read its pyramid -- header
only, seconds.
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
from WsiSplit import SPLIT_JOB, make_split, split_path              # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+',
                    default=['bracs/test', 'ki67_with_photo'])
    ap.add_argument('--val-n-wsi', type=int, default=10,
                    help='WSIs held out per dataset for val; the rest is test')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cache-job', default=None,
                    help=f'the <made_by> of result/cache/<made_by>_split/. '
                         f'Default: this job (SLURM_JOB_NAME or {SPLIT_JOB})')
    args = ap.parse_args()

    made_by = args.cache_job or job_name(SPLIT_JOB)
    for dataset_id in args.datasets:
        path = split_path(made_by, dataset_id)
        existed = path.exists()
        val, test = make_split(dataset_id, args.val_n_wsi, path, seed=args.seed)
        print(f'{dataset_id:17s} {"kept   " if existed else "written"} {path}\n'
              f'{"":17s} val {len(val)}: {", ".join(val)}\n'
              f'{"":17s} test {len(test)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
