#!/usr/bin/env python3
"""Per-slide, per-rung: is this WSI's own pyramid NATIVE at each rung, and
how much does `base_mpp` itself vary across a dataset -- both without
rendering anything (no GPU, no Camera, no encoder), just the level-choice
arithmetic `QueryFromWSI.__init__` already does.

Moved here 2026-09-18 from `training/MppRoutingHead/datasets/
diagnose_native_split.py`, which was written as a one-off to answer why
bracs/test's val split showed n_native=220 against n_resampled=870 -- see
`training/MppRoutingHead/spec.md`'s "QueryFromWSI.reads_natively was too
strict" for that story. Its own docstring said "kept... in case the same
question comes up again", and it has: the native-rung question and "how
much does base_mpp vary WSI to WSI" (2026-09-18, generalizing to any WSI
without retraining) are the same underlying per-slide metadata scan, asked
two ways. Generalized here: any registered dataset, not one hardcoded id;
every WSI in it by default, not only a recorded val split; plus a
`base_mpp` distribution summary. Refactored again the same day into
`WsiScaleCheck`, the shape `wsi_health_check.py` calls this and every other
diagnostic in this directory through -- see `WsiSelection.py`'s own
docstring for the "which WSIs" half of that shape.

Read-only, cheap: opens each WSI once for its pyramid metadata (`base_mpp`,
`level_downsamples`) plus `QueryFromWSI`'s own level-choice arithmetic --
no pixel reads.

Usage:
    python utilities/cli/diagnostics/diag_wsi_scale.py                      # every registered dataset, every WSI
    python utilities/cli/diagnostics/diag_wsi_scale.py --dataset ki67_pure bracs/test
    python utilities/cli/diagnostics/diag_wsi_scale.py --dataset bracs/test --val-only  # only the recorded val split -- the original tool's own scope
    python utilities/cli/diagnostics/diag_wsi_scale.py --wsi /path/to/one.svs /path/to/two.mrxs
    python utilities/cli/diagnostics/diag_wsi_scale.py --quiet              # summaries only, skip the per-slide/per-rung dump
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

from DsLadder import DEFAULT_RUNGS                                   # noqa: E402
from QueryFromWSI import QueryFromWSI                                 # noqa: E402
from SafeSlide import SafeSlide                                      # noqa: E402
from WsiSelection import resolve_wsi_paths                            # noqa: E402


class WsiScaleCheck:
    """Per-slide, per-rung nativity + `base_mpp`. `run(entries)` is the
    shape every diagnostic in this directory shares (see `WsiSelection.py`):
    a list of resolved `{'dataset', 'wsi_name', 'path'}` in, a list of row
    dicts out -- this tool's own rows carry `base_mpp` and a `native`
    sub-dict keyed by rung, one bool each.
    """

    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def run_one(self, entry: dict):
        """One resolved WSI entry -> a row dict, or `None` on a read
        failure (printed, not raised -- one bad slide should not stop a
        batch of hundreds)."""
        try:
            wsi = SafeSlide(entry['path'])
        except Exception as exc:                                   # noqa: BLE001
            print(f'  [SKIP] {entry.get("dataset")}/{entry["wsi_name"]}: '
                 f'{type(exc).__name__}: {exc}')
            return None
        try:
            row = dict(dataset=entry.get('dataset'),
                      wsi_name=entry['wsi_name'], base_mpp=wsi.base_mpp)
            # FULL precision -- round(d, 2) hid the exact difference that
            # flips a rung's native flag between slides whose ds prints
            # identically at 2 decimals (the original bug this tool found).
            dss = [float(d) for d in wsi.level_downsamples]
            if not self.quiet:
                print(f'{entry["wsi_name"]:28s} base_mpp={wsi.base_mpp!r}')
                print(f'    levels(ds)={dss!r}')
            native = {}
            for rung in DEFAULT_RUNGS:
                mpp = wsi.base_mpp * rung
                qfw = QueryFromWSI(wsi, wh_ratio='1:1',
                                   MPixels=(256 ** 2) / 1e6, mpp=mpp)
                is_native = qfw.reads_natively
                native[rung] = is_native
                if not self.quiet:
                    chosen_ds = dss[qfw.chosen_level]
                    diff = abs(chosen_ds - rung) / chosen_ds
                    print(f'    rung {rung:>5g}  chosen_level={qfw.chosen_level}  '
                         f'chosen_ds={chosen_ds!r}  diff={diff:.6f}  '
                         f'native={is_native}')
            row['native'] = native
            return row
        finally:
            wsi.close()

    def run(self, entries: list) -> list:
        return [r for r in (self.run_one(e) for e in entries) if r is not None]


def print_mpp_summary(rows: list) -> None:
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r['dataset'], []).append(r['base_mpp'])

    print(f'  {"dataset":20s}{"n":>5s}{"min":>10s}{"max":>10s}{"mean":>10s}'
         f'{"median":>10s}{"stdev":>10s}{"max/min":>10s}')
    for dataset_id, values in sorted(by_dataset.items(), key=lambda kv: kv[0] or ''):
        lo, hi = min(values), max(values)
        print(f'  {str(dataset_id):20s}{len(values):>5d}{lo:>10.4f}{hi:>10.4f}'
             f'{statistics.mean(values):>10.4f}{statistics.median(values):>10.4f}'
             f'{(statistics.stdev(values) if len(values) > 1 else 0.0):>10.4f}'
             f'{(hi / lo if lo else float("nan")):>10.2f}')

    all_values = [r['base_mpp'] for r in rows]
    if len(by_dataset) > 1 and all_values:
        lo, hi = min(all_values), max(all_values)
        print(f'  {"(all datasets)":20s}{len(all_values):>5d}{lo:>10.4f}'
             f'{hi:>10.4f}{statistics.mean(all_values):>10.4f}'
             f'{statistics.median(all_values):>10.4f}'
             f'{(statistics.stdev(all_values) if len(all_values) > 1 else 0.0):>10.4f}'
             f'{(hi / lo if lo else float("nan")):>10.2f}')


def print_native_summary(rows: list) -> None:
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r['dataset'], []).append(r)
    for dataset_id, drows in sorted(by_dataset.items(), key=lambda kv: kv[0] or ''):
        n = len(drows)
        print(f'\n{dataset_id} ({n} slide(s)):')
        for rung in DEFAULT_RUNGS:
            native_n = sum(1 for r in drows if r['native'][rung])
            print(f'  rung {rung:>5g}: native on {native_n}/{n} slides, '
                 f'resampled on {n - native_n}/{n} slides')


def write_csv(rows: list, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fieldnames = ['dataset', 'wsi_name', 'base_mpp'] + \
        [f'native_{r:g}' for r in DEFAULT_RUNGS]
    with open(out_path, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            flat = dict(dataset=r['dataset'], wsi_name=r['wsi_name'],
                       base_mpp=r['base_mpp'])
            flat.update({f'native_{rung:g}': r['native'][rung]
                        for rung in DEFAULT_RUNGS})
            wr.writerow(flat)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', nargs='+', default=None,
                    help='default (with no --wsi): every registered dataset '
                         '(AccessDatasets.dataset_ids())')
    ap.add_argument('--wsi', nargs='+', default=None,
                    help='explicit WSI path(s), for a slide not in the '
                         'registry -- combined with --dataset if both given')
    ap.add_argument('--val-only', action='store_true',
                    help="only the recorded val split (needs "
                         "utilities/cli/build_cache/make_split.py to have "
                         "written it) -- the original diagnose_native_split.py's "
                         "own scope. Default: every WSI in the dataset")
    ap.add_argument('--quiet', action='store_true',
                    help='skip the per-slide/per-rung dump, print only the '
                         'summaries')
    ap.add_argument('--out', default=None,
                    help='CSV path for the per-WSI rows. Default: '
                         'result/<SLURM_JOB_NAME or DiagWsiScale>/base_mpp.csv')
    args = ap.parse_args()

    from AccessDatasets import dataset_ids                          # noqa: PLC0415
    datasets = args.dataset or (None if args.wsi else dataset_ids())
    entries = resolve_wsi_paths(dataset=datasets, wsi=args.wsi,
                                val_only=args.val_only)
    print(f'{len(entries)} WSI(s)'
         f'{f" across {len(datasets)} dataset(s)" if datasets else ""}'
         f'{"  (--val-only)" if args.val_only else ""}')

    rows = WsiScaleCheck(quiet=args.quiet).run(entries)
    if not rows:
        print('\nnothing scanned')
        return 1

    print('\n' + '=' * 74)
    print('base_mpp distribution')
    print('=' * 74)
    print_mpp_summary(rows)

    print('\n' + '=' * 74)
    print('native vs resampled, per rung')
    print('=' * 74)
    print_native_summary(rows)

    out_path = args.out or os.path.join(
        _paths.job_result_dir('DiagWsiScale'), 'base_mpp.csv')
    write_csv(rows, out_path)
    print(f'\n{out_path}  ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
