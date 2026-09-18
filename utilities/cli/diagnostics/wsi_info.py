#!/usr/bin/env python3
"""Dump one or more WSIs' own metadata: vendor, level count, per-level
size/downsample/mpp, and (by default) every raw openslide property.

Usage:
    python utilities/cli/diagnostics/wsi_info.py <wsi_path>            # original, single-file calling convention
    python utilities/cli/diagnostics/wsi_info.py --wsi <path> [<path> ...]
    python utilities/cli/diagnostics/wsi_info.py --dataset ki67_pure bracs/test
    python utilities/cli/diagnostics/wsi_info.py --dataset bracs/test --no-properties --out result/wsi_info.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

import openslide                                                    # noqa: E402

from SlideProbe import get_mpp                                       # noqa: E402
from WsiSelection import resolve_wsi_paths                            # noqa: E402


class WsiInfoCheck:
    """`run(entries)` is the shape every diagnostic in this directory
    shares (see `WsiSelection.py`). `full=True` (default) prints every raw
    openslide property per slide, same as the tool's original single-file
    behaviour -- turn it off (`--no-properties`) once `entries` is more than
    a handful, or that dump is most of the output."""

    def __init__(self, full: bool = True):
        self.full = full

    def run_one(self, entry: dict) -> dict:
        wsi = openslide.OpenSlide(entry['path'])
        try:
            p = wsi.properties
            mpp_x, mpp_y = get_mpp(p)
            W0, H0 = wsi.level_dimensions[0]

            print(f'\n{"─" * 52}')
            print(f'  File     : {entry["path"]}')
            print(f'  Format   : {p.get("openslide.vendor", "unknown")}')
            print(f'  Levels   : {wsi.level_count}')
            print(f'  L0 size  : {W0} x {H0} px')
            if mpp_x:
                print(f'  MPP (x,y): {mpp_x:.4f}, {mpp_y:.4f} um/px')
                print(f'  L0 FoV   : {W0 * mpp_x / 1000:.2f} x '
                     f'{H0 * mpp_y / 1000:.2f} mm')
            else:
                print('  MPP      : not found in metadata')

            print(f'\n  {"Level":>5}  {"Width":>10}  {"Height":>10}  '
                 f'{"Downsample":>10}  {"MPP-x":>8}')
            print(f'  {"─" * 5}  {"─" * 10}  {"─" * 10}  {"─" * 10}  {"─" * 8}')
            for lv in range(wsi.level_count):
                W, H = wsi.level_dimensions[lv]
                ds = wsi.level_downsamples[lv]
                mpp = f'{mpp_x * ds:.4f}' if mpp_x else 'N/A'
                print(f'  {lv:>5}  {W:>10}  {H:>10}  {ds:>10.2f}  {mpp:>8}')

            if self.full:
                print(f'\n  {"─" * 52}')
                print('  All properties:')
                for k, v in sorted(p.items()):
                    print(f'    {k} = {v}')
                print()

            return dict(dataset=entry.get('dataset'), wsi_name=entry['wsi_name'],
                       path=entry['path'],
                       vendor=p.get('openslide.vendor', 'unknown'),
                       levels=wsi.level_count, width=W0, height=H0,
                       mpp_x=mpp_x, mpp_y=mpp_y)
        finally:
            wsi.close()

    def run(self, entries: list) -> list:
        return [self.run_one(e) for e in entries]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wsi_path', nargs='?', default=None,
                    help='original single-file calling convention; '
                         'combined with --wsi/--dataset if those are also given')
    ap.add_argument('--dataset', nargs='+', default=None)
    ap.add_argument('--wsi', nargs='+', default=None,
                    help='explicit WSI path(s)')
    ap.add_argument('--val-only', action='store_true')
    ap.add_argument('--no-properties', action='store_true',
                    help='skip the full raw-property dump per slide -- '
                         'default is to show it, which gets long fast over '
                         'many slides')
    ap.add_argument('--out', default=None,
                    help='CSV path for the summary rows -- written whenever '
                         'more than one WSI is scanned, or always with this '
                         'flag set. Default: result/<SLURM_JOB_NAME or '
                         'WsiInfo>/wsi_info.csv')
    args = ap.parse_args()

    wsi_paths = list(args.wsi or [])
    if args.wsi_path:
        wsi_paths.append(args.wsi_path)
    if not args.dataset and not wsi_paths:
        ap.error('need --dataset and/or --wsi (or a single positional path)')

    entries = resolve_wsi_paths(dataset=args.dataset, wsi=wsi_paths or None,
                                val_only=args.val_only)
    rows = WsiInfoCheck(full=not args.no_properties).run(entries)

    if args.out or len(rows) > 1:
        out_path = args.out or os.path.join(
            _paths.job_result_dir('WsiInfo'), 'wsi_info.csv')
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f'\n{out_path}  ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
