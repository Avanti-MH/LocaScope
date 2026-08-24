#!/usr/bin/env python3
"""Re-plot an existing bench_locascope metrics.csv — no GPU, no WSI, no torch.

The bench already writes these figures at the end of a run; this script exists
so you can iterate on the visualisation (or re-plot a filtered subset) without
paying for another pipeline run.

Usage:
    python utilities/cli/plot_locascope_metrics.py \\
        result/BenchLocaScope/metrics.csv                       # in place
    python utilities/cli/plot_locascope_metrics.py \\
        result/BenchLocaScope/metrics.csv --out result/Replot    # elsewhere

Filters (applied before plotting, all optional):
    --wsi SUBSTR        keep rows whose wsi_path contains SUBSTR
    --level N [N ...]   keep only these source levels
    --rot   D [D ...]   keep only these ground-truth rotations
    --only-routed-ok    keep only rows where routed_level == level
    --only-sift-ok      keep only rows where refine_success

Outputs (into --out, default = the metrics.csv's own directory):
    summary.txt              aggregate stats, level routing + rotation recall
    confusions.png           level routing + rotation confusion matrices
    mpp_scatter.png          est_mpp vs effective_mpp (log-log)
    stage_progression.png    per-shot retrieval -> refine
    stage{1,2,3}_*_cdf.png   CDFs, aggregate + per WSI
    heatmap.png              (WSI, level) x stage median error
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # utilities/

from dump_function._locascope_plots import (load_metrics_csv,   # noqa: E402
                                            render_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('metrics_csv')
    ap.add_argument('--out', default=None,
                    help="Output dir (default: the metrics.csv's own directory)")
    ap.add_argument('--wsi',   default=None, metavar='SUBSTR')
    ap.add_argument('--level', type=int, nargs='+', default=None)
    ap.add_argument('--rot',   type=int, nargs='+', default=None)
    ap.add_argument('--only-routed-ok', action='store_true')
    ap.add_argument('--only-sift-ok',   action='store_true')
    args = ap.parse_args()

    metrics = load_metrics_csv(args.metrics_csv)
    print(f'Loaded {len(metrics)} rows from {args.metrics_csv}')

    if args.wsi:
        metrics = [m for m in metrics if args.wsi in m['wsi_path']]
    if args.level is not None:
        metrics = [m for m in metrics if m['level'] in set(args.level)]
    if args.rot is not None:
        metrics = [m for m in metrics if m['gt_rot_deg'] in set(args.rot)]
    if args.only_routed_ok:
        metrics = [m for m in metrics if m['routed_level'] == m['level']]
    if args.only_sift_ok:
        metrics = [m for m in metrics if m['refine_success']]

    if not metrics:
        sys.exit('No rows left after filtering.')
    print(f'Plotting {len(metrics)} rows')

    out_dir = args.out or os.path.dirname(os.path.abspath(args.metrics_csv))
    print(f'Output -> {out_dir}\n')
    render_all(metrics, out_dir)


if __name__ == '__main__':
    main()
