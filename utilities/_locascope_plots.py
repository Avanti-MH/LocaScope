"""metrics.csv -> summary + figures. Pure post-processing; no torch / no WSI.

Lives in utilities/ rather than beside either caller because both need it:
`test_modules/bench_locascope.py` (writes metrics during the run) and
`cli/plot_locascope_metrics.py` (re-plots an existing metrics.csv offline).

Every function takes `metrics: List[dict]` — the same row shape that
bench_locascope.compute_metrics produces, and that `load_metrics_csv`
reconstructs from disk (typed, with '' restored to None).
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── CSV round-trip ────────────────────────────────────────────────────────────

_BOOL_COLS = {'unusable_level', 'rot_correct', 'refine_success'}
_INT_COLS  = {'level', 'gt_x', 'gt_y', 'routed_level', 'gt_rot_deg',
              'retr_rotation', 'retr_x0', 'retr_y0', 'refine_x0', 'refine_y0',
              'refine_inliers', 'refine_matches'}
_STR_COLS  = {'filename', 'wsi_path', 'error'}


def _coerce(key: str, val: str):
    if key in _STR_COLS:
        return val
    if val == '' or val is None:
        return None
    if key in _BOOL_COLS:
        return val == 'True'
    try:
        return int(val) if key in _INT_COLS else float(val)
    except ValueError:
        return val


def load_metrics_csv(path: str) -> List[dict]:
    with open(path) as f:
        return [{k: _coerce(k, v) for k, v in row.items()}
                for row in csv.DictReader(f)]


def write_metrics_csv(metrics: List[dict], out_path: str) -> None:
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)
    print(f'metrics.csv -> {out_path}  ({len(metrics)} rows)')


def wsi_tag(m: dict) -> str:
    return os.path.splitext(os.path.basename(m['wsi_path']))[0]


# ── summary.txt ───────────────────────────────────────────────────────────────

def write_summary(metrics: List[dict], out_path: str) -> None:
    lines: List[str] = []
    lines.append(f'Total shots     : {len(metrics)}')
    lines.append(f'Stage errors    : {sum(1 for m in metrics if m["error"])}')
    lines.append(f'Unusable levels : {sum(1 for m in metrics if m["unusable_level"])}')

    routed = [m for m in metrics if m['routed_level'] is not None]
    if routed:
        n_ok = sum(1 for m in routed if m['routed_level'] == m['level'])
        lines.append(f'Level routing   : {n_ok}/{len(routed)} '
                     f'({100.0 * n_ok / len(routed):.1f}%) routed_level == level')
    rot_known = [m for m in metrics if m['rot_correct'] is not None]
    if rot_known:
        n_ok = sum(1 for m in rot_known if m['rot_correct'])
        lines.append(f'Rotation recall : {n_ok}/{len(rot_known)} '
                     f'({100.0 * n_ok / len(rot_known):.1f}%) retr_rotation == gt_rot_deg')
    lines.append(f'SIFT success    : {sum(1 for m in metrics if m["refine_success"])}'
                 f'/{len(metrics)}')
    lines.append('')

    for key, label, unit in [
        ('mpp_err_rel',          'Stage 1 (mpp err, relative)',                  ''),
        ('retr_center_err_px',   'Stage 2 (retrieval CENTRE err)',               ' px'),
        ('retr_center_err_um',   'Stage 2 (retrieval CENTRE err)',               ' um'),
        ('refine_center_err_px', 'Stage 3 (refine CENTRE err)',                  ' px'),
        ('refine_center_err_um', 'Stage 3 (refine CENTRE err)',                  ' um'),
        ('retr_err_px',          'Stage 2 (retrieval top-left err, rot-naive)',  ' px'),
        ('refine_err_px',        'Stage 3 (refine top-left err, rot-naive)',     ' px'),
    ]:
        vals = np.array([m[key] for m in metrics if m.get(key) is not None], dtype=float)
        lines.append(label)
        if vals.size == 0:
            lines.append('  no data')
            lines.append('')
            continue
        lines.append(f'  n     = {vals.size}')
        lines.append(f'  min   = {vals.min():.4f}{unit}')
        lines.append(f'  p50   = {np.percentile(vals, 50):.4f}{unit}')
        if vals.size >= 20:
            lines.append(f'  p90   = {np.percentile(vals, 90):.4f}{unit}')
        if vals.size >= 50:
            lines.append(f'  p95   = {np.percentile(vals, 95):.4f}{unit}')
        if vals.size < 20:
            lines.append(f'  (n<20: p90/p95 omitted — they would be pure '
                         f'interpolation toward max)')
        lines.append(f'  max   = {vals.max():.4f}{unit}')
        lines.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'summary.txt -> {out_path}')


# ── CDF ───────────────────────────────────────────────────────────────────────

def _plot_cdf_line(ax, sorted_vals: np.ndarray, xlabel: str) -> None:
    n = sorted_vals.size
    ax.step(sorted_vals, np.arange(1, n + 1) / n, where='post', linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    # Small n: show the raw samples as rug ticks, percentiles are meaningless
    if n < 20:
        ax.plot(sorted_vals, np.zeros(n), '|', color='crimson', ms=10, mew=1.4)
    p50 = float(np.percentile(sorted_vals, 50))
    ax.axvline(p50, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.text(p50, 0.03, f'p50={p50:.3g}', fontsize=7, color='gray',
            rotation=90, va='bottom', ha='right')
    if n >= 20:
        p90 = float(np.percentile(sorted_vals, 90))
        ax.axvline(p90, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
        ax.text(p90, 0.03, f'p90={p90:.3g}', fontsize=7, color='gray',
                rotation=90, va='bottom', ha='right')


def plot_stage_cdfs(metrics: List[dict], stage_key: str, xlabel: str,
                    title: str, out_path: str) -> None:
    all_vals = np.array(sorted(m[stage_key] for m in metrics
                               if m.get(stage_key) is not None), dtype=float)
    if all_vals.size == 0:
        print(f'  [skip plot] {stage_key}: no data')
        return

    per_wsi: Dict[str, List[float]] = defaultdict(list)
    for m in metrics:
        if m.get(stage_key) is not None:
            per_wsi[wsi_tag(m)].append(m[stage_key])

    tags   = sorted(per_wsi)
    n_cols = min(max(len(tags), 1), 3)
    n_rows = math.ceil(len(tags) / n_cols) if tags else 0

    fig = plt.figure(figsize=(max(9, n_cols * 4.5), 3.5 + n_rows * 3.0))
    gs  = fig.add_gridspec(1 + n_rows, n_cols)

    ax_all = fig.add_subplot(gs[0, :])
    _plot_cdf_line(ax_all, all_vals, xlabel)
    note = '  (n<20: red ticks = raw samples)' if all_vals.size < 20 else ''
    ax_all.set_title(f'Aggregate  (n={all_vals.size}){note}')

    for i, tag in enumerate(tags):
        ax = fig.add_subplot(gs[1 + i // n_cols, i % n_cols])
        _plot_cdf_line(ax, np.array(sorted(per_wsi[tag]), dtype=float), xlabel)
        ax.set_title(f'{tag}  (n={len(per_wsi[tag])})', fontsize=10)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)} -> {out_path}')


# ── heatmap: (WSI, level) x stage ─────────────────────────────────────────────

def plot_heatmap(metrics: List[dict], out_path: str) -> None:
    per_key: Dict[tuple, List[dict]] = defaultdict(list)
    for m in metrics:
        per_key[(wsi_tag(m), m['level'])].append(m)

    keys       = sorted(per_key)
    stages     = ['mpp_err_rel', 'retr_center_err_px', 'refine_center_err_px']
    stage_lbls = ['mpp err (rel)', 'retr ctr err (px)', 'refine ctr err (px)']

    grid = np.full((len(keys), 3), np.nan)
    for i, key in enumerate(keys):
        for j, s in enumerate(stages):
            vals = [r[s] for r in per_key[key] if r.get(s) is not None]
            if vals:
                grid[i, j] = float(np.median(vals))

    color = np.zeros_like(grid)
    for j in range(3):
        col = grid[:, j]
        v = ~np.isnan(col)
        if v.any() and col[v].max() > col[v].min():
            color[:, j] = (col - col[v].min()) / (col[v].max() - col[v].min())
    color[np.isnan(grid)] = np.nan

    fig, ax = plt.subplots(figsize=(7, max(3, len(keys) * 0.42)))
    ax.imshow(color, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(stage_lbls)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f'{w}  L{l}' for w, l in keys], fontsize=8)
    for i in range(len(keys)):
        for j in range(3):
            v = grid[i, j]
            txt = 'N/A' if np.isnan(v) else (f'{v:.3f}' if j == 0 else f'{v:.0f}')
            c = 'white' if (not np.isnan(color[i, j]) and color[i, j] > 0.55) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color=c)
    ax.set_title('Per-(WSI, level) median error  (color = column-normalized)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)} -> {out_path}')


# ── confusion matrices: level routing + rotation ──────────────────────────────

def _confusion(ax, pairs, labels, title, xlabel, ylabel):
    """pairs = [(truth, pred), ...]; labels = sorted unique values to show."""
    idx = {v: i for i, v in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)))
    for t, p in pairs:
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1

    ax.imshow(mat, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    vmax = mat.max() if mat.size else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            if mat[i, j]:
                ax.text(j, i, int(mat[i, j]), ha='center', va='center',
                        fontsize=10,
                        color='white' if mat[i, j] > vmax * 0.55 else 'black')
    # Highlight the diagonal (correct predictions)
    for i in range(len(labels)):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor='limegreen', lw=1.8))
    n_ok = int(np.trace(mat)); n_all = int(mat.sum())
    acc = f'{100.0 * n_ok / n_all:.0f}%' if n_all else 'n/a'
    ax.set_title(f'{title}\n{n_ok}/{n_all} on diagonal ({acc})')


def plot_confusions(metrics: List[dict], out_path: str) -> None:
    """Level routing + rotation recovery — the two discrete decisions.

    These are the honest view at small n, where percentiles are meaningless.
    """
    lvl_pairs = [(m['level'], m['routed_level']) for m in metrics
                 if m['routed_level'] is not None]
    rot_pairs = [(m['gt_rot_deg'], m['retr_rotation']) for m in metrics
                 if m.get('retr_rotation') is not None]
    if not lvl_pairs and not rot_pairs:
        print('  [skip plot] confusions: no data')
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    if lvl_pairs:
        lvls = sorted({v for p in lvl_pairs for v in p})
        _confusion(axes[0], lvl_pairs, lvls,
                   'Stage 1 -> level routing', 'routed_level', 'true level')
    else:
        axes[0].set_title('level routing — no data'); axes[0].axis('off')

    if rot_pairs:
        rots = sorted({v for p in rot_pairs for v in p})
        _confusion(axes[1], rot_pairs, rots,
                   'Stage 2 -> rotation recovery', 'retr_rotation', 'gt_rot_deg')
    else:
        axes[1].set_title('rotation recovery — no data'); axes[1].axis('off')

    fig.suptitle('Discrete decisions (green box = correct)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)} -> {out_path}')


# ── mpp scatter ───────────────────────────────────────────────────────────────

def plot_mpp_scatter(metrics: List[dict], out_path: str) -> None:
    pts = [(m['effective_mpp'], m['est_mpp'], m['level'],
            m['routed_level'] == m['level'])
           for m in metrics if m.get('est_mpp') is not None]
    if not pts:
        print('  [skip plot] mpp scatter: no data')
        return

    eff = np.array([p[0] for p in pts])
    est = np.array([p[1] for p in pts])
    ok  = np.array([p[3] for p in pts])

    fig, ax = plt.subplots(figsize=(6.4, 6))
    lo = min(eff.min(), est.min()) * 0.6
    hi = max(eff.max(), est.max()) * 1.6
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='perfect')
    ax.scatter(eff[ok],  est[ok],  s=55, c='seagreen', marker='o',
               edgecolors='black', linewidths=0.5, label='level routed OK')
    ax.scatter(eff[~ok], est[~ok], s=75, c='crimson', marker='X',
               edgecolors='black', linewidths=0.5, label='level routed WRONG')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('effective_mpp (ground truth, um/px)')
    ax.set_ylabel('est_mpp (predicted, um/px)')
    ax.set_title('Stage 1 — mpp estimate vs truth\n'
                 'est_mpp is level-quantized, so points land on discrete rows')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)} -> {out_path}')


# ── stage progression: retrieval -> refine, per shot ──────────────────────────

def plot_stage_progression(metrics: List[dict], out_path: str) -> None:
    rows = [m for m in metrics
            if m.get('retr_center_err_px') is not None
            and m.get('refine_center_err_px') is not None]
    if not rows:
        print('  [skip plot] stage progression: no data')
        return
    rows.sort(key=lambda m: m['retr_center_err_px'])

    r = np.array([m['retr_center_err_px']   for m in rows])
    s = np.array([m['refine_center_err_px'] for m in rows])
    ok = np.array([bool(m['refine_success']) for m in rows])
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: paired per-shot slope chart
    ax = axes[0]
    for i in range(len(rows)):
        ax.plot([0, 1], [r[i], s[i]],
                color=('seagreen' if s[i] < r[i] else 'crimson'),
                alpha=0.65, lw=1.2, marker='o', ms=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['retrieval', 'SIFT refine'])
    ax.set_yscale('symlog', linthresh=10)
    ax.set_ylabel('centre error (level-0 px)')
    n_better = int((s < r).sum())
    ax.set_title(f'Per-shot stage 2 -> 3\n'
                 f'green = SIFT improved ({n_better}/{len(rows)})')
    ax.grid(True, alpha=0.3, axis='y')

    # Right: sorted bars, retrieval vs refine
    ax = axes[1]
    ax.bar(x - 0.2, r, width=0.4, label='retrieval', color='goldenrod')
    ax.bar(x + 0.2, s, width=0.4,
           color=np.where(ok, 'dodgerblue', 'lightgray'),
           label='refine (gray = SIFT failed)')
    ax.set_yscale('symlog', linthresh=10)
    ax.set_xlabel('shots (sorted by retrieval error)')
    ax.set_ylabel('centre error (level-0 px)')
    ax.set_title('Error per shot')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Stage progression — does SIFT actually refine?', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {os.path.basename(out_path)} -> {out_path}')


# ── one call to make everything ───────────────────────────────────────────────

def render_all(metrics: List[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    write_summary(metrics, os.path.join(out_dir, 'summary.txt'))
    plot_confusions(metrics, os.path.join(out_dir, 'confusions.png'))
    plot_mpp_scatter(metrics, os.path.join(out_dir, 'mpp_scatter.png'))
    plot_stage_progression(metrics, os.path.join(out_dir, 'stage_progression.png'))
    plot_stage_cdfs(metrics, 'mpp_err_rel',
                    xlabel='|est - effective| / effective',
                    title='Stage 1 — MPP estimation error',
                    out_path=os.path.join(out_dir, 'stage1_mpp_cdf.png'))
    plot_stage_cdfs(metrics, 'retr_center_err_px',
                    xlabel='retrieval centre err  (level-0 pixels)',
                    title='Stage 2 — retrieval error (rotation-invariant centre)',
                    out_path=os.path.join(out_dir, 'stage2_retr_cdf.png'))
    plot_stage_cdfs(metrics, 'refine_center_err_px',
                    xlabel='refine centre err  (level-0 pixels)',
                    title='Stage 3 — SIFT+RANSAC error (rotation-invariant centre)',
                    out_path=os.path.join(out_dir, 'stage3_refine_cdf.png'))
    plot_heatmap(metrics, os.path.join(out_dir, 'heatmap.png'))
