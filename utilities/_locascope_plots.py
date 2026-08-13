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
              'refine_inliers', 'refine_matches',
              'retr_topk_n', 'retr_hit_rank', 'retr_hit_rank_strict',
              'sift_topk_n', 'sift_hit_rank', 'sift_verified_rank',
              'sift_best_inliers', 'sift_best_rank'}
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


def append_metrics_row(m: dict, out_path: str,
                       fieldnames: Optional[List[str]] = None) -> List[str]:
    """Append one row, writing the header into an empty file. Returns the
    fieldnames to hand back on the next call.

    Per shot rather than once at the end, because the end is not guaranteed to
    arrive. A bench over a whole corpus is hours of retriever builds and query
    encodes, and a crash at the last shot used to return nothing at all -- the
    same way a batch that died on its fifth slide used to throw away four
    slides of images. compute_metrics declares every field up front with None
    defaults, so the first row fixes the columns and the rest align with it;
    pass the returned list back so a later row cannot silently re-order them.
    """
    if fieldnames is None:
        fieldnames = list(m.keys())
    write_header = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
    with open(out_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(m)
    return fieldnames


def wsi_tag(m: dict) -> str:
    return os.path.splitext(os.path.basename(m['wsi_path']))[0]


# ── stage 2 recall@K ──────────────────────────────────────────────────────────

K_GRID = (1, 3, 5, 10, 20)


def _recall_at_k(metrics: List[dict], key: str = 'retr_hit_rank') -> List[tuple]:
    """[(K, hits, n, fraction), ...] over the shots that recorded a top-K.

    A shot counts at K when the truth first appeared at rank K or better, so
    the curve is monotone and recall@1 is exactly the retrieval accuracy the
    winner-only metric already reports.

    The denominator is every scored shot at every K, including shots that could
    not enumerate K candidates at all -- a region barely larger than the query
    offers only a handful of placements. For those, recall@20 is decided by the
    few that exist, which is the honest reading of "was it in the top 20
    proposed": there was nothing else to propose. Dropping the K instead would
    let one small region delete a column for the whole corpus.
    """
    scored = [m for m in metrics if m.get('retr_topk_n')]
    if not scored:
        return []
    n = len(scored)
    out = []
    for k in K_GRID:
        hits = sum(1 for m in scored
                   if m.get(key) is not None and m[key] <= k)
        out.append((k, hits, n, hits / n))
    return out


def _recall_at_k_lines(metrics: List[dict]) -> List[str]:
    curve = _recall_at_k(metrics)
    if not curve:
        return []
    tol = next((m['retr_hit_tol_px'] for m in metrics
                if m.get('retr_hit_tol_px') is not None), None)
    depths = [m['retr_topk_n'] for m in metrics if m.get('retr_topk_n')]
    lines = ['Stage 2 recall@K  (truth within '
             f'{tol:.0f} px of a candidate centre, i.e. inside the crop SIFT '
             'would search)',
             f'  candidates enumerated per shot: min {min(depths)}, '
             f'median {int(np.median(depths))}, max {max(depths)}']
    for k, hits, n, frac in curve:
        lines.append(f'  K={k:<3d} {hits:5d}/{n:<5d}  {100 * frac:5.1f}%')
    strict = _recall_at_k(metrics, 'retr_hit_rank_strict')
    if strict:
        lines.append('  strict (window itself right, within one tile)')
        for k, hits, n, frac in strict:
            lines.append(f'    K={k:<3d} {hits:5d}/{n:<5d}  {100 * frac:5.1f}%')
    # The gap between recall@20 and recall@1 is the whole argument for adding a
    # verification pass; if it is small the features are the problem, not the
    # ranking.
    if len(curve) > 1:
        gain = 100 * (curve[-1][3] - curve[0][3])
        lines.append(f'  headroom from ranking: +{gain:.1f} points between '
                     f'K=1 and K={curve[-1][0]}')
    lines.append('')
    lines.extend(_sift_topk_lines(metrics))
    return lines


def _sift_topk_lines(metrics: List[dict]) -> List[str]:
    """SIFT run over the top-K candidates: the ceiling, and what is reachable.

    Three numbers, and the third is the one that decides whether this design
    survives contact with photographs that have no ground truth:

      hit@K       SIFT localised some candidate correctly. The ceiling.
      verified@K  SIFT accepted some candidate on its own inlier count. What a
                  system without ground truth can actually reach.
      agree       of the shots where it accepted one, how often that same
                  candidate was the correct one. Below 1.0 means the loop would
                  confidently return a wrong position.
    """
    scored = [m for m in metrics if m.get('sift_topk_n')]
    if not scored:
        return []
    n = len(scored)
    depth = max(m['sift_topk_n'] for m in scored)
    lines = [f'Stage 2+3 SIFT over the top {depth} candidates  (n={n})']
    for k in [k for k in K_GRID if k <= depth] or [depth]:
        hit = sum(1 for m in scored
                  if m.get('sift_hit_rank') and m['sift_hit_rank'] <= k)
        ver = sum(1 for m in scored
                  if m.get('sift_verified_rank') and m['sift_verified_rank'] <= k)
        lines.append(f'  K={k:<3d} hit {100 * hit / n:5.1f}%   '
                     f'verified {100 * ver / n:5.1f}%')
    accepted = [m for m in scored if m.get('sift_verified_rank')]
    if accepted:
        agree = sum(1 for m in accepted
                    if m.get('sift_hit_rank') == m['sift_verified_rank'])
        lines.append(f'  accepted-and-correct: {agree}/{len(accepted)} '
                     f'({100 * agree / len(accepted):.1f}%) -- the inlier count '
                     f'picked the right candidate this often')
    lines.append('')
    lines.extend(_picker_lines(scored, depth))
    return lines


def _picker_lines(scored: List[dict], depth: int) -> List[str]:
    """Does top-K beat rank-1, once something has to actually choose?

    recall@K is a ceiling, not a result: it says the truth was somewhere in the
    list, not that the system could tell which one it was. A deployable loop
    has to commit to one candidate with no ground truth to consult, so it is
    scored here against the baseline it would replace.

      baseline        SIFT on rank 1, which is what the pipeline does today.
      first accepted  scan by score, take the first candidate SIFT accepts.
      most inliers    verify all K, take the highest inlier count.
      ceiling         a perfect picker. The gap above the two real pickers is
                      what a better verifier could still buy.

    REGRESSIONS is the number that decides whether to ship it. A shot where
    rank 1 was already right and the picker chose someone else is a case the
    loop actively broke, and a net gain can hide a lot of them.
    """
    n = len(scored)
    if not n:
        return []

    def correct(m, rank_key):
        r = m.get(rank_key)
        return r is not None and m.get('sift_hit_rank') == r

    base    = sum(1 for m in scored if m.get('sift_hit_rank') == 1)
    first   = sum(1 for m in scored if correct(m, 'sift_verified_rank'))
    most    = sum(1 for m in scored if correct(m, 'sift_best_rank'))
    ceiling = sum(1 for m in scored if m.get('sift_hit_rank') is not None)

    lines = [f'Picker comparison over the top {depth}  (n={n})']
    for label, hits in [('baseline  (rank 1 only)', base),
                        ('first accepted        ', first),
                        ('most inliers          ', most),
                        ('ceiling  (perfect pick)', ceiling)]:
        delta = ('' if hits == base
                 else f'   {100 * (hits - base) / n:+.1f} pts vs baseline')
        lines.append(f'  {label} : {hits:5d}/{n:<5d} {100 * hits / n:5.1f}%{delta}')

    for label, key in [('first accepted', 'sift_verified_rank'),
                       ('most inliers  ', 'sift_best_rank')]:
        broke = sum(1 for m in scored
                    if m.get('sift_hit_rank') == 1 and not correct(m, key))
        lines.append(f'  regressions, {label}: {broke} shot(s) where rank 1 was '
                     f'already correct and the picker did not choose it')
    lines.append('')
    return lines


def plot_recall_at_k(metrics: List[dict], out_path: str) -> None:
    """Recall@K overall and split by routed level.

    The split matters more than the total: a level the mpp stage mis-routes
    cannot be rescued by looking further down that level's ranking, and mixing
    the two hides which of the two is failing.
    """
    curve = _recall_at_k(metrics)
    if not curve:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ks = [c[0] for c in curve]
    axes[0].plot(ks, [100 * c[3] for c in curve], 'o-', label='in SIFT crop')
    strict = _recall_at_k(metrics, 'retr_hit_rank_strict')
    if strict:
        axes[0].plot([c[0] for c in strict], [100 * c[3] for c in strict],
                     's--', label='window exact')
    axes[0].set_title(f'Stage 2 recall@K   (n={curve[0][2]})')
    axes[0].legend()

    by_level: Dict[int, List[dict]] = defaultdict(list)
    for m in metrics:
        if m.get('retr_topk_n') and m.get('routed_level') is not None:
            by_level[m['routed_level']].append(m)
    for lv in sorted(by_level):
        sub = _recall_at_k(by_level[lv])
        if sub:
            axes[1].plot([c[0] for c in sub], [100 * c[3] for c in sub], 'o-',
                         label=f'routed L{lv}  (n={sub[0][2]})')
    axes[1].set_title('by routed level')
    if by_level:
        axes[1].legend()

    for ax in axes:
        ax.set_xlabel('K')
        ax.set_ylabel('recall (%)')
        ax.set_xticks(ks)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'recall_at_k.png -> {out_path}')


# ── summary.txt ───────────────────────────────────────────────────────────────

def write_summary(metrics: List[dict], out_path: str) -> None:
    lines: List[str] = []
    lines.append(f'Total shots     : {len(metrics)}')
    lines.append(f'Stage errors    : {sum(1 for m in metrics if m["error"])}')
    lines.append(f'Unusable levels : {sum(1 for m in metrics if m["unusable_level"])}')

    # Routing is reported with a SIGN, not as an equality, because the two
    # directions do not cost the same. A window one level coarser than the
    # query still contains the truth -- SIFT reads a crop at full resolution
    # and absorbs the scale gap (L2 -> L3 retrieval centre err 5188 um vs
    # L2 -> L2's 1670 um, yet 9.1 um vs 7.4 um after refine). A window one
    # level finer covers LESS than the FoV, so no window can hold the truth
    # and no amount of ranking recovers it. Reporting only `== level` scored
    # 42.4% on a corpus whose wrongly-routed shots succeeded 60.3% of the
    # time, which reads as a failure rate and is not one.
    routed = [m for m in metrics if m['routed_level'] is not None]
    if routed:
        n = len(routed)
        finer   = sum(1 for m in routed if m['routed_level'] <  m['level'])
        exact   = sum(1 for m in routed if m['routed_level'] == m['level'])
        coarser = sum(1 for m in routed if m['routed_level'] >  m['level'])
        lines.append(f'Level routing   : finer {finer} ({100.0 * finer / n:.1f}%)   '
                     f'exact {exact} ({100.0 * exact / n:.1f}%)   '
                     f'coarser {coarser} ({100.0 * coarser / n:.1f}%)   (n={n})')
        lines.append('                  finer is the fatal direction; coarser is '
                     'absorbed by SIFT')

    # Derived from the raw columns rather than read off `rot_correct`, because
    # that column was written by an equality that cannot hold: best_rotation is
    # the rotation applied to the QUERY to bring it onto the reference, i.e. the
    # inverse of the camera's. The correct answer therefore sits on the
    # anti-diagonal -- gt 90 -> retr 270 (288 shots), gt 270 -> retr 90 (228) --
    # and a direct comparison can only be true at gt 0 and 180, halving the
    # figure by construction (34.5% reported vs 52.8% actual). Deriving it here
    # also means every CSV already on disk reports the corrected number without
    # being regenerated.
    rot_known = [m for m in metrics
                 if m.get('retr_rotation') is not None
                 and m.get('gt_rot_deg') is not None]
    if rot_known:
        n_ok = sum(1 for m in rot_known
                   if int(m['retr_rotation']) == (360 - int(m['gt_rot_deg'])) % 360)
        lines.append(f'Rotation recall : {n_ok}/{len(rot_known)} '
                     f'({100.0 * n_ok / len(rot_known):.1f}%) '
                     f'retr_rotation == (-gt_rot_deg) mod 360')
    lines.append(f'SIFT success    : {sum(1 for m in metrics if m["refine_success"])}'
                 f'/{len(metrics)}')
    ref = _ref_marker(metrics, 'retr_center_err_px', 'retr_tile_l0', '1 tile')
    if ref:
        _, frac, text = ref
        n_scored = sum(1 for m in metrics
                       if m.get('retr_center_err_px') is not None
                       and m.get('retr_tile_l0'))
        lines.append(f'Stage 2 <= 1 tile: {round(frac * n_scored)}/{n_scored} '
                     f'({100 * frac:.1f}%)   [{text}]')
    lines.append('')
    lines.extend(_recall_at_k_lines(metrics))

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

def _ref_marker(rows: List[dict], stage_key: str, ref_key: str,
                label: str) -> Optional[tuple]:
    """(x, fraction, label) for a per-shot reference width such as one tile.

    The FRACTION is exact: every shot is compared against its own ref_key, so a
    corpus spanning several pyramid levels is still counted correctly. The LINE
    can only sit at one x, so it goes at the median and the label says so when
    the shots disagree -- reading the line as the threshold would otherwise be
    wrong for every shot not routed to the median level.
    """
    pairs = [(m[stage_key], m[ref_key]) for m in rows
             if m.get(stage_key) is not None and m.get(ref_key)]
    if not pairs:
        return None
    refs = [r for _, r in pairs]
    frac = sum(1 for v, r in pairs if v <= r) / len(pairs)
    med  = float(np.median(refs))
    varies = max(refs) - min(refs) > 1e-6
    text = (f'{label} (median {med:.0f} px)' if varies else f'{label} = {med:.0f} px')
    return (med, frac, text)


def _plot_cdf_line(ax, sorted_vals: np.ndarray, xlabel: str,
                   ref: Optional[tuple] = None) -> None:
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

    if ref is not None:
        ref_x, frac, text = ref
        ax.axvline(ref_x, color='seagreen', linewidth=1.3, alpha=0.9)
        ax.plot([ref_x], [frac], 'o', color='seagreen', ms=5, zorder=5)
        ax.annotate(f'{text}\n{100 * frac:.1f}%',
                    xy=(ref_x, frac), xytext=(6, -2),
                    textcoords='offset points', fontsize=7.5,
                    color='seagreen', va='top', ha='left')


def plot_stage_cdfs(metrics: List[dict], stage_key: str, xlabel: str,
                    title: str, out_path: str,
                    ref_key: Optional[str] = None, ref_label: str = '') -> None:
    """CDF of one stage's error, aggregate plus one panel per WSI.

    ref_key names a per-shot width to mark on the curve -- retr_tile_l0 for
    stage 2, so the plot answers "what percentile is one tile" directly instead
    of leaving it to be eyeballed against the axis.
    """
    all_vals = np.array(sorted(m[stage_key] for m in metrics
                               if m.get(stage_key) is not None), dtype=float)
    if all_vals.size == 0:
        print(f'  [skip plot] {stage_key}: no data')
        return

    per_wsi: Dict[str, List[dict]] = defaultdict(list)
    for m in metrics:
        if m.get(stage_key) is not None:
            per_wsi[wsi_tag(m)].append(m)

    tags   = sorted(per_wsi)
    n_cols = min(max(len(tags), 1), 3)
    n_rows = math.ceil(len(tags) / n_cols) if tags else 0

    fig = plt.figure(figsize=(max(9, n_cols * 4.5), 3.5 + n_rows * 3.0))
    gs  = fig.add_gridspec(1 + n_rows, n_cols)

    ref_all = _ref_marker(metrics, stage_key, ref_key, ref_label) if ref_key else None
    ax_all = fig.add_subplot(gs[0, :])
    _plot_cdf_line(ax_all, all_vals, xlabel, ref=ref_all)
    note = '  (n<20: red ticks = raw samples)' if all_vals.size < 20 else ''
    ax_all.set_title(f'Aggregate  (n={all_vals.size}){note}')

    for i, tag in enumerate(tags):
        ax = fig.add_subplot(gs[1 + i // n_cols, i % n_cols])
        rows = per_wsi[tag]
        vals = np.array(sorted(m[stage_key] for m in rows), dtype=float)
        ref  = _ref_marker(rows, stage_key, ref_key, ref_label) if ref_key else None
        _plot_cdf_line(ax, vals, xlabel, ref=ref)
        ax.set_title(f'{tag}  (n={len(rows)})', fontsize=10)

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
           color=np.where(ok, 'dodgerblue', '#8FA8C9'),
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
                    out_path=os.path.join(out_dir, 'stage2_retr_cdf.png'),
                    ref_key='retr_tile_l0', ref_label='1 tile')
    plot_stage_cdfs(metrics, 'refine_center_err_px',
                    xlabel='refine centre err  (level-0 pixels)',
                    title='Stage 3 — SIFT+RANSAC error (rotation-invariant centre)',
                    out_path=os.path.join(out_dir, 'stage3_refine_cdf.png'))
    plot_heatmap(metrics, os.path.join(out_dir, 'heatmap.png'))
    plot_recall_at_k(metrics, os.path.join(out_dir, 'recall_at_k.png'))
