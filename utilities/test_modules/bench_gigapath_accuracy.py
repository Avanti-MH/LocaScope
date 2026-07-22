'''
GigaPath fp16 / ToMe accuracy analysis — Level 1 + Level 2.

Compares embeddings from optimized configs against fp32 baseline on
tissue-region tiles from real WSIs (1 SVS + 1 MRXS).

Sampling
--------
  * 1 SVS (BRACS_1228) + 1 MRXS (Ki67 S1104043) — different level counts
  * HESTSegFunc tissue mask (ds=32 by default)
  * TileSampler.sample(n=per_level) — per-level distribution
  * Total 200 patches split by WSI, then by level

Configs
-------
  Main 4  (L1 + L2):
    baseline fp32 / fp16 only / ToMe r=8 fp32 / fp16+ToMe r=8
  ToMe r sweep (L1 only, fp32):
    r ∈ {4, 6, 12}  (r=0 = baseline, r=8 already in main)

Level 1 — embedding fidelity (intrinsic)
  Per-patch cosine similarity vs baseline → distribution
  (mean / std / p1 / p5 / p50 / p95 / p99) + overlay histogram.

Level 2 — ranking preservation (intrinsic)
  N×N pairwise cos matrix per config; compare to baseline via:
    - Spearman correlation on upper-triangle scores
    - Top-K neighbor overlap (K ∈ {1, 5, 10, 50})
    - Rank shift: where does baseline's top-1 sit in the other's ranking?

Outputs
------
  result/<SLURM_JOB_NAME>/    (final analysis products; defaults to 'AccuracyV1' locally)
    summary.txt       — L1 + L2 tables + ToMe sweep (CSV-parseable)
    cos_hist.png      — L1 overlay histogram (main 4)
    tome_sweep.png    — mean cos sim vs ToMe r (fp32)

  log/tmp/                  (byproducts — reproducibility + raw data)
    tiles/<wsi>.json  — sampled TileInfo per WSI
    embeddings.pt     — dict[cfg → (N, 1536) fp32 tensor]

  log/<slurm-name>          (SLURM stdout/stderr)
'''
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import openslide
import torch

from _paths import setup_import_paths, PROJECT_ROOT, job_result_dir
setup_import_paths()

from TissuesRegionsMask import TissuesRegionsMask
from TileSampler import TileSampler
from GigaPathFunc import (
    gigapath_model, gigapath_apply_tome,
    make_gigapath_encoder,
)
from HESTSegFunc import hest_seg_model, make_hest_method


# ── Config table ──────────────────────────────────────────────────────────────

_MAIN_CONFIGS = [
    # label,             dtype,           tome_r
    ('baseline fp32',    torch.float32,   0),
    ('fp16 only',        torch.float16,   0),
    ('ToMe r=8',         torch.float32,   8),
    ('fp16 + ToMe r=8',  torch.float16,   8),
]

_BASE_LABEL = 'baseline fp32'
_TOPK       = (1, 5, 10, 50)


def _load_model(device, tome_r):
    '''Fresh GigaPath model. flash-attn on (default when TIMM_FUSED_ATTN unset).'''
    os.environ.pop('TIMM_FUSED_ATTN', None)
    model = gigapath_model(device)
    if tome_r > 0:
        model = gigapath_apply_tome(model, r=tome_r)
    return model


# ── Sampling ─────────────────────────────────────────────────────────────────

def sample_wsi(wsi_path, per_wsi, hest_method, hest_ds, hest_max_pixels,
               hest_overlap, tile_size, tissue_ratio, seed, tile_json, resume):
    '''
    Open one WSI, build tissue mask, sample per-level tiles, save JSON.
    Returns list of PIL images (in-order matching the JSON).
    '''
    print(f'\n── {wsi_path.name} ──', flush=True)
    wsi = openslide.OpenSlide(str(wsi_path))
    n_lv = wsi.level_count
    per_level = max(1, per_wsi // n_lv)
    print(f'  levels={n_lv}  wsi_budget={per_wsi}  per_level={per_level}',
          flush=True)

    print(f'  building HEST tissue mask (ds={hest_ds}, '
          f'max_pixels={hest_max_pixels/1e6:.1f}M) ...', flush=True)
    mask = TissuesRegionsMask.from_wsi(
        wsi, ds=hest_ds, method=hest_method,
        max_pixels=hest_max_pixels, overlap=hest_overlap,
    )
    print(f'  tissue_fraction={mask.tissue_fraction() * 100:.1f}%  '
          f'regions={len(mask)}', flush=True)

    if resume and tile_json.exists():
        sampler = TileSampler.from_json(wsi, mask, tile_json,
                                        tile_size=tile_size, seed=seed)
        print(f'  [resume] loaded {len(sampler)} tiles from {tile_json.name}',
              flush=True)
    else:
        sampler = TileSampler(wsi, mask, tile_size=tile_size, seed=seed)
        sampler.sample(n=per_level, tissue_ratio=tissue_ratio)
        sampler.save(tile_json)
        print(f'  saved tile coords -> {tile_json}', flush=True)

    sampler.summary()
    images = sampler.read_all()
    wsi.close()
    return images


# ── Encoding ─────────────────────────────────────────────────────────────────

def encode_config(images, device, dtype, tome_r, batch_size, label):
    print(f'\n[encode] {label}', flush=True)
    t0 = time.perf_counter()
    model   = _load_model(device, tome_r)
    encoder = make_gigapath_encoder(model, device,
                                    batch_size=batch_size, dtype=dtype)
    feats = encoder(images)   # (N, D) fp32 unit-normalized
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    n_nan = int(torch.isnan(feats).any(dim=-1).sum())
    n_inf = int(torch.isinf(feats).any(dim=-1).sum())
    if n_nan or n_inf:
        print(f'  [WARN] {label}: NaN={n_nan}  Inf={n_inf} / {feats.shape[0]} '
              f'patches (fp16 overflow / normalize-by-zero suspected)',
              flush=True)
    print(f'  time={time.perf_counter() - t0:6.1f}s   feats={tuple(feats.shape)}',
          flush=True)
    return feats


# ── Level 1 ──────────────────────────────────────────────────────────────────

def level1_stats(base, other):
    '''Per-patch cos sim vs baseline. Inputs are unit-normalized.'''
    cos = (base * other).sum(dim=-1).numpy()
    p = np.nanpercentile(cos, [1, 5, 50, 95, 99])
    stats = {
        'mean': float(np.nanmean(cos)), 'std': float(np.nanstd(cos)),
        'p1':  float(p[0]), 'p5':  float(p[1]), 'p50': float(p[2]),
        'p95': float(p[3]), 'p99': float(p[4]),
    }
    return stats, cos


def print_level1(l1):
    print('\n' + '=' * 88, flush=True)
    print('  Level 1 — Embedding fidelity vs baseline fp32', flush=True)
    print('=' * 88, flush=True)
    print(f"  {'config':<20}  {'mean':>7}  {'std':>7}  {'p1':>7}  {'p5':>7}"
          f"  {'p50':>7}  {'p95':>7}  {'p99':>7}", flush=True)
    print('-' * 88, flush=True)
    for k, s in l1.items():
        print(f"  {k:<20}  {s['mean']:>7.5f}  {s['std']:>7.5f}  "
              f"{s['p1']:>7.5f}  {s['p5']:>7.5f}  {s['p50']:>7.5f}  "
              f"{s['p95']:>7.5f}  {s['p99']:>7.5f}", flush=True)
    print('-' * 88, flush=True)


# ── Level 2 ──────────────────────────────────────────────────────────────────

def level2_ranking(base, other, ks=_TOPK):
    '''
    Pairwise cos matrices + top-K neighbor overlap + Spearman + rank shift.
    Inputs are unit-normalized. Diagonal excluded (self-neighbor).
    NaN/Inf rows in `other` are zeroed (their cos rows become 0 → uniform bad ranking)
    so downstream argpartition / argmax stay defined.
    '''
    from scipy.stats import spearmanr

    n = base.shape[0]
    other = other.clone()
    bad_rows = ~torch.isfinite(other).all(dim=-1)
    if bad_rows.any():
        print(f'  [WARN] level2: zeroing {int(bad_rows.sum())} NaN/Inf rows in other',
              flush=True)
        other[bad_rows] = 0.0
    m_b = (base  @ base.T ).numpy()
    m_o = (other @ other.T).numpy()
    np.fill_diagonal(m_b, -np.inf)
    np.fill_diagonal(m_o, -np.inf)

    iu = np.triu_indices(n, k=1)
    sp = spearmanr(m_b[iu], m_o[iu]).statistic

    overlap = {}
    for k in ks:
        t_b = np.argpartition(-m_b, k, axis=1)[:, :k]
        t_o = np.argpartition(-m_o, k, axis=1)[:, :k]
        overlap[k] = float(np.mean([
            len(set(t_b[i]) & set(t_o[i])) / k for i in range(n)
        ]))

    b_top1 = m_b.argmax(axis=1)
    o_rank = (-m_o).argsort(axis=1)   # position 0 = best neighbor
    rank_of_top1 = np.array([
        int(np.where(o_rank[i] == b_top1[i])[0][0]) for i in range(n)
    ])
    return {
        'spearman':          float(sp),
        'top_k_overlap':     overlap,
        'rank_shift_median': float(np.median(rank_of_top1)),
        'rank_shift_p95':    float(np.percentile(rank_of_top1, 95)),
    }


def print_level2(l2, ks=_TOPK):
    print('\n' + '=' * 88, flush=True)
    print('  Level 2 — Ranking preservation vs baseline fp32', flush=True)
    print('=' * 88, flush=True)
    tk_hdr = '  '.join(f'top{k:<3}'.rjust(7) for k in ks)
    print(f"  {'config':<20}  {'spearman':>9}  {tk_hdr}"
          f"  {'rank_med':>8}  {'rank_p95':>8}", flush=True)
    print('-' * 88, flush=True)
    for k, r in l2.items():
        tk = '  '.join(f"{r['top_k_overlap'][kk]:>7.3f}" for kk in ks)
        print(f"  {k:<20}  {r['spearman']:>9.5f}  {tk}  "
              f"{r['rank_shift_median']:>8.1f}  {r['rank_shift_p95']:>8.1f}",
              flush=True)
    print('-' * 88, flush=True)


# ── Plots ────────────────────────────────────────────────────────────────────

def _mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save_cos_hist(cos_by_cfg, out_path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, cos in cos_by_cfg.items():
        finite = cos[np.isfinite(cos)]
        if len(finite) < len(cos):
            print(f'  [WARN] {label}: dropping {len(cos) - len(finite)} '
                  f'NaN/Inf values from histogram', flush=True)
        if len(finite) == 0:
            print(f'  [SKIP] {label}: all NaN/Inf, cannot plot', flush=True)
            continue
        ax.hist(finite, bins=50, alpha=0.5, label=label)
    ax.set_xlabel('cosine similarity vs baseline fp32')
    ax.set_ylabel('# patches')
    ax.set_title('Level 1 — embedding fidelity distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}', flush=True)


def save_tome_sweep(rs_to_mean, out_path):
    plt = _mpl()
    rs = sorted(rs_to_mean.keys())
    means = [rs_to_mean[r] for r in rs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rs, means, 'o-', color='C0')
    ax.set_xlabel('ToMe r (tokens merged per layer)')
    ax.set_ylabel('mean cos sim vs baseline fp32')
    ax.set_title('ToMe r sweep — accuracy trade-off (fp32)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}', flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    default_svs  = '/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs'
    default_mrxs = '/work/u26130998/datasets/Ki67/S1104043,G7E,110207.mrxs'
    default_out  = Path(job_result_dir('AccuracyV1'))
    default_tmp  = Path(PROJECT_ROOT) / 'log' / 'tmp'

    p = argparse.ArgumentParser(description='GigaPath accuracy L1+L2')
    p.add_argument('--svs',  default=default_svs)
    p.add_argument('--mrxs', default=default_mrxs)
    p.add_argument('--total-patches', type=int,   default=200)
    p.add_argument('--hest-ds',           type=float, default=32.0)
    p.add_argument('--hest-max-pixels',   type=int,   default=4_000_000,
                   help='cap per-tile pixels for HEST inference (adaptive halving)')
    p.add_argument('--hest-overlap',      type=int,   default=128,
                   help='per-tile margin (px) trimmed after stitching')
    p.add_argument('--tile-size',     type=int,   default=256)
    p.add_argument('--tissue-ratio',  type=float, default=0.5)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--batch-size',    type=int,   default=128)
    p.add_argument('--tome-r-sweep',  default='1,2,3,4,8',
                   help='fp32 ToMe r values (comma-sep). r=8 already in main; '
                        'duplicates are skipped.')
    p.add_argument('--out-dir',       type=Path, default=default_out,
                   help='final products: summary.txt + PNGs')
    p.add_argument('--tmp-dir',       type=Path, default=default_tmp,
                   help='byproducts: tiles/ + embeddings.pt')
    p.add_argument('--resume-tiles',  action='store_true',
                   help='reuse sampled tile JSONs if present')
    p.add_argument('--skip-l2',       action='store_true',
                   help='skip Level 2 (useful when N is large)')
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = args.tmp_dir / 'tiles'
    tiles_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    if device.type == 'cuda':
        print(f'GPU 0 : {torch.cuda.get_device_name(0)}', flush=True)

    # ── 1. Sample tiles per WSI (share one HEST model) ────────────────────
    wsi_paths = [Path(args.svs), Path(args.mrxs)]
    per_wsi   = args.total_patches // len(wsi_paths)
    print(f'\nSampling target: {args.total_patches} patches across '
          f'{len(wsi_paths)} WSIs ({per_wsi} per WSI)', flush=True)

    print('\nLoading HEST tissue seg model ...', flush=True)
    hest        = hest_seg_model(device)
    hest_method = make_hest_method(hest, device)

    images = []
    for wp in wsi_paths:
        stem      = wp.stem.replace(',', '_')
        tile_json = tiles_dir / f'{stem}.json'
        images += sample_wsi(
            wp, per_wsi, hest_method,
            hest_ds=args.hest_ds,
            hest_max_pixels=args.hest_max_pixels,
            hest_overlap=args.hest_overlap,
            tile_size=args.tile_size,
            tissue_ratio=args.tissue_ratio, seed=args.seed,
            tile_json=tile_json, resume=args.resume_tiles,
        )

    del hest
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    N = len(images)
    if N == 0:
        print('ERROR: no tiles sampled', flush=True)
        sys.exit(1)
    print(f'\nTotal sampled: {N} patches', flush=True)

    # ── 2. Encode: main 4 configs + ToMe r sweep ──────────────────────────
    embeddings = {}
    for label, dtype, r in _MAIN_CONFIGS:
        embeddings[label] = encode_config(images, device, dtype, r,
                                          args.batch_size, label)

    sweep_rs = [int(x) for x in args.tome_r_sweep.split(',') if x.strip()]
    for r in sweep_rs:
        label = f'ToMe r={r}'
        if label in embeddings:
            continue
        embeddings[label] = encode_config(images, device,
                                          torch.float32, r,
                                          args.batch_size, label)

    torch.save(embeddings, args.tmp_dir / 'embeddings.pt')
    print(f'\nsaved embeddings -> {args.tmp_dir / "embeddings.pt"}', flush=True)

    # ── 3. Level 1 — per-patch cos sim vs baseline ────────────────────────
    base = embeddings[_BASE_LABEL]
    l1, cos_by_cfg = {}, {}
    for label, emb in embeddings.items():
        if label == _BASE_LABEL:
            continue
        s, cos = level1_stats(base, emb)
        l1[label]         = s
        cos_by_cfg[label] = cos
    print_level1(l1)

    # ── 4. Level 2 — ranking preservation (main 4 only) ───────────────────
    l2 = {}
    if not args.skip_l2:
        for label, _, _ in _MAIN_CONFIGS:
            if label == _BASE_LABEL:
                continue
            l2[label] = level2_ranking(base, embeddings[label], ks=_TOPK)
        print_level2(l2)

    # ── 5. Plots ──────────────────────────────────────────────────────────
    main_labels = [c[0] for c in _MAIN_CONFIGS if c[0] != _BASE_LABEL]
    save_cos_hist(
        {k: cos_by_cfg[k] for k in main_labels if k in cos_by_cfg},
        args.out_dir / 'cos_hist.png',
    )

    tome_curve = {0: 1.0}
    if 'ToMe r=8' in l1:
        tome_curve[8] = l1['ToMe r=8']['mean']
    for r in sweep_rs:
        label = f'ToMe r={r}'
        if label in l1:
            tome_curve[r] = l1[label]['mean']
    save_tome_sweep(tome_curve, args.out_dir / 'tome_sweep.png')

    # ── 6. summary.txt (also CSV-parseable) ───────────────────────────────
    with open(args.out_dir / 'summary.txt', 'w') as f:
        f.write(f'GigaPath accuracy L1+L2  N={N} patches  WSIs={len(wsi_paths)}\n\n')
        f.write('=== Level 1 — Embedding fidelity vs baseline fp32 ===\n')
        f.write('config,mean,std,p1,p5,p50,p95,p99\n')
        for k, s in l1.items():
            f.write(f'{k},{s["mean"]},{s["std"]},{s["p1"]},{s["p5"]},'
                    f'{s["p50"]},{s["p95"]},{s["p99"]}\n')
        f.write('\n=== ToMe r sweep — mean cos sim vs baseline (fp32) ===\n')
        f.write('r,mean_cos_sim\n')
        for r in sorted(tome_curve.keys()):
            f.write(f'{r},{tome_curve[r]}\n')
        if l2:
            f.write('\n=== Level 2 — Ranking preservation vs baseline fp32 ===\n')
            f.write('config,spearman,top1,top5,top10,top50,rank_shift_median,rank_shift_p95\n')
            for k, r in l2.items():
                ok = r['top_k_overlap']
                f.write(f'{k},{r["spearman"]},{ok[1]},{ok[5]},{ok[10]},{ok[50]},'
                        f'{r["rank_shift_median"]},{r["rank_shift_p95"]}\n')
    print(f'saved summary -> {args.out_dir / "summary.txt"}', flush=True)

    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
