#!/usr/bin/env python3
"""GigaPath optimization configs -- speed AND accuracy, one script.

Merge of `bench_gigapath_accuracy.py` (does fp16 change the ANSWER) and
`bench_gigapath_infer.py` (does fp16/flash-attn/compile change the SPEED).
Both existed to answer the one CLAUDE.md sentence that cites them together:
"fp16 alone went to production with cos=0.99995 and a 5.5x speedup" -- one
number from each file. They shared a near-identical 6-config table
(baseline fp32 / fp16 / flash-attn / compile / fp16+flash / ALL) and each
built its own encoder-loading helper and its own tissue mask; this file
shares both instead.

    python bench_gigapath_configs.py --mode accuracy
    python bench_gigapath_configs.py --mode speed --compare
    python bench_gigapath_configs.py --mode speed          # standard sweep

Speed and accuracy do NOT share a sampling path. Accuracy wants a
level-stratified sample (does fp16 degrade differently at different
scales?), sampled once and encoded under every config; speed wants a plain
patch grid at a fixed level/overlap, timed under every config -- forcing one
onto the other would break the question either mode is actually asking.
What they DO share: the config table below, `_load_encoder`, tissue-mask
construction, and one argparse.

MODE accuracy
-------------
  Level 1 -- embedding fidelity: per-patch cosine similarity vs baseline fp32
             (mean/std/p1/p5/p50/p95/p99) + overlay histogram.
  Level 2 -- ranking preservation: N x N pairwise cosine matrix per config,
             vs baseline via Spearman correlation, top-K neighbour overlap,
             and rank shift of baseline's own top-1.
  Sampling: `TileSampler`, level-stratified, HEST tissue mask, richness caps
            from `KnnEstMpp.REFERENCE_BANK_RICHNESS` -- see "NOT
            --tissue-ratio" below.
  Outputs: result/<SLURM_JOB_NAME or AccuracyV1>/{summary.txt,cos_hist.png}

MODE speed
----------
  --compare: sweep every config in `_CONFIGS` x batch sizes, on synthetic
             patches (Part 1) and a real WSI at a fixed level (Part 2).
  standard (no --compare): ONE config (flags pick flash-attn/compile) swept
             over --batch-sizes x --dtypes x --levels x --overlaps, with a
             cpu/gpu time split per point and a bottleneck verdict.

NOT `--tissue-ratio`. The old accuracy bench took a `--tissue-ratio` float
and ran it through `TileSampler.caps_for_tissue_ratio` -- a function whose
own docstring called it "the RETIRED tissue_ratio gate" even before this
file used it. `KnnEstMpp.py` hit the same call this session and replaced it
with a richness policy spelled out directly (`REFERENCE_BANK_RICHNESS`:
floors all zero, caps admit background < 50%, no preference between the
three buckets that clears). This file reuses THAT constant rather than
inventing a second one, since the two benches want the identical policy --
"any admissible tile, no preference" -- for the identical reason (a fair
per-config comparison should not also be biased toward busy tiles).

Two bugs fixed while merging, found because each file failed to IMPORT under
the old dependency list, not merely to run:
  accuracy bench  `del hest` referenced a name that was never bound (the
                  variable is `hest_method`) -- would have raised NameError
                  the first time this file ever reached that line.
  speed bench     `_CPU_TRANSFORM = TransformConfig().build()` ran at import
                  time with no `TransformConfig` import anywhere in the file
                  -- NameError before `main()` is ever reached.
Neither bug is reachable from the other bench, which is presumably why each
went unnoticed independently.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import nullcontext
from itertools import groupby
from pathlib import Path

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from _paths import setup_import_paths, RESULT_DIR, job_result_dir
setup_import_paths()

from SafeSlide import SafeSlide                                     # noqa: E402
from PatchingLib import region_grids                                 # noqa: E402
from TissueMaskConfig import MaskMaker, add_mask_args, mask_cfg_from_args  # noqa: E402
from TileSampler import OverlapConfig, SamplerConfig, TileSampler, native_plans  # noqa: E402
from ReadGeometry import ReadSpec                                                # noqa: E402
from SlideReader import SlideReader                                             # noqa: E402
from GigaPathFunc import GigaPathEncoderConfig                        # noqa: E402
from TileEncoderFunc import TransformConfig                          # noqa: E402
from stage1_estimation.KnnEstMpp import REFERENCE_BANK_RICHNESS                        # noqa: E402


# ── shared config table ─────────────────────────────────────────────────────

#: (label, flash, dtype, compile). Speed mode's own table (was
#: `_COMPARE_CONFIGS`) -- accuracy mode uses the first two entries by
#: default (was `_MAIN_CONFIGS`), since flash-attn/compile change SPEED, not
#: the arithmetic, and the old accuracy bench never compared them.
#: `--configs` lets accuracy mode check that assumption against more of the
#: table instead of asserting it.
_CONFIGS = [
    ('baseline  fp32',                 False,  torch.float32,  False),
    ('fp16  only',                     False,  torch.float16,  False),
    ('flash-attn  only (fp32)',        True,   torch.float32,  False),
    ('compile  only (fp32)',           False,  torch.float32,  True),
    ('fp16 + flash',                   True,   torch.float16,  False),
    ('ALL  fp16+flash+compile',        True,   torch.float16,  True),
]
_BASE_LABEL = _CONFIGS[0][0]
_TOPK = (1, 5, 10, 50)
_LABEL_W = 36


def _dt(d) -> str:
    return 'fp16' if d is torch.float16 else 'fp32'


def _load_encoder(device, use_flash: bool, use_compile: bool, batch_size=None,
                  dtype=None, multi_gpu: bool = False):
    '''One encoder per config. `compile` is a `GigaPathEncoderConfig` field,
    so `build()` owns when it is applied rather than the caller.'''
    if use_flash:
        os.environ.pop('TIMM_FUSED_ATTN', None)
    else:
        os.environ['TIMM_FUSED_ATTN'] = '0'
    cfg = GigaPathEncoderConfig(compile=use_compile,
                                **({'batch_size': batch_size} if batch_size else {}))
    encoder = cfg.build(device, multi_gpu=multi_gpu)
    if dtype is not None:
        encoder = encoder.variant(dtype=_dt(dtype))
    return encoder


def _tissue_mask(wsi, mask_cfg, device):
    '''Speed mode needs real tissue tiles, so blank glass has to be excluded;
    `--seg` names the recipe (`--seg hsv` for a model-free one).'''
    return mask_cfg.build(wsi, device)


def read_region_tiles(wsi, mask, ds, level, overlap, tile=256):
    '''Every tile of every patchable region at a level's own ds, main and
    offset lattice, as one list of uint8 arrays per region -- through
    `SlideReader.read_grid`, the read stage 2's build uses, in blocks rather
    than one whole-region read. The order inside a region is block by block,
    which a timing does not care about.'''
    regions = mask.patchable(tile * ds).tissue_regions
    grids = region_grids(regions, ds=ds, level=level, tile_size=tile,
                         overlap=overlap)
    per_region = [[] for _ in regions]
    for block in SlideReader(wsi).read_grid(regions, grids, ds, tile=tile,
                                            offset=overlap, level=level):
        per_region[block.region].extend(t.numpy() for t in block.main)
        per_region[block.region].extend(t.numpy() for t in block.offset)
    return per_region


# ═════════════════════════════════════════════════════════════════════════
#  MODE accuracy
# ═════════════════════════════════════════════════════════════════════════

def sample_wsi(wsi_path, per_wsi, masks, tile_size, seed):
    '''Open one WSI, build its tissue mask, sample per-level tiles.'''
    print(f'\n-- {wsi_path.name} --', flush=True)
    wsi = SafeSlide(str(wsi_path))
    n_lv = wsi.level_count
    per_level = max(1, per_wsi // n_lv)
    print(f'  levels={n_lv}  wsi_budget={per_wsi}  per_level={per_level}',
         flush=True)

    print(f'  building tissue mask ({masks.cfg.seg_id()}) ...', flush=True)
    mask, _ = masks.mask(wsi)
    print(f'  tissue_fraction={mask.tissue_fraction() * 100:.1f}%  '
         f'regions={len(mask)}', flush=True)

    # One rung per PYRAMID level: this bench compares the SAME tiles across
    # encoder configs, so the magnifications are the slide's own.
    cfg = SamplerConfig(n_per_rung=per_level, seed=seed,
                        richness=REFERENCE_BANK_RICHNESS, overlap=OverlapConfig())
    sampler = TileSampler(wsi, mask, cfg)
    sampler.sample(native_plans(wsi, tile_size))
    sampler.summary()
    images = SlideReader(wsi, resize='area').read_samples(
        sampler, ReadSpec(tile_size, tile_size))
    wsi.close()
    return images


def encode_config(images, device, dtype, batch_size, label):
    print(f'\n[encode] {label}', flush=True)
    t0 = time.perf_counter()
    encoder = _load_encoder(device, use_flash=True, use_compile=False,
                            batch_size=batch_size, dtype=dtype)
    feats = encoder(images)   # (N, D) fp32 unit-normalized
    del encoder
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    n_nan = int(torch.isnan(feats).any(dim=-1).sum())
    n_inf = int(torch.isinf(feats).any(dim=-1).sum())
    if n_nan or n_inf:
        print(f'  [WARN] {label}: NaN={n_nan}  Inf={n_inf} / {feats.shape[0]} '
             f'patches (fp16 overflow / normalize-by-zero suspected)', flush=True)
    print(f'  time={time.perf_counter() - t0:6.1f}s   feats={tuple(feats.shape)}',
         flush=True)
    return feats


def level1_stats(base, other):
    '''Per-patch cos sim vs baseline. Inputs are unit-normalized.'''
    cos = (base * other).sum(dim=-1).cpu().numpy()
    p = np.nanpercentile(cos, [1, 5, 50, 95, 99])
    return {'mean': float(np.nanmean(cos)), 'std': float(np.nanstd(cos)),
           'p1': float(p[0]), 'p5': float(p[1]), 'p50': float(p[2]),
           'p95': float(p[3]), 'p99': float(p[4])}, cos


def print_level1(l1):
    print('\n' + '=' * 88)
    print('  Level 1 -- Embedding fidelity vs baseline fp32')
    print('=' * 88)
    print(f"  {'config':<20}  {'mean':>7}  {'std':>7}  {'p1':>7}  {'p5':>7}"
         f"  {'p50':>7}  {'p95':>7}  {'p99':>7}")
    print('-' * 88)
    for k, s in l1.items():
        print(f"  {k:<20}  {s['mean']:>7.5f}  {s['std']:>7.5f}  "
             f"{s['p1']:>7.5f}  {s['p5']:>7.5f}  {s['p50']:>7.5f}  "
             f"{s['p95']:>7.5f}  {s['p99']:>7.5f}")
    print('-' * 88)


def level2_ranking(base, other, ks=_TOPK):
    '''Pairwise cos matrices + top-K neighbour overlap + Spearman + rank shift.'''
    from scipy.stats import spearmanr

    n = base.shape[0]
    other = other.clone()
    bad_rows = ~torch.isfinite(other).all(dim=-1)
    if bad_rows.any():
        print(f'  [WARN] level2: zeroing {int(bad_rows.sum())} NaN/Inf rows in other')
        other[bad_rows] = 0.0
    m_b = (base  @ base.T ).cpu().numpy()
    m_o = (other @ other.T).cpu().numpy()
    np.fill_diagonal(m_b, -np.inf)
    np.fill_diagonal(m_o, -np.inf)

    iu = np.triu_indices(n, k=1)
    sp = spearmanr(m_b[iu], m_o[iu]).statistic

    overlap = {}
    for k in ks:
        t_b = np.argpartition(-m_b, k, axis=1)[:, :k]
        t_o = np.argpartition(-m_o, k, axis=1)[:, :k]
        overlap[k] = float(np.mean(
            [len(set(t_b[i]) & set(t_o[i])) / k for i in range(n)]))

    b_top1 = m_b.argmax(axis=1)
    o_rank = (-m_o).argsort(axis=1)
    rank_of_top1 = np.array([int(np.where(o_rank[i] == b_top1[i])[0][0])
                             for i in range(n)])
    return {'spearman': float(sp), 'top_k_overlap': overlap,
           'rank_shift_median': float(np.median(rank_of_top1)),
           'rank_shift_p95': float(np.percentile(rank_of_top1, 95))}


def print_level2(l2, ks=_TOPK):
    print('\n' + '=' * 88)
    print('  Level 2 -- Ranking preservation vs baseline fp32')
    print('=' * 88)
    tk_hdr = '  '.join(f'top{k:<3}'.rjust(7) for k in ks)
    print(f"  {'config':<20}  {'spearman':>9}  {tk_hdr}"
         f"  {'rank_med':>8}  {'rank_p95':>8}")
    print('-' * 88)
    for k, r in l2.items():
        tk = '  '.join(f"{r['top_k_overlap'][kk]:>7.3f}" for kk in ks)
        print(f"  {k:<20}  {r['spearman']:>9.5f}  {tk}  "
             f"{r['rank_shift_median']:>8.1f}  {r['rank_shift_p95']:>8.1f}")
    print('-' * 88)


def save_cos_hist(cos_by_cfg, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, cos in cos_by_cfg.items():
        finite = cos[np.isfinite(cos)]
        if len(finite) == 0:
            print(f'  [SKIP] {label}: all NaN/Inf, cannot plot')
            continue
        ax.hist(finite, bins=50, alpha=0.5, label=label)
    ax.set_xlabel('cosine similarity vs baseline fp32')
    ax.set_ylabel('# patches')
    ax.set_title('Level 1 -- embedding fidelity distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'  saved {out_path}')


def run_accuracy(args, out_dir: Path) -> int:
    tmp_dir = Path(RESULT_DIR) / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU 0 : {torch.cuda.get_device_name(0)}')

    configs = [_CONFIGS[i] for i in args.configs] if args.configs else _CONFIGS[:2]
    wsi_paths = [Path(args.svs), Path(args.mrxs)]
    per_wsi = args.total_patches // len(wsi_paths)
    print(f'\nSampling target: {args.total_patches} patches across '
         f'{len(wsi_paths)} WSIs ({per_wsi} per WSI)')

    images = []
    with MaskMaker(mask_cfg_from_args(args), device=device) as masks:
        for wp in wsi_paths:
            images += sample_wsi(wp, per_wsi, masks,
                                 tile_size=args.tile_size, seed=args.seed)

    n = len(images)
    if n == 0:
        print('ERROR: no tiles sampled')
        return 1
    print(f'\nTotal sampled: {n} patches')

    embeddings = {}
    for label, flash, dtype, compile_ in configs:
        embeddings[label] = encode_config(images, device, dtype, args.batch_size,
                                          label)
    torch.save(embeddings, tmp_dir / 'embeddings.pt')
    print(f'\nsaved embeddings -> {tmp_dir / "embeddings.pt"}')

    base = embeddings[_BASE_LABEL]
    l1, cos_by_cfg = {}, {}
    for label, emb in embeddings.items():
        if label == _BASE_LABEL:
            continue
        s, cos = level1_stats(base, emb)
        l1[label] = s
        cos_by_cfg[label] = cos
    print_level1(l1)

    l2 = {}
    if not args.skip_l2:
        for label in l1:
            l2[label] = level2_ranking(base, embeddings[label])
        print_level2(l2)

    save_cos_hist(cos_by_cfg, out_dir / 'cos_hist.png')

    with open(out_dir / 'summary.txt', 'w') as f:
        f.write(f'GigaPath accuracy L1+L2  N={n} patches  WSIs={len(wsi_paths)}\n\n')
        f.write('=== Level 1 -- Embedding fidelity vs baseline fp32 ===\n')
        f.write('config,mean,std,p1,p5,p50,p95,p99\n')
        for k, s in l1.items():
            f.write(f'{k},{s["mean"]},{s["std"]},{s["p1"]},{s["p5"]},'
                   f'{s["p50"]},{s["p95"]},{s["p99"]}\n')
        if l2:
            f.write('\n=== Level 2 -- Ranking preservation vs baseline fp32 ===\n')
            f.write('config,spearman,top1,top5,top10,top50,rank_shift_median,rank_shift_p95\n')
            for k, r in l2.items():
                ok = r['top_k_overlap']
                f.write(f'{k},{r["spearman"]},{ok[1]},{ok[5]},{ok[10]},{ok[50]},'
                       f'{r["rank_shift_median"]},{r["rank_shift_p95"]}\n')
    print(f'saved summary -> {out_dir / "summary.txt"}')
    return 0


# ═════════════════════════════════════════════════════════════════════════
#  MODE speed
# ═════════════════════════════════════════════════════════════════════════

#: GigaPath's own validated preprocessing (`GigaPathFunc._GIGAPATH_BASELINE`),
#: spelled out here rather than relying on `TransformConfig()`'s bare
#: defaults happening to match it -- they do today, but a bare default is
#: not a promise the way naming the actual baseline values is.
_CPU_TRANSFORM = TransformConfig(
    scale_size=256, crop_size=224, interpolation='bicubic',
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    preprocess='none').build()


def _run_batch_loop(encoder, ctx, device, batch_size, images):
    '''Per-batch two-phase timing: Phase A (t_cpu) CPU transform only, Phase B
    (t_gpu) H2D + forward + pool + normalize + D2H.'''
    m = getattr(encoder.model, 'module', encoder.model)
    if not images:
        return torch.empty(0), 0.0, 0.0
    t_cpu = t_gpu = 0.0
    outputs = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        t0 = time.perf_counter()
        batch_cpu = torch.stack([
            _CPU_TRANSFORM(img if isinstance(img, Image.Image) else Image.fromarray(img))
            for img in chunk])
        t_cpu += time.perf_counter() - t0

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad(), ctx:
            feats = m.pool(m(batch_cpu.to(device)), pool_type='token')
        feat_cpu = F.normalize(feats.float(), dim=-1).cpu()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t_gpu += time.perf_counter() - t0
        outputs.append(feat_cpu)
    return torch.cat(outputs, dim=0), t_cpu, t_gpu


def run_encode_timed(encoder, device, batch_size, dtype, images):
    ctx = (torch.autocast(device_type=device.type, dtype=dtype)
          if dtype != torch.float32 else nullcontext())
    _, t_cpu, t_gpu = _run_batch_loop(encoder, ctx, device, batch_size, images)
    return t_cpu, t_gpu


def dtype_label(dtype):
    return {torch.float32: 'fp32', torch.float16: 'fp16',
           torch.bfloat16: 'bf16'}.get(dtype, str(dtype))

def peak_gpu_mb(device):
    return torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else 0.0

def reset_peak(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

def ratio_note(ratio):
    if ratio > 0.8: return 'CPU bottleneck -> DataLoader will help'
    if ratio < 0.3: return 'GPU bottleneck -> DataLoader marginal'
    return               'balanced       -> DataLoader worth trying'

def sep(char='-', w=72):
    print(char * w)


def _time_encoder(encoder, patches, device, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        encoder(patches)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def _sweep_configs(device, batch_sizes, encode_fn):
    '''Run every `_CONFIGS` entry, one fresh encoder per config, sweep
    batch_sizes. `encode_fn(encoder, bs) -> dict` with at least
    {'pps': float, 'mem': float}.'''
    all_results, baseline_pps = {}, {}
    for label, use_flash, dtype, use_compile in _CONFIGS:
        if use_compile:
            print(f'  [torch.compile warmup for: {label}]')
        base = _load_encoder(device, use_flash, use_compile)
        pps_by_bs = {bs: encode_fn(base.variant(batch_size=bs, dtype=_dt(dtype)), bs)
                    for bs in batch_sizes}
        all_results[label] = pps_by_bs
        if label == _BASE_LABEL:
            baseline_pps = {bs: v['pps'] for bs, v in pps_by_bs.items()}
        del base
    return all_results, baseline_pps


def _print_compare_flat(all_results, baseline_pps, batch_sizes, *, show_encode_s=False):
    print(f'\n  [Flat -- (config, bs) rows]')
    if show_encode_s:
        print(f'  {"config":<{_LABEL_W}}  {"bs":>5}  {"encode_s":>9}  {"patches/s":>10}  {"vs baseline":>11}  {"GPU MB":>8}')
    else:
        print(f'  {"config":<{_LABEL_W}}  {"bs":>5}  {"patches/s":>10}  {"vs baseline":>11}  {"GPU MB":>8}')
    sep()
    for label, pps_by_bs in all_results.items():
        is_base = label == _BASE_LABEL
        for bs in batch_sizes:
            if bs not in pps_by_bs:
                print(f'  {label:<{_LABEL_W}}  {bs:>5}  {"OOM":>{9 if show_encode_s else 10}}')
                continue
            v = pps_by_bs[bs]
            speedup = v['pps'] / baseline_pps[bs] if bs in baseline_pps else 1.0
            mark = ' <-' if is_base else ''
            if show_encode_s:
                print(f'  {label:<{_LABEL_W}}  {bs:>5}  {v["encode_s"]:>9.1f}  {v["pps"]:>10.1f}'
                     f'  {speedup:>10.2f}x  {v["mem"]:>8.0f}{mark}')
            else:
                print(f'  {label:<{_LABEL_W}}  {bs:>5}  {v["pps"]:>10.1f}  {speedup:>10.2f}x  {v["mem"]:>8.0f}{mark}')
        if not is_base:
            sep('.')
    sep()


def _print_compare_matrix(all_results, baseline_pps, batch_sizes):
    def _bs_label(bs):
        return f'{bs // 1000}K' if bs >= 1000 else str(bs)
    col_w = 7
    hdrs = [f'bs={_bs_label(bs)}'.rjust(col_w) for bs in batch_sizes]
    print(f'\n  [Matrix -- patches/s for baseline; xspeedup for others]')
    print(f'  {"config":<{_LABEL_W}}  ' + '  '.join(hdrs))
    sep()
    for label, pps_by_bs in all_results.items():
        is_base = label == _BASE_LABEL
        cells = []
        for bs in batch_sizes:
            if bs not in pps_by_bs:
                cells.append('OOM'.rjust(col_w))
            elif is_base:
                cells.append(f'{pps_by_bs[bs]["pps"]:>{col_w}.0f}')
            else:
                b = baseline_pps.get(bs)
                val = pps_by_bs[bs]['pps'] / b if b else float('nan')
                cells.append(f'{val:>{col_w - 1}.2f}x')
        suffix = '  p/s' if is_base else '  x  '
        print(f'  {label:<{_LABEL_W}}  ' + '  '.join(cells) + suffix)
    sep()


def bench_compare(device, n_patches, batch_sizes, warmup, repeats=3):
    print('\n' + '=' * 72)
    print(f'  Part 1 -- Comparison Sweep  n={n_patches} synthetic patches')
    print('=' * 72)
    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(n_patches)]

    def encode_fn(encoder, bs):
        for _ in range(warmup):
            encoder(patches[:bs])
        reset_peak(device)
        t = _time_encoder(encoder, patches, device, repeats)
        return {'pps': n_patches / t, 'mem': peak_gpu_mb(device)}

    all_results, baseline_pps = _sweep_configs(device, batch_sizes, encode_fn)
    _print_compare_flat(all_results, baseline_pps, batch_sizes)
    _print_compare_matrix(all_results, baseline_pps, batch_sizes)


def bench_wsi_compare(device, wsi_path, batch_sizes, level, overlap, warmup,
                      mask_cfg):
    print('\n' + '=' * 72)
    print(f'  Part 2 -- WSI Comparison  level={level}  overlap={overlap}'
         f'  {os.path.basename(wsi_path)}')
    print('=' * 72)
    wsi = SafeSlide(wsi_path)
    ds = wsi.level_downsamples[level]
    mask = _tissue_mask(wsi, mask_cfg, device)
    tiles = read_region_tiles(wsi, mask, ds, level, overlap)
    n_patches = sum(len(tp) for tp in tiles)
    print(f'  n_patches={n_patches}  regions={len(tiles)}  ds={ds:.2f}')

    def encode_fn(encoder, bs):
        for _ in range(warmup):
            encoder(tiles[0][:bs])
        reset_peak(device)
        t_encode = 0.0
        for tp_patches in tiles:
            if not tp_patches:
                continue
            t0 = time.perf_counter()
            encoder(tp_patches)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            t_encode += time.perf_counter() - t0
        pps = n_patches / t_encode if t_encode > 0 else 0.0
        return {'pps': pps, 'mem': peak_gpu_mb(device), 'encode_s': t_encode}

    all_results, baseline_pps = _sweep_configs(device, batch_sizes, encode_fn)
    wsi.close()
    _print_compare_flat(all_results, baseline_pps, batch_sizes, show_encode_s=True)
    _print_compare_matrix(all_results, baseline_pps, batch_sizes)


def bench_synthetic(base, device, batch_sizes, dtypes, n_patches, warmup, repeats):
    print('\n' + '=' * 72)
    print(f'  Part 1 -- Synthetic Sweep  ({n_patches} random 256x256 patches)')
    print('=' * 72)
    print(f'  {"batch":>5}  {"dtype":>5}  {"patches/s":>10}  {"ms/batch":>9}'
         f'  {"cpu_s":>6}  {"gpu_s":>6}  {"ratio":>6}  {"note":<28}  {"GPU MB":>8}')
    sep()
    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(n_patches)]
    results = {}
    for dtype in dtypes:
        for bs in batch_sizes:
            encoder = base.variant(batch_size=bs, dtype=_dt(dtype))
            for _ in range(warmup):
                encoder(patches[:bs])
            reset_peak(device)
            cpu_runs, gpu_runs = [], []
            for _ in range(repeats):
                tc, tg = run_encode_timed(base, device, bs, dtype, patches)
                cpu_runs.append(tc); gpu_runs.append(tg)
            t_cpu = sum(cpu_runs) / len(cpu_runs)
            t_gpu = sum(gpu_runs) / len(gpu_runs)
            avg = t_cpu + t_gpu
            n_batches = (n_patches + bs - 1) // bs
            pps = n_patches / avg
            ms_b = avg / n_batches * 1000
            ratio = t_cpu / t_gpu if t_gpu > 0 else float('inf')
            mem = peak_gpu_mb(device)
            dl = dtype_label(dtype)
            print(f'  {bs:>5}  {dl:>5}  {pps:>10.1f}  {ms_b:>9.1f}'
                 f'  {t_cpu:>6.1f}  {t_gpu:>6.1f}  {ratio:>6.2f}  {ratio_note(ratio):<28}  {mem:>8.0f}')
            results[(dl, bs)] = {'pps': pps, 'ms_b': ms_b, 't_cpu': t_cpu,
                                 't_gpu': t_gpu, 'ratio': ratio, 'mem': mem}
    sep()
    return results


def bench_wsi(base, device, wsi_path, levels, overlaps, batch_sizes, dtypes, warmup,
              mask_cfg):
    print('\n' + '=' * 72)
    print(f'  Part 2 -- WSI Pipeline  {os.path.basename(wsi_path)}')
    print('=' * 72)
    wsi = SafeSlide(wsi_path)
    base_mpp = wsi.base_mpp
    ds_list = wsi.level_downsamples
    n_levels = len(ds_list)
    print(f'  levels={n_levels}  downsamples={[f"{d:.2f}" for d in ds_list]}')
    if base_mpp:
        print(f'  MPP/level={[f"{base_mpp * d:.3f}" for d in ds_list]}')

    base_mask = _tissue_mask(wsi, mask_cfg, device)
    n_all = len(base_mask.tissue_regions)
    print(f'  Tissue regions: {n_all}')

    all_wsi_results = []
    for level in levels:
        if level >= n_levels:
            print(f'\n  [SKIP] level {level} exceeds WSI max level {n_levels - 1}')
            continue
        ds = ds_list[level]
        mask = base_mask.patchable(256 * ds)
        if len(mask.tissue_regions) < n_all:
            print(f'  patchable: {n_all} -> {len(mask.tissue_regions)} regions at ds={ds:.2f}')

        for overlap in overlaps:
            print(f'\n  -- level={level}  ds={ds:.2f}'
                 + (f'  mpp~{base_mpp * ds:.3f}' if base_mpp else '')
                 + f'  overlap={overlap} --')
            t0 = time.perf_counter()
            tiles = read_region_tiles(wsi, mask, ds, level, overlap)
            t_extract = time.perf_counter() - t0
            n_patches = sum(len(tp) for tp in tiles)
            print(f'  Extract: {t_extract:.1f}s   patches={n_patches}  regions={len(tiles)}')
            print(f'  {"batch":>5}  {"dtype":>5}  {"encode_s":>9}  {"total_s":>8}'
                 f'  {"patches/s":>10}  {"cpu_s":>6}  {"gpu_s":>6}  {"ratio":>6}'
                 f'  {"note":<28}  {"GPU MB":>8}')
            sep('.')

            wsi_results = []
            warmup_patches = tiles[0][:max(batch_sizes)]
            for dtype in dtypes:
                encoder = base.variant(batch_size=max(batch_sizes), dtype=_dt(dtype))
                for _ in range(warmup):
                    encoder(warmup_patches[:batch_sizes[0]])
                for bs in batch_sizes:
                    reset_peak(device)
                    t_cpu_total = t_gpu_total = 0.0
                    for patches_tp in tiles:
                        if not patches_tp:
                            continue
                        tc, tg = run_encode_timed(base, device, bs, dtype, patches_tp)
                        t_cpu_total += tc; t_gpu_total += tg
                    t_encode = t_cpu_total + t_gpu_total
                    total = t_extract + t_encode
                    pps = n_patches / t_encode if t_encode > 0 else 0
                    ratio = t_cpu_total / t_gpu_total if t_gpu_total > 0 else float('inf')
                    mem = peak_gpu_mb(device)
                    dl = dtype_label(dtype)
                    print(f'  {bs:>5}  {dl:>5}  {t_encode:>9.1f}  {total:>8.1f}'
                         f'  {pps:>10.1f}  {t_cpu_total:>6.1f}  {t_gpu_total:>6.1f}'
                         f'  {ratio:>6.2f}  {ratio_note(ratio):<28}  {mem:>8.0f}')
                    wsi_results.append({'level': level, 'ds': ds, 'overlap': overlap,
                                        'dtype': dl, 'bs': bs, 't_extract': t_extract,
                                        't_encode': t_encode, 't_cpu': t_cpu_total,
                                        't_gpu': t_gpu_total, 'n_patches': n_patches,
                                        'pps': pps, 'ratio': ratio})
            sep('.')
            all_wsi_results.extend(wsi_results)
    wsi.close()
    return all_wsi_results


def print_summary(synthetic, wsi_results):
    print('\n' + '=' * 72)
    print('  Final Bottleneck Summary')
    print('=' * 72)
    if synthetic:
        print(f'\n  [Synthetic -- best throughput / min GPU time per dtype]')
        print(f'  {"dtype":>5}  {"best p/s":>10}  {"@bs":>5}  {"min gpu_s":>10}  {"@bs":>5}  {"ratio":>6}  note')
        sep('.')
        by_dtype = {}
        for (dtype, bs), v in synthetic.items():
            if dtype not in by_dtype:
                by_dtype[dtype] = {'best_pps': v, 'best_pps_bs': bs, 'min_gpu': v, 'min_gpu_bs': bs}
            else:
                if v['pps'] > by_dtype[dtype]['best_pps']['pps']:
                    by_dtype[dtype]['best_pps'] = v; by_dtype[dtype]['best_pps_bs'] = bs
                if v['t_gpu'] < by_dtype[dtype]['min_gpu']['t_gpu']:
                    by_dtype[dtype]['min_gpu'] = v; by_dtype[dtype]['min_gpu_bs'] = bs
        for dtype, d in by_dtype.items():
            bp, mg = d['best_pps'], d['min_gpu']
            print(f'  {dtype:>5}  {bp["pps"]:>10.1f}  {d["best_pps_bs"]:>5}'
                 f'  {mg["t_gpu"]:>10.1f}  {d["min_gpu_bs"]:>5}'
                 f'  {bp["ratio"]:>6.2f}  {ratio_note(bp["ratio"])}')
        sep('.')
    if wsi_results:
        print(f'\n  [WSI Pipeline -- best encode per (level, overlap)]')
        print(f'  {"level":>5}  {"overlap":>7}  {"dtype":>5}  {"bs":>4}'
             f'  {"encode_s":>9}  {"extract_s":>10}  {"cpu_s":>6}  {"gpu_s":>6}'
             f'  {"ext%":>5}  {"gpu%":>5}  overall bottleneck')
        sep('.')
        keyfn = lambda r: (r['level'], r['overlap'])
        for (level, overlap), group in groupby(sorted(wsi_results, key=keyfn), key=keyfn):
            best = max(group, key=lambda r: r['pps'])
            total = best['t_extract'] + best['t_encode']
            ext_pct = best['t_extract'] / total * 100 if total > 0 else 0
            gpu_pct = best['t_gpu'] / total * 100 if total > 0 else 0
            cpu_pct = best['t_cpu'] / total * 100 if total > 0 else 0
            verdict = max({'extract': ext_pct, 'CPU transform': cpu_pct,
                           'GPU fwd': gpu_pct}.items(), key=lambda kv: kv[1])[0]
            print(f'  {level:>5}  {str(overlap):>7}  {best["dtype"]:>5}  {best["bs"]:>4}'
                 f'  {best["t_encode"]:>9.1f}  {best["t_extract"]:>10.1f}'
                 f'  {best["t_cpu"]:>6.1f}  {best["t_gpu"]:>6.1f}'
                 f'  {ext_pct:>4.1f}%  {gpu_pct:>4.1f}%  {verdict}')
        sep('.')
        min_gpu = min(wsi_results, key=lambda r: r['t_gpu'])
        print(f'\n  Min GPU time: {min_gpu["t_gpu"]:.1f}s'
             f'  @ level={min_gpu["level"]} overlap={min_gpu["overlap"]}'
             f'  {min_gpu["dtype"]} bs={min_gpu["bs"]}')
    print()


def run_speed(args) -> int:
    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    dtypes = [dtype_map[d] for d in args.dtypes]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    wsi_exists = os.path.exists(args.wsi)

    print(f'Device : {device}')
    if device.type == 'cuda':
        for i in range(n_gpus):
            print(f'GPU {i}  : {torch.cuda.get_device_name(i)}')

    if args.compare:
        bench_compare(device, n_patches=args.compare_patches,
                      batch_sizes=args.compare_bs, warmup=args.warmup,
                      repeats=args.repeats)
        if not args.no_wsi:
            if wsi_exists:
                bench_wsi_compare(device, args.wsi, batch_sizes=args.wsi_compare_bs,
                                  level=args.wsi_compare_level,
                                  overlap=args.wsi_compare_overlap, warmup=args.warmup,
                                  mask_cfg=mask_cfg_from_args(args))
            else:
                print(f'\n[SKIP Part 2] WSI not found: {args.wsi}')
        print('Done.')
        return 0

    print('Loading GigaPath model...')
    if args.compile:
        print('torch.compile... (first warmup ~2-5 min)')
    base = _load_encoder(device, use_flash=not args.no_flash_attn,
                         use_compile=args.compile, multi_gpu=n_gpus > 1)
    if n_gpus > 1:
        print(f'DataParallel across {n_gpus} GPUs')
    opts = [o for o in ['flash-attn' if not args.no_flash_attn else None,
                        'compile' if args.compile else None] if o]
    print(f'Optimizations : {", ".join(opts) if opts else "none (baseline)"}')

    synthetic = bench_synthetic(base, device, args.batch_sizes, dtypes,
                                args.n_patches, args.warmup, args.repeats)
    wsi_results = []
    if not args.no_wsi:
        if wsi_exists:
            wsi_results = bench_wsi(base, device, args.wsi, args.levels, args.overlaps,
                                    args.batch_sizes, dtypes, args.warmup,
                                    mask_cfg_from_args(args))
        else:
            print(f'\n[SKIP Part 2] WSI not found: {args.wsi}')

    print_summary(synthetic, wsi_results)
    print('Done.')
    return 0


# ═════════════════════════════════════════════════════════════════════════

def parse_bool(s):
    return s.strip().lower() in ('true', '1', 'yes', 'on')


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['accuracy', 'speed'], default='accuracy')
    ap.add_argument('--wsi', default=(
        '/work/u26130998/datasets/histoimage.na.icar.cnr.it'
        '/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs'))

    # ── accuracy mode ──
    ap.add_argument('--svs', default=(
        '/work/u26130998/datasets/histoimage.na.icar.cnr.it'
        '/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs'))
    ap.add_argument('--mrxs', default=(
        '/work/u26130998/datasets/Ki67_with_photo/S1104043_G7E_110207_mrxs/'
        'S1104043,G7E,110207.mrxs'))
    ap.add_argument('--total-patches', type=int, default=200)
    ap.add_argument('--tile-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--configs', type=int, nargs='+', default=None,
                    help='indices into the 6-entry config table to compare '
                         '(default: just [0, 1], baseline fp32 vs fp16 only)')
    ap.add_argument('--skip-l2', action='store_true')
    ap.add_argument('--out-dir', type=Path, default=None)

    # ── speed mode ──
    ap.add_argument('--compare', action='store_true',
                    help='6-config comparison sweep (Part 1 + Part 2)')
    ap.add_argument('--compare-bs', type=int, nargs='+',
                    default=[8, 16, 64, 128, 512, 1024, 4096])
    ap.add_argument('--compare-patches', type=int, default=4096)
    ap.add_argument('--wsi-compare-bs', type=int, nargs='+', default=[32, 128, 512])
    ap.add_argument('--wsi-compare-level', type=int, default=0)
    ap.add_argument('--wsi-compare-overlap', type=parse_bool, default=False, metavar='BOOL')
    ap.add_argument('--n-patches', type=int, default=4096)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument('--dtypes', nargs='+', default=['fp32', 'fp16'],
                    choices=['fp32', 'fp16', 'bf16'])
    ap.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--overlaps', type=parse_bool, nargs='+', default=[True, False], metavar='BOOL')
    ap.add_argument('--no-flash-attn', action='store_true')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--no-wsi', action='store_true')
    # the tissue mask both modes sample inside (--seg, default hest)
    add_mask_args(ap)
    args = ap.parse_args()

    if args.mode == 'accuracy':
        out_dir = args.out_dir or Path(job_result_dir('AccuracyV1'))
        out_dir.mkdir(parents=True, exist_ok=True)
        return run_accuracy(args, out_dir)
    return run_speed(args)


if __name__ == '__main__':
    sys.exit(main())
