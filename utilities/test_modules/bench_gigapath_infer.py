#!/usr/bin/env python3
"""
GigaPath inference speed benchmark.

════════════════════════════════════════════════════════════════════════
  --compare mode  (6 optimization configs vs baseline)
════════════════════════════════════════════════════════════════════════

  6 configs compared:
    baseline fp32
    fp16 only
    flash-attn only (fp32)
    compile only (fp32)
    fp16 + flash
    ALL fp16+flash+compile

  Part 1 — bench_compare  (synthetic patches)
    Sweep: 6 configs × --compare-bs  (default [8,16,64,128,512,1024,4096])
    Output 1 — Flat table:  (config, bs) per row → patches/s, ×speedup, GPU MB
    Output 2 — Matrix:      config × bs; baseline shows p/s, others show ×speedup

  Part 2 — bench_wsi_compare  (real WSI, fixed level/overlap)
    Sweep: 6 configs × --wsi-compare-bs  (default [32,128,512])
           fixed --wsi-compare-level 0  --wsi-compare-overlap false
    Output 1 — Flat table:  (config, bs) per row → encode_s, patches/s, ×speedup, GPU MB
    Output 2 — Matrix:      same format as Part 1

════════════════════════════════════════════════════════════════════════
  Standard mode  (single model config, detailed sweep)
════════════════════════════════════════════════════════════════════════

  Part 1 — bench_synthetic
    Sweep: --batch-sizes × --dtypes
    Output: patches/s, ms/batch, cpu_s, gpu_s, cpu/gpu ratio,
            ratio note (CPU bottleneck / GPU bottleneck / balanced)

  Part 2 — bench_wsi
    Sweep: --levels × --overlaps × --batch-sizes × --dtypes
    Output: encode_s, total_s, patches/s, cpu_s, gpu_s, ratio
    Final print_summary: best config per (level, overlap) + bottleneck verdict

  Model flags (standard mode only; --compare covers all combos automatically):
    --no-flash-attn   set TIMM_FUSED_ATTN=0 before model load
    --compile         torch.compile with mode=reduce-overhead (~2-5 min warmup)

════════════════════════════════════════════════════════════════════════
  Usage
════════════════════════════════════════════════════════════════════════

    python bench_gigapath_infer.py --compare
    python bench_gigapath_infer.py --compare --no-wsi
    python bench_gigapath_infer.py --compare --compare-bs 64 128 512 --wsi-compare-bs 64 128
    python bench_gigapath_infer.py
    python bench_gigapath_infer.py --compile --no-wsi
    python bench_gigapath_infer.py --no-flash-attn
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

# _paths holds the one definition of OUTPUT_ROOT for every package, so it lives
# in utilities/ rather than beside this file. That directory goes on sys.path
# here, because setup_import_paths -- which puts the rest there -- is inside it.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from _paths import setup_import_paths
setup_import_paths()

import openslide
from SafeSlide import SafeSlide
from PatchingLib import WsiTissuesContainer
from TissuesRegionsMask import TissuesRegionsMask
from GigaPathFunc import (
    GigaPathEncoderConfig,
)


# ── encode helpers ────────────────────────────────────────────────────────────

def _run_batch_loop(encoder, ctx, device, batch_size, images):
    '''Core encode loop used by standard mode for cpu/gpu split timing.
    Returns (features [N,D], t_cpu, t_gpu) in seconds.

    Per-batch two-phase timing:
      Phase A (t_cpu) — CPU transform only
      Phase B (t_gpu) — H2D + model forward + pool + normalize + D2H

    The pool is explicit because the model is built with global_pool='': it
    hands back [B, 197, D], and taking the CLS here is what keeps phase B
    measuring the same work it measured before, rather than a 197x larger D2H.
    '''
    m = getattr(encoder.model, 'module', encoder.model)
    if not images:
        return torch.empty(0), 0.0, 0.0
    t_cpu = t_gpu = 0.0
    outputs = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]

        # Phase A: CPU transform
        t0 = time.perf_counter()
        batch_cpu = torch.stack([
            _CPU_TRANSFORM(img if isinstance(img, Image.Image) else Image.fromarray(img))
            for img in chunk
        ])
        t_cpu += time.perf_counter() - t0

        # Phase B: H2D + forward + normalize + D2H
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
    '''Returns (t_cpu, t_gpu) in seconds. Used by standard mode.'''
    ctx = (torch.autocast(device_type=device.type, dtype=dtype)
           if dtype != torch.float32 else nullcontext())
    _, t_cpu, t_gpu = _run_batch_loop(encoder, ctx, device, batch_size, images)
    return t_cpu, t_gpu


# ── utils ─────────────────────────────────────────────────────────────────────

def dtype_label(dtype):
    return {torch.float32: 'fp32', torch.float16: 'fp16', torch.bfloat16: 'bf16'}.get(dtype, str(dtype))

def peak_gpu_mb(device):
    return torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else 0.0

def reset_peak(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

def ratio_note(ratio):
    if ratio > 0.8: return 'CPU bottleneck → DataLoader will help'
    if ratio < 0.3: return 'GPU bottleneck → DataLoader marginal'
    return               'balanced       → DataLoader worth trying'

def sep(char='─', w=72):
    print(char * w)


# ── compare mode: configs and shared loop ─────────────────────────────────────

_COMPARE_CONFIGS = [
    # (label,                          flash,  dtype,          compile)
    ('baseline  fp32',                 False,  torch.float32,  False),
    ('fp16  only',                     False,  torch.float16,  False),
    ('flash-attn  only (fp32)',        True,   torch.float32,  False),
    ('compile  only (fp32)',           False,  torch.float32,  True),
    ('fp16 + flash',                   True,   torch.float16,  False),
    ('ALL  fp16+flash+compile',        True,   torch.float16,  True),
]

_LABEL_W = 36


#: torch dtype -> the string GigaPathEncoderConfig takes. This bench sweeps precisions
#: and holds them as torch dtypes throughout; the config is a string so that a
#: hash of it is stable across torch versions.
def _dt(d) -> str:
    return 'fp16' if d is torch.float16 else 'fp32'


_CPU_TRANSFORM = TransformConfig().build()


def _load_encoder_for_config(device, use_flash, use_compile):
    '''One encoder per config. compile is a GigaPathEncoderConfig field, so
    build() owns when it is applied rather than the caller.'''
    os.environ.pop('TIMM_FUSED_ATTN', None) if use_flash else os.environ.__setitem__('TIMM_FUSED_ATTN', '0')
    return GigaPathEncoderConfig(compile=use_compile).build(device)


def _sweep_configs(device, batch_sizes, encode_fn):
    '''
    Run all 6 _COMPARE_CONFIGS, load a fresh model per config, sweep batch_sizes.

    encode_fn(encoder, bs) -> dict  must contain at least {'pps': float, 'mem': float}.
    The callback owns warmup, timing, and any extra metric collection.

    Returns:
        all_results  : {label: {bs: result_dict}}
        baseline_pps : {bs: float}  (pps of the baseline fp32 config)
    '''
    all_results  = {}
    baseline_pps = {}
    for label, use_flash, dtype, use_compile in _COMPARE_CONFIGS:
        if use_compile:
            print(f'  [torch.compile warmup for: {label}]')
        base = _load_encoder_for_config(device, use_flash, use_compile)
        # variant(), not a second build: the sweep changes batch size and
        # precision, neither of which rebuilds a model, and reloading 4.5 GB per
        # point would be most of what this bench claims to measure.
        pps_by_bs = {
            bs: encode_fn(base.variant(batch_size=bs, dtype=_dt(dtype)), bs)
            for bs in batch_sizes
        }
        all_results[label] = pps_by_bs
        if label.startswith('baseline'):
            baseline_pps = {bs: v['pps'] for bs, v in pps_by_bs.items()}
        del base
    return all_results, baseline_pps


def _time_encoder(encoder, patches, device, repeats):
    '''Run encoder on all patches `repeats` times; return mean elapsed seconds.'''
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        encoder(patches)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


# ── compare mode: printing ────────────────────────────────────────────────────

def _print_compare_flat(all_results, baseline_pps, batch_sizes, *, show_encode_s=False):
    '''Flat (config, bs) rows. show_encode_s=True adds encode_s column (WSI mode).'''
    print(f'\n  [Flat — (config, bs) rows]')
    if show_encode_s:
        print(f'  {"config":<{_LABEL_W}}  {"bs":>5}  {"encode_s":>9}  {"patches/s":>10}  {"vs baseline":>11}  {"GPU MB":>8}')
    else:
        print(f'  {"config":<{_LABEL_W}}  {"bs":>5}  {"patches/s":>10}  {"vs baseline":>11}  {"GPU MB":>8}')
    sep()
    for label, pps_by_bs in all_results.items():
        is_base = label.startswith('baseline')
        for bs in batch_sizes:
            if bs not in pps_by_bs:
                print(f'  {label:<{_LABEL_W}}  {bs:>5}  {"OOM":>{"9" if show_encode_s else "10"}}')
                continue
            v       = pps_by_bs[bs]
            speedup = v['pps'] / baseline_pps[bs] if bs in baseline_pps else 1.0
            mark    = ' ←' if is_base else ''
            if show_encode_s:
                print(f'  {label:<{_LABEL_W}}  {bs:>5}  {v["encode_s"]:>9.1f}  {v["pps"]:>10.1f}'
                      f'  {speedup:>10.2f}x  {v["mem"]:>8.0f}{mark}')
            else:
                print(f'  {label:<{_LABEL_W}}  {bs:>5}  {v["pps"]:>10.1f}  {speedup:>10.2f}x  {v["mem"]:>8.0f}{mark}')
        if not is_base:
            sep('·')
    sep()


def _print_compare_matrix(all_results, baseline_pps, batch_sizes):
    '''Matrix: config × bs; baseline row shows p/s, other rows show xspeedup.'''
    def _bs_label(bs):
        return f'{bs // 1000}K' if bs >= 1000 else str(bs)

    col_w = 7
    hdrs  = [f'bs={_bs_label(bs)}'.rjust(col_w) for bs in batch_sizes]
    print(f'\n  [Matrix — patches/s for baseline; xspeedup for others]')
    print(f'  {"config":<{_LABEL_W}}  ' + '  '.join(hdrs))
    sep()
    for label, pps_by_bs in all_results.items():
        is_base = label.startswith('baseline')
        cells = []
        for bs in batch_sizes:
            if bs not in pps_by_bs:
                cells.append('OOM'.rjust(col_w))
            elif is_base:
                cells.append(f'{pps_by_bs[bs]["pps"]:>{col_w}.0f}')
            else:
                base = baseline_pps.get(bs)
                val  = pps_by_bs[bs]['pps'] / base if base else float('nan')
                cells.append(f'{val:>{col_w - 1}.2f}x')
        suffix = '  p/s' if is_base else '  x  '
        print(f'  {label:<{_LABEL_W}}  ' + '  '.join(cells) + suffix)
    sep()


# ── Part 1 comparison ─────────────────────────────────────────────────────────

def bench_compare(device, n_patches, batch_sizes, warmup, repeats=3):
    '''7 configs x batch_sizes sweep on synthetic patches.'''
    print('\n' + '=' * 72)
    print(f'  Part 1 — Comparison Sweep  n={n_patches} synthetic patches')
    print('=' * 72)

    rng     = np.random.default_rng(0)
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


# ── Part 1 standard ───────────────────────────────────────────────────────────

def bench_synthetic(base, device, batch_sizes, dtypes, n_patches, warmup, repeats):
    '''Single model config x batch_sizes x dtypes, with cpu/gpu split timing.'''
    print('\n' + '=' * 72)
    print(f'  Part 1 — Synthetic Sweep  ({n_patches} random 256x256 patches)')
    print('=' * 72)
    print(f'  {"batch":>5}  {"dtype":>5}  {"patches/s":>10}  {"ms/batch":>9}'
          f'  {"cpu_s":>6}  {"gpu_s":>6}  {"ratio":>6}  {"note":<28}  {"GPU MB":>8}')
    sep()

    rng     = np.random.default_rng(0)
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

            t_cpu     = sum(cpu_runs) / len(cpu_runs)
            t_gpu     = sum(gpu_runs) / len(gpu_runs)
            avg       = t_cpu + t_gpu
            n_batches = (n_patches + bs - 1) // bs
            pps       = n_patches / avg
            ms_b      = avg / n_batches * 1000
            ratio     = t_cpu / t_gpu if t_gpu > 0 else float('inf')
            mem       = peak_gpu_mb(device)

            dl = dtype_label(dtype)
            print(f'  {bs:>5}  {dl:>5}  {pps:>10.1f}  {ms_b:>9.1f}'
                  f'  {t_cpu:>6.1f}  {t_gpu:>6.1f}  {ratio:>6.2f}  {ratio_note(ratio):<28}  {mem:>8.0f}')
            results[(dl, bs)] = {
                'pps': pps, 'ms_b': ms_b, 't_cpu': t_cpu,
                't_gpu': t_gpu, 'ratio': ratio, 'mem': mem,
            }

    sep()
    return results


# ── Part 2 comparison (WSI) ───────────────────────────────────────────────────

def bench_wsi_compare(device, wsi_path, batch_sizes, level, overlap, warmup):
    '''7 configs x batch_sizes on a real WSI (fixed level/overlap).'''
    print('\n' + '=' * 72)
    print(f'  Part 2 — WSI Comparison  level={level}  overlap={overlap}'
          f'  {os.path.basename(wsi_path)}')
    print('=' * 72)

    wsi  = SafeSlide(wsi_path)
    ds   = wsi.level_downsamples[level]
    from TissueSegFunc import TissueSegConfig
    # hsv: these need real tissue tiles, so blank glass has to
    # be excluded. Retrieval does not -- see GigaPathSlidingWinSim.
    mask = TissuesRegionsMask.from_wsi(
        wsi, method=TissueSegConfig('hsv').build())
    wtc  = WsiTissuesContainer(wsi, ds=ds, level=level,
                               tile_size=256, overlap=overlap, mask=mask)
    n_patches = sum(len(tp) for tp in wtc)
    print(f'  n_patches={n_patches}  regions={len(wtc)}  ds={ds:.2f}')

    def encode_fn(encoder, bs):
        for _ in range(warmup):
            encoder(list(wtc[0])[:bs])
        reset_peak(device)
        t_encode = 0.0
        for tp in wtc:
            tp_patches = list(tp)
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


# ── Part 2 standard (WSI) ────────────────────────────────────────────────────

def bench_wsi(base, device, wsi_path, levels, overlaps, batch_sizes, dtypes, warmup):
    '''Single model config x level x overlap x batch_sizes x dtypes, with cpu/gpu split.'''
    print('\n' + '=' * 72)
    print(f'  Part 2 — WSI Pipeline  {os.path.basename(wsi_path)}')
    print('=' * 72)

    wsi      = SafeSlide(wsi_path)
    base_mpp = wsi.base_mpp  # SafeSlide.base_mpp: mean of mpp-x/y, one definition
    ds_list  = wsi.level_downsamples
    n_levels = len(ds_list)

    print(f'  levels={n_levels}  downsamples={[f"{d:.2f}" for d in ds_list]}')
    if base_mpp:
        print(f'  MPP/level={[f"{base_mpp * d:.3f}" for d in ds_list]}')

    from TissueSegFunc import TissueSegConfig
    # hsv: these need real tissue tiles, so blank glass has to
    # be excluded. Retrieval does not -- see GigaPathSlidingWinSim.
    mask = TissuesRegionsMask.from_wsi(
        wsi, method=TissueSegConfig('hsv').build())
    n_all = len(mask.tissue_regions)
    print(f'  Tissue regions: {n_all}')

    all_wsi_results = []
    for level in levels:
        if level >= n_levels:
            print(f'\n  [SKIP] level {level} exceeds WSI max level {n_levels - 1}')
            continue
        ds = ds_list[level]

        # regions_resume re-runs the connected-component search and, unlike a
        # bare _search_tissue_regions call, carries mask.origin_* so the boxes
        # come back in absolute level-0 coordinates on a cropped mask.
        mask.regions_resume()
        mask.filter_patchable(tile_size=256, ds=ds)
        if len(mask.tissue_regions) < n_all:
            print(f'  filter_patchable: {n_all} → {len(mask.tissue_regions)} regions at ds={ds:.2f}')

        for overlap in overlaps:
            print(f'\n  ── level={level}  ds={ds:.2f}'
                  + (f'  mpp~{base_mpp * ds:.3f}' if base_mpp else '')
                  + f'  overlap={overlap} ──')

            t0  = time.perf_counter()
            wtc = WsiTissuesContainer(wsi, ds=ds, level=level,
                                      tile_size=256, overlap=overlap, mask=mask)
            t_extract = time.perf_counter() - t0
            n_patches = sum(len(tp) for tp in wtc)
            print(f'  Extract: {t_extract:.1f}s   patches={n_patches}  regions={len(wtc)}')

            print(f'  {"batch":>5}  {"dtype":>5}  {"encode_s":>9}  {"total_s":>8}'
                  f'  {"patches/s":>10}  {"cpu_s":>6}  {"gpu_s":>6}  {"ratio":>6}'
                  f'  {"note":<28}  {"GPU MB":>8}')
            sep('·')

            wsi_results    = []
            warmup_patches = list(wtc[0])[:max(batch_sizes)]
            for dtype in dtypes:
                encoder = base.variant(batch_size=max(batch_sizes), dtype=_dt(dtype))
                for _ in range(warmup):
                    encoder(warmup_patches[:batch_sizes[0]])

                for bs in batch_sizes:
                    reset_peak(device)
                    t_cpu_total = t_gpu_total = 0.0
                    for tp in wtc:
                        patches_tp = list(tp)
                        if not patches_tp:
                            continue
                        tc, tg = run_encode_timed(base, device, bs, dtype, patches_tp)
                        t_cpu_total += tc; t_gpu_total += tg

                    t_encode = t_cpu_total + t_gpu_total
                    total    = t_extract + t_encode
                    pps      = n_patches / t_encode if t_encode > 0 else 0
                    ratio    = t_cpu_total / t_gpu_total if t_gpu_total > 0 else float('inf')
                    mem      = peak_gpu_mb(device)
                    dl       = dtype_label(dtype)

                    print(f'  {bs:>5}  {dl:>5}  {t_encode:>9.1f}  {total:>8.1f}'
                          f'  {pps:>10.1f}  {t_cpu_total:>6.1f}  {t_gpu_total:>6.1f}'
                          f'  {ratio:>6.2f}  {ratio_note(ratio):<28}  {mem:>8.0f}')
                    wsi_results.append({
                        'level': level, 'ds': ds, 'overlap': overlap, 'dtype': dl, 'bs': bs,
                        't_extract': t_extract, 't_encode': t_encode,
                        't_cpu': t_cpu_total, 't_gpu': t_gpu_total,
                        'n_patches': n_patches, 'pps': pps, 'ratio': ratio,
                    })
            sep('·')
            all_wsi_results.extend(wsi_results)

    wsi.close()
    return all_wsi_results


# ── Summary (standard mode) ───────────────────────────────────────────────────

def print_summary(synthetic, wsi_results):
    print('\n' + '=' * 72)
    print('  Final Bottleneck Summary')
    print('=' * 72)

    if synthetic:
        print(f'\n  [Synthetic — best throughput / min GPU time per dtype]')
        print(f'  {"dtype":>5}  {"best p/s":>10}  {"@bs":>5}  {"min gpu_s":>10}  {"@bs":>5}  {"ratio":>6}  note')
        sep('·')
        by_dtype = {}
        for (dtype, bs), v in synthetic.items():
            if dtype not in by_dtype:
                by_dtype[dtype] = {'best_pps': v, 'best_pps_bs': bs, 'min_gpu': v, 'min_gpu_bs': bs}
            else:
                if v['pps']   > by_dtype[dtype]['best_pps']['pps']:
                    by_dtype[dtype]['best_pps'] = v; by_dtype[dtype]['best_pps_bs'] = bs
                if v['t_gpu'] < by_dtype[dtype]['min_gpu']['t_gpu']:
                    by_dtype[dtype]['min_gpu'] = v; by_dtype[dtype]['min_gpu_bs'] = bs
        for dtype, d in by_dtype.items():
            bp, mg = d['best_pps'], d['min_gpu']
            print(f'  {dtype:>5}  {bp["pps"]:>10.1f}  {d["best_pps_bs"]:>5}'
                  f'  {mg["t_gpu"]:>10.1f}  {d["min_gpu_bs"]:>5}'
                  f'  {bp["ratio"]:>6.2f}  {ratio_note(bp["ratio"])}')
        sep('·')

    if wsi_results:
        print(f'\n  [WSI Pipeline — best encode per (level, overlap)]')
        print(f'  {"level":>5}  {"overlap":>7}  {"dtype":>5}  {"bs":>4}'
              f'  {"encode_s":>9}  {"extract_s":>10}  {"cpu_s":>6}  {"gpu_s":>6}'
              f'  {"ext%":>5}  {"gpu%":>5}  overall bottleneck')
        sep('·')
        from itertools import groupby
        keyfn = lambda r: (r['level'], r['overlap'])
        for (level, overlap), group in groupby(sorted(wsi_results, key=keyfn), key=keyfn):
            best    = max(group, key=lambda r: r['pps'])
            total   = best['t_extract'] + best['t_encode']
            ext_pct = best['t_extract'] / total * 100 if total > 0 else 0
            gpu_pct = best['t_gpu']     / total * 100 if total > 0 else 0
            verdict = max({'extract': ext_pct,
                           'CPU transform': best['t_cpu'] / total * 100,
                           'GPU fwd': gpu_pct}, key=lambda k: {'extract': ext_pct,
                               'CPU transform': best['t_cpu']/total*100, 'GPU fwd': gpu_pct}[k])
            print(f'  {level:>5}  {str(overlap):>7}  {best["dtype"]:>5}  {best["bs"]:>4}'
                  f'  {best["t_encode"]:>9.1f}  {best["t_extract"]:>10.1f}'
                  f'  {best["t_cpu"]:>6.1f}  {best["t_gpu"]:>6.1f}'
                  f'  {ext_pct:>4.1f}%  {gpu_pct:>4.1f}%  {verdict}')
        sep('·')
        min_gpu = min(wsi_results, key=lambda r: r['t_gpu'])
        print(f'\n  Min GPU time: {min_gpu["t_gpu"]:.1f}s'
              f'  @ level={min_gpu["level"]} overlap={min_gpu["overlap"]}'
              f'  {min_gpu["dtype"]} bs={min_gpu["bs"]}')

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_bool(s):
    return s.strip().lower() in ('true', '1', 'yes', 'on')


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--wsi', default=(
        '/work/u26130998/datasets/histoimage.na.icar.cnr.it'
        '/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs'
    ))

    # ── comparison mode ──
    ap.add_argument('--compare',         action='store_true',
                    help='run 7-config comparison sweep (Part 1 + Part 2)')
    ap.add_argument('--compare-bs',      type=int, nargs='+',
                    default=[8, 16, 64, 128, 512, 1024, 4096],
                    help='batch sizes for Part 1 comparison sweep')
    ap.add_argument('--compare-patches', type=int, default=4096,
                    help='synthetic patches per timing run in comparison mode')
    ap.add_argument('--wsi-compare-bs',      type=int,        nargs='+', default=[32, 128, 512],
                    help='batch sizes for Part 2 WSI comparison')
    ap.add_argument('--wsi-compare-level',   type=int,        default=0)
    ap.add_argument('--wsi-compare-overlap', type=parse_bool, default=False, metavar='BOOL')

    # ── standard mode ──
    ap.add_argument('--n-patches',   type=int, default=4096)
    ap.add_argument('--warmup',      type=int, default=2)
    ap.add_argument('--repeats',     type=int, default=3)
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument('--dtypes',      nargs='+', default=['fp32', 'fp16'],
                    choices=['fp32', 'fp16', 'bf16'])
    ap.add_argument('--levels',   type=int,        nargs='+', default=[0, 1, 2])
    ap.add_argument('--overlaps', type=parse_bool, nargs='+', default=[True, False], metavar='BOOL')

    # ── model flags (standard mode only) ──
    ap.add_argument('--no-flash-attn', action='store_true',
                    help='disable flash-attn (TIMM_FUSED_ATTN=0)')
    ap.add_argument('--compile', action='store_true',
                    help='torch.compile (first warmup ~2-5 min)')

    ap.add_argument('--no-wsi', action='store_true', help='skip Part 2')
    args = ap.parse_args()

    dtype_map  = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    dtypes     = [dtype_map[d] for d in args.dtypes]
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus     = torch.cuda.device_count() if device.type == 'cuda' else 0
    wsi_exists = os.path.exists(args.wsi)

    print(f'Device : {device}')
    if device.type == 'cuda':
        for i in range(n_gpus):
            print(f'GPU {i}  : {torch.cuda.get_device_name(i)}')

    # ── comparison mode ──────────────────────────────────────────────────────
    if args.compare:
        bench_compare(device,
                      n_patches=args.compare_patches,
                      batch_sizes=args.compare_bs,
                      warmup=args.warmup,
                      repeats=args.repeats)
        if not args.no_wsi:
            if wsi_exists:
                bench_wsi_compare(device, args.wsi,
                                  batch_sizes=args.wsi_compare_bs,
                                  level=args.wsi_compare_level,
                                  overlap=args.wsi_compare_overlap,
                                  warmup=args.warmup)
            else:
                print(f'\n[SKIP Part 2] WSI not found: {args.wsi}')
        print('Done.')
        return

    # ── standard mode ────────────────────────────────────────────────────────
    if args.no_flash_attn:
        os.environ['TIMM_FUSED_ATTN'] = '0'

    print('Loading GigaPath model...')
    if args.compile:
        print('torch.compile... (first warmup ~2-5 min)')
    base = GigaPathEncoderConfig(compile=args.compile).build(
        device, multi_gpu=n_gpus > 1)
    if n_gpus > 1:
        print(f'DataParallel across {n_gpus} GPUs')

    opts = ([o for o in ["flash-attn" if not args.no_flash_attn else None,
                         'compile' if args.compile else None] if o])
    print(f'Optimizations : {", ".join(opts) if opts else "none (baseline)"}')

    synthetic = bench_synthetic(base, device, args.batch_sizes, dtypes,
                                args.n_patches, args.warmup, args.repeats)

    wsi_results = []
    if not args.no_wsi:
        if wsi_exists:
            wsi_results = bench_wsi(base, device, args.wsi, args.levels, args.overlaps,
                                    args.batch_sizes, dtypes, args.warmup)
        else:
            print(f'\n[SKIP Part 2] WSI not found: {args.wsi}')

    print_summary(synthetic, wsi_results)
    print('Done.')


if __name__ == '__main__':
    main()
