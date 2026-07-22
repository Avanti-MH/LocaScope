#!/usr/bin/env python3
"""
GigaPath inference speed benchmark.

════════════════════════════════════════════════════════════════════════
  --compare mode  (7 optimization configs vs baseline)
════════════════════════════════════════════════════════════════════════

  7 configs compared:
    baseline fp32
    fp16 only
    flash-attn only (fp32)
    ToMe r=8 only (fp32)
    compile only (fp32)
    fp16 + flash
    ALL fp16+flash+tome+compile

  Part 1 — bench_compare  (synthetic patches)
    Sweep: 7 configs × --compare-bs  (default [8,16,64,128,512,1024,4096])
    Output 1 — Flat table:  (config, bs) per row → patches/s, ×speedup, GPU MB
    Output 2 — Matrix:      config × bs; baseline shows p/s, others show ×speedup

  Part 2 — bench_wsi_compare  (real WSI, fixed level/overlap)
    Sweep: 7 configs × --wsi-compare-bs  (default [32,128,512])
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
    --tome            apply Token Merging (gigapath_apply_tome)
    --tome-r N        tokens merged per layer (default 8)
    --compile         torch.compile with mode=reduce-overhead (~2-5 min warmup)

════════════════════════════════════════════════════════════════════════
  Usage
════════════════════════════════════════════════════════════════════════

    python bench_gigapath_infer.py --compare
    python bench_gigapath_infer.py --compare --no-wsi
    python bench_gigapath_infer.py --compare --compare-bs 64 128 512 --wsi-compare-bs 64 128
    python bench_gigapath_infer.py
    python bench_gigapath_infer.py --tome --compile --no-wsi
    python bench_gigapath_infer.py --no-flash-attn
"""

import argparse
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from _paths import setup_import_paths
setup_import_paths()

import openslide
from PatchingLib import WsiTissuesContainer
from TissuesRegionsMask import TissuesRegionsMask
from GigaPathFunc import (
    gigapath_model, gigapath_compile, gigapath_apply_tome,
    make_gigapath_encoder, _TRANSFORM,
)


# ── encode helpers ────────────────────────────────────────────────────────────

def _run_batch_loop(model, ctx, device, batch_size, images):
    '''Core encode loop used by standard mode for cpu/gpu split timing.
    Returns (features [N,D], t_cpu, t_gpu) in seconds.

    Per-batch two-phase timing:
      Phase A (t_cpu) — CPU transform only
      Phase B (t_gpu) — H2D + model forward + normalize + D2H
    '''
    if not images:
        return torch.empty(0), 0.0, 0.0
    t_cpu = t_gpu = 0.0
    outputs = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]

        # Phase A: CPU transform
        t0 = time.perf_counter()
        batch_cpu = torch.stack([
            _TRANSFORM(img if isinstance(img, Image.Image) else Image.fromarray(img))
            for img in chunk
        ])
        t_cpu += time.perf_counter() - t0

        # Phase B: H2D + forward + normalize + D2H
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad(), ctx:
            feats = model(batch_cpu.to(device))
        feat_cpu = F.normalize(feats.float(), dim=-1).cpu()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t_gpu += time.perf_counter() - t0

        outputs.append(feat_cpu)
    return torch.cat(outputs, dim=0), t_cpu, t_gpu


def run_encode_timed(model, device, batch_size, dtype, images):
    '''Returns (t_cpu, t_gpu) in seconds. Used by standard mode.'''
    ctx = (torch.autocast(device_type=device.type, dtype=dtype)
           if dtype != torch.float32 else nullcontext())
    _, t_cpu, t_gpu = _run_batch_loop(model, ctx, device, batch_size, images)
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
    # (label,                          flash,  dtype,          tome,  compile)
    ('baseline  fp32',                 False,  torch.float32,  False, False),
    ('fp16  only',                     False,  torch.float16,  False, False),
    ('flash-attn  only (fp32)',        True,   torch.float32,  False, False),
    ('ToMe r=8  only (fp32)',          False,  torch.float32,  True,  False),
    ('compile  only (fp32)',           False,  torch.float32,  False, True),
    ('fp16 + flash',                   True,   torch.float16,  False, False),
    ('ALL  fp16+flash+tome+compile',   True,   torch.float16,  True,  True),
]

_LABEL_W = 36


def _load_model_for_config(device, use_flash, use_tome, use_compile, tome_r):
    os.environ.pop('TIMM_FUSED_ATTN', None) if use_flash else os.environ.__setitem__('TIMM_FUSED_ATTN', '0')
    model = gigapath_model(device)
    if use_tome:    model = gigapath_apply_tome(model, r=tome_r)
    if use_compile: model = gigapath_compile(model)
    return model


def _sweep_configs(device, tome_r, batch_sizes, encode_fn):
    '''
    Run all 7 _COMPARE_CONFIGS, load a fresh model per config, sweep batch_sizes.

    encode_fn(encoder, bs) -> dict  must contain at least {'pps': float, 'mem': float}.
    The callback owns warmup, timing, and any extra metric collection.

    Returns:
        all_results  : {label: {bs: result_dict}}
        baseline_pps : {bs: float}  (pps of the baseline fp32 config)
    '''
    all_results  = {}
    baseline_pps = {}
    for label, use_flash, dtype, use_tome, use_compile in _COMPARE_CONFIGS:
        if use_compile:
            print(f'  [torch.compile warmup for: {label}]')
        model = _load_model_for_config(device, use_flash, use_tome, use_compile, tome_r)
        pps_by_bs = {
            bs: encode_fn(make_gigapath_encoder(model, device, batch_size=bs, dtype=dtype), bs)
            for bs in batch_sizes
        }
        all_results[label] = pps_by_bs
        if label.startswith('baseline'):
            baseline_pps = {bs: v['pps'] for bs, v in pps_by_bs.items()}
        del model
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

def bench_compare(device, tome_r, n_patches, batch_sizes, warmup, repeats=3):
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

    all_results, baseline_pps = _sweep_configs(device, tome_r, batch_sizes, encode_fn)
    _print_compare_flat(all_results, baseline_pps, batch_sizes)
    _print_compare_matrix(all_results, baseline_pps, batch_sizes)


# ── Part 1 standard ───────────────────────────────────────────────────────────

def bench_synthetic(model, device, batch_sizes, dtypes, n_patches, warmup, repeats):
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
            encoder = make_gigapath_encoder(model, device, batch_size=bs, dtype=dtype)
            for _ in range(warmup):
                encoder(patches[:bs])

            reset_peak(device)
            cpu_runs, gpu_runs = [], []
            for _ in range(repeats):
                tc, tg = run_encode_timed(model, device, bs, dtype, patches)
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

def bench_wsi_compare(device, wsi_path, tome_r, batch_sizes, level, overlap, warmup):
    '''7 configs x batch_sizes on a real WSI (fixed level/overlap).'''
    print('\n' + '=' * 72)
    print(f'  Part 2 — WSI Comparison  level={level}  overlap={overlap}'
          f'  {os.path.basename(wsi_path)}')
    print('=' * 72)

    wsi  = openslide.OpenSlide(wsi_path)
    ds   = wsi.level_downsamples[level]
    mask = TissuesRegionsMask.from_wsi(wsi)
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

    all_results, baseline_pps = _sweep_configs(device, tome_r, batch_sizes, encode_fn)
    wsi.close()
    _print_compare_flat(all_results, baseline_pps, batch_sizes, show_encode_s=True)
    _print_compare_matrix(all_results, baseline_pps, batch_sizes)


# ── Part 2 standard (WSI) ────────────────────────────────────────────────────

def bench_wsi(model, device, wsi_path, levels, overlaps, batch_sizes, dtypes, warmup):
    '''Single model config x level x overlap x batch_sizes x dtypes, with cpu/gpu split.'''
    print('\n' + '=' * 72)
    print(f'  Part 2 — WSI Pipeline  {os.path.basename(wsi_path)}')
    print('=' * 72)

    wsi      = openslide.OpenSlide(wsi_path)
    base_mpp = float(wsi.properties.get('openslide.mpp-x', 0))
    ds_list  = wsi.level_downsamples
    n_levels = len(ds_list)

    print(f'  levels={n_levels}  downsamples={[f"{d:.2f}" for d in ds_list]}')
    if base_mpp:
        print(f'  MPP/level={[f"{base_mpp * d:.3f}" for d in ds_list]}')

    mask  = TissuesRegionsMask.from_wsi(wsi)
    n_all = len(mask.tissue_regions)
    print(f'  Tissue regions: {n_all}')

    all_wsi_results = []
    for level in levels:
        if level >= n_levels:
            print(f'\n  [SKIP] level {level} exceeds WSI max level {n_levels - 1}')
            continue
        ds = ds_list[level]

        mask.tissue_regions = TissuesRegionsMask._search_tissue_regions(
            mask.main_mask, mask.mask_ds_x, mask.mask_ds_y
        )
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
                encoder = make_gigapath_encoder(model, device, batch_size=max(batch_sizes), dtype=dtype)
                for _ in range(warmup):
                    encoder(warmup_patches[:batch_sizes[0]])

                for bs in batch_sizes:
                    reset_peak(device)
                    t_cpu_total = t_gpu_total = 0.0
                    for tp in wtc:
                        patches_tp = list(tp)
                        if not patches_tp:
                            continue
                        tc, tg = run_encode_timed(model, device, bs, dtype, patches_tp)
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
    ap.add_argument('--tome',    action='store_true')
    ap.add_argument('--tome-r',  type=int, default=8)
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
        bench_compare(device, args.tome_r,
                      n_patches=args.compare_patches,
                      batch_sizes=args.compare_bs,
                      warmup=args.warmup,
                      repeats=args.repeats)
        if not args.no_wsi:
            if wsi_exists:
                bench_wsi_compare(device, args.wsi, args.tome_r,
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
    model = gigapath_model(device)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f'DataParallel across {n_gpus} GPUs')
    if args.tome:
        model = gigapath_apply_tome(model, r=args.tome_r)
        print(f'ToMe applied (r={args.tome_r})')
    if args.compile:
        print('torch.compile... (first warmup ~2-5 min)')
        model = gigapath_compile(model)

    opts = ([o for o in ['flash-attn' if not args.no_flash_attn else None,
                         f'ToMe(r={args.tome_r})' if args.tome else None,
                         'compile' if args.compile else None] if o])
    print(f'Optimizations : {", ".join(opts) if opts else "none (baseline)"}')

    synthetic = bench_synthetic(model, device, args.batch_sizes, dtypes,
                                args.n_patches, args.warmup, args.repeats)

    wsi_results = []
    if not args.no_wsi:
        if wsi_exists:
            wsi_results = bench_wsi(model, device, args.wsi, args.levels, args.overlaps,
                                    args.batch_sizes, dtypes, args.warmup)
        else:
            print(f'\n[SKIP Part 2] WSI not found: {args.wsi}')

    print_summary(synthetic, wsi_results)
    print('Done.')


if __name__ == '__main__':
    main()
