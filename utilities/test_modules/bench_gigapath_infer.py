#!/usr/bin/env python3
"""
GigaPath inference speed benchmark.

Part 1 — synthetic patches : sweep batch_size × dtype  →  patches/sec
Part 2 — full WSI pipeline : sweep level × overlap × batch_size × dtype

Usage:
    python test_modules/bench_gigapath_infer.py
    python test_modules/bench_gigapath_infer.py --levels 0 1 --overlaps false --batch-sizes 64 128 256
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
from GigaPathFunc import gigapath_model, build_transform


# ── encode helpers ────────────────────────────────────────────────────────────

_TRANSFORM = build_transform()


def _make_ctx(device, dtype):
    return (torch.autocast(device_type=device.type, dtype=dtype)
            if dtype != torch.float32 else nullcontext())


def _run_batch_loop(model, ctx, device, batch_size: int, images):
    '''Core encode loop. Returns (features, t_cpu, t_gpu).
    features: normalized [N, D] tensor; t_cpu/t_gpu in seconds.
    '''
    if not images:
        return torch.empty(0), 0.0, 0.0
    t_cpu = t_gpu = 0.0
    outputs = []
    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        t0 = time.perf_counter()
        batch = torch.stack([
            _TRANSFORM(img if isinstance(img, Image.Image) else Image.fromarray(img))
            for img in batch_imgs
        ])
        t_cpu += time.perf_counter() - t0
        batch = batch.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            with ctx:
                feats = model(batch)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t_gpu += time.perf_counter() - t0
        outputs.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0), t_cpu, t_gpu


def make_encoder(model, device, batch_size: int, dtype: torch.dtype):
    ctx = _make_ctx(device, dtype)

    @torch.no_grad()
    def encode(images):
        features, _, _ = _run_batch_loop(model, ctx, device, batch_size, images)
        return features

    return encode


def run_encode_timed(model, device, batch_size: int, dtype: torch.dtype, images):
    '''Returns (t_cpu, t_gpu) in seconds.'''
    ctx = _make_ctx(device, dtype)
    _, t_cpu, t_gpu = _run_batch_loop(model, ctx, device, batch_size, images)
    return t_cpu, t_gpu


def dtype_label(dtype: torch.dtype) -> str:
    return {torch.float32: 'fp32', torch.float16: 'fp16', torch.bfloat16: 'bf16'}.get(dtype, str(dtype))


def peak_gpu_mb(device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else 0.0


def reset_peak(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)


def ratio_note(ratio: float) -> str:
    if ratio > 0.8:  return 'CPU bottleneck → DataLoader will help'
    if ratio < 0.3:  return 'GPU bottleneck → DataLoader marginal'
    return                  'balanced       → DataLoader worth trying'


def sep(char='─', w=72):
    print(char * w)


# ── Part 1: synthetic patches ─────────────────────────────────────────────────

def bench_synthetic(model, device, batch_sizes, dtypes, n_patches, warmup, repeats):
    print('\n' + '═' * 72)
    print(f'  Part 1 — Synthetic Inference  ({n_patches} random 256×256 patches)')
    print('═' * 72)
    print(f'  {"batch":>5}  {"dtype":>5}  {"patches/s":>10}  {"ms/batch":>9}'
          f'  {"cpu_s":>6}  {"gpu_s":>6}  {"ratio":>6}  {"note":<28}  {"GPU MB":>8}')
    sep()

    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(n_patches)]

    results = {}
    for dtype in dtypes:
        for bs in batch_sizes:
            encoder = make_encoder(model, device, bs, dtype)
            for _ in range(warmup):
                encoder(patches[:bs])

            reset_peak(device)
            cpu_list, gpu_list = [], []
            for _ in range(repeats):
                tc, tg = run_encode_timed(model, device, bs, dtype, patches)
                cpu_list.append(tc)
                gpu_list.append(tg)

            t_cpu = sum(cpu_list) / len(cpu_list)
            t_gpu = sum(gpu_list) / len(gpu_list)
            avg   = t_cpu + t_gpu
            n_batches = (n_patches + bs - 1) // bs
            pps  = n_patches / avg
            ms_b = avg / n_batches * 1000
            ratio = t_cpu / t_gpu if t_gpu > 0 else float('inf')
            mem  = peak_gpu_mb(device)

            dl = dtype_label(dtype)
            print(f'  {bs:>5}  {dl:>5}  {pps:>10.1f}  {ms_b:>9.1f}'
                  f'  {t_cpu:>6.1f}  {t_gpu:>6.1f}  {ratio:>6.2f}  {ratio_note(ratio):<28}  {mem:>8.0f}')
            results[(dl, bs)] = {
                'pps': pps, 'ms_b': ms_b, 't_cpu': t_cpu, 't_gpu': t_gpu,
                'ratio': ratio, 'mem': mem,
            }

    sep()
    return results


# ── Part 2: WSI pipeline ──────────────────────────────────────────────────────

def bench_wsi(model, device, wsi_path, levels, overlaps, batch_sizes, dtypes, warmup):
    print('\n' + '═' * 72)
    print(f'  Part 2 — WSI Pipeline  {os.path.basename(wsi_path)}')
    print('═' * 72)

    wsi = openslide.OpenSlide(wsi_path)
    base_mpp = float(wsi.properties.get('openslide.mpp-x', 0))
    n_levels = len(wsi.level_downsamples)
    ds_list = wsi.level_downsamples

    print(f'  levels={n_levels}  downsamples={[f"{d:.2f}" for d in ds_list]}')
    if base_mpp:
        print(f'  MPP/level={[f"{base_mpp*d:.3f}" for d in ds_list]}')

    all_wsi_results = []
    mask = TissuesRegionsMask.from_wsi(wsi)
    n_all = len(mask.tissue_regions)
    print(f'  Tissue regions: {n_all}')

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
                  + (f'  mpp≈{base_mpp*ds:.3f}' if base_mpp else '')
                  + f'  overlap={overlap} ──')

            t0 = time.perf_counter()
            wtc = WsiTissuesContainer(wsi, ds=ds, level=level,
                                      tile_size=256, overlap=overlap, mask=mask)
            t_extract = time.perf_counter() - t0

            n_patches = sum(len(tp) for tp in wtc)
            print(f'  Extract: {t_extract:.1f}s   patches={n_patches}  regions={len(wtc)}')

            print(f'  {"batch":>5}  {"dtype":>5}  {"encode_s":>9}  {"total_s":>8}'
                  f'  {"patches/s":>10}  {"cpu_s":>6}  {"gpu_s":>6}  {"ratio":>6}'
                  f'  {"note":<28}  {"GPU MB":>8}')
            sep('·')

            wsi_results = []
            warmup_patches = list(wtc[0])[:max(batch_sizes)]
            for dtype in dtypes:
                encoder = make_encoder(model, device, max(batch_sizes), dtype)
                for _ in range(warmup):
                    encoder(warmup_patches[:batch_sizes[0]])

                for bs in batch_sizes:
                    reset_peak(device)
                    t_cpu_total = 0.0
                    t_gpu_total = 0.0
                    for tp in wtc:
                        patches_tp = list(tp)
                        if not patches_tp:
                            continue
                        tc, tg = run_encode_timed(model, device, bs, dtype, patches_tp)
                        t_cpu_total += tc
                        t_gpu_total += tg

                    t_encode = t_cpu_total + t_gpu_total
                    total    = t_extract + t_encode
                    pps      = n_patches / t_encode if t_encode > 0 else 0
                    ratio    = t_cpu_total / t_gpu_total if t_gpu_total > 0 else float('inf')
                    mem      = peak_gpu_mb(device)

                    dl = dtype_label(dtype)
                    print(f'  {bs:>5}  {dl:>5}  {t_encode:>9.1f}  {total:>8.1f}'
                          f'  {pps:>10.1f}  {t_cpu_total:>6.1f}  {t_gpu_total:>6.1f}'
                          f'  {ratio:>6.2f}  {ratio_note(ratio):<28}  {mem:>8.0f}')
                    wsi_results.append({
                        'level': level, 'ds': ds, 'overlap': overlap,
                        'dtype': dl, 'bs': bs,
                        't_extract': t_extract, 't_encode': t_encode,
                        't_cpu': t_cpu_total, 't_gpu': t_gpu_total,
                        'n_patches': n_patches, 'pps': pps, 'ratio': ratio,
                    })
            sep('·')
            all_wsi_results.extend(wsi_results)

    wsi.close()
    return all_wsi_results


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(synthetic, wsi_results):
    print('\n' + '═' * 72)
    print('  Final Bottleneck Summary')
    print('═' * 72)

    # ── Synthetic: best throughput + min GPU time per dtype ──
    if synthetic:
        print(f'\n  [Synthetic — best throughput / min GPU time per dtype]')
        print(f'  {"dtype":>5}  {"best p/s":>10}  {"@bs":>5}  {"min gpu_s":>10}  {"@bs":>5}  {"ratio":>6}  note')
        sep('·')
        by_dtype: dict = {}
        for (dtype, bs), v in synthetic.items():
            if dtype not in by_dtype:
                by_dtype[dtype] = {'best_pps': v, 'best_pps_bs': bs,
                                   'min_gpu': v, 'min_gpu_bs': bs}
            else:
                if v['pps'] > by_dtype[dtype]['best_pps']['pps']:
                    by_dtype[dtype]['best_pps'] = v
                    by_dtype[dtype]['best_pps_bs'] = bs
                if v['t_gpu'] < by_dtype[dtype]['min_gpu']['t_gpu']:
                    by_dtype[dtype]['min_gpu'] = v
                    by_dtype[dtype]['min_gpu_bs'] = bs
        for dtype, d in by_dtype.items():
            bp = d['best_pps']
            mg = d['min_gpu']
            ratio = bp['ratio']
            print(f'  {dtype:>5}  {bp["pps"]:>10.1f}  {d["best_pps_bs"]:>5}'
                  f'  {mg["t_gpu"]:>10.1f}  {d["min_gpu_bs"]:>5}'
                  f'  {ratio:>6.2f}  {ratio_note(ratio)}')
        sep('·')

    # ── WSI: best encode per (level, overlap) + stage breakdown ──
    if wsi_results:
        print(f'\n  [WSI Pipeline — best encode per (level, overlap)]')
        print(f'  {"level":>5}  {"overlap":>7}  {"dtype":>5}  {"bs":>4}'
              f'  {"encode_s":>9}  {"extract_s":>10}  {"cpu_s":>6}  {"gpu_s":>6}'
              f'  {"ext%":>5}  {"gpu%":>5}  overall bottleneck')
        sep('·')
        from itertools import groupby
        keyfn = lambda r: (r['level'], r['overlap'])
        for key, group in groupby(sorted(wsi_results, key=keyfn), key=keyfn):
            level, overlap = key
            rows = list(group)
            best = max(rows, key=lambda r: r['pps'])
            total   = best['t_extract'] + best['t_encode']
            ext_pct = best['t_extract'] / total * 100 if total > 0 else 0
            gpu_pct = best['t_gpu']     / total * 100 if total > 0 else 0
            parts   = {'extract': ext_pct, 'CPU transform': best['t_cpu']/total*100,
                       'GPU fwd': gpu_pct}
            verdict = max(parts, key=parts.get)
            print(f'  {level:>5}  {str(overlap):>7}  {best["dtype"]:>5}  {best["bs"]:>4}'
                  f'  {best["t_encode"]:>9.1f}  {best["t_extract"]:>10.1f}'
                  f'  {best["t_cpu"]:>6.1f}  {best["t_gpu"]:>6.1f}'
                  f'  {ext_pct:>4.1f}%  {gpu_pct:>4.1f}%  {verdict}')
        sep('·')

        # min GPU time across all WSI configs
        min_gpu_row = min(wsi_results, key=lambda r: r['t_gpu'])
        print(f'\n  Min GPU time: {min_gpu_row["t_gpu"]:.1f}s'
              f'  @ level={min_gpu_row["level"]} overlap={min_gpu_row["overlap"]}'
              f'  {min_gpu_row["dtype"]} bs={min_gpu_row["bs"]}')

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_bool(s: str) -> bool:
    return s.strip().lower() in ('true', '1', 'yes', 'on')


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--wsi', default=(
        '/work/u26130998/datasets/histoimage.na.icar.cnr.it'
        '/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs'
    ))
    # Part 1
    ap.add_argument('--n-patches',  type=int, default=4096,
                    help='number of synthetic patches for Part 1')
    ap.add_argument('--warmup',     type=int, default=2)
    ap.add_argument('--repeats',    type=int, default=3,
                    help='repeat runs per config (Part 1 only)')
    # shared
    ap.add_argument('--batch-sizes', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument('--dtypes',      nargs='+', default=['fp32', 'fp16'],
                    choices=['fp32', 'fp16', 'bf16'])
    # Part 2
    ap.add_argument('--levels',   type=int,        nargs='+', default=[0, 1, 2])
    ap.add_argument('--overlaps', type=parse_bool, nargs='+', default=[True, False],
                    metavar='BOOL', help='e.g. --overlaps true false')
    ap.add_argument('--no-wsi', action='store_true', help='skip Part 2')
    args = ap.parse_args()

    dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
    dtypes = [dtype_map[d] for d in args.dtypes]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    print(f'Device : {device}')
    if device.type == 'cuda':
        for i in range(n_gpus):
            print(f'GPU {i}  : {torch.cuda.get_device_name(i)}')

    print('Loading GigaPath model...')
    model = gigapath_model(device)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f'DataParallel across {n_gpus} GPUs')

    synthetic = bench_synthetic(model, device, args.batch_sizes, dtypes,
                                args.n_patches, args.warmup, args.repeats)

    wsi_results = []
    if args.no_wsi:
        pass
    elif not os.path.exists(args.wsi):
        print(f'\n[SKIP Part 2] WSI not found: {args.wsi}')
    else:
        wsi_results = bench_wsi(model, device, args.wsi, args.levels, args.overlaps,
                                args.batch_sizes, dtypes, args.warmup)

    print_summary(synthetic, wsi_results)
    print('Done.')


if __name__ == '__main__':
    main()
