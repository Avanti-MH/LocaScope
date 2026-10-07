#!/usr/bin/env python3
"""How much memory does EACH stage-1 method actually need -- one model's
parameters (exact, no run needed) plus one real forward pass's PEAK GPU
memory (measured, not guessed -- see the module docstring on why a formula
cannot answer this one). Takes the exact same `--knn-encoder`/
`--classifier-weights` method specs `bench_stage1_mpp.py`
does, because the question only matters for
whichever methods a real run actually names.

Two numbers, and they answer different questions:

    ONE AT A TIME (max across methods)   what stage1_compare's own
                                         method-outer loop needs at peak,
                                         with only one method's encoder
                                         resident at once.
    ALL AT ONCE (sum across methods)     what building every method up
                                         front would need.

Peak forward memory is measured on a SYNTHETIC batch (random uint8 tiles),
not a real query -- the transformer forward pass is the expensive part and
it costs the same whether the pixels are real or noise; a real WSI read
would only add I/O time this number does not care about.

For `ClassifierEstMpp`, only the ENCODER's forward is measured, not the head
on top of it -- the head's own parameters are already counted in the params
total, and its forward cost (a Linear/MLP/ArcFace/AttentionPool layer over
an already-computed feature) is negligible next to the foundation-model
backbone underneath it. Measuring the encoder alone is also what lets this
script use `TileEncoder`'s own public `.features()` call rather than reaching
into `KnnEstMpp`/`ClassifierEstMpp`'s private `_raw_of`/`_num_prefix` --
pooling is a cheap step on top of the same forward activations, so the peak
this reports is the same either way.

Usage:
    python utilities/cli/diagnostics/estimate_method_memory.py \\
        --knn-encoder gigapath uni2 \\
        --classifier-weights result/MppRoutingHead/weights/*_best.pt \\
        --batch 128 --tile 256
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

from stage1_estimation.KnnEstMpp import KnnEstMpp, KnnEstMppConfig      # noqa: E402
from stage1_estimation.ClassifierEstMpp import ClassifierEstMpp, ClassifierEstMppConfig  # noqa: E402
from TileSampler import SAMPLER_RECIPES                               # noqa: E402


def _bytes(n) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:,.1f} {unit}'
        n /= 1024


def _param_bytes(*modules) -> int:
    return sum(p.numel() * p.element_size()
              for m in modules for p in m.parameters())


def measure(spec: dict, args, device) -> dict:
    """(params_bytes, peak_forward_bytes) for one method spec, built and
    freed within this call -- never resident alongside another method's."""
    if spec['kind'] == 'knn':
        cfg = KnnEstMppConfig(
            encoder=spec['encoder'],
            sampler_cfg=replace(SAMPLER_RECIPES['reference-bank'],
                                n_per_rung=args.knn_samples),
            k=args.knn_k, tile_size=args.tile)
        est = KnnEstMpp(cfg, device)
        modules = (est.encoder.model,)
        tile = args.tile
    else:
        cfg = ClassifierEstMppConfig.from_checkpoint(spec['weights_path'])
        est = ClassifierEstMpp(cfg, device)
        modules = (est.encoder.model, est.head)
        tile = cfg.tile_size

    params_bytes = _param_bytes(*modules)

    images = [np.random.randint(0, 255, (tile, tile, 3), dtype=np.uint8)
             for _ in range(args.batch)]
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        est.encoder(images)   # TileEncoder.__call__ -> .features(): the
                              # same forward pass encode_raw/trunk_raw run,
                              # read through the pooled exit -- see module
                              # docstring for why that is enough here.
    peak_bytes = (torch.cuda.max_memory_allocated(device)
                 if device.type == 'cuda' else 0)

    del est
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return dict(params_bytes=params_bytes, peak_forward_bytes=peak_bytes)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--knn-encoder', nargs='+', default=[])
    ap.add_argument('--knn-samples', type=int, default=40)
    ap.add_argument('--knn-k', type=int, default=5)
    ap.add_argument('--classifier-weights', nargs='+', default=[])
    ap.add_argument('--tile', type=int, default=256,
                    help='KnnEstMpp only -- ClassifierEstMpp reads its own '
                         "tile size off each checkpoint's extra dict")
    ap.add_argument('--batch', type=int, default=128,
                    help='synthetic patches per forward pass, measured once')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if not args.knn_encoder and not args.classifier_weights:
        ap.error('need --knn-encoder and/or --classifier-weights')
    device = torch.device(args.device)

    specs = [dict(kind='knn', encoder=n, weights_path=None) for n in args.knn_encoder]
    for weights in args.classifier_weights:
        specs.append(dict(kind='classifier', encoder=os.path.basename(weights),
                          weights_path=weights))

    print(f'{len(specs)} method(s), batch={args.batch}, device={device}\n')
    print(f'  {"method":40s}{"params":>12s}{"peak fwd":>12s}')
    results = []
    for spec in specs:
        try:
            r = measure(spec, args, device)
        except Exception as exc:                             # noqa: BLE001
            print(f'  {spec["encoder"]:40s}  [FAIL] {type(exc).__name__}: {exc}')
            continue
        results.append(r)
        print(f'  {spec["encoder"]:40s}{_bytes(r["params_bytes"]):>12s}'
             f'{_bytes(r["peak_forward_bytes"]):>12s}')

    if not results:
        print('\nnothing measured')
        return 1

    one_at_a_time = max(r['params_bytes'] + r['peak_forward_bytes'] for r in results)
    all_at_once = sum(r['params_bytes'] + r['peak_forward_bytes'] for r in results)
    print(f'\n  one at a time (max): {_bytes(one_at_a_time)}   '
         f'-- what the method-outer loop needs at peak')
    print(f'  all at once   (sum): {_bytes(all_at_once)}   '
         f'-- what building every method up front would need '
         f'(job 346494\'s OOM)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
