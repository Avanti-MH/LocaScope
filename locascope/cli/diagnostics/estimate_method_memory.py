#!/usr/bin/env python3
"""How much memory does EACH stage-1 method actually need -- one model's
parameters (exact, no run needed) plus one real forward pass's PEAK GPU
memory (measured, not guessed -- see the module docstring on why a formula
cannot answer this one). Takes the same `--stage1 <method>:<recipe>`
specs `bench_stage1_mpp.py` does, because the question only matters for
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
        --stage1 knn:gigapath knn:uni2 classifier:gigapath-arcface --batch 128
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import stage1_estimation                                            # noqa: E402


def _bytes(n) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or unit == 'GB':
            return f'{n:,.1f} {unit}'
        n /= 1024


def _param_bytes(*modules) -> int:
    return sum(p.numel() * p.element_size()
               for m in modules for p in m.parameters())


def measure(cls, cfg, batch: int, device) -> dict:
    """(params_bytes, peak_forward_bytes) for one method, built and freed
    within this call -- never resident alongside another method's. A method
    with no encoder (classic) costs nothing here."""
    est = cls(cfg, device=device)
    encoder = getattr(est, 'encoder', None)
    if encoder is None:
        return dict(params_bytes=0, peak_forward_bytes=0)
    head = getattr(est, 'head', None)
    modules = (encoder.model,) + ((head,) if isinstance(head, torch.nn.Module) else ())
    params_bytes = _param_bytes(*modules)
    tile = int(getattr(cfg, 'tile_size', 256))

    images = [np.random.randint(0, 255, (tile, tile, 3), dtype=np.uint8)
              for _ in range(batch)]
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        encoder(images)   # TileEncoder.__call__ -> .features(): the
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
    ap.add_argument('--stage1', nargs='+', required=True,
                    help='<method>:<recipe>, as bench_stage1_mpp takes them')
    ap.add_argument('--batch', type=int, default=128,
                    help='synthetic patches per forward pass, measured once')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = torch.device(args.device)

    print(f'{len(args.stage1)} method(s), batch={args.batch}, device={device}\n')
    print(f'  {"method":40s}{"params":>12s}{"peak fwd":>12s}')
    results = []
    for spec in args.stage1:
        try:
            _, _, cfg, cls = stage1_estimation.recipe(spec)
            r = measure(cls, cfg, args.batch, device)
        except Exception as exc:                             # noqa: BLE001
            print(f'  {spec:40s}  [FAIL] {type(exc).__name__}: {exc}')
            continue
        results.append(r)
        print(f'  {spec:40s}{_bytes(r["params_bytes"]):>12s}'
              f'{_bytes(r["peak_forward_bytes"]):>12s}')

    if not results:
        print('\nnothing measured')
        return 1

    one_at_a_time = max(r['params_bytes'] + r['peak_forward_bytes'] for r in results)
    all_at_once = sum(r['params_bytes'] + r['peak_forward_bytes'] for r in results)
    print(f'\n  one at a time (max): {_bytes(one_at_a_time)}   '
          f'-- what the method-outer loop needs at peak')
    print(f'  all at once   (sum): {_bytes(all_at_once)}   '
          f'-- what building every method up front would need')
    return 0


if __name__ == '__main__':
    sys.exit(main())
