#!/bin/bash
#SBATCH --job-name=DiagMultiGPU          # -> log/<name>
#SBATCH --partition=normal               # Partition
#SBATCH --time=01:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=2                # the point of the test
#SBATCH --cpus-per-task=8                # one transform loop feeds every card
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH --mem=600G                       # RAM
#SBATCH -o ./log/DiagMultiGPU            # STDOUT
#SBATCH -e ./log/DiagMultiGPU            # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Does DataParallel change GigaPath, and does it help? ------
#
# gigapath_model(device, multi_gpu=True) has been in GigaPathFunc.py all along
# with no caller ever setting it. bench_locascope now has --multi-gpu and
# BenchLocaScope.sh asks for 4 cards, so this answers the two questions that
# should be answered before a multi-hour bench depends on it:
#
#   1. CORRECTNESS. Does wrapping in DataParallel change the features? It
#      should not -- the wrapper only splits a batch and gathers the outputs --
#      but GigaPath is a ViT-g with LayerScale, and this codebase has already
#      been bitten once by a wrapper that silently dropped ls1/ls2 and turned
#      the embedding into noise (see MILESTONE M3). Cosine against the
#      single-GPU features is the check.
#
#   2. SPEEDUP, and where it goes. gigapath_encode preprocesses on ONE thread:
#
#        batch = torch.stack([_TRANSFORM(_to_pil(img)) for img in ...]).to(device)
#
#      Four cards do not make that loop faster. The test times the transform
#      separately from the forward pass, so a speedup well under 4x can be
#      attributed rather than guessed at.
#
# The python is inline rather than a module under utilities/cli/ because it
# answers a question about the environment, not about LocaScope, and it stops
# being interesting the moment it is answered.

python - <<'PY'
import os, sys, time
import numpy as np
import torch

_ROOT = os.path.abspath('.')
for _d in ('utilities', 'aiNNModel'):
    p = os.path.join(_ROOT, _d)
    if p not in sys.path:
        sys.path.insert(0, p)

from GigaPathFunc import gigapath_model, make_gigapath_encoder, _TRANSFORM, _to_pil

N_PATCHES = 2048
BATCHES   = (512, 2048, 8192)
TILE      = 256

dev   = torch.device('cuda')
n_gpu = torch.cuda.device_count()
print(f'visible GPUs : {n_gpu}')
for i in range(n_gpu):
    pr = torch.cuda.get_device_properties(i)
    print(f'  [{i}] {pr.name}  {pr.total_memory / 2**30:.0f} GiB')
print(f'patches      : {N_PATCHES} x {TILE}x{TILE}x3 uint8')
print(f'batch sizes  : {BATCHES}\n', flush=True)

# Random pixels, not tissue: this measures throughput and numerics, and the
# content changes neither. Same array every run so the comparison is exact.
rng = np.random.default_rng(0)
patches = [rng.integers(0, 256, (TILE, TILE, 3), dtype=np.uint8)
           for _ in range(N_PATCHES)]

# ── how much of a batch is CPU-side preprocessing ────────────────────────────
t0 = time.perf_counter()
_ = torch.stack([_TRANSFORM(_to_pil(im)) for im in patches])
t_tf = time.perf_counter() - t0
print(f'[transform] {N_PATCHES} patches on one thread: {t_tf:.2f}s '
      f'({N_PATCHES / t_tf:.0f} patch/s)')
print('            this is the floor no number of GPUs can go below\n', flush=True)

results = {}
feats   = {}
for multi in (False, True):
    tag = f'multi_gpu={multi}'
    if multi and n_gpu < 2:
        print(f'[{tag}] skipped, only {n_gpu} GPU visible\n', flush=True)
        continue
    print(f'--- {tag} ---', flush=True)
    model = gigapath_model(dev, multi_gpu=multi)
    print(f'    wrapped: {type(model).__name__}', flush=True)
    for bs in BATCHES:
        enc = make_gigapath_encoder(model, dev, batch_size=bs,
                                    dtype=torch.float16)
        enc(patches[:min(bs, 64)])            # warm up cudnn / autotune
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f  = enc(patches)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        results[(multi, bs)] = dt
        feats.setdefault(multi, {})[bs] = f
        print(f'    bs={bs:5d}  {dt:6.2f}s  {N_PATCHES / dt:7.0f} patch/s'
              f'   (transform is {100 * t_tf / dt:.0f}% of it)', flush=True)
    del model
    torch.cuda.empty_cache()
    print(flush=True)

# ── correctness: does the wrapper change the embedding ───────────────────────
if True in feats:
    print('--- features, multi vs single (both fp16 autocast) ---')
    for bs in BATCHES:
        a, b = feats[False][bs], feats[True][bs]
        cos  = (a * b).sum(-1)                # rows are L2-normalised
        print(f'    bs={bs:5d}  cos min={cos.min():.6f}  mean={cos.mean():.6f}'
              f'  max_abs_diff={(a - b).abs().max():.2e}')
    print('    anything below about 0.9999 means DataParallel is NOT transparent')
    print()

    print('--- speedup ---')
    for bs in BATCHES:
        s = results[(False, bs)] / results[(True, bs)]
        print(f'    bs={bs:5d}  {s:.2f}x  on {n_gpu} GPUs')
    print(f'    a speedup far below {n_gpu}x with a large transform share is')
    print('    the single-threaded preprocessing, not the model')
PY

echo ""
echo "exit=$?"
