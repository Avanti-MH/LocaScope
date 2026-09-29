#!/bin/bash
#SBATCH --job-name=WsiHealthCheck         # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (do not set 0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/WsiHealthCheck           # STDOUT
#SBATCH -e /work/u26130998/log/WsiHealthCheck           # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# Runs write outside the checkout; see utilities/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# =============================================================================
# One SLURM job, every WSI diagnostic, in funnel order -- routed through
# utilities/cli/diagnostics/wsi_health_check.py rather than calling
# scan_wsi_holes.py alone the way this jobscript (formerly WsiHoles.sh) used
# to. See that script's own docstring for why the order is
# metadata -> holes -> mask -> yield, with scale running alongside rather
# than in the funnel.
#
# STOP_AFTER controls depth (not forced to run everything):
#   STOP_AFTER=metadata   just the sanity check, seconds
#   STOP_AFTER=holes      + the exhaustive block scan below, the slow tier
#   STOP_AFTER=mask       + tissue-mask validity (GPU for a model SEG)
#   STOP_AFTER=yield      (default) + tile-yield, needs a mask STORE already
#                         built by utilities/cli/build_cache/build_mask_store.py
#   SKIP_SCALE=1          drop the parallel base_mpp/native-rung check
#
# GPU line above is only there because the partition wants one -- metadata,
# holes and scale are CPU/IO bound; only mask (with a model SEG) touches the
# GPU.
# =============================================================================

# ---------------- Parameters ----------------
# Dataset ids (utilities/AccessDatasets.py dataset_ids()), not raw filesystem
# roots -- this is what replaces the old script's own `ls *_mrxs/*.mrxs` glob
# and its hand-rolled cross-root duplicate-name guard: AccessDatasets already
# keeps ki67_pure and ki67_with_photo apart by dataset id, and
# WsiSelection.resolve_wsi_paths carries that id through as `dataset` on every
# resolved entry, so two datasets sharing a slide NAME no longer collide under
# one key the way scanning raw roots could.
#   sbatch --export=ALL,DATASET="ki67_pure" jobscripts/WsiHealthCheck.sh
DATASET="${DATASET:-ki67_pure ki67_with_photo}"
WSI="${WSI:-}"                  # explicit path(s), combined with DATASET if both set
VAL_ONLY="${VAL_ONLY:-0}"

STOP_AFTER="${STOP_AFTER:-yield}"
SKIP_SCALE="${SKIP_SCALE:-0}"
SEG="${SEG:-hest}"                              # mask recipe: the mask tier segments with it, the yield tier reads its cache
MASK_CACHE_JOB="${MASK_CACHE_JOB:-BuildMaskStore}"  # result/cache/<this>_mask/

LEVELS="${LEVELS:-0 1 2 3 4}"   # one column per level in the holes grid figure
# 1024, not the 128 this script used while it pointed at Ki67_with_photo's 19
# slides alone. Measured on 2026-09-16 at BLOCK=1024: 69 + 27 + 4 + 1 + 0 s for
# the five levels of S1103037, about 100 s/slide -- 145 slides is roughly 4 h.
# Step 1 below re-measures the per-read cost on the day; prefer that number.
#
#   BLOCK   10 slides (walltime table)   145 slides (scaled)   24 h walltime
#    1024     5 - 20 min                   1.2 - 4.8 h         fits (~4 h measured)
#     512    20 - 81 min                   4.8 -  20 h         tight
#     256    81 min - 5.4 h                 20 -  78 h         no
#     128    5.4 - 21.6 h                   78 - 313 h         no
# Halving BLOCK quadruples the reads; --sweep still derives 2x/4x/... for free
# by pooling, so starting coarse costs only the resolution below it.
BLOCK="${BLOCK:-1024}"
SWEEP="${SWEEP:-5}"
# 'holed' draws only slides with a broken block; 'all' draws every scanned
# slide -- at 145 slides that PNG hit matplotlib's raster limit outright.
FIGURE_SLIDES="${FIGURE_SLIDES:-holed}"

OUT="${OUT:-$RESULT_ROOT/WsiHealthCheck}"
mkdir -p "$OUT"

ARGS=(--dataset $DATASET --stop-after "$STOP_AFTER" --seg "$SEG" --mask-cache-job "$MASK_CACHE_JOB"
      --levels $LEVELS --block "$BLOCK" --sweep "$SWEEP"
      --figure-slides "$FIGURE_SLIDES" --out "$OUT")
[ -n "$WSI" ] && ARGS+=(--wsi $WSI)
[ "$VAL_ONLY" = "1" ] && ARGS+=(--val-only)
[ "$SKIP_SCALE" = "1" ] && ARGS+=(--skip-scale)

# ---------------- Step 1: how expensive is one read? (holes tier only) ------
# Turns the walltime table above into wall-clock time for THIS filesystem and
# THIS slide, so a bad BLOCK choice is caught in seconds rather than at the
# 24h walltime limit. Skipped when the run will not reach the holes tier --
# it answers a question STOP_AFTER=metadata never asks.
#
# Two things this has to get right, both learned the hard way:
#   * SPREAD the samples over the whole scanned rectangle -- a small corner
#     measures reads that land where no tile exists, which openslide answers
#     in microseconds without decoding anything.
#   * REOPEN after a failure -- a handle that has raised once is dead for
#     every later call, so a benchmark without this dies at the first hole.
run_holes=0
case "$STOP_AFTER" in holes|mask|yield) run_holes=1 ;; esac

if [ "$run_holes" = "1" ]; then
    FIRST=$(python - "$DATASET" <<'EOF'
import sys
sys.path.insert(0, 'utilities/cli/diagnostics')
sys.path.insert(0, 'utilities')
import _paths; _paths.setup_import_paths()
from WsiSelection import resolve_wsi_paths
datasets = sys.argv[1].split()
entries = resolve_wsi_paths(dataset=datasets)
print(entries[0]['path'] if entries else '')
EOF
)
    if [ -n "$FIRST" ]; then
        echo ""
        echo "======== read cost (calibrates BLOCK for the holes tier) ========"
        python - "$FIRST" <<'EOF'
import sys, time, openslide
import numpy as np

path = sys.argv[1]
w = openslide.OpenSlide(path)
p = w.properties
bx = int(p.get('openslide.bounds-x', 0))
by = int(p.get('openslide.bounds-y', 0))
bw = int(p.get('openslide.bounds-width',  w.dimensions[0]))
bh = int(p.get('openslide.bounds-height', w.dimensions[1]))
print('probing %s' % path.split('/')[-1])
print('  bounds %d x %d at (%d, %d)' % (bw, bh, bx, by))
print('%8s %10s %10s %12s %18s'
      % ('size', 'decoded', 'blank', 'ms/decoded', 'per 1M decoded'))

rng = np.random.default_rng(0)
for size in (64, 128, 256, 512, 1024, 4096):
    n = 60
    xs = rng.integers(bx, max(bx + 1, bx + bw - size), n)
    ys = rng.integers(by, max(by + 1, by + bh - size), n)
    t_dec = t_blk = 0.0
    n_dec = n_blk = n_bad = 0
    for x, y in zip(xs, ys):
        t0 = time.time()
        try:
            im = w.read_region((int(x), int(y)), 0, (size, size))
        except Exception:
            n_bad += 1
            w.close()
            w = openslide.OpenSlide(path)     # handle is dead, replace it
            continue
        dt = time.time() - t0
        # alpha 0 everywhere means openslide returned fill, decoding nothing
        if im.getextrema()[3][1] == 0:
            t_blk += dt; n_blk += 1
        else:
            t_dec += dt; n_dec += 1
    ms = (t_dec / n_dec * 1000) if n_dec else float('nan')
    print('%8d %10d %10d %12.2f %15.1f min'
          % (size, n_dec, n_blk, ms, ms * 1e6 / 1000 / 60))
    if n_bad:
        print('%8s %s' % ('', '(%d of %d samples hit a hole)' % (n_bad, n)))
w.close()
EOF
    fi
fi

# ---------------- Step 2: the funnel ----------------
echo ""
echo "======== wsi_health_check.py  dataset=[$DATASET] stop-after=$STOP_AFTER ========"
python utilities/cli/diagnostics/wsi_health_check.py "${ARGS[@]}"
rc=$?

echo ""
echo "======== done ========"
for f in "$OUT"/metadata/wsi_info.csv "$OUT"/holes/holes.csv \
         "$OUT"/holes/slides_with_holes.csv "$OUT"/holes/holes_grid*.png \
         "$OUT"/holes/holes_sweep.png "$OUT"/mask/*_coverage.png \
         "$OUT"/yield/tile_yield.csv "$OUT"/yield/tile_yield.png \
         "$OUT"/scale/base_mpp.csv; do
    [ -e "$f" ] && echo "  $f"
done
exit $rc
