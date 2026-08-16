#!/bin/bash
#SBATCH --job-name=WsiHoles               # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (do not set 0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/WsiHoles                 # STDOUT
#SBATCH -e /work/u26130998/log/WsiHoles                 # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath


# Runs write outside the checkout; see utilities/test_modules/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# =============================================================================
# Where does read_region fail, for every Ki67 slide and every pyramid level?
#
# No GPU is used; the GPU line above is only there because the partition wants
# one. The work is openslide decoding JPEG: CPU and IO bound, single threaded,
# so extra cores do not help.
#
# ONE PASS ONLY. It is tempting to run a coarse scan and then a fine one, but
# the coarse answer is DERIVED from the fine scan for free: a coarse block is
# readable exactly when every fine block inside it is, so --sweep pools the map
# instead of re-reading. Scan at the finest block you can afford; every coarser
# size comes out of the same data.
#
# COST. Halving BLOCK quadruples the reads. For bounds 67328 x 189440 (other
# slides are within about 20 percent of this), 10 slides x 5 levels:
#
#   BLOCK   cell    blocks/level      total reads     @0.5ms      @2ms
#    1024   0.25mm        12,210          610,500       5 min    20 min
#     512   0.12mm        48,840        2,442,000      20 min    81 min
#     256    62um        194,620        9,731,000      81 min     5.4 h
#     128    31um        778,480       38,924,000       5.4 h    21.6 h
#      64    16um      3,113,920      155,696,000      21.6 h      86 h   <- no
#
# Step 1 measures the real per-read cost, so pick BLOCK from that rather than
# from the guesses above. A 64 px read is not 16x cheaper than a 256 px one:
# MIRAX tiles are a few hundred px, so a small read still decodes a whole tile,
# which is why the cheap-looking small blocks are the expensive column.
#
# For hole SHAPE at fine resolution, do not shrink BLOCK over the whole slide --
# scan one region instead. That needs a --rect option the tool does not have yet.
# =============================================================================

# ---------------- Parameters ----------------
DATA=/work/u26130998/datasets/Ki67

LEVELS="0,1,2,3,4"   # one column per level in the grid figure
BLOCK=128            # level-0 px; the finest size actually scanned
SWEEP=5              # report BLOCK, 2x, 4x, ... this many doublings
OUT="$RESULT_ROOT"/WsiHoles

mkdir -p "$OUT"

# Every .mrxs in the dataset, one figure row each. An array keeps the commas in
# the filenames from being resplit.
mapfile -t SLIDES < <(ls -1 "$DATA"/*.mrxs)
echo "found ${#SLIDES[@]} slides in $DATA"
printf '  %s\n' "${SLIDES[@]}"

# ---------------- Step 1: how expensive is one read? ----------------
# Turns the table above into wall-clock time for THIS filesystem. If the numbers
# say the chosen BLOCK will not finish, stop and raise it rather than finding
# out at the walltime limit.
#
# Two things this has to get right, both learned the hard way:
#
#   * SPREAD the samples over the whole scanned rectangle. Walking a small
#     corner measures reads that land where no tile exists, which openslide
#     answers in microseconds without decoding anything -- 0.02 ms per read is
#     the signature of measuring empty space, not of a fast filesystem.
#
#   * REOPEN after a failure. A handle that has raised once is dead for every
#     later call, so a benchmark without this dies at the first hole, and any
#     timing after that point would be meaningless anyway.
#
# Blank and decoded reads are timed separately because they differ by orders of
# magnitude, and only the decoded number predicts the scan.
echo ""
echo "======== read cost ========"
FIRST="${SLIDES[0]}"
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

# ---------------- Step 2: the scan ----------------
echo ""
echo "======== scan: block=$BLOCK level-0 px, levels=$LEVELS ========"
python utilities/cli/scan_wsi_holes.py \
  "${SLIDES[@]}" \
  --levels "$LEVELS" \
  --block "$BLOCK" \
  --sweep "$SWEEP" \
  --out "$OUT"

echo ""
echo "======== done ========"
echo "  $OUT/holes_grid.png    ${#SLIDES[@]} rows (slides) x levels"
echo "  $OUT/holes_sweep.png   damage vs block size, derived by pooling"
echo "  $OUT/holes.csv         every broken block, slide + level + level-0 xy"
