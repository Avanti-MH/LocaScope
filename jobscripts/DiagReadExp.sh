#!/bin/bash
#SBATCH --job-name=DiagReadExp            # -> log/DiagReadExp/%x, result/DiagReadExp/%x/
#SBATCH --partition=dev                   # diagnostics go to dev
#SBATCH --time=01:00:00                   # two slides, each level-0 grid read REPEATS times
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0) -- nothing here encodes
#SBATCH --cpus-per-task=8                 # read_grid uses the CpuBudget's workers
#SBATCH --mem=96G                         # a region read whole at level 1
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/DiagReadExp/%x   # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/DiagReadExp/%x   # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  utilities/cli/diagnostics/diag_read_exp.py -- read-only
#
#  How fast the read path is, flow by flow, on the same slides every time:
#    grid      SlideReader.read_grid per level (tiles/s, CpuBudget workers)
#    capture   training render_row, routing-query (rows/s)
#    tiles     SlideReader.read_samples, native reference tiles (tiles/s)
#    fov       FovSupply, the window bench's still FoV (shots/s)
#    pca       the UNI2-PCA segmenter's reads against the retired WsiTileLoader
#              (tiles/s, and PASS only if every tile is identical); not in the
#              default FLOWS, since it is a comparison, not a timing
#    s1photo   bench_stage1_mpp's Render photo against the frozen sensor read
#              it replaced: PASS only if identical with no gap; the reflected
#              side bands of a 92 degree turn; photos/s for both. Not default
#    phase     how openslide samples a level (floor / round / bilinear), per
#              slide at PHASE_LEVELS, on MASK_CACHE_JOB's hest masks
#  Best of REPEATS. Run it before and after a change to the read path and
#  compare the two speed.csv files. Correctness is TestReadPath.sh's.
#
#    sbatch jobscripts/DiagReadExp.sh
#    FLOWS="grid" GRID_LEVELS="1" sbatch --job-name=DiagReadExpGridL1 jobscripts/DiagReadExp.sh
#    FLOWS="pca" sbatch --job-name=DiagReadExpPca jobscripts/DiagReadExp.sh
#    FLOWS="s1photo" sbatch --job-name=DiagReadExpS1Photo jobscripts/DiagReadExp.sh
#    FLOWS="phase" sbatch --job-name=DiagReadPhase jobscripts/DiagReadExp.sh
#
#  Writes nothing to any cache but MASK_CACHE_JOB's mask cache (phase segments
#  a slide it lacks). Output: log/DiagReadExp/<job>,
#  result/DiagReadExp/<job>/{speed,phase}.csv
#  (The log directory must exist before sbatch: mkdir -p log/DiagReadExp.)
# =============================================================================
SLIDES="${SLIDES:-BRACS_1228 S1104233,G7E,110208}"
FLOWS="${FLOWS:-grid capture tiles fov}"
N="${N:-6}"
GRID_LEVELS="${GRID_LEVELS:-0 1}"
REPEATS="${REPEATS:-2}"
PHASE_LEVELS="${PHASE_LEVELS:-1 2 3}"
MASK_CACHE_JOB="${MASK_CACHE_JOB:-MppRoutingHead}"

JOB="${SLURM_JOB_NAME:-DiagReadExp}"
RES_DIR=/work/u26130998/result/DiagReadExp/$JOB
mkdir -p "$RES_DIR"

ARGS=(--slides $SLIDES --flows $FLOWS --n "$N" --grid-levels $GRID_LEVELS
      --repeats "$REPEATS" --out "$RES_DIR"
      --phase-levels $PHASE_LEVELS --mask-cache-job "$MASK_CACHE_JOB")
[ -n "${BLOCK_ROWS:-}" ] && ARGS+=(--block-rows "$BLOCK_ROWS")

echo "======== DiagReadExp ========"
echo "  slides  $SLIDES"
echo "  flows   $FLOWS   n $N   grid levels $GRID_LEVELS   repeats $REPEATS"
echo "  cpus allowed $(nproc)"
echo "  out     $RES_DIR"

python utilities/cli/diagnostics/diag_read_exp.py "${ARGS[@]}"
rc=$?
echo ""
echo "======== done (exit $rc) ========"
exit $rc
