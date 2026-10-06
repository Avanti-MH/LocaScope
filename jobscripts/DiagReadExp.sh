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
#    capture   training render_row, CAMERA_FULL (rows/s)
#    tiles     SlideReader.read_samples, native reference tiles (tiles/s)
#    fov       FovSupply.bank, the window bench's still FoV (shots/s)
#  Best of REPEATS. Run it before and after a change to the read path and
#  compare the two speed.csv files. Correctness is TestReadPath.sh's.
#
#    sbatch jobscripts/DiagReadExp.sh
#    FLOWS="grid" GRID_LEVELS="1" sbatch --job-name=DiagReadExpGridL1 jobscripts/DiagReadExp.sh
#
#  Writes nothing to any cache. Output: log/DiagReadExp/<job>,
#  result/DiagReadExp/<job>/speed.csv
#  (The log directory must exist before sbatch: mkdir -p log/DiagReadExp.)
# =============================================================================
SLIDES="${SLIDES:-BRACS_1228 S1104233,G7E,110208}"
FLOWS="${FLOWS:-grid capture tiles fov}"
N="${N:-6}"
GRID_LEVELS="${GRID_LEVELS:-0 1}"
REPEATS="${REPEATS:-2}"

JOB="${SLURM_JOB_NAME:-DiagReadExp}"
RES_DIR=/work/u26130998/result/DiagReadExp/$JOB
mkdir -p "$RES_DIR"

ARGS=(--slides $SLIDES --flows $FLOWS --n "$N" --grid-levels $GRID_LEVELS
      --repeats "$REPEATS" --out "$RES_DIR")
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
