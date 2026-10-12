#!/bin/bash
#SBATCH --job-name=AnalyzeLocaScope       # -> log/%x, result/%x/
#SBATCH --partition=dev                   # reads csv files, no model
#SBATCH --time=01:00:00
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0) -- nothing here uses it
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh

# =============================================================================
#  utilities/cli/metrics/analyze_locascope.py -- read-only
#
#  Where a bench_locascope run loses its FoVs: every FoV in one outcome (success, or the first
#  stage that failed it), each stage on its own population, stage 3 by inlier count, one row per
#  slide, and -- only when the oracle route is in the cache -- the FoVs stage 1 routed wrong run
#  again at the placed level. Tables (report.txt + csv) and figures (png) in result/<job>/.
#
#  The run flags are BenchLocaScope.sh's: the same ones give the same cache addresses, so keep
#  them equal to the bench's (a mismatch shows as "miss" / "stale" lines, never as wrong numbers).
#
#    sbatch jobscripts/AnalyzeLocaScope.sh
#    TOL_UM=50 sbatch jobscripts/AnalyzeLocaScope.sh
# =============================================================================
DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
SPLIT="${SPLIT:-val}"
N_WSI="${N_WSI:-10}"
PER_LEVEL="${PER_LEVEL:-50}"
SEG="${SEG:-hest}"
STAGE1="${STAGE1:-knn:gigapath}"
STAGE2="${STAGE2:-slidewin:gigapath}"
STAGE3="${STAGE3:-sift:default}"
LIMIT="${LIMIT:-0}"
STAGE_JOB="${STAGE_JOB:-BenchLocaScope}"   # whose cache is read
TOL_UM="${TOL_UM:-100}"                    # the final answer is a success within this many um of the truth
EXTRA="${EXTRA:-}"

python -u utilities/cli/metrics/analyze_locascope.py \
  --datasets $DATASETS --split "$SPLIT" --n-wsi "$N_WSI" \
  --sampler-n-per-rung "$PER_LEVEL" \
  --stage1 "$STAGE1" --stage2 "$STAGE2" --stage3 "$STAGE3" \
  --seg "$SEG" --limit "$LIMIT" --stage-cache-job "$STAGE_JOB" \
  --tol-um "$TOL_UM" $EXTRA
