#!/bin/bash
#SBATCH --job-name=TestLocaScopeStages    # Job name
#SBATCH --partition=dev               # Partition
#SBATCH --time=02:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                 # read_workers = CpuBudget workers (7), reading alongside the encoder
#SBATCH --mem=600G                        # see the FILTER_SWEEP note below
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x      # STDOUT
#SBATCH -e /work/u26130998/log/%x      # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# Replaces THREE jobscripts, one per test file that merged into
# utilities/test_modules/test_locascope_stages.py: KnnEstiMppTestBench.sh
# (test_gigapath_knn_esti_mpp.py), SlideWinTest.sh (test_gigapath_slide_win_
# sim.py) and SiftRansacTest.sh (test_sift_ransac.py). One jobscript per
# merged file, same as the python side -- not three jobscripts each
# pointing at a file that no longer exists.
#
# Two modes:
#   default            one run through --stages (1,2,3 unless overridden)
#   FILTER_SWEEP=1     SlideWinTest.sh's own 4-way (overlap x filter) sweep
#                      at --stages 1,2 -- see the memory note below for why
#                      this mode is what --mem=600G is actually sized for

# ---------------- Memory (FILTER_SWEEP mode) ----------------
#
# This wants a lot, and the reason is not the tiles.
#
#   job 301949   no --mem   ReqMem 200G   MaxRSS 199.2 GB   2 rounds OOM
#   job 301972   --mem=128G ReqMem 128G   MaxRSS 127.7 GB   4 rounds OOM
#
# The second is the cautionary one. This script carried no --mem and the site
# default turned out to be 200G, so writing 128G to "give it memory" LOWERED
# the ceiling and killed the two --no-filter rounds that had been passing.
# Read the default before overriding it; sacct prints it as ReqMem.
#
# Measured on BRACS_1228, the slide this used to hard-code, at mpp 0.252
# against its own 0.2524, so every round built at LEVEL 0 (RUNG=1 does the
# same on whichever slide is picked). Tile pixels are 9.5 GB with the filter
# and 10.9 GB without -- nearly equal, so the tiles are not what differs. The regions are: the filter leaves
# 5 merged ones, no-filter leaves 2988, each reading its own bounding box, and
# those boxes overlap heavily, so the total read is a multiple of the tissue
# area rather than equal to it.
#
# 600G because the nodes hold 1.9 TB and DiagMultiGPU has completed at that
# size. It is not measured -- 200G was not enough and the true ceiling is
# unknown, so this is headroom, not a number anyone derived. If a bigger slide
# ever OOMs here, the fix is to stop holding every region's pixels at once, not
# to raise this again.
#
# The default (non-sweep) mode never approaches this: one filtered mask, one
# query. --mem stays at 600G anyway rather than being conditional on the mode,
# since a job that OOMs in default mode is cheap to notice and a job that OOMs
# 40 minutes into a sweep is not.

# ---------------- Parameters ----------------
# A val slide (every one has a mask in the MppRoutingHead cache) and one FoV
# drawn on it at RUNG -- nothing is segmented. PICK_SEED picks both.
DATASET="${DATASET:-bracs/test#val}"   # or ki67_with_photo#val
RUNG="${RUNG:-1}"
PICK_SEED="${PICK_SEED:-0}"
SENSOR="1440 1024"
TILE=256
SAMPLES="${SAMPLES:-100}"
K="${K:-11}"
BATCH="${BATCH:-4096}"
MIN_REGION_RATIO=0.10
PADDING="${PADDING:-2}"
MIN_INLIERS="${MIN_INLIERS:-10}"
STAGES="${STAGES:-1,2,3}"
FILTER_SWEEP="${FILTER_SWEEP:-0}"
# CHECK_SIMS=1: stage 2's maps against the frozen pre-2026-10-06 path and
# against a feature-cache read, bit for bit (check_sims). Both modes take it;
# with FILTER_SWEEP=1 it covers no-filter (many regions) and no-overlap too.
CHECK_SIMS="${CHECK_SIMS:-0}"
# PRECISION: the stage 2 encoder, fp16 (production) or fp32.
PRECISION="${PRECISION:-fp16}"

# The recorded val/test split the slides are taken from. Written once, by its
# one writer, under MakeSplit; an existing split is kept as it is.
python utilities/cli/build_cache/make_split.py --cache-job MakeSplit || exit $?

BASE_ARGS="
  --dataset $DATASET --rung $RUNG --pick-seed $PICK_SEED
  --sensor $SENSOR
  --tile $TILE --samples $SAMPLES --k $K --batch $BATCH --precision $PRECISION
  --min-region-ratio $MIN_REGION_RATIO
  --padding $PADDING --min-inliers $MIN_INLIERS
"
[ "$CHECK_SIMS" = "1" ] && BASE_ARGS="$BASE_ARGS --check-sims"

if [ "$FILTER_SWEEP" -eq 1 ]; then
  echo "======== [1/4] overlap + filter ========"
  python utilities/test_modules/test_locascope_stages.py \
    --stages 1,2 $BASE_ARGS --overlap --filter

  echo "======== [2/4] overlap + no-filter ========"
  python utilities/test_modules/test_locascope_stages.py \
    --stages 1,2 $BASE_ARGS --overlap --no-filter

  echo "======== [3/4] no-overlap + filter ========"
  python utilities/test_modules/test_locascope_stages.py \
    --stages 1,2 $BASE_ARGS --no-overlap --filter

  echo "======== [4/4] no-overlap + no-filter ========"
  python utilities/test_modules/test_locascope_stages.py \
    --stages 1,2 $BASE_ARGS --no-overlap --no-filter
else
  python utilities/test_modules/test_locascope_stages.py \
    --stages "$STAGES" $BASE_ARGS --overlap --filter
fi
