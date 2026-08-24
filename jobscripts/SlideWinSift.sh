#!/bin/bash
#SBATCH --job-name=SlideWinSift           # Job name
#SBATCH --partition=normal                # Partition
#SBATCH --time=24:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0) -- unused here
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/SlideWinSift             # STDOUT
#SBATCH -e /work/u26130998/log/SlideWinSift             # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# Runs write outside the checkout; see utilities/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# ---------------- Brute-force SIFT over a whole WSI ----------------
# No encoder, no KNN, no mask, no pipeline. Slide the query image itself across
# every position of one pyramid level, score each window by SIFT+RANSAC inlier
# count, keep the k best windows. Two questions only: how fast, and does it find
# the right place.
#
# NO GPU IS USED. The line above is there because the scheduler rejects 0.
#
# Cost, measured from the tissue extent of S1103037 (94000 x 208316 @ level-0)
# with a 1440x1024 query and a half-frame stride:
#
#     L0  130 x 406 = 52780 windows    77.8 Gpx read
#     L1   65 x 203 = 13195            19.5
#     L2   32 x 101 =  3232             4.8
#     L3   16 x  50 =   800             1.2
#
# Every window pays a read AND a SIFT detect on 1.5 MP. At an assumed 300 ms per
# window a full L0 scan is over four hours FOR ONE PHOTO, so PROBE=1 exists to
# replace that assumption with a measurement before anything long is submitted.
#
# The probe samples windows spread ACROSS the slide rather than taking the first
# few hundred. The first few hundred are the blank top-left corner, where SIFT
# finds almost no keypoints and both detect and match run far faster than they
# ever will on tissue -- timing them would understate the real cost badly.

set -u
cd /work/u26130998/LocaScope

# ---------------- Knobs, shared by both runs ----------------
RUN_GOLDEN=0         # 1 = run the scored query_sim pass
RUN_REAL=1           # 1 = run the real-photo pass afterwards
PROBE=0              # 1 = time a sample of windows only; 0 = full scan
PROBE_WINDOWS=300    # how many windows the probe samples, spread over the slide
TOP=5                # how many best windows to keep and draw
MIN_INLIERS=10       # below this a window is not a hit at all
STRIDE=1             # take every Nth photo, to spread the sample

# One invocation. Both passes go through here so a flag added to the ARGS line
# below cannot end up applying to only one of them.
run_swsift () {
  local tag="$1" photo="$2" wsi="$3" gt="$4" level="$5" n="$6"
  local out="$RESULT_ROOT/SlideWinSift/$tag"
  mkdir -p "$out"

  local args="--level $level --top $TOP --min-inliers $MIN_INLIERS --out $out"
  args="$args --stride $STRIDE"
  [ "$n" -gt 0 ]     && args="$args --limit $n"
  [ -n "$gt" ]       && args="$args --gt-csv $gt --figures"
  [ "$PROBE" -eq 1 ] && args="$args --sample-windows $PROBE_WINDOWS"

  echo "================ $tag ================"
  echo "photo: $photo"
  echo "wsi  : $wsi"
  echo "out  : $out"
  echo "args : $args"
  echo
  python utilities/cli/slide_win_sift.py "$photo" "$wsi" $args
  echo "[$tag] exit=$?"
  echo
}

# ---- 1. GOLDEN: query_sim shots, position recorded --------------------------
# images/ is flat and holds all 7 slides x 3-4 levels = 2500 files. The CLI
# keeps only the ones whose gt row names THIS wsi at THIS level and says how
# many it dropped, so pointing it at the whole folder is correct rather than a
# shortcut. BRACS_1228 at L0 is 17731 windows -- about 71 min per photo at the
# 239 ms/window measured on 2026-08-06.
if [ "$RUN_GOLDEN" -eq 1 ]; then
  run_swsift golden \
    "$RESULT_ROOT"/MultiBatch1440/images \
    /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs \
    "$RESULT_ROOT"/MultiBatch1440/gt.csv \
    0 1
fi

# ---- 1b. GOLDEN, on a shot the full pipeline could not locate ---------------
# BRACS_1936_L0_syn00051. LocaScopePipeline put it 21898 px from the truth, SIFT
# refinement failed on 6 inliers, and none of the top-5 candidates verified --
# a complete miss, from bench_locascope's metrics.csv of 2026-08-06.
#
# Why this shot and not one of BRACS_1228's. That slide fails 30 of 100 at L0
# and ALL 30 sit in one narrow band, x 10786..26524 of a 107568-wide slide,
# while its 70 successes span the whole thing. Those failures are one tissue
# region misbehaving -- the retrieval size bias already in log/TODO.log -- so a
# brute-force scan of one of them would measure that region, not the method.
# BRACS_1936 fails only 5 of 100 and those 5 are scattered through the same area
# its successes cover, which makes one of them an ordinary hard shot rather than
# a symptom. syn00051 is additionally the only one of the five at rot=0, so a
# wrong answer cannot be blamed on rotation.
#
# Passing the FILE rather than the folder is what selects it: a single path
# skips the listing, and the gt filter still checks it belongs to this slide and
# level, so a typo fails loudly instead of silently scanning the wrong shot.
if [ "$RUN_GOLDEN" -eq 1 ]; then
  run_swsift golden_pipeline_miss \
    "$RESULT_ROOT"/MultiBatch1440/images/BRACS_1936_L0_syn00051.png \
    /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_FEA/BRACS_1936.svs \
    "$RESULT_ROOT"/MultiBatch1440/gt.csv \
    0 1
fi

# ---- 2. NO GOLDEN: real microscope photos, position unknown -----------------
# An empty gt argument is what puts the tool in figures-only mode, and that is
# all that is possible here: nothing records where on the slide these photos
# were taken, so there is no distance to report and the inlier count plus the
# blend/checkerboard panels are the only verdict.
#
# The level must still match the photos' mpp -- these are 0.2425 um/px, which is
# this slide's level 0, and nothing in the tool rescales the query. There is no
# cheap level to rehearse on for that reason.
#
# THIS ONE IS THE LONG POLE. The Ki67 MRXS is 52780 windows at L0 against
# BRACS_1228's 17731, so a full scan is about 3.5 h PER PHOTO. Keep the count at
# 1 until a figure has been looked at.
if [ "$RUN_REAL" -eq 1 ]; then
  run_swsift real \
    /work/u26130998/datasets/Ki67/S1103627_ki67/2.bmp \
    "/work/u26130998/datasets/Ki67/S1103627,G7E,110127.mrxs" \
    "" \
    0 1
fi
if [ "$RUN_REAL" -eq 1 ]; then
  run_swsift real \
    /work/u26130998/datasets/Ki67/S1103627_ki67/1.bmp \
    "/work/u26130998/datasets/Ki67/S1103627,G7E,110127.mrxs" \
    "" \
    0 1
fi
