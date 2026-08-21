#!/bin/bash
#SBATCH --job-name=SlidewinPooling        # -> log/<name>, result/<name>/
#SBATCH --partition=normal                # Partition
#SBATCH --time=24:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # one card: encode + HEST segmentation
#SBATCH --cpus-per-task=8                 # openslide reads + the CPU transform
#SBATCH --mem=256G                        # a level-0 region is read whole
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/SlidewinPooling    # STDOUT
#SBATCH -e /work/u26130998/log/SlidewinPooling    # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# Runs write outside the checkout; see utilities/test_modules/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# ---------------- pooling x window score, through stage 2 --------------------
#
# Production keeps one of GigaPath's 197 tokens (the CLS) and turns the query's
# per-tile cosines into a window score with an arithmetic mean. Neither choice
# has been measured against an alternative AT THE WINDOW LEVEL, and retrieval's
# largest failure bucket is "the truth was never proposed" -- 32.3% of 1404
# shots. Five poolings x three scores = 15 arms; cls+mean IS production and is
# the baseline every other arm is paired against, query by query.
#
# SMOKE FIRST. The full run is 21 (slide, level) combinations and reads whole
# tissue regions; a level-0 BRACS region is 6.58 Gpx, which is 19.7 GB of uint8
# before the tile list copies it again -- that is what --mem 256G is for, and it
# is the cost this bench deliberately accepted rather than reading tile by tile
# the way bench_offgrid_score does.
#
# So MODE=smoke runs two slides and two levels, chosen to be opposite in the
# two ways that have historically mattered here:
#
#   BRACS_1228   H&E    4x pyramid   tissue 38.2%   1 region after merge
#   S1104233     Ki67   2x pyramid   tissue  3.5%   23 regions after merge
#
# The region count is the point. `rank_of` accumulates a candidate pool across
# regions and `sample_fovs` picks one to sit in; a single-region slide never
# exercises either. Levels 1 and 2 walk every code path in minutes -- level 0
# adds read time and memory, not logic, so it stays out of the smoke test.
#
# GATES RUN FIRST, on 32 tiles read straight from the middle of the first
# slide, before the model has been asked for anything expensive:
#
#   baseline is production   pool_tokens(...,'cls') == gigapath_encode
#   concat identity          cos(concat of normalised slots) == mean of the
#                            per-slot cosines, the identity that lets five
#                            multi-slot poolings run through an unmodified
#                            SlidingWindowSimilarity
#   grid geometry            d(nearest main) <= 181.02, d(closer of the two)
#                            <= 128.00 -- and the second bound caught its own
#                            first version, which said 90.51
#
# Then per (slide, level) the log prints `truth_pctile`, the truth window's
# mean percentile. A uniformly random window sits at 0.5000; if this is near
# 0.5 the coordinate mapping is broken and every arm is ranking noise, which
# would otherwise read as "no pooling helps".

MODE="${MODE:-smoke}"

# Resolved once. The bench puts this name in the CSV's filename AND in every
# row, so the echo lines at the bottom have to spell the same default the ARGS
# line does -- two places that must agree is one place too many.
ENCODER="${ENCODER:-gigapath}"

if [ "$MODE" = "smoke" ]; then
  BRACS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test
  KI67=/work/u26130998/datasets/Ki67
  SLIDES=(
    "$BRACS/Group_AT/Type_ADH/BRACS_1228.svs"
    "$KI67/S1104233,G7E,110208.mrxs"
  )
  ARGS="--slides ${SLIDES[*]} --levels 1 2 --n-fov ${N_FOV:-25}"
  ARGS="$ARGS --encoder $ENCODER"
  OUT="$RESULT_ROOT/SlidewinPooling/smoke"
else
  ARGS="--levels ${LEVELS:-0 1 2} --n-fov ${N_FOV:-100} --batch-size ${BATCH_SIZE:-2048}"
  ARGS="$ARGS --encoder $ENCODER"
  OUT="$RESULT_ROOT/SlidewinPooling"
fi

echo "======== mode=$MODE  encoder=$ENCODER ========"
echo "out : $OUT/slidewin_pooling_$ENCODER.csv"
echo ""

python utilities/test_modules/bench_slidewin_pooling.py \
  $ARGS \
  --out "$OUT"

echo ""
echo "======== done ========"
echo "  $OUT/slidewin_pooling_$ENCODER.csv"
echo ""
echo "  Read the gates FIRST -- a failure there means no number below is worth"
echo "  reading. Then truth_pctile per (slide, level): 0.5 = broken mapping."
echo ""
echo "  Tables, narrowest first:"
echo "    單片單層   absolute numbers, one pool each"
echo "    同層跨片   PRIMARY -- pools within a level differ ~6x, across ~250x"
echo "    單片跨層   only worth reading if H&E and Ki67 split"
echo "    全部       one row per arm, the conclusion"
echo ""
echo "  Every metric is derived from two stored integers per (query, arm), so"
echo "  re-tabulating costs no GPU:"
echo ""
echo "    python utilities/test_modules/bench_slidewin_pooling.py --report-only \\"
echo "        $OUT/slidewin_pooling_$ENCODER.csv"
echo ""
echo "  One encoder per report. Feeding two CSVs at once is refused: every"
echo "  table averages over rows, so the merge would print one comparison"
echo "  where there are two."
