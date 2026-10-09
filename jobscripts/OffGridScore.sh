#!/bin/bash
#SBATCH --job-name=OffGridScore           # Job name -> log/<name>, result/<name>
#SBATCH --partition=8gpus                 # Partition
#SBATCH --time=24:00:00                   # step 4 is 1089 captures per point
#SBATCH --array=0-9                       # ONE SLIDE PER TASK: N_WSI per dataset
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # one card: encode + HEST segmentation
#SBATCH --cpus-per-task=8                 # openslide reads + the CPU transform
#SBATCH --mem=128G                        # per-tile reads, never a whole region
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x_%a          # STDOUT, one file per slide
#SBATCH -e /work/u26130998/log/%x_%a          # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# Runs write outside the checkout; see utilities/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# ---------------- What does landing off the grid cost? -----------------------
#
# Retrieval scores a query against a fixed lattice: a main grid at (256i, 256j)
# and an overlap grid half a tile off it. A real photograph lands wherever it
# lands. The pooling bench recorded that displacement but never measured what it
# costs, and "the truth was never proposed" is retrieval's largest failure
# bucket -- so how fast the score falls between grid points is a direct question
# about whether the lattice is fine enough.
#
# Per grid point the reference is encoded ONCE and never moves: two R x C
# windows, one anchored at (x, y) and one at (x+128, y+128). Then the QUERY
# moves -- Camera photographs a fresh FoV at every displacement in
# [0,128] x [0,128] and each is scored against those same two windows. Nothing
# slides; the score is production's own window score at a single position.
#
# The two curves MUST cross. At (0,0) the query sits on the main point; at
# (128,128) it has walked onto the overlap point and they trade places. Where
# they cross is the answer to "does the half-tile overlap grid earn its keep":
# a crossing near the middle with both scores high means the lattice covers the
# plane; both dipping hard before they cross means a query landing in that
# hollow is described badly by either grid.
#
# That crossing is what the gates check, because the geometry forces it. If it
# does not appear the grid pairing or the coordinates are wrong, and no score in
# the output means anything.
#
# COST, measured rather than assumed. The first run did 810 captures in about
# 760 s of scanning: 0.94 s each, and that IS the whole cost -- those 810
# captures needed 26k patch encodes, which is 21 s. Adding the greyscale pass
# doubles the encoding and changes the total by under 3%.
#
# At --step 4 the scan is 33 x 33 = 1089 displacements per grid point, so one
# slide at 10 points across 3 levels is 32,670 captures ~ 8.5 h. That is why
# this is an ARRAY job: seven slides in parallel, one per task, rather than 60 h
# in series. --time is 24 h to leave room for the MRXS slides, which have more
# usable levels than the BRACS ones.
#
# Whether step 4 resolves anything is an open question, not a settled one:
# CenterCrop(224) means the model never sees the outer 16 px of a tile, and the
# domain gap is redrawn on every capture, so two displacements 4 px apart may
# differ by less than the exposure noise between them. The aggregate over 10
# points is what would show it.
#
# Rotation is fixed at 0 on purpose. This measures displacement alone, and
# mixing rotation into it would leave one number answering two questions.
# (The rotation-recovery checks that used to cover SlidingWinSimRot's
# rotation path -- test_gigapath_slide_win_sim.py step 5 -- were dropped when
# that file was merged into test_locascope_stages.py, 2026-09-17; nothing
# currently covers rotation on its own.)
#
# Two axes ride along at almost no cost:
#
#   preprocess  none (RGB, what production encodes) and grey (luminance, both
#               sides). Each needs its own forward pass, but the FoV is captured
#               ONCE and encoded twice -- and capture dominates, so this is
#               roughly +30% wall clock, not +100%.
#
#   combiner    how the query's per-tile cosines become one window score. mean
#               is additive and is what production ships; geomean is
#               multiplicative, which is the AND of the same measurement; min is
#               the hardest AND. All three reduce the SAME tensor, so they are
#               columns of one row and cost nothing at all.

# The slides are the first N_WSI of each dataset's recorded TEST split (the
# file make_split.py wrote), in its file order -- the slides Stage1MppBench
# scores under SPLIT=test. Every one has a mask in the MppRoutingHead cache
# (BRACS test's first 5, all 6 of Ki67 test), so nothing is segmented. Task IDX
# takes dataset IDX / N_WSI and slide IDX % N_WSI, so --array must be
# 0 .. (datasets x N_WSI - 1).
DATASETS=(${DATASETS:-bracs/test#test ki67_with_photo#test})
N_WSI="${N_WSI:-5}"
MASK_CACHE_JOB="${MASK_CACHE_JOB:-MppRoutingHead}"

# One slide per array task. Each writes its own result directory, so nothing
# races on a shared CSV; merge afterwards with the awk line at the bottom.
IDX=${SLURM_ARRAY_TASK_ID:-0}
DS_IDX=$(( IDX / N_WSI ))
WSI_IDX=$(( IDX % N_WSI ))
if [ "$DS_IDX" -ge "${#DATASETS[@]}" ]; then
  echo "[skip] array index $IDX >= ${#DATASETS[@]} datasets x $N_WSI slides"
  exit 0
fi
DATASET="${DATASETS[$DS_IDX]}"
# The name, for the result directory; the bench resolves the path itself.
STEM=$(python -c "
import sys; sys.path.insert(0, 'utilities')
from _paths import setup_import_paths; setup_import_paths()
from AccessDatasets import list_names
print(list_names(dataset='$DATASET')[$WSI_IDX])")
# Resolved once: the bench puts this name in every row it stores, so the echo
# lines below must spell the same default.
#
# The tag goes ABOVE the slide, and the bench takes --out verbatim -- it adds
# the encoder level only to the directory it derives itself. Composing it here
# is what keeps the layout result/<job>/<encoder>/<slide>/ rather than burying
# the encoder one level down under each slide.
ENCODER="${ENCODER:-gigapath}"
HEAD="${HEAD:-}"
TAG="$ENCODER${HEAD:+_$HEAD}"
OUT_DIR="${OUT:-"$RESULT_ROOT"/${SLURM_JOB_NAME:-OffGridScore}}/$TAG/$STEM"

echo "======== [$IDX] $STEM  encoder=$TAG ========"
echo "wsi : $DATASET [$WSI_IDX] $STEM"
echo "out : $OUT_DIR"

python utilities/bench_modules/bench_offgrid_score.py "$STEM" \
  --dataset "$DATASET" --mask-cache-job "$MASK_CACHE_JOB" \
  --encoder "$ENCODER"${HEAD:+ --head "$HEAD"} \
  --step "${STEP:-16}" \
  --points "${POINTS:-5}" \
  --white-max 0.15 \
  --fov-sensor 1440 1024 \
  --domain-gap \
  ${LEVELS:+--levels $LEVELS} \
  --out "$OUT_DIR"

echo ""
echo "======== done ========"
echo "  result/${SLURM_JOB_NAME:-OffGridScore}/$TAG/<slide>/offgrid_scores.csv  one row per (slide, level,"
echo "                                               point, dx, dy, preprocess)"
echo "  result/${SLURM_JOB_NAME:-OffGridScore}/$TAG/<slide>/offgrid_gates.csv   geometry gates, AGGREGATED"
echo "                                               over points -- per point the"
echo "                                               endpoints are two separate"
echo "                                               exposures and their order is"
echo "                                               noise, not geometry"
echo "  result/${SLURM_JOB_NAME:-OffGridScore}/$TAG/<slide>/offgrid_definitions.csv  every name, and what it is"
echo ""
echo "  Read the gates FIRST. Both are forced by the geometry:"
echo "    main_decays    main(0,0) > main(128,128)          on the mean combiner"
echo "    overlap_rises  overlap(128,128) > overlap(0,0)"
echo ""
echo "  Then the figures, all covering every slide at once:"
echo "    offgrid_heatmap.png  2D maps, none|grey x main|overlap, plus ALL"
echo "    offgrid_surface.png  one 3D panel per preprocess, where the two"
echo "                         windows trade places"
echo "    offgrid_profile.png  along dx = dy, one panel per combiner:"
echo "                         mean (additive, production) vs geomean"
echo "                         (multiplicative = AND) vs min (hardest AND)"
echo ""
echo "  ONE set of figures covering every slide, once the array finishes."
echo "  --plot-only reads the CSVs and redraws; no GPU, no WSI, no model, so"
echo "  it runs on a login node in seconds and can be repeated freely:"
echo ""
echo "    python utilities/bench_modules/bench_offgrid_score.py --plot-only \\"
echo "        result/${SLURM_JOB_NAME:-OffGridScore}/$TAG/*/offgrid_scores.csv \\"
echo "        --out result/${SLURM_JOB_NAME:-OffGridScore}/$TAG/merged"
echo ""
echo "  That also writes the merged offgrid_gates.csv, aggregated across"
echo "  all seven slides rather than one at a time. Keep the glob to ONE encoder:"
echo "  the figures average over rows, so a mixed set is refused rather than"
echo "  drawn as a picture of neither model."
