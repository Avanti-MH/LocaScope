#!/bin/bash
#SBATCH --job-name=WindowRetrievalBench # -> log/<name>, result/<name>/
#SBATCH --partition=normal                # Partition
#SBATCH --time=24:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # one card: encode + HEST segmentation
#SBATCH --cpus-per-task=8                 # openslide reads + the CPU transform
#SBATCH --mem=600G                        # this partition's ceiling; the reference streams a tile row at a time
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x          # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x          # STDERR
#
# %x, so a second encoder's run does not overwrite the first one's log the way
# it would have overwritten its CSV. #SBATCH cannot read a shell variable --
# SLURM parses these before the script runs -- so the name comes from the
# command line, next to the encoder it belongs to:
#
#   ENCODER=conch_vit HEAD=trunk sbatch --job-name=ConchWindowRetrievalBench \
#       jobscripts/Benchmarks/WindowRetrievalBench.sh

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# Runs write outside the checkout; see utilities/_paths.py
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
ENCODER="${ENCODER:-uni2}"

# Which exit of the model. Empty for encoders with one, which is gigapath and
# uni2. CONCH has two and needs HEAD=trunk to run here AT ALL: its default is
# the attentional pooler, one 512-d vector with no token axis, so all five arms
# this bench compares are inadmissible and it stops before loading anything.
#
# trunk is the bare ViT at 28x28 of 768 -- the same shape the other two have,
# which is what makes the three comparable rather than merely all present.
#
# TAG is what the CSV is named after, and it carries the head because
# identity_id does: two heads are two vectors of two widths, and a table
# averaging both would read as one comparison. Empty head leaves the names
# gigapath and uni2 already wrote unchanged.
HEAD="${HEAD:-}"
TAG="$ENCODER${HEAD:+_$HEAD}"

# ---------------- what a run uses ---------------------------------------------
#
# The values are in bench_window_retrieval.py's CONFIG block, right after its
# imports: which datasets and how many slides, the FoV sampler (the richness
# buckets and their caps, overlap, how many FoVs per (slide, level)), the camera,
# the tissue mask, the encoder, the arms. Edit them there.
#
# Nothing below sets one of them unless you name it, so the CONFIG block is what
# runs. An environment variable, when set, becomes the matching flag and wins:
#
#   DATASETS  N_WSI  MAX_DS  SEED  N_FOV  RICHNESS  ROTATION  SCALE_MIN  SCALE_MAX
#   ARMS  SEG  MASK_CACHE_JOB  SPLIT_CACHE_JOB  BATCH_SIZE
#
# and EXTRA_ARGS takes ANY flag of the bench, which is how a single field is
# changed from sbatch -- every field of the sampler, camera and encoder configs
# has one (`--richness-caps`, `--camera-noise-sigma`, `--encoder-batch-size`, ...):
#
#   EXTRA_ARGS="--richness-caps 0.15 0.25 0.6 0 0 0 0 --camera-noise-sigma 0" \
#       sbatch jobscripts/Benchmarks/WindowRetrievalBench.sh
#
# DATASETS names pools the way AccessDatasets does: a real dataset (the whole
# pool) or a recorded split of one, `<id>#<split>`. Quote a value that holds a `#`:
# DATASETS="bracs/test#val ki67_with_photo#val".
#
# ROTATION / SCALE_MIN / SCALE_MAX: anything but 0 and 1 is scored against an
# UPRIGHT reference window, so recall falls for a reason unrelated to pooling; the
# entry exists to look at that fall.

# ---------------- which arms, and where the masks come from -----------------
#
# ARMS: which poolings to compare, each on its own lattices. No suffix is the
# full grid (main + the grid offset by half a tile); `-M` (capital) is main tiles
# only, e.g. ARMS="cls cls_avg-M rings3-M". A run whose arms are all `-M` never
# encodes an offset tile. `cls` on the full grid is the baseline. Empty: every
# pooling on the full grid.
#
# Nothing of a slide's tile features is kept: the reference is streamed one tile
# row at a time and scored as it goes, so memory does not grow with the slide.
#
# The masks are read from MASK_CACHE_JOB's cache; unset, THIS job's own.

# --encoder is always passed: it names the output directory (TAG) above, so the
# script and the bench have to agree on it.
COMMON="--encoder $ENCODER${HEAD:+ --head $HEAD}"
[ -n "${DATASETS:-}" ]  && COMMON="$COMMON --datasets $DATASETS"
[ -n "${SEED:-}" ]      && COMMON="$COMMON --seed $SEED"
[ -n "${ROTATION:-}" ]  && COMMON="$COMMON --rotation $ROTATION"
[ -n "${SCALE_MIN:-}" ] && COMMON="$COMMON --scale-min $SCALE_MIN"
[ -n "${SCALE_MAX:-}" ] && COMMON="$COMMON --scale-max $SCALE_MAX"
[ -n "${RICHNESS:-}" ]  && COMMON="$COMMON --richness $RICHNESS"
[ -n "${N_FOV:-}" ]     && COMMON="$COMMON --n-fov $N_FOV"
[ -n "${BATCH_SIZE:-}" ] && COMMON="$COMMON --batch-size $BATCH_SIZE"
# READ_WORKERS: reference-grid readers per shard (default: the shard's cpu share
# minus one -- CpuBudget). BLOCK_ROWS: tile rows per read (default 8).
[ -n "${READ_WORKERS:-}" ] && COMMON="$COMMON --read-workers $READ_WORKERS"
[ -n "${BLOCK_ROWS:-}" ] && COMMON="$COMMON --block-rows $BLOCK_ROWS"
[ "${GATES_ONLY:-0}" = "1" ] && COMMON="$COMMON --gates-only"   # the seconds-long checks, then stop
[ -n "${ARMS:-}" ] && COMMON="$COMMON --arms $ARMS"
[ -n "${SEG:-}" ] && COMMON="$COMMON --seg $SEG"
[ -n "${MASK_CACHE_JOB:-}" ] && COMMON="$COMMON --mask-cache-job $MASK_CACHE_JOB"
[ -n "${SPLIT_CACHE_JOB:-}" ] && COMMON="$COMMON --split-cache-job $SPLIT_CACHE_JOB"
[ -n "${EXTRA_ARGS:-}" ] && COMMON="$COMMON $EXTRA_ARGS"

if [ "$MODE" = "smoke" ]; then
  # One slide per dataset, the two finest levels the pyramid has, five FoVs:
  # walks every code path in minutes. It names its own numbers; anything else
  # still comes from CONFIG.
  ARGS="$COMMON --n-wsi 1 --max-ds 4"
  [ -z "${N_FOV:-}" ] && ARGS="$ARGS --n-fov 5"
  OUT="$RESULT_ROOT/WindowRetrievalBench/smoke"
else
  ARGS="$COMMON"
  [ -n "${N_WSI:-}" ] && ARGS="$ARGS --n-wsi $N_WSI"
  [ -n "${MAX_DS:-}" ] && ARGS="$ARGS --max-ds $MAX_DS"
  OUT="$RESULT_ROOT/WindowRetrievalBench"
fi

echo "======== mode=$MODE  encoder=$TAG ========"
echo "out : $OUT/$TAG/window_retrieval.csv"
echo ""

BENCH=utilities/bench_modules/bench_window_retrieval.py

# MODE=report re-tabulates the finished CSV and stops: no model, no slide read,
# no GPU. The log of this job gets the tables. Ask for a small job, since the
# #SBATCH lines above are for the full run:
#
#     MODE=report sbatch --job-name=WindowRetrievalReport --gpus-per-node=0 \
#         --cpus-per-task=2 --mem=32G --time=00:30:00 <this script>
#
# PER_SLIDE=1 adds the one-slide-across-levels tables.
if [ "$MODE" = "report" ]; then
  python $BENCH --report-only ${PER_SLIDE:+--per-slide} "$OUT/$TAG/window_retrieval.csv"
  exit $?
fi

# SHARDS=2 runs two processes, one per card, each taking every other slide. They
# write into the SAME parts directory (a slide belongs to one shard, so no file is
# written twice) and neither assembles the CSV: a third, model-free call does, once
# both are done. Ask for the cards on the command line -- the #SBATCH lines above
# are for one:
#
#     SHARDS=2 sbatch --gpus-per-node=2 --cpus-per-task=16 <this script>
#
# Each shard's stdout goes to its own file, <log>.shard<i>; this file gets the
# summary and the tables.
SHARDS="${SHARDS:-1}"
if [ "$SHARDS" -le 1 ]; then
  python $BENCH $ARGS --out "$OUT/$TAG"
  status=$?
else
  LOG="/work/u26130998/log/${SLURM_JOB_NAME:-WindowRetrievalBench}"
  CARDS=$(nvidia-smi -L | wc -l)
  if [ "$CARDS" -lt "$SHARDS" ]; then
    echo "SHARDS=$SHARDS but this job sees $CARDS card(s): ask for them with"
    echo "  sbatch --gpus-per-node=$SHARDS ..."
    exit 2
  fi
  pids=()
  for ((i = 0; i < SHARDS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python $BENCH $ARGS --out "$OUT/$TAG" \
        --shard "$i/$SHARDS" > "$LOG.shard$i" 2>&1 &
    pids+=($!)
    echo "shard $i/$SHARDS: pid $! on card $i, log $LOG.shard$i"
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  echo "all shards finished (exit $status); assembling"
  python $BENCH $ARGS --out "$OUT/$TAG" --assemble || status=1
fi

echo ""
echo "======== done (exit $status) ========"
echo "  $OUT/$TAG/window_retrieval.csv"
echo ""
echo "  Each finished (slide, level) is saved as its own file under"
echo "  $OUT/$TAG/parts-<id>/. If the job is killed, submit the SAME command"
echo "  again: finished ones are skipped. The CSV above is assembled from them."
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
echo "    python utilities/bench_modules/bench_window_retrieval.py --report-only \\"
echo "        $OUT/$TAG/window_retrieval.csv"
echo ""
echo "  One encoder per report. Feeding two CSVs at once is refused: every"
echo "  table averages over rows, so the merge would print one comparison"
echo "  where there are two."
exit $status
