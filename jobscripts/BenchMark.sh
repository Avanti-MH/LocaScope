#!/bin/bash
#SBATCH --job-name=BenchMarkV2            # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=08:00:00                   # per TASK; bracs is the long one (~2h)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                 # see PASS 3 -- was 2, deliberately raised
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH --array=0-3                       # one (slide, encoder) per task
#SBATCH -o /work/u26130998/log/%x_%a                      # STDOUT, one per task
#SBATCH -e /work/u26130998/log/%x_%a                      # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# ============================================================================
#  TEMP-MEASURE  --  where does a real-photo run actually spend its time?
#  DELETE THIS SCRIPT'S BODY WHEN THE QUESTION IS ANSWERED. See
#  utilities/_tempmeasure.py, which carries the same instruction and whose
#  removal is `sed -i '/TEMP-MEASURE/d'` over the tagged files.
# ============================================================================
#
# The question, from the ten RealTest_uni2_* runs:
#
#     summed t_total_s over ten slides     616 min
#     the jobs' own wall clock            1278 min
#
# About half the run is outside the only timer there is, and t_total_s closes
# BEFORE the figures are drawn while realtest.sh runs --figures all.
#
#     build (one per slide)          56 min     4.4%
#     inside t_total_s              616 min    48%
#     measured by nothing           662 min    52%
#
# Two hypotheses, and this script separates them rather than confirming either:
#
#   H1  matplotlib.  The unmeasured gap is 18-25 s per photo and does NOT scale
#       with slide size, while the measured part varies 3-45 s and does.
#       Constant-per-photo is what figure drawing looks like.
#
#   H2  the sliding-window einsum.  TileEncoderFunc._run hands features back
#       with .cpu() and nothing in the retrieval path moves them, so
#       _sim_tensors runs on the host -- with 2 CPUs allocated. That would be
#       linear in WSI tile count, which is what the 3-45 s spread looks like.
#
# ---------------------------------------------------------------------------
#  THREE AXES, one variable at a time
# ---------------------------------------------------------------------------
#
#   PASS      within one job: figures on/off, then threads 2/8
#   MODE      which slide, and therefore how many WSI tiles   -> H2
#   ENCODER   which model                                     -> is any of this
#                                                                encoder-shaped?
#
# PASS is the inner loop because both of its steps are free. MODE and ENCODER
# are separate submissions, because each rebuilds the whole feature map.
#
#   PASS 1   figures all,  2 threads    the historical configuration
#   PASS 2   figures none, 2 threads    PASS 1 minus matplotlib     -> H1
#   PASS 3   figures none, 8 threads    PASS 2 minus the thread cap -> H2
#
# Each pass is its own process, so each prints its own TEMP-MEASURE table at
# exit. Read the tables against each other, never one in isolation.
#
#   MODE=ki67    S1104233, 71 REAL photos, MRXS.  The smallest corpus of the
#                ten AND the largest unmeasured fraction: 4.8 min of t_total
#                against 26.8 min of wall clock, 5.6x. Strongest signal,
#                cheapest run.
#
#   MODE=bracs   BRACS_1228, 50 SYNTHETIC shots at L0, SVS. There are no real
#                photographs for the BRACS slides -- the corpus is query_sim's
#                MultiBatch1440, which is 1440x1024 like the real ones, so the
#                per-photo work is the same shape.
#
#                This is the H2 lever, not merely another slide: BRACS_1228's
#                L0 grid is 162,159 tiles (RefStore pre-flight), while
#                S1104233 is the fastest build of the ten at 65 s and so is far
#                smaller. If sim.einsum is linear in tile count, the two modes
#                differ by that ratio and nothing else does.
#
#                ONE level on purpose. The corpus holds L0/L1/L2 for this
#                slide, and mixing them would build three retrievers -- three
#                whole-slide encodes -- and put three different tile counts in
#                one average, which is the reading this exists to prevent.
#
#                Pyramid note: BRACS SVS steps 4x per level, Ki67 MRXS steps 2x
#                (CLAUDE.md). Another reason not to compare across levels here.
#
#   ENCODER      uni2 by default, which is what the ten RealTest runs used.
#                gigapath is 1536-d like uni2, so the einsum's D is unchanged
#                and a difference between them is NOT about the similarity;
#                conch_vit trunk is 768-d and halves it, which makes it the
#                interesting third point if H2 survives.
#
# ---------------------------------------------------------------------------
#  HOW TO SUBMIT
# ---------------------------------------------------------------------------
#     sbatch jobscripts/BenchMark.sh          # all four combinations at once
#
# An ARRAY and not a loop inside one job, because the four are independent and
# the serial cost is real: locate_photo never passes feature_store_root, so
# every process rebuilds the whole slide's feature map. Three passes times four
# combinations is twelve builds -- about 5.4 h in series against roughly 2 h
# for the longest task in parallel. The four logs also stay separate, which
# matters when each ends with a table meant to be read against the others.
#
#     task 0   ki67  + uni2         ~40 min
#     task 1   ki67  + gigapath     ~40 min
#     task 2   bracs + uni2         ~2 h
#     task 3   bracs + gigapath     ~2 h
#
# One combination on its own, or one outside the table:
#     sbatch --array=2 jobscripts/BenchMark.sh
#     MODE=bracs ENCODER=conch_vit HEAD=trunk sbatch --array=0 jobscripts/BenchMark.sh
#
# An explicit MODE / ENCODER / HEAD in the environment WINS over the table, so
# the second form works whatever index it lands on. SLURM exports the
# submitting environment by default (--export=ALL). Every combination writes
# its own directory, so nothing races.

RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"
DATA=/work/u26130998/datasets/Ki67
BRACS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test
CORPUS="$RESULT_ROOT/MultiBatch1440"

# One row per array task: MODE ENCODER [HEAD]. A missing third field leaves
# HEAD empty, which is what gigapath and uni2 want -- they have one exit.
COMBOS=(
  "ki67  uni2"
  "ki67  gigapath"
  "bracs uni2"
  "bracs gigapath"
)
IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#COMBOS[@]}" ]; then
  echo "[skip] array index $IDX >= ${#COMBOS[@]} combinations"
  exit 0
fi
read -r _MODE _ENCODER _HEAD <<< "${COMBOS[$IDX]}"

# The environment wins over the table, so a one-off combination can be run at
# any index without editing the table for it.
MODE="${MODE:-$_MODE}"
ENCODER="${ENCODER:-$_ENCODER}"
HEAD="${HEAD:-$_HEAD}"
TAG="$ENCODER${HEAD:+_$HEAD}"
ENC_FLAG="--encoder $ENCODER${HEAD:+ --head $HEAD}"

OUT="$RESULT_ROOT/BenchMarkV2/$MODE/$TAG"

case "$MODE" in
  ki67)
    STEM=S1104233
    WSI=$(ls "$DATA/$STEM",*.mrxs 2>/dev/null | head -1)
    PHOTO_DIR="$DATA/${STEM}_ki67"
    KIND="real photographs"
    ;;
  bracs)
    STEM=BRACS_1228
    WSI="$BRACS/Group_AT/Type_ADH/$STEM.svs"
    # A symlink farm, because collect_photos reads EVERY image in the folder it
    # is given and MultiBatch1440/images holds all seven slides at three
    # levels. Links and not copies: 50 shots at ~1.9 MB is 95 MB that already
    # exists on this filesystem.
    PHOTO_DIR="$OUT/photos_L0"
    mkdir -p "$PHOTO_DIR"
    find "$PHOTO_DIR" -maxdepth 1 -type l -delete     # only links, never files
    if ! ls "$CORPUS"/images/"$STEM"_L0_syn*.png >/dev/null 2>&1; then
      echo "[abort] no L0 shots for $STEM under $CORPUS/images"
      echo "        regenerate with query_sim/cli/multi_batch.py"
      exit 1
    fi
    ln -s "$CORPUS"/images/"$STEM"_L0_syn*.png "$PHOTO_DIR"/
    KIND="synthetic shots (query_sim MultiBatch1440, L0 only)"
    ;;
  *)
    echo "[abort] MODE must be ki67 or bracs, got '$MODE'"
    exit 1
    ;;
esac

if [ -z "$WSI" ] || [ ! -f "$WSI" ]; then
  echo "[abort] no slide for MODE=$MODE: '$WSI'"
  exit 1
fi

echo "task    : $IDX of ${#COMBOS[@]}"
echo "mode    : $MODE   ($KIND)"
echo "wsi     : $WSI"
echo "photos  : $(ls "$PHOTO_DIR" | wc -l)"
echo "encoder : $TAG"
echo "out     : $OUT"
echo ""

run_pass () {          # $1 = label   $2 = --figures value   $3 = OMP threads
  echo ""
  echo "############################################################"
  echo "# PASS $1   mode=$MODE  encoder=$TAG  figures=$2  threads=$3"
  echo "############################################################"
  # OMP_NUM_THREADS and not --cpus-per-task: the allocation is fixed for the
  # whole job, and this has to vary WITHIN it or the two numbers would come
  # from two queue slots on two machines.
  OMP_NUM_THREADS=$3 MKL_NUM_THREADS=$3 \
  python utilities/cli/locate_photo.py \
    "$PHOTO_DIR" \
    "$WSI" \
    --out "$OUT/pass$1" \
    --figures "$2" \
    --no-resume \
    $ENC_FLAG \
    --precision fp16 --batch-size 1024
}

# --no-resume on every pass: resume skips on photo name alone and all three
# passes photograph the same files, so without it passes 2 and 3 would print
# "nothing to do" and exit 0.
run_pass 1 all  2
run_pass 2 none 2
run_pass 3 none 8

echo ""
echo "======== how to read the three tables ========"
echo "  PASS 1 - PASS 2  = what matplotlib costs        -> H1"
echo "  PASS 2 - PASS 3  = what the 2-thread cap costs  -> H2"
echo "  this MODE vs the other = what WSI tile count costs -> H2"
echo ""
echo "  Inside one table the buckets NEST, so they sum past 100%:"
echo "    photo.run     contains every run.*"
echo "    run.sim       contains sim.einsum + sim.window_slice"
echo "    wsi.container contains wsi.read_region + wsi.from_pil + wsi.extract_all"
echo "  Read a parent against its own children only."
echo ""
echo "  sim.einsum carries the device it ran on -- sim.einsum[cpu] against"
echo "  sim.einsum[cuda] -- so the table says where the features live rather"
echo "  than leaving it to be argued about."
echo ""
echo "  encode[...] and patch.list[...] are labelled by container class, so the"
echo "  WSI side and the query side land in separate buckets from one call site."
echo ""
echo "  build.retriever fires ONCE per level. In MODE=bracs that is one level"
echo "  by construction; in MODE=ki67 every real photo routed to L0, so it is"
echo "  one there too. A second build.retriever call means a photo routed"
echo "  somewhere else and the per-photo averages below it are mixing scales."


# ---------------- the encoder throughput sweep this file used to drive -------
# Kept because it is a different question -- batch size and dtype against
# tiles/s, with no WSI and no pipeline -- and its parameters took a while to
# settle. Restore this body when the timing question above is answered and the
# TEMP-MEASURE instrument is removed.
#
# WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs
# WARMUP=2
# COMPARE_PATCHES=40960
# COMPARE_BS="8 16 64 128 512 1024 4096"
#
# python utilities/bench_modules/bench_gigapath_infer.py \
#   --compare \
#   --no-wsi \
#   --compare-patches  $COMPARE_PATCHES \
#   --compare-bs       $COMPARE_BS \
#   --warmup $WARMUP
