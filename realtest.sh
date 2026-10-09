#!/bin/bash
#SBATCH --job-name=RealTest              # Job name -> log/<name>_<task> and result/cache/<name>/
#SBATCH --partition=8gpus                # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # GPUs per node (不要設0)
#SBATCH --cpus-per-task=12               # CPU cores per task (the H200 cap per GPU)
#SBATCH --mem=200G                       # host RAM (the H200 cap per GPU)
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH --array=0-15                     # One task per slide (16 slides of ki67_with_photo)
#SBATCH -o /work/u26130998/log/%x_%a             # STDOUT
#SBATCH -e /work/u26130998/log/%x_%a             # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# ---------------- Real photos through the three stages ----------------
#
# The mainstream run: every real microscope photo of ki67_with_photo (all 16
# slides, no val / test split), no ground truth. Each stage's tables for a
# slide are one cache entry, as the bench's are:
#   result/cache/<job>/slide=/seg=/region=/photos=<id>/
#       shots/  stage1/  stage1=<s1>/stage2/  .../stage2=<s2>/stage3/
# and `answer` in stage 3's entry is the position of each photo. See the
# docstring of utilities/cli/driver/locate_photo.py.
#
# ONE ARRAY TASK PER SLIDE. The pipeline segments the slide (or reads its mask
# cache) and the retriever encodes the whole level a photo is routed to; all of
# it is per slide, none per photo, so a task loops over its slide's photos.
# A slide whose entries all hit loads no model. A task killed mid-slide redoes
# that slide: there is no resume inside one.
#
# VARIABLES, by what they belong to (a value left empty is the recipe's own):
#
#   which photos
#     LIMIT              first N photos of each slide (a smoke run); 0 = all
#
#   which stages (<method>:<recipe>)
#     STAGE1  STAGE2  STAGE3      any other recipe field is a --stageN-<field>
#                                 flag in EXTRA, e.g. EXTRA="--stage2-min-sep-tiles 2"
#     SEG                         the mask stages 1 and 2 search (MASK_RECIPES)
#
#   stage 2 (retrieval)
#     STAGE2_K           how many candidates stage 2 KEEPS per photo (--stage2-k;
#                        the recipe's is 100). Part of stage 2's identity.
#     STAGE2_BATCH       the encoder's batch for the whole-level encode
#                        (--batch-size). Not identity: no cache entry changes.
#
#   stage 3 (localization)
#     STAGE3_TOPK        how many of stage 2's candidates stage 3 VERIFIES, the
#                        first N by stage 2's rank (--stage3-topk; the
#                        recipe's is 100). Part of stage 3's identity: another N
#                        is another stage-3 entry, and stages 1 and 2 are read
#                        back. It is at most STAGE2_K. There is no reranker in
#                        this run, so no other top-k exists.
#
#   anything else
#     EXTRA              flags passed to locate_photo.py as they are
#
# To run one slide only:  sbatch --array=5 realtest.sh
# A smoke run:            LIMIT=3 sbatch --array=4 realtest.sh
# Bad nodes:              sbatch --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn006 ...

# which photos
LIMIT="${LIMIT:-0}"

# which stages
STAGE1="${STAGE1:-knn:gigapath}"
STAGE2="${STAGE2:-slidewin:gigapath}"
STAGE3="${STAGE3:-sift:default}"
SEG="${SEG:-hest}"

# stage 2
STAGE2_K="${STAGE2_K:-}"
STAGE2_BATCH="${STAGE2_BATCH:-1024}"

# stage 3
STAGE3_TOPK="${STAGE3_TOPK:-100}"

EXTRA="${EXTRA:-}"

SLIDE_ARG=""
[ -n "$SLURM_ARRAY_TASK_ID" ] && SLIDE_ARG="--slide-index $SLURM_ARRAY_TASK_ID"
STAGE2_K_ARG=""
[ -n "$STAGE2_K" ] && STAGE2_K_ARG="--stage2-k $STAGE2_K"

echo "======== real photos  task=${SLURM_ARRAY_TASK_ID:-all}  limit=$LIMIT ========"
echo "stages $STAGE1 | $STAGE2 | $STAGE3   seg $SEG"
echo "stage 2: k=${STAGE2_K:-recipe}  batch=$STAGE2_BATCH   stage 3: topk=$STAGE3_TOPK"

python utilities/cli/driver/locate_photo.py \
  $SLIDE_ARG \
  --stage1 "$STAGE1" --stage2 "$STAGE2" --stage3 "$STAGE3" \
  --seg "$SEG" --limit "$LIMIT" \
  $STAGE2_K_ARG --batch-size "$STAGE2_BATCH" \
  --stage3-topk "$STAGE3_TOPK" \
  $EXTRA || exit $?

echo ""
echo "======== done -> result/cache/${SLURM_JOB_NAME:-LocatePhoto}/ ========"
echo "to plot it: REAL=1 STAGE3_TOPK=$STAGE3_TOPK STAGE2_K=${STAGE2_K} sbatch jobscripts/PlotLocaScope.sh"
