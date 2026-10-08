#!/bin/bash
#SBATCH --job-name=BenchLocaScope        # Job name -> log/<name> and result/cache/<name>/
#SBATCH --partition=normal2              # Partition
#SBATCH --time=48:00:00                  # partition normal caps at 2 days
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=4                # >1 so --multi-gpu has cards to use
#SBATCH --cpus-per-task=8                # photos render on the cpus while the GPU encodes
#SBATCH --mem=600G
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH -o /work/u26130998/log/%x          # STDOUT
#SBATCH -e /work/u26130998/log/%x          # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# ---------------- End-to-end bench: every stage's results into the cache ----
#
# Synthetic FoVs (FOV recipe, PER_LEVEL per native level, the first N_WSI
# slides of each dataset's recorded SPLIT) through stage 1, 2 and 3. Each
# stage's tables for a slide are one cache entry under
#   result/cache/<job>/slide=/seg=/region=/plan=/draw=/render=<gap>/
#       stage1/  stage1=<s1|oracle>/stage2/  .../stage2=<s2>/stage3/
# written when the slide is done; a rerun reads what is there and loads no
# model for a stage that hits. Nothing is scored here: PlotLocaScope.sh
# reads the same entries with the same flags.
#
# STAGE1/2/3 are <method>:<recipe> (KNN_RECIPES, SLIDEWIN_RECIPES,
# SIFT_RECIPES); any recipe field is a --stageN-<field> flag in EXTRA, e.g.
#   EXTRA="--stage2-k 20 --stage3-n-verify 5"
# ROUTE: stage1 (stage 1's level), oracle (the placed level) or both.

DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
SPLIT="${SPLIT:-test}"
N_WSI="${N_WSI:-5}"            # per dataset
PER_LEVEL="${PER_LEVEL:-50}"   # FoVs per native level per slide
SEG="${SEG:-hest}"             # the mask stages 1 and 2 search (MASK_RECIPES)
STAGE1="${STAGE1:-knn:gigapath}"
STAGE2="${STAGE2:-slidewin:gigapath}"
STAGE3="${STAGE3:-sift:default}"
ROUTE="${ROUTE:-both}"
LIMIT="${LIMIT:-0}"            # first N FoVs per slide (smoke); 0 = all
SAVE_PHOTOS="${SAVE_PHOTOS:-0}"  # 1 = keep every photo beside its record
FEATURES_CACHE_JOB="${FEATURES_CACHE_JOB:-BenchLocaScope}"
FEATURE_STORE_MODE="${FEATURE_STORE_MODE:-rw}"
EXTRA="${EXTRA:-}"

echo "======== $DATASETS #$SPLIT  n_wsi=$N_WSI  per_level=$PER_LEVEL  limit=$LIMIT ========"
echo "stages $STAGE1 | $STAGE2 | $STAGE3   route $ROUTE   seg $SEG"

python utilities/bench_modules/bench_locascope.py \
  --datasets $DATASETS --split "$SPLIT" --n-wsi "$N_WSI" \
  --sampler-n-per-rung "$PER_LEVEL" \
  --stage1 "$STAGE1" --stage2 "$STAGE2" --stage3 "$STAGE3" --route "$ROUTE" \
  --seg "$SEG" --limit "$LIMIT" \
  --multi-gpu --batch-size 8192 \
  $([ "$SAVE_PHOTOS" = "1" ] && echo --save-photos) \
  --features-cache-job "$FEATURES_CACHE_JOB" \
  --feature-store-mode "$FEATURE_STORE_MODE" \
  $EXTRA || exit $?

echo ""
echo "======== done -> result/cache/${SLURM_JOB_NAME:-BenchLocaScope}/ ========"
