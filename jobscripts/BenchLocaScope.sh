#!/bin/bash
#SBATCH --job-name=BenchLocaScope        # Job name -> log/<name> and result/cache/<name>/
#SBATCH --partition=8gpus                # Partition
#SBATCH --time=48:00:00                  # partition 8gpus caps at 2 days
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # one card per slide; stage 3 is CPU work
#SBATCH --cpus-per-task=12               # the H200 cap per GPU: stage 3 and the renders run on them
#SBATCH --mem=200G                       # the H200 cap per GPU
#SBATCH --array=0-19%8                   # one task per slide: 2 datasets x N_WSI 10 (8 at a time)
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH -o /work/u26130998/log/%x_%a        # STDOUT, one log per slide
#SBATCH -e /work/u26130998/log/%x_%a        # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
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
#   EXTRA="--stage2-k 20 --stage3-topk 5"
# ROUTE: stage1 (stage 1's level), oracle (the placed level) or both.
#
# ONE ARRAY TASK PER SLIDE (the datasets in order, N_WSI of each: task i is the
# i-th). Every task writes the same cache job (the job name), each its own
# slide's entries, so they never touch each other's files. A task whose index is
# past the last slide does nothing: with N_WSI=1 there are 2 slides, ask for
#   --array=0-1
# If the scheduler refuses 20 tasks (8 running + 12 pending is over the pending
# limit), submit twice:  --array=0-9  and later  --array=10-19.
#
# MODE=smoke is the same run made small, to measure what the full run costs: the
# same datasets, split, stages, routes, mask and levels, only 1 slide per dataset
# and 5 FoV per level (N_WSI, PER_LEVEL still win when set). Its log ends with the
# time of every step, the speed of every stage and of the whole pipeline, and how
# long encoding a whole level of a slide takes. Name it apart so it neither
# overwrites the full run's log nor shares its cache:
#
#   MODE=smoke sbatch -J BenchLocaScopeSmoke --array=0-1 --time=04:00:00 \
#       --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn006 jobscripts/BenchLocaScope.sh
MODE="${MODE:-full}"
if [ "$MODE" = "smoke" ]; then SMOKE_WSI=1; SMOKE_PER_LEVEL=5; else SMOKE_WSI=10; SMOKE_PER_LEVEL=50; fi

DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
SPLIT="${SPLIT:-val}"          # val: where methods are chosen; test: the choice confirmed
N_WSI="${N_WSI:-$SMOKE_WSI}"   # per dataset (10; 1 in smoke)
PER_LEVEL="${PER_LEVEL:-$SMOKE_PER_LEVEL}"   # FoVs per native level per slide (50; 5 in smoke)
SEG="${SEG:-hest}"             # the mask stages 1 and 2 search (MASK_RECIPES)
STAGE1="${STAGE1:-knn:gigapath}"
STAGE2="${STAGE2:-slidewin:gigapath}"
STAGE3="${STAGE3:-sift:default}"
ROUTE="${ROUTE:-both}"
LIMIT="${LIMIT:-0}"            # first N FoVs per slide (smoke); 0 = all
SAVE_PHOTOS="${SAVE_PHOTOS:-0}"  # 1 = keep every photo beside its record
EXTRA="${EXTRA:-}"
BATCH="${BATCH:-1024}"          # stage 2 encoder batch (not identity)

SLIDE_ARG=""
[ -n "$SLURM_ARRAY_TASK_ID" ] && SLIDE_ARG="--slide-index $SLURM_ARRAY_TASK_ID"

echo "======== mode=$MODE  $DATASETS #$SPLIT  n_wsi=$N_WSI  per_level=$PER_LEVEL  limit=$LIMIT ========"
echo "stages $STAGE1 | $STAGE2 | $STAGE3   route $ROUTE   seg $SEG   task ${SLURM_ARRAY_TASK_ID:-all}"

python utilities/bench_modules/bench_locascope.py \
  --datasets $DATASETS --split "$SPLIT" --n-wsi "$N_WSI" \
  --sampler-n-per-rung "$PER_LEVEL" \
  --stage1 "$STAGE1" --stage2 "$STAGE2" --stage3 "$STAGE3" --route "$ROUTE" \
  --seg "$SEG" --limit "$LIMIT" \
  $SLIDE_ARG --batch-size "$BATCH" \
  $([ "$SAVE_PHOTOS" = "1" ] && echo --save-photos) \
  $EXTRA || exit $?

echo ""
echo "======== done -> result/cache/${SLURM_JOB_NAME:-BenchLocaScope}/ ========"
