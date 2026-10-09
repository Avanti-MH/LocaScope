#!/bin/bash
#SBATCH --job-name=Stage1MppBench             # -> log/%x, result/%x/
#SBATCH --partition=8gpus                     # Partition
#SBATCH --time=48:00:00                       # 13+ methods, one encoder resident at a time
#SBATCH --account=MST114560                   # Account
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gpus-per-node=1                     # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                     # openslide reads for the FoV crops
#SBATCH --mem=400G                             # host RSS peaks while an encoder + masks are resident
#SBATCH --ntasks-per-node=1                   # Tasks per node
#SBATCH -o /work/u26130998/log/%x             # STDOUT
#SBATCH -e /work/u26130998/log/%x             # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  utilities/bench_modules/bench_stage1_mpp.py, then its analysis
# =============================================================================
#
# Every STAGE1 method (<method>:<recipe>: KNN_RECIPES, CLASSIFIER_RECIPES,
# PROTOTYPE_RECIPES, CLASSIC_RECIPES) on the same FoVs -- the first N_WSI
# slides of each dataset's recorded SPLIT, the FOV recipe's draw -- one method
# resident at a time. Each (slide, method) is one stage-1 cache entry under
#   result/cache/<job>/slide=/seg=/region=/plan=/draw=/render=<gap>/stage1/
# so a rerun scores only what is missing, and a new recipe costs only itself.
# The reference banks are draw entries of the same job, their features beside
# them. The analysis takes the same flags, joins the ground truth and writes
# result/<job>/stage1_<split>.csv and its tables.
#
# FoV_Vote.md fixes every risk threshold on val before test is looked at:
#   SPLIT=val sbatch jobscripts/Benchmarks/Stage1MppBench.sh     fits them
#   sbatch jobscripts/Benchmarks/Stage1MppBench.sh               reads them back
# Smoke: LIMIT=5 N_WSI=1 (the first 5 FoVs of each slide; recorded in the
# entries, so a full run does not take them for its own).

DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
SPLIT="${SPLIT:-test}"
N_WSI="${N_WSI:-5}"            # per dataset; val and test must match
FOV="${FOV:-bench}"            # query_sim/FovSupply.FOV_RECIPES
PER_LEVEL="${PER_LEVEL:-}"     # replaces the recipe's n_per_rung when set
SEG="${SEG:-hest}"             # the mask the reference banks are drawn on
STAGE1="${STAGE1:-knn:gigapath knn:uni2 classic:default}"
LIMIT="${LIMIT:-0}"
MASK_CACHE_JOB="${MASK_CACHE_JOB:-MppRoutingHead}"
EXTRA="${EXTRA:-}"

ARGS=(--datasets $DATASETS --split "$SPLIT" --n-wsi "$N_WSI" --fov "$FOV"
      --seg "$SEG" --mask-cache-job "$MASK_CACHE_JOB" --limit "$LIMIT"
      --stage1 $STAGE1)
[ -n "$PER_LEVEL" ] && ARGS+=(--sampler-n-per-rung "$PER_LEVEL")
ARGS+=($EXTRA)

echo "======== [memory estimate] ========"
python utilities/cli/diagnostics/estimate_method_memory.py --stage1 $STAGE1

echo ""
echo "======== Stage1MppBench: $STAGE1 ========"
echo "  $DATASETS #$SPLIT  n_wsi=$N_WSI  fov=$FOV  seg=$SEG  limit=$LIMIT"
python -u utilities/bench_modules/bench_stage1_mpp.py "${ARGS[@]}"
status=$?

echo ""
echo "======== [analysis] ========"
if [ "$SPLIT" = "val" ]; then
  python utilities/cli/metrics/analyze_stage1_metrics.py "${ARGS[@]}" --fit-thresholds
else
  python utilities/cli/metrics/analyze_stage1_metrics.py "${ARGS[@]}" --thresholds auto
fi
rc=$?
[ $status -eq 0 ] && status=$rc

echo ""
echo "======== done (exit $status) -> result/${SLURM_JOB_NAME:-Stage1MppBench}/ ========"
exit $status
