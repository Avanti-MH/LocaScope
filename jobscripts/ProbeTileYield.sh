#!/bin/bash
#SBATCH --job-name=ProbeTileYield         # -> log/%x, result/%x/
#SBATCH --partition=dev               # Partition
#SBATCH --time=02:00:00                   # mask arithmetic; the encoder is built once on the CPU
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0) -- unused here
#SBATCH --cpus-per-task=4                 # CPU cores per task
#SBATCH --mem=64G                         # one slide's mask at a time, plus the CPU-built encoder
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  utilities/cli/diagnostics/probe_tile_yield.py
#
#  FEATURE_MAP=only (default here)  how big the WHOLE-SLIDE feature map is at
#                                   every pyramid level: tiles and GB, with and
#                                   without the mask, with and without the
#                                   overlap grid. Seconds per slide, no pixels.
#  FEATURE_MAP=on                   that AND the sampler cells (minutes)
#  FEATURE_MAP=off                  the sampler cells only -- the tool's original
#                                   job: how many TRAINING tiles a sampler draws
#
#    sbatch jobscripts/ProbeTileYield.sh
#    ENCODER=gigapath TILE_SIZE=256 sbatch jobscripts/ProbeTileYield.sh
#    DATASET=bracs/test FP32=1 sbatch jobscripts/ProbeTileYield.sh   # narrows the slides
#
#  WHICH SLIDES: every slide that has a mask in result/cache/<MASK_CACHE_JOB>_mask/
#  <seg_id>/ -- by default the hest masks MppRoutingHead made -- so every row has
#  both the no-mask and the mask columns. DATASET="ki67_with_photo bracs/test"
#  (or WSI=path...) narrows to those instead, and a slide of them with no mask
#  then gets its no-mask columns only.
# =============================================================================

DATASET="${DATASET:-}"          # empty: every slide the mask cache holds
SEG="${SEG:-hest}"
MASK_CACHE_JOB="${MASK_CACHE_JOB:-MppRoutingHead}"
FEATURE_MAP="${FEATURE_MAP:-only}"
ENCODER="${ENCODER:-uni2}"
TILE_SIZE="${TILE_SIZE:-256}"        # what FeatureMapCache stores; the probe's own default also lists 512 and 1024

ARGS=(--seg "$SEG" --mask-cache-job "$MASK_CACHE_JOB"
      --feature-map "$FEATURE_MAP" --encoder "$ENCODER" --tile-size $TILE_SIZE)
[ -n "$DATASET" ] && ARGS+=(--dataset $DATASET)
[ -n "${WSI:-}" ] && ARGS+=(--wsi $WSI)
[ -n "${HEAD:-}" ] && ARGS+=(--head "$HEAD")
[ "${FP32:-0}" = "1" ] && ARGS+=(--fp32)
[ -n "${DS:-}" ] && ARGS+=(--ds $DS)

echo "======== ProbeTileYield ${ARGS[*]} ========"
python utilities/cli/diagnostics/probe_tile_yield.py "${ARGS[@]}"
status=$?

echo ""
echo "======== done (exit $status) ========"
echo "  feature map -> result/${SLURM_JOB_NAME:-ProbeTileYield}/feature_map_size.csv"
echo "  names       -> feature_map_size_definitions.csv beside it"
exit $status
