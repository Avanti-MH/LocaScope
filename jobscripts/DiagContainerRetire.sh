#!/bin/bash
#SBATCH --job-name=DiagContainerRetire     # -> log/%x, result/%x/
#SBATCH --partition=dev                    # diagnostics go to dev
#SBATCH --time=02:00:00                    # container builds + three encodes + a mask census
#SBATCH --account=MST114560                # Account
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gpus-per-node=1                  # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                  # read_grid workers
#SBATCH --mem=96G                          # the container holds every region at level 1
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH -o /work/u26130998/log/%x          # STDOUT
#SBATCH -e /work/u26130998/log/%x          # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  utilities/cli/diagnostics/diag_container_retire.py -- read-only
#
#    pixels   read_grid tiles against the container's, pixel for pixel; the
#             feature difference beside the encoder's own noise floor
#    crops    the on-demand stage-3 crop against the container's
#    phase    how openslide samples a level at a non-multiple level-0 point:
#             floor / round / bilinear, by a noise-free read-and-compare
#    origins  frac(region origin / ds) over cached masks, per dataset and level
#
#    sbatch jobscripts/DiagContainerRetire.sh
#    CHECKS="phase origins" sbatch --job-name=DiagReadPhase jobscripts/DiagContainerRetire.sh
#
#  Masks come from MppRoutingHead's cache (MASK_CACHE_JOB); the default slides
#  are in it, so nothing is segmented. origins only reads cached masks and
#  skips a slide that is not cached.
# =============================================================================

SLIDES="${SLIDES:-bracs/test:BRACS_1413:1 bracs/test:BRACS_1413:2 ki67_with_photo:S1130983,G7E,110816:1 ki67_with_photo:S1130983,G7E,110816:3}"
CHECKS="${CHECKS:-pixels crops phase origins}"
ENCODER="${ENCODER:-uni2}"
MASK_CACHE_JOB="${MASK_CACHE_JOB:-MppRoutingHead}"
ORIGIN_PER_DATASET="${ORIGIN_PER_DATASET:-8}"

ARGS=(--encoder "$ENCODER" --mask-cache-job "$MASK_CACHE_JOB" --workers 4
      --checks $CHECKS --origin-per-dataset "$ORIGIN_PER_DATASET")
for s in $SLIDES; do ARGS+=(--slide "$s"); done

echo "======== DiagContainerRetire  checks: ${CHECKS} ========"
python utilities/cli/diagnostics/diag_container_retire.py "${ARGS[@]}"
status=$?
echo "======== done (exit $status) ========"
exit $status
