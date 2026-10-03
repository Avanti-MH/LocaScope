#!/bin/bash
#SBATCH --job-name=StoreTest              # -> log/%x, result/%x/
#SBATCH --partition=dev               # Partition
#SBATCH --time=01:00:00                   # unit tests are seconds; WITH_MODEL reads real tiles
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0) -- WITH_MODEL uses it
#SBATCH --cpus-per-task=2                 # CPU cores per task
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
#  utilities/test_modules/test_store.py -- utilities/Store.py: addresses,
#  validation, the feature-map cache, pre-tile rungs. No slide, no model.
#
#    sbatch jobscripts/StoreTest.sh                    the unit tests
#    WITH_MODEL=1 sbatch jobscripts/StoreTest.sh       + how far a pooling computed
#                                                      live is from the same one
#                                                      taken from a stored raw
#                                                      output, on real tiles
#    ONLY_MODEL=1 sbatch jobscripts/StoreTest.sh       just that measurement
#  ENCODER (default uni2), TILES (per slide and level, default 200), LEVELS, SLIDES,
#  SEG (default hest), MASK_CACHE_JOB (default MppRoutingHead: its hest masks are
#  reused; empty = this job's own cache).
# =============================================================================

status=0

ARGS=()
[ "${WITH_MODEL:-0}" = "1" ] && ARGS+=(--with-model)
[ "${ONLY_MODEL:-0}" = "1" ] && ARGS+=(--only-model)
# Masks are read from an existing cache: MppRoutingHead already segmented most of
# the slides with hest, and a hit builds no segmenter. MASK_CACHE_JOB= (empty)
# uses this job's own cache instead, and segments what it does not find.
ARGS+=(--encoder "${ENCODER:-uni2}" --tiles "${TILES:-200}")
MASK_CACHE_JOB="${MASK_CACHE_JOB-MppRoutingHead}"
[ -n "$MASK_CACHE_JOB" ] && ARGS+=(--mask-cache-job "$MASK_CACHE_JOB")
[ -n "${SEG:-}" ] && ARGS+=(--seg "$SEG")
[ -n "${SLIDES:-}" ] && ARGS+=(--slides $SLIDES)
[ -n "${LEVELS:-}" ] && ARGS+=(--levels $LEVELS)

echo "======== test_store.py ${ARGS[*]} ========"
python utilities/test_modules/test_store.py "${ARGS[@]}"
rc=$?
[ $rc -ne 0 ] && status=$rc

echo ""
echo "======== done (exit $status) ========"
exit $status
