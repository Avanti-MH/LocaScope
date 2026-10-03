#!/bin/bash
#SBATCH --job-name=AiNNModelTest                  # Job name
#SBATCH --partition=dev                        # Partition
#SBATCH --time=00:30:00                           # three real encoders + GigaPath, weights loaded once each
#SBATCH --account=MST114560                       # Account
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --gpus-per-node=1                         # GPUs per node (不要設0)
#SBATCH --cpus-per-task=4                         # CPU cores per task
#SBATCH --mem=64G                                 # weights on the host while they load
#SBATCH --ntasks-per-node=1                       # Tasks per node
#SBATCH -o /work/u26130998/log/%x      # STDOUT
#SBATCH -e /work/u26130998/log/%x      # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  aiNNModel/ -- the shared model code, in one job.
#
#    sbatch jobscripts/AiNNModelTest.sh
#    ONLY="tile_encoder resume" sbatch jobscripts/AiNNModelTest.sh
#    ENCODER=uni2 NO_DUAL_LOAD=1 ONLY=encoders sbatch jobscripts/AiNNModelTest.sh
#
#  The tests live in utilities/test_modules/AiNNModelTest/, named after this script.
#
#    tile_encoder  TileEncoderFunc, the template, against fake models (a vector,
#                  a token and a spatial one): no GPU, seconds. Its last section
#                  downloads vit_tiny (5 MB) the first time -- if a compute node
#                  has no network, put the weights in HF_HOME beforehand
#    resume        Resume.py: crash-resume state, a toy model, bit for bit. No GPU
#    encoders      the three real encoders (GigaPath, UNI2, CONCH) against the
#                  weights they actually load. GPU
#    gigapath      GigaPath's current path against the free functions it replaced
#                  (GigaPathFunc_old.py). GPU. Delete it when the old file goes
#
#  Knobs (environment):
#    ONLY          which of the four, space separated; default all
#    ENCODER       encoders to check, space separated (uni2 gigapath conch_vit);
#                  default all three
#    NO_DUAL_LOAD  1 skips the check that loads a second copy (4.5 GB)
#    DTYPE         fp16 or fp32 for the encoders check; default per encoder
#    TILES         tiles per check in the GigaPath equivalence test; default 8
# =============================================================================

TESTS=utilities/test_modules/AiNNModelTest
ONLY="${ONLY:-tile_encoder resume encoders gigapath}"

ENCODERS_ARGS=""
[ -n "${ENCODER:-}" ] && ENCODERS_ARGS="$ENCODERS_ARGS --encoder $ENCODER"
[ "${NO_DUAL_LOAD:-0}" = "1" ] && ENCODERS_ARGS="$ENCODERS_ARGS --no-dual-load"
[ -n "${DTYPE:-}" ] && ENCODERS_ARGS="$ENCODERS_ARGS --dtype $DTYPE"
GIGAPATH_ARGS=""
[ -n "${TILES:-}" ] && GIGAPATH_ARGS="--tiles $TILES"

status=0
run () {   # run <label> <command...>
  echo ""
  echo "======== $1 ========"
  shift
  "$@" || status=1
}

echo "======== AiNNModelTest  tests: $ONLY ========"

for t in $ONLY; do
  case "$t" in
    tile_encoder) run "tile_encoder  (TileEncoderFunc, fake models)" \
                    python "$TESTS"/test_tile_encoder.py ;;
    resume)       run "resume  (Resume.py, bit for bit)" \
                    python "$TESTS"/test_resume.py ;;
    encoders)     run "encoders  (the real weights)${ENCODERS_ARGS}" \
                    python "$TESTS"/test_encoders.py $ENCODERS_ARGS ;;
    gigapath)     run "gigapath  (against GigaPathFunc_old)${GIGAPATH_ARGS:+  $GIGAPATH_ARGS}" \
                    python "$TESTS"/test_gigapath_equivalence.py $GIGAPATH_ARGS ;;
    *) echo "unknown test '$t' -- one of: tile_encoder resume encoders gigapath"
       status=1 ;;
  esac
done

echo ""
echo "======== done (exit $status) ========"
exit $status
