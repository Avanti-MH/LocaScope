#!/bin/bash
#SBATCH --job-name=TestConfig             # -> log/%x
#SBATCH --partition=dev                   # Partition
#SBATCH --time=00:20:00                   # both are seconds
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0) -- weights_id checks .cuda() when there is one
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  The two config layers every module is named and configured through. No
#  slide, no model weights.
#    test_config_identity  utilities/ConfigIdentity.py: identity parts, nested
#                          configs, weights_id (content, device, the wrapper),
#                          the registry, the json round trip
#    test_config_args      utilities/ConfigArgs.py: one flag per config field,
#                          the sampler's and the camera's flags
#
#    sbatch jobscripts/TestConfig.sh
#    ONLY="test_config_args" sbatch jobscripts/TestConfig.sh
# =============================================================================
ONLY="${ONLY:-test_config_identity test_config_args}"

status=0
for t in $ONLY; do
  echo "======== $t ========"
  python "utilities/test_modules/$t.py"
  rc=$?
  [ $rc -ne 0 ] && status=$rc
  echo ""
done

echo "======== done (exit $status) ========"
exit $status
