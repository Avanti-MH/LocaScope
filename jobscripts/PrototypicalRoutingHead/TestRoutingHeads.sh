#!/bin/bash
#SBATCH --job-name=TestRoutingHeads               # Job name
#SBATCH --partition=dev                        # Partition
#SBATCH --time=00:10:00                           # toy data, no slide, no model
#SBATCH --account=MST114560                       # Account
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --gpus-per-node=1                         # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                         # CPU cores per task
#SBATCH --ntasks-per-node=1                       # Tasks per node
#SBATCH -o /work/u26130998/log/%x   # STDOUT
#SBATCH -e /work/u26130998/log/%x   # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  The tests of the two routing-head training packages, one job.
#
#    sbatch jobscripts/PrototypicalRoutingHead/TestRoutingHeads.sh
#    ONLY="ordinal" sbatch jobscripts/PrototypicalRoutingHead/TestRoutingHeads.sh
#
#  They live in utilities/test_modules/TestRoutingHeads/, named after this
#  script: one job owns all three.
#
#    episodes   training/PrototypicalRoutingHead/Episodes.py -- the mixed-pool
#               draws training and validation make (also reads
#               MppRoutingHead/Datasets.py's ManifestRow)
#    kxk        kxk_report in PrototypicalRoutingHead/cli/train.py -- the
#               validation accuracy the checkpoints are chosen on
#    ordinal    the ordinal losses in BOTH places they are written,
#               PrototypicalRoutingHead/Losses.compute_loss and
#               MppRoutingHead/cli/train._compute_loss, held to the same numbers
#
#  No slide and no model: fake pools and toy tensors, seconds. The GPU line above
#  is there because this cluster wants one, not because the assertions need it.
#
#  crash-resume state (Resume.py), which both packages use, is tested in
#  jobscripts/TestAiNNModel.sh -- it belongs to the shared aiNNModel code.
#
#  ONLY picks a subset, space separated; the default is all three.
# =============================================================================

TESTS=utilities/test_modules/TestRoutingHeads
ONLY="${ONLY:-episodes kxk ordinal}"

status=0
run () {   # run <label> <command...>
  echo ""
  echo "======== $1 ========"
  shift
  "$@" || status=1
}

echo "======== TestRoutingHeads  tests: $ONLY ========"

for t in $ONLY; do
  case "$t" in
    episodes) run "episodes  (PrototypicalRoutingHead/Episodes.py)" \
                python "$TESTS"/test_episodes.py ;;
    kxk)      run "kxk  (kxk_report)" \
                python "$TESTS"/test_kxk_report.py ;;
    ordinal)  run "ordinal  (the two loss implementations agree)" \
                python "$TESTS"/test_ordinal_loss.py ;;
    *) echo "unknown test '$t' -- one of: episodes kxk ordinal"; status=1 ;;
  esac
done

echo ""
echo "======== done (exit $status) ========"
exit $status
