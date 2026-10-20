#!/bin/bash
#SBATCH --job-name=TestRerank             # -> log/%x
#SBATCH --partition=dev                   # Partition
#SBATCH --time=00:30:00
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=4                 # CPU cores per task
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh

# =============================================================================
#  The tests of the rerank work (stage 2 phase two, its training, and the base
#  modules it changed). No slide, no model weights.
#
#    sbatch jobscripts/RerankHead/TestRerank.sh
# =============================================================================
ONLY="${ONLY:-test_context test_rerank test_neural test_arm_spec test_naming test_retriever_arms test_arm_tables}"

status=0
for t in $ONLY; do
  echo "======== $t ========"
  python "utilities/test_modules/TestRerank/$t.py"
  rc=$?
  [ $rc -ne 0 ] && status=$rc
done
echo ""
if [ $status -eq 0 ]; then echo "======== ALL PASS ========"; else echo "======== FAILED (exit $status) ========"; fi
exit $status
