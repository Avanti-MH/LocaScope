#!/bin/bash
#SBATCH --job-name=TestGloRet             # -> log/%x
#SBATCH --partition=dev                   # Partition
#SBATCH --time=00:30:00
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=4                 # CPU cores per task
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source deploy/slurm/_env.sh

# =============================================================================
#  The tests of GloRet and LogLoRet (locascope/core/stage2_retrieval/proposal.md): the networks, their retrievers, their training
#  (locascope/training/GloRet), the patch tokens of TilePool attnpool and their stages, the resource meter, the wandb tools.
#  No slide, no model weights. Its own job and log (log/TestGloRet), apart from TestRerank's (the stage 2 re-rankers
#  and RerankHead), so the one never overwrites the other.
#
#    sbatch deploy/slurm/GloRet/TestGloRet.sh
# =============================================================================
ONLY="${ONLY:-test_gloret}"

status=0
for t in $ONLY; do
  echo "======== $t ========"
  python "tests/TestGloRet/$t.py"
  rc=$?
  [ $rc -ne 0 ] && status=$rc
done
echo ""
if [ $status -eq 0 ]; then echo "======== ALL PASS ========"; else echo "======== FAILED (exit $status) ========"; fi
exit $status
