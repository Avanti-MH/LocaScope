#!/bin/bash
#SBATCH --job-name=ExpGloRetFull          # Job name -> log/<name>, result/<name>/ (an experiment: nothing of it is in a formal name)
#SBATCH --partition=dev                   # Partition
#SBATCH --time=04:00:00
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=3                 # the largest of the three settings below
#SBATCH --cpus-per-task=36                # 12 per GPU, the H200 cap
#SBATCH --mem=600G                        # 200G per GPU, the H200 cap
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083
#SBATCH -o /work/u26130998/log/%x         # STDOUT
#SBATCH -e /work/u26130998/log/%x         # STDERR

# =============================================================================
#  How long a step takes when its softmax is over EVERY window of the level (`--top-p 1.0`, locascope/training/GloRet/Full.py),
#  and what it needs of a card: GloRet-SelfAttn-AttnPool on gigapath cls, a few visits of each of three settings, one
#  after the other in this job (a visit prints its steps' windows and the seconds of their parts: t_keys, t_loss,
#  t_back, and the card's peak memory):
#
#    1  1 GPU                              a gradient cache
#    2  2 GPUs, one group (--window-split 2)  each encodes half of the windows, with a gradient cache
#    3  3 GPUs, one group (--window-split 3 --no-gradcache)   each encodes a third WITH its graph: it may run out of memory,
#                                          and that is an answer
#
#  The logs: the Slurm log has setting 1 and the headers; the ranks' (log/<job>.r<N>) are copied to
#  log/<job>.split2.r<N> and log/<job>.split3.r<N> after each. The weights go to result/<job>/ (--out), W&B is off.
#
#    sbatch deploy/slurm/GloRet/ExpGloRetFull.sh          # no argument: one name, re-sent over itself
# =============================================================================
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate locascope
source deploy/slurm/_env.sh    # HF_HOME; must be exported before python starts
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore:Plan failed with a cudnnException:UserWarning"

JOB="${SLURM_JOB_NAME:-ExpGloRetFull}"
LOGS="/work/u26130998/log"
COMMON="--heads GloRet-SelfAttn-AttnPool --encoder gigapath --pooling cls --epochs 1 --n-train-wsi 3 --n-val-wsi 1 \
  --top-p 1.0 --fovs-per-step 64 --steps-per-visit 2 --fov-per-visit 64 --val-fov-per-visit 4 --max-visits 8 \
  --tag full --out /work/u26130998/result/$JOB --wandb-mode disabled"

echo "==== setting 1: 1 GPU, a gradient cache ===="
CUDA_VISIBLE_DEVICES=0 python -u -m locascope.training.GloRet.train $COMMON
echo "==== setting 1 ended: exit $? ===="

echo "==== setting 2: 2 GPUs, --window-split 2 ===="
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m locascope.training.GloRet.train $COMMON --window-split 2
echo "==== setting 2 ended: exit $? ===="
for r in 0 1; do cp "$LOGS/$JOB.r$r" "$LOGS/$JOB.split2.r$r" 2>/dev/null; done

echo "==== setting 3: 3 GPUs, --window-split 3 --no-gradcache ===="
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 -m locascope.training.GloRet.train $COMMON --window-split 3 --no-gradcache
echo "==== setting 3 ended: exit $? ===="
for r in 0 1 2; do cp "$LOGS/$JOB.r$r" "$LOGS/$JOB.split3.r$r" 2>/dev/null; done
