#!/bin/bash
#SBATCH --job-name=RerankHead             # Job name -> log/<name>, result/<name>/
#SBATCH --partition=8gpus                 # Partition
#SBATCH --time=48:00:00                   # partition 8gpus caps at 2 days
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # one card; the heads share every level they encode
#SBATCH --cpus-per-task=12                # the H200 cap per GPU: slide reads and photo renders
#SBATCH --mem=200G                        # the H200 cap per GPU
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  training/RerankHead -- the second phase of stage 2 (stage2_retrieval/spec.md)
#
#  Several heads are trained in ONE process on the SAME FoVs: a level of a slide
#  is encoded once and every head trains on it, and nothing of it is kept on disk
#  (no feature store, no candidate lists: they are made when used).
#
#    sbatch jobscripts/RerankHead/RerankHead.sh                       # tilegt, uni2-avg (K = 500 of its list)
#    HEADS="tilegt corrnet winembed" TAG=run1 sbatch jobscripts/RerankHead/RerankHead.sh
#    DRY_RUN=1 bash jobscripts/RerankHead/RerankHead.sh                # the plan; opens nothing
#    SMOKE=1 sbatch jobscripts/RerankHead/RerankHead.sh                # 2 slides a dataset, 3 visits, 1 epoch
#
#  The weights are written to <results>/RerankHead/weights/ and read from there by
#  a recipe (RerankHead/weights/<file>.pt). One checkpoint per head: `_best` (the
#  epoch best at SELECT_K on the held-out FoVs) and `_last`.
#
#  Every flag of cli/train.py can be given in EXTRA, e.g.
#    EXTRA="--sampler-n-per-rung 100 --fov-per-visit 96" sbatch ...
# =============================================================================
HEADS="${HEADS:-tilegt}"
RETRIEVER="${RETRIEVER:-uni2-avg}"
EPOCHS="${EPOCHS:-30}"
TAG="${TAG:-}"
SELECT_K="${SELECT_K:-30}"
N_TRAIN_WSI="${N_TRAIN_WSI:-0}"          # slides per training dataset; 0 = all
N_VAL_WSI="${N_VAL_WSI:-10}"            # slides of each dataset @legacy-val draws from
EXTRA="${EXTRA:-}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$SMOKE" = "1" ]; then
  EPOCHS=1; N_TRAIN_WSI=2; N_VAL_WSI=2
  EXTRA="--max-visits 3 --fov-per-visit 8 --val-fov-per-visit 4 --steps-per-visit 4 $EXTRA"
fi
[ "$DRY_RUN" = "1" ] && EXTRA="--dry-run $EXTRA"

python -u training/RerankHead/cli/train.py \
  --heads $HEADS --retriever "$RETRIEVER" --epochs "$EPOCHS" --tag "$TAG" --select-k "$SELECT_K" \
  --n-train-wsi "$N_TRAIN_WSI" --n-val-wsi "$N_VAL_WSI" $EXTRA
