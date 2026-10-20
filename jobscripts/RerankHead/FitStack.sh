#!/bin/bash
#SBATCH --job-name=FitStack               # Job name -> log/<name>
#SBATCH --partition=8gpus                 # Partition
#SBATCH --time=24:00:00
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh

# =============================================================================
#  training/RerankHead/cli/fit_stack.py -- the stacker (S7) over the columns the
#  rerankers of an arm wrote. Run it AFTER the heads are trained (RerankHead.sh): the
#  arm's learned rerankers read their weights. It is fitted on the held-out slides, not
#  the training ones (see the script's docstring), so confirm on `@legacy-test` in
#  the stage 2 retriever bench (DATASETS=@legacy-test).
#
#    sbatch jobscripts/RerankHead/FitStack.sh
#    COLUMNS="first dual csls" ARM='uni2:SlidingWindow::cls_avg+mean@500::Dual::cls_avg+dual::CSLS::cls_avg+csls' \
#        TAG=a sbatch jobscripts/RerankHead/FitStack.sh
# =============================================================================
ARM="${ARM:-uni2:SlidingWindow::cls_avg+mean@500::Dual::cls_avg+dual::CSLS::cls_avg+csls::Corr::cls_avg+sinkhorn::Corr::cls_avg+contrast::Corr::cls_avg+mnn::CandGraph::raw+prominence::TileGT::cls_avg+tilegt::CorrNet::cls_avg}"
COLUMNS="${COLUMNS:-first dual csls sinkhorn contrast mnn nbr_support prominence rot_margin tilegt corrnet}"
TAG="${TAG:-}"
EXTRA="${EXTRA:-}"

python -u training/RerankHead/cli/fit_stack.py \
  --arm "$ARM" --columns $COLUMNS --tag "$TAG" $EXTRA
