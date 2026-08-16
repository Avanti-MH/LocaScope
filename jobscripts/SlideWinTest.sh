#!/bin/bash
#SBATCH --job-name=SlideWinTest           # Job name
#SBATCH --partition=normal                # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=2                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --mem=600G                        # see the note below
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/SlideWinTest             # STDOUT
#SBATCH -e /work/u26130998/log/SlideWinTest             # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath


# ---------------- Memory ----------------
#
# This wants a lot, and the reason is not the tiles.
#
#   job 301949   no --mem   ReqMem 200G   MaxRSS 199.2 GB   2 rounds OOM
#   job 301972   --mem=128G ReqMem 128G   MaxRSS 127.7 GB   4 rounds OOM
#
# The second is the cautionary one. This script carried no --mem and the site
# default turned out to be 200G, so writing 128G to "give it memory" LOWERED the
# ceiling and killed the two --filter rounds that had been passing. Read the
# default before overriding it; sacct prints it as ReqMem.
#
# MPP below is 0.252 against the slide's own 0.2524, so every round builds at
# LEVEL 0. Tile pixels are 9.5 GB with the filter and 10.9 GB without -- nearly
# equal, so the tiles are not what differs. The regions are: the filter leaves
# 5 merged ones, no-filter leaves 2988, each reading its own bounding box, and
# those boxes overlap heavily, so the total read is a multiple of the tissue
# area rather than equal to it.
#
# Nothing here regressed. Before WsiTissuesContainer.from_ds, the mask went to
# the container unnarrowed, and the first region too small to host a tile
# reached gigapath_encode as an empty batch and died in torch.cat -- so the
# --no-filter rounds never ran to the end and never allocated what they needed.
# The requirement was always this size; the crash used to arrive first.
#
# 600G because the nodes hold 1.9 TB and DiagMultiGPU has completed at that
# size. It is not measured -- 200G was not enough and the true ceiling is
# unknown, so this is headroom, not a number anyone derived. If a bigger slide
# ever OOMs here, the fix is to stop holding every region's pixels at once, not
# to raise this again.
#
# --cpus-per-task is still 2 while the sibling scripts use 8. Left alone on
# purpose: raising it is a scheduling trade, not a correctness one, and this
# job waits for a GPU either way.

# ---------------- Parameters ----------------
WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs
X=31700
Y=33600
MPP=0.252
RATIO=45:32
MPIXELS=1.475
TILE=256
BATCH=4096
MIN_REGION_RATIO=0.10

BASE_ARGS="
  --wsi $WSI
  --x $X --y $Y
  --mpp $MPP
  --ratio $RATIO
  --mpixels $MPIXELS
  --tile $TILE
  --batch $BATCH
  --min-region-ratio $MIN_REGION_RATIO
"

echo "======== [1/4] overlap + filter ========"
python utilities/test_modules/test_gigapath_slide_win_sim.py \
  $BASE_ARGS --overlap --filter

echo "======== [2/4] overlap + no-filter ========"
python utilities/test_modules/test_gigapath_slide_win_sim.py \
  $BASE_ARGS --overlap --no-filter

echo "======== [3/4] no-overlap + filter ========"
python utilities/test_modules/test_gigapath_slide_win_sim.py \
  $BASE_ARGS --no-overlap --filter

echo "======== [4/4] no-overlap + no-filter ========"
python utilities/test_modules/test_gigapath_slide_win_sim.py \
  $BASE_ARGS --no-overlap --no-filter
