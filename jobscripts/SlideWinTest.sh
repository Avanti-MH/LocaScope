#!/bin/bash
#SBATCH --job-name=SlideWinTest           # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/SlideWinTest             # STDOUT
#SBATCH -e ./log/SlideWinTest             # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

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
