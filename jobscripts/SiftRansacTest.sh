#!/bin/bash
#SBATCH --job-name=SiftRansacTest         # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/SiftRansacTest           # STDOUT
#SBATCH -e ./log/SiftRansacTest           # STDERR

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
PADDING=2
MIN_INLIERS=10

python utilities/test_modules/test_sift_ransac.py \
  --wsi $WSI \
  --x $X --y $Y \
  --mpp $MPP \
  --ratio $RATIO \
  --mpixels $MPIXELS \
  --tile $TILE \
  --batch $BATCH \
  --min-region-ratio $MIN_REGION_RATIO \
  --padding $PADDING \
  --min-inliers $MIN_INLIERS \
  --overlap --filter \
  --h-decomp --patch-grid --trans-arrow
