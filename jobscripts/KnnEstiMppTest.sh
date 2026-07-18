#!/bin/bash
#SBATCH --job-name=KnnEstiMppTest         # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/KnnEstiMppTest           # STDOUT
#SBATCH -e ./log/KnnEstiMppTest           # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Parameters ----------------
WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1003691.svs
X=31700
Y=33600
TILE=256
SAMPLES=100
K=11
MPIXELS=1.475
BATCH_SIZE=4096

python utilities/test_modules/test_gigapath_knn_esti_mpp.py \
  $WSI \
  --x $X --y $Y \
  --tile $TILE \
  --samples $SAMPLES \
  --k $K \
  --mpixels $MPIXELS \
  --batch-size $BATCH_SIZE
