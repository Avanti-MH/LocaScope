#!/bin/bash
#SBATCH --job-name=PatchInfoCoordsTest    # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=01:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/PatchInfoCoordsTest      # STDOUT
#SBATCH -e ./log/PatchInfoCoordsTest      # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Parameters ----------------
SIZE=128

python utilities/test_modules/test_patch_info_coords.py \
  --size $SIZE
