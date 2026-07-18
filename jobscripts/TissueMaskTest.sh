#!/bin/bash
#SBATCH --job-name=TissueMaskTest         # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/TissueMaskTest           # STDOUT
#SBATCH -e ./log/TissueMaskTest           # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Parameters ----------------
WSI=/work/u26130998/datasets/Ki67/S1151088,G7E,111220.mrxs
BBOX_LW=0.5

python utilities/test_modules/test_tissues_regions_mask.py \
  --wsi $WSI \
  --sweep \
  --no-hest \
  --no-region-index \
  --bbox-lw $BBOX_LW
