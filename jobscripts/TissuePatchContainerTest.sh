#!/bin/bash
#SBATCH --job-name=TissuePatchContainerTest # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/TissuePatchContainerTest # STDOUT
#SBATCH -e ./log/TissuePatchContainerTest # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Parameters ----------------
SIZE=128
RSIZE=256
QUERY=/work/u26130998/datasets/Ki67/S1103037_ki67/2.bmp
ROI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_RoI/latest_version/test/0_N/BRACS_264_N_5.png
WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1003691.svs
LEVEL=3
OPENSLIDE_LEVEL=9

python utilities/test_modules/test_tissue_patch_container.py \
  --size $SIZE \
  --rsize $RSIZE \
  --query $QUERY \
  --roi $ROI \
  --wsi $WSI \
  --level $LEVEL \
  --openslide-level $OPENSLIDE_LEVEL
