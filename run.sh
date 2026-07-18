#!/bin/bash
#SBATCH --job-name=TissueMaskTest # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/TissueMaskTest # STDOUT
#SBATCH -e ./log/TissueMaskTest # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Run WSI ROI Thumbnail script ----------------
# python utilities/test_modules/test_estimate_mpp.py \
#   /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1003691.svs \
#   --x 31700 --y 33600\
#   --tile 256 \
#   --samples 100 \
#   --k 11 \
#   --mpixels 2

# python utilities/test_modules/test_tissue_patch_container.py \
#   --wsi /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1003691.svs --level 3

BASE_ARGS="--batch 4096"

# echo "======== [1/4] overlap + filter ========"
# python utilities/test_modules/test_gigapath_slide_win_sim.py \
#   $BASE_ARGS --overlap --filter

# echo "======== [2/4] overlap + no-filter ========"
# python utilities/test_modules/test_gigapath_slide_win_sim.py \
#   $BASE_ARGS --overlap --no-filter

# echo "======== [3/4] no-overlap + filter ========"
# python utilities/test_modules/test_gigapath_slide_win_sim.py \
#   $BASE_ARGS --no-overlap --filter

# echo "======== [4/4] no-overlap + no-filter ========"
# python utilities/test_modules/test_gigapath_slide_win_sim.py \
#   $BASE_ARGS --no-overlap --no-filter

# echo "======== [5/5] SIFT+RANSAC (overlap + filter) ========"
# python utilities/test_modules/test_sift_ransac.py \
#   $BASE_ARGS --overlap --filter

# python utilities/test_modules/test_tissue_patch_container.py \
#   --wsi /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_FEA/BRACS_1936.svs
python utilities/test_modules/test_tissues_regions_mask.py --sweep --no-region-index --wsi /work/u26130998/datasets/Ki67/S1151088,G7E,111220.mrxs