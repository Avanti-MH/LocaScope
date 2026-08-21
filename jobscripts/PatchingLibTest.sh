#!/bin/bash
#SBATCH --job-name=PatchingLibTest        # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                   # the containers section reads a WSI
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# ---------------- PatchingLib, all four sections ----------------------------
#
# One script for what used to be four. PatchGridIndexTest, PatchInfoCoordsTest
# and TissuePatchContainerTest all invoked THIS SAME test_patching_lib.py with
# a different `--only`, and had said so in their own job-name comments for a
# while ("legacy alias -> PatchingLibTest grid"). Four files that differ by one
# flag are four places to update when an argument changes, and the arguments
# had already drifted: two of them carried the real-data paths, two did not.
#
#   grid        PatchGrid.from_size indexing -- synthetic, seconds
#   coords      PatchInfo coordinate round-trips -- synthetic, seconds
#   containers  the real-data path: query BMP, RoI PNG, WSI at --level
#   scale       cross-level scaling; never had a jobscript of its own
#
# ONLY picks a subset; the default is all four. The names are checked by
# argparse, so a typo fails at parse time rather than silently running nothing:
#
#   sbatch jobscripts/PatchingLibTest.sh                        # all four
#   ONLY=grid sbatch jobscripts/PatchingLibTest.sh              # was PatchGridIndexTest
#   ONLY=coords sbatch jobscripts/PatchingLibTest.sh            # was PatchInfoCoordsTest
#   ONLY=containers sbatch jobscripts/PatchingLibTest.sh        # was TissuePatchContainerTest
#   ONLY="grid coords" sbatch jobscripts/PatchingLibTest.sh     # the synthetic pair
#
# TWO SUBSETS WRITE TO ONE PLACE unless you say otherwise. The log is `%x` and
# _paths.job_result_dir reads SLURM_JOB_NAME, so --job-name moves BOTH at once
# -- one knob, no pair of names that has to be kept in agreement:
#
#   ONLY=grid sbatch --job-name=PatchingLibTest_grid --time=01:00:00 \
#       jobscripts/PatchingLibTest.sh
#
# --time on the command line overrides the directive above, which is set for
# the containers section. grid and coords are synthetic and finish in seconds;
# asking for 24 h only makes them queue longer.

ONLY="${ONLY:-grid coords containers scale}"

SIZE=128
RSIZE=256
QUERY=/work/u26130998/datasets/Ki67/S1103037_ki67/2.bmp
ROI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_RoI/latest_version/test/0_N/BRACS_264_N_5.png
WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1003691.svs
LEVEL=3
OPENSLIDE_LEVEL=9

echo "======== PatchingLibTest  sections: $ONLY ========"
echo ""

# $ONLY unquoted on purpose: --only takes nargs='+', so ONLY="grid coords" has
# to word-split into two arguments. Everything else is quoted.
#
# Every section gets every argument, including the ones it ignores -- coords
# never opens the WSI. That is cheaper than four argument lists to keep in
# agreement, and keeping them in agreement is exactly what the split versions
# failed to do.
python utilities/test_modules/test_patching_lib.py \
  --only $ONLY \
  --size $SIZE \
  --tile $SIZE \
  --rsize $RSIZE \
  --query "$QUERY" \
  --roi "$ROI" \
  --wsi "$WSI" \
  --level $LEVEL \
  --openslide-level $OPENSLIDE_LEVEL

echo ""
echo "======== done ========"
echo "  figures -> result/\$SLURM_JOB_NAME/  (patch_grid__index.png,"
echo "             patch_info__coords.png, and the container/scale figures)"
