#!/bin/bash
#SBATCH --job-name=InspectPcaSeg          # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=04:00:00                   # measured: 3.5-6 min per slide at level 0
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                 # the tile loader's workers
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


# =============================================================================
#  Which side of PC1 is tissue -- and is PC1 tissue at all?
# =============================================================================
#
# NOT a test. It asserts nothing and cannot fail; it draws a figure for a human
# and prints how to read it. `TestSuperPathPoint.sh` is where the assertions
# live, and this deliberately is not a stage of it.
#
# THE QUESTION. `Uni2PcaSegConfig.larger_pca_as_fg`. The sign of a principal
# component is arbitrary, nothing in the fit chooses it, and the run of
# 2026-08-25 reported a foreground fraction of 68.0 percent on BRACS_1228 where
# that slide's measured tissue is 20.8 (SlideWinTest log, hsv mask at ds 32).
# 100 - 68 = 32 lands inside the 23-38 percent the three BRACS slides cover, so
# the flag being backwards is the hypothesis with the most explanatory power --
# and a hypothesis is not a finding. The mask decides it.
#
# TWO OUTCOMES, and the second is why the histogram panel exists:
#
#   A  one polarity lies on the tissue, the other on the glass, and the first
#      one's agree_hsv is well above the other -> set that value
#   B  neither lies on the tissue, OR the PC1 histogram is unimodal -> PC1 is
#      not the tissue/glass axis here, the flag is the wrong knob, and what to
#      reach for is the fit sample, feature_norm, or a component past the first
#
# Two masks always look like a choice between two answers even when neither is
# one. The histogram is what says which case you are in.
#
# BOTH DATASETS, because they stain differently. BRACS is H&E and Ki67 is DAB
# on a blue counterstain, and if the polarity comes out opposite on the two then
# it is not one flag but a per-stain one -- which would be a real problem for a
# single ladder covering both (spec.md 6.5).
#
# ONE MAGNIFICATION, because the segmenter has none. Level 0, mask at ds 14 --
# finer than the ds 32 hsv masks this project uses, and a derived fact rather
# than a choice. See Uni2PcaSegFunc.LEVEL.

BRACS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test
KI67=/work/u26130998/datasets/Ki67

SLIDES=(
  "$BRACS/Group_AT/Type_ADH/BRACS_1228.svs"     # 20.8% tissue measured, the one
                                                # the 68.0% came from
  "$KI67/S1104233,G7E,110208.mrxs"              # DAB brown, and a 2x pyramid
)

# 200 rather than the config default of 1000: the basis only has to be good
# enough to answer a sign question. It IS a hashed field, so this run and a
# production run correctly get different identity ids.
FIT_TILES=200

# The reading is the cost, not the model. From the EoMTest run of 2026-08-24:
# 86,940 tiles at ~410/s and 270,570 at ~783/s, whole slide, level 0.
WORKERS=8

echo "======== InspectPcaSeg ========"
echo "  slides: ${#SLIDES[@]}   level 0, whole slide, mask ds 14"
echo ""

python utilities/cli/inspect_pca_seg.py \
  "${SLIDES[@]}" \
  --workers $WORKERS \
  --fit-tiles $FIT_TILES

status=$?

echo ""
echo "======== done  (exit $status) ========"
echo "  figures -> result/\${SLURM_JOB_NAME}/pca_seg_polarity__<slide>_L0.png"
echo "  table   -> result/\${SLURM_JOB_NAME}/pca_seg_polarity.csv"
echo "  names   -> the _definitions.csv beside each figure"
echo ""
echo "  Look at the histogram FIRST. If it is unimodal, the two mask panels are"
echo "  a choice between two wrong answers and the flag is not the knob."

exit $status
