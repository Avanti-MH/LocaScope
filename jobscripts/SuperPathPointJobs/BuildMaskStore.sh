#!/bin/bash
#SBATCH --job-name=BuildMaskStore         # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=06:00:00                   # 6 slides x 3.5-6 min measured, x2 margin
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
#  spec.md 12 steps 3a and 3b: fill the mask store, then probe what it yields
# =============================================================================
#
# TWO STEPS IN ONE JOB because the second is the first one's only consumer here
# and costs minutes against the first one's half hour. Splitting them would mean
# a second queue wait to read a table that is already sitting in memory's reach.
#
# STEP 3a  build_mask_store.py -- one UNI2-PCA tissue mask per slide, written to
#          result/cache/masks/ under the segmenter's identity_id(). The mask
#          costs 3.5 to 6 min of GPU per slide (measured, Uni2PcaSegFunc.LEVEL)
#          and THREE later steps read it: this probe, the pre-tile extraction of
#          3c, and any bench that wants the regions the training tiles came from.
#
#          Each mask now carries its COMPONENTS as a second tensor -- the field
#          the bit was thresholded from, 581 to 814 MB per slide. That is what
#          makes `background_threshold` answerable later without re-encoding:
#          a sweep over candidate values is seconds off disk, and the alternative
#          is 3.5 to 6 min of GPU per slide per candidate.
#
# STEP 3b  probe_tile_yield.py -- how many tiles rejection sampling actually
#          returns per (slide, tile_size, ds), AND in which richness bucket.
#          3 tile sizes x 6 rungs x 12 slides = 216 cells, mask arithmetic
#          only, no model. This decides three things spec.md left open:
#
#            how many tiles per cell 3c extracts,
#            which rung-balancing mode the trainer uses -- `align-min` if the
#            worst cell still has hundreds, `loss-weight` if it has forty, and
#            whether bg30_50's 50 per cent FLOOR is reachable at all.
#
#          The third is new on 2026-08-27 and it is the one with no prior
#          measurement behind it: the 2026-08-26 corpus put a single 15 per
#          cent cap over what are now two buckets, so its histogram records the
#          cap and not the slide. `supply_<bucket>` is the candidate pool
#          before any cap; `got_<bucket>` is what the fill took; the run prints
#          every cell where a floor went unmet.
#
#          The probe decides the switch; the switch does not decide the probe.
#
# WHAT IS ALREADY SETTLED and so is not a parameter here: the polarity
# (`larger_pca_as_fg=True`, InspectPcaSeg of 2026-08-26, both stains, four
# magnifications) is the config's own default now, so this script does not
# repeat it -- see build_mask_store.py's --larger-pca-as-fg help.
#
# WHAT IS NOT SETTLED: `background_threshold` stays at the notebook's 0.5 for
# this build. It is not a guess being frozen -- it is the value whose components
# this run puts on disk so the question can be asked properly.

BRACS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test
KI67=/work/u26130998/datasets/Ki67

# The six slides of spec.md 6.5. Four train, two held out -- but the mask store
# does not know or care which is which: a mask is a property of the slide, and
# splitting it here would mean two stores to keep in step.
# Five per stain in TRAIN, plus one per stain HELD OUT from outside the ten.
# Each addition answers something the earlier set of six could not:
#
#   Group_BT was absent entirely -- Type_N is normal tissue, the furthest in
#     architecture from ADH / FEA / DCIS, and keypoint survival is a question
#     about architecture
#   Type_IC is the largest type in the test set and invasive carcinoma is
#     unlike the other four
#   the three Ki67 slides were 110208, 110208 and 111220, i.e. two batches.
#     Scanner and batch drift is a real axis and it was almost unsampled.
#
# Within a type the pick is the FIRST filename in sort order -- a rule rather
# than a choice, so nobody has to remember why this one.
#
# TWO KI67 SLIDES ARE DELIBERATELY ABSENT. `S1137178` (spec.md 6.5) and
# `S1103037` (SafeSlide.py:33) are the only two known here with scanner holes.
# A hole reads back as the slide's background colour, and the straight
# high-contrast edge that leaves is a perfect corner -- a detector fires on it,
# scores it highly, and the result looks like a model that has learned
# something. When they come back in, it is with
# `SafeSlide.read_region_valid` masking the invalid area out of HA.
SLIDES=(
  "$BRACS/Group_AT/Type_ADH/BRACS_1228.svs"      # train, and the sanity-check
                                                 # slide every earlier bench ran
  "$BRACS/Group_MT/Type_DCIS/BRACS_1476.svs"     # train
  "$BRACS/Group_AT/Type_FEA/BRACS_1936.svs"      # train
  "$BRACS/Group_BT/Type_N/BRACS_1579.svs"        # train  (new: Group_BT)
  "$BRACS/Group_MT/Type_IC/BRACS_1284.svs"       # train  (new: Type_IC)
  "$KI67/S1104233,G7E,110208.mrxs"               # train
  "$KI67/S1104360,G7E,110208.mrxs"               # train
  "$KI67/S1151088,G7E,111220.mrxs"               # train
  "$KI67/S1103520,G7E,110126.mrxs"               # train  (new: earliest batch)
  "$KI67/S1140701,G7E,111018.mrxs"               # train  (new: middle batch)
  "$BRACS/Group_BT/Type_N/BRACS_1598.svs"        # HELD OUT  (seen type)
  "$KI67/S1103627,G7E,110127.mrxs"               # HELD OUT
)

# The config default, unlike InspectPcaSeg's 200. That run only had to answer a
# sign question; these masks are what every tile downstream is sampled through.
FIT_TILES=1000

# The reading is the cost, not the model. From the EoMTest run of 2026-08-24:
# 86,940 tiles at ~410/s and 270,570 at ~783/s, whole slide, level 0.
WORKERS=8

echo "======== 3a  BuildMaskStore ========"
echo "  slides: ${#SLIDES[@]}   level 0, mask ds 14, fit on $FIT_TILES tiles"
echo ""

python utilities/cli/build_mask_store.py \
  "${SLIDES[@]}" \
  --fit-tiles $FIT_TILES \
  --workers $WORKERS
status=$?

if [ $status -ne 0 ]; then
  echo ""
  echo "======== 3a failed (exit $status); not probing ========"
  echo "  A partial store is still valid -- build_mask_store writes per slide."
  echo "  Re-run this script; slides already in the store are skipped."
  exit $status
fi

echo ""
echo "======== 3b  ProbeTileYield ========"
echo "  216 cells: 12 slides x 3 tile_size x 6 ds   (no ratio axis: the gate"
echo "             is gone, the richness caps are it)"
echo "  no model, no GPU -- mask arithmetic and the rejection sampler"
echo ""

# No --out: both CLIs resolve theirs with job_result_dir(), which prefers
# SLURM_JOB_NAME, so the mask table and the yield table land in the same
# result/BuildMaskStore/ without either of them being told where that is.
python utilities/cli/probe_tile_yield.py
probe=$?

echo ""
echo "======== done  (3a $status, 3b $probe) ========"
echo "  masks   -> result/cache/masks/<slide>__uni2-pca-seg__<cfg8>.safetensors"
echo "  table   -> result/\${SLURM_JOB_NAME}/build_mask_store.csv"
echo "  yield   -> result/\${SLURM_JOB_NAME}/tile_yield.csv + tile_yield.png"
echo "  names   -> tile_yield_definitions.csv beside the figure"
echo ""
echo "  Read the WORST cell of the yield figure first. It is the one that"
echo "  decides align-min against loss-weight, and it is the one 3c has to"
echo "  live with."

exit $probe
