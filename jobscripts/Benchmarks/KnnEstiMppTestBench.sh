#!/bin/bash
#SBATCH --job-name=KnnEstiMppTest         # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/KnnEstiMppTest           # STDOUT
#SBATCH -e /work/u26130998/log/KnnEstiMppTest           # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# =============================================================================
#  TODO 2026-08-27: the reference bank is sampled by an UNMEASURED config
# =============================================================================
#
# `GigaPathKnnEstiMpp.build_samples` used to draw a region uniformly and a
# position uniformly inside it. It now goes through the rewritten
# `utilities/TileSampler.py` with a DISJOINT LATTICE and nothing else -- no
# richness buckets, no inheritance, `SamplerConfig(overlap=OverlapConfig())`.
#
# That is a stopgap, not a decision. What it changed, measured on this run:
#
#     reference tiles: 293      (--samples 100 x 4 levels asks for 400)
#     KNN err 0.0%              so the estimator still works at 293
#
# The lattice trades yield for non-overlap and the deep levels are where it
# binds -- a level-3 tile covers 8x the level-0 area, so a region holds far
# fewer disjoint positions. 293 is not obviously wrong: `ReferenceSampler`'s
# own docstring records that 40 per level is already saturated for this KNN
# (result/MppEstimate: 40 -> 640 buys 2.7 points). But nobody chose 293.
#
# WHAT IS ACTUALLY UNDECIDED
#   * how many tiles per level a reference bank wants, now that they no longer
#     repeat -- the old count was inflated by up to 13 per cent bit-identical
#     twins in the deep banks (ReferenceSampler's docstring again)
#   * whether the background-fraction BUCKETS belong here. A bank that is all
#     tissue-dense tiles routes a blank query badly, which is the argument
#     ReferenceSampler was written on
#   * whether this should use ReferenceSampler at all. That module exists to
#     choose reference banks well and is currently used by nothing; the plan to
#     retire it into TileSampler's config is in log/TODO.log 2026-08-27, and
#     this jobscript is its first real consumer.
#
# THE ERROR THE FIRST RUN AFTER THE MIGRATION HIT, AND HOW IT WAS FIXED
# ----------------------------------------------------------------------
# Recorded here rather than only in the commit, because it is the shape of
# every remaining one: the SAMPLING migrated cleanly and a FIGURE did not.
#
#     Traceback (most recent call last):
#       File "utilities/test_modules/test_gigapath_knn_esti_mpp.py", line 313
#         main()
#       File ".../test_gigapath_knn_esti_mpp.py", line 272, in main
#         plot_knn_debug(est, gt_mpp, n_patches=8, ...)
#       File ".../test_gigapath_knn_esti_mpp.py", line 152, in plot_knn_debug
#         legend_handles = [
#     AttributeError: 'TileSampler' object has no attribute 'level_mpps'
#
# Everything before it worked -- 293 reference tiles encoded, KNN err 0.0% on
# the first level -- and the run died drawing the legend.
#
# WHAT level_mpps WAS. The old sampler built it in __init__:
#
#     self.level_mpps = [base_mpp * wsi.level_downsamples[lv]
#                        for lv in range(wsi.level_count)]
#
# A list cached at construction, i.e. a STORED COPY OF A DERIVED NUMBER. The
# rewrite dropped it for the same reason it dropped `TileInfo.mpp`: a stored
# copy goes stale against the handle it came from, and these are the KNN's
# LABELS -- the one thing in this estimator that must not drift.
#
# THE FIX, and it is permanent rather than a stopgap:
#
#     def _level_mpp(est, level):
#         return float(est.wsi.base_mpp * est.wsi.level_downsamples[level])
#
# One helper in the test file, derived from the slide at use. The legend and
# the per-tile titles now share it -- before the fix they computed the same
# quantity in two places, which is the shape that lets two numbers drift apart
# in the first place.
#
# WHAT TO DISCUSS LATER. Whether `mpp` should be on `SampleMeta` after all.
# The argument against is above; the argument for is that every consumer now
# writes the same two-term product, and a fourth copy of it is a matter of time.
# If it goes back, it goes back as a PROPERTY that reads the slide, not as a
# field written at construction.
#
# Read `sampler.summary()` in the output before touching anything: it prints
# `cand -> gate -> took` per level, so which level fell short is a number here
# rather than a guess.

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
