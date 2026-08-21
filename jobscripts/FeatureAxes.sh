#!/bin/bash
#SBATCH --job-name=FeatureAxes            # Job name -> log/<name>, result/<name>
#SBATCH --partition=normal2               # Partition
#SBATCH --time=02:00:00                   # seven slides; eigh is seconds each
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0) -- unused here
#SBATCH --cpus-per-task=8                 # covariance and eigh
#SBATCH --mem=64G                         # one token store in flight per level
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/FeatureAxes              # STDOUT
#SBATCH -e /work/u26130998/log/FeatureAxes              # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# Runs write outside the checkout; see utilities/test_modules/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# ---------------- What is in the feature space, and where is mpp in it? -------
#
# Every stage-1 experiment so far treated the 1536 numbers per tile as a black
# box and asked whether some method estimates mpp better. Three answers came
# back closed. This asks about the box instead.
#
# Five numbers, each with its own null:
#
#   n_significant   components beating a parallel-analysis null -- the same data
#                   with each dimension shuffled on its own, which keeps every
#                   marginal and destroys the correlation between them. That is
#                   a count, where a scree elbow is a judgement.
#
#   var_between     the share of the variance that is level rather than tissue.
#
#   corr_logmpp     how much each component tracks scale.
#
#   corr_white      how much it tracks BACKGROUND instead. The median grid
#                   position inside a tissue region is 72% background and 46%
#                   are pure background, so "this component is mpp" and "this
#                   component is emptiness" are easy to confuse and only one is
#                   worth following. Needs a store from the quota sampler; older
#                   stores have no white_frac and the run says so.
#
#   r2(r)           how much of log mpp the first r components explain, against
#                   r RANDOM directions as the decoy. If random does as well,
#                   any r dimensions would, and the subspace is not special.
#
# The consequence, if r turns out small: a reference bank is a few thousand
# tiles in 1536 dimensions, which is a badly conditioned place for a
# nearest-neighbour search. The same KNN in three dimensions would be far better
# posed -- a bigger change than any arithmetic on top of the full space.
#
# axes_extremes.csv is for looking. A direction means nothing until something is
# seen along it: the correlations say whether a component tracks scale or
# emptiness, only the images say whether it is fat against stroma. Every field
# needed to read the tile back is in the row:
#
#     SafeSlide(wsi_path).read_region_rgb((x, y), level, (tile_size, tile_size))

# All seven. Each is analysed on its own -- nothing here averages across
# slides, because whether PC1 means the same thing on two slides is a separate
# question and answering it by accident would be worse than not answering it.
SLIDES=(
  BRACS_1228 BRACS_1476 BRACS_1936
  "S1104233,G7E,110208" "S1104360,G7E,110208"
  "S1137178,G7E,110926" "S1151088,G7E,111220"
)

python utilities/bench_modules/bench_feature_axes.py "${SLIDES[@]}" \
  --stores "$RESULT_ROOT"/cache/reference_features \
  --pooling cls

echo ""
echo "======== done ========"
echo "  result/FeatureAxes/axes_summary.csv      the five numbers, one row"
echo "  result/FeatureAxes/axes_components.csv   per component"
echo "  result/FeatureAxes/axes_r2_decoy.csv     r2 against random subspaces"
echo "  result/FeatureAxes/axes_projection.csv   per tile, for plotting offline"
echo "  result/FeatureAxes/axes_extremes.csv     the tiles at each axis end"
echo ""
echo "  Four figures, all derivable from those CSVs:"
echo "    axes_scree.png        how many directions are real"
echo "    axes_correlation.png  scale against emptiness, per component"
echo "    axes_r2.png           how few dimensions hold the scale"
echo "    axes_scatter.png      the same tiles read two ways"
