#!/bin/bash
#SBATCH --job-name=RefStore               # Job name -> log/<name>
#SBATCH --partition=normal2               # Partition
#SBATCH --time=06:00:00                   # mask segmentation dominates
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # one card: encode + HEST segmentation
#SBATCH --cpus-per-task=8                 # openslide reads
#SBATCH --mem=400G                         # one level's tiles in flight
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/RefStore                 # STDOUT
#SBATCH -e /work/u26130998/log/RefStore                 # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# ---------------- Build the stage-1 reference under quota ---------------------
#
# 1000 tiles per level per slide, chosen against an explicit background quota
# rather than uniformly, and stored with the reason each one is there.
#
# Read the pre-flight block FIRST. It is produced before any tile is read and it
# answers, per level: how many grid positions exist, how the background fraction
# is distributed over them, which percentile each fixed threshold lands on, what
# each bucket wants against what it can have, and whether the level will fall
# short. A level that will abandon says so there, in milliseconds, instead of an
# hour into encoding.
#
# The percentile line is the one to watch. A tile's footprint grows with ds, so
# at a deep level a 256 px tile covers half a millimetre and almost always
# contains background: the same "<15% white" rule that selects two thirds of the
# grid at level 0 may select a tenth of it at level 2. When that happens the
# report shows it as a percentile, not merely as a failure, and the choice is
# between lowering the target for that level and defining the buckets as
# quantiles of each level's own distribution.
#
# What the pre-flight cannot cover: holes and unscanned canvas. Whether a tile
# was photographed is a property of (location, level) -- a corrupt stored tile at
# level 0 says nothing about level 3 -- so it only surfaces on read. Every tile
# is checked with read_region_valid at encode time and rejects are replaced from
# their OWN bucket, because holes come in contiguous patches and topping up from
# anywhere would move the background mix the quotas exist to hold.
#
# --pooling cls keeps one vector per tile, about 61 MB per slide across ten
# levels. --pooling tokens keeps all 197 and costs about 6 GB. Encoding time is
# identical; the difference is disk and every later read.

WSIS=(
  /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs
)

# Spelled once, and passed to BOTH invocations. The store root and the report
# directory both carry it, so a dry run that names one encoder and a build that
# names another would describe a directory the build never wrote to.
#
# conch_vit needs `HEAD=trunk`: this writes POOLED features, and pooling needs a
# token axis that CONCH's default attentional pooler does not have.
ENCODER="${ENCODER:-gigapath}"
HEAD="${HEAD:-}"
TAG="$ENCODER${HEAD:+_$HEAD}"
ENC_FLAG="--encoder $ENCODER${HEAD:+ --head $HEAD}"

echo "======== dry run: quotas only, no tile is read ========"
python utilities/cli/build_reference_store.py "${WSIS[@]}" \
  $ENC_FLAG \
  --dry-run

echo ""
echo "======== build ========"
python utilities/cli/build_reference_store.py "${WSIS[@]}" \
  $ENC_FLAG \
  --pooling tokens

echo ""
echo "======== done ========"
echo "  result/cache/features/$TAG/*.safetensors"
echo ""
echo "  Every tile carries why it is there: white_frac, bucket, origin"
echo "  (grid / displaced / inherited), parent_x/parent_y, inherit_id, valid_frac."
echo "  inherit_id is the same number at every level for one physical location,"
echo "  so cross-level correspondence is an index lookup rather than a search."
echo ""
echo "  Inspect with:  python utilities/cli/inspect_feature_store.py \\"
echo "                        result/cache/features/$TAG"
echo ""
echo "  The encoder names a directory of its own, and the readers glob one"
echo "  level without recursing -- so point them at the encoder, not the cache."
