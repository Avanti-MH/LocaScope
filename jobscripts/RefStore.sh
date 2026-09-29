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


# ---------------- Build the stage-1 reference ---------------------------------
#
# 1000 tiles per level per slide, drawn by TileSampler under the reference-bank
# richness contract (KnnEstMpp.REFERENCE_BANK_RICHNESS), one native rung per
# pyramid level, half of each level carried by chains -- the same level-0 centre
# at every level -- and stored with the reason each tile is there.
#
# The dry run prints each level's supply before any tile is read: how many
# candidates the lattice offers, how many the richness gate admits, and per
# bucket what the pool has. A level that will come in thin says so there, in
# seconds, instead of an hour into encoding.
#
# What the dry run cannot cover: holes. Whether a tile was photographed is a
# property of (location, level), so it only surfaces on read; a tile below
# --min-valid is dropped (not replaced), and refstore_levels.csv says how many.
#
# --pooling cls keeps one vector per tile, about 61 MB per slide; --pooling
# tokens keeps all 197 and costs about 6 GB.

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

echo "======== dry run: supply only, no tile is read ========"
python utilities/cli/build_cache/build_reference_store.py "${WSIS[@]}" \
  $ENC_FLAG \
  --dry-run

echo ""
echo "======== build ========"
python utilities/cli/build_cache/build_reference_store.py "${WSIS[@]}" \
  $ENC_FLAG \
  --pooling tokens

echo ""
echo "======== done ========"
echo "  result/cache/RefStore_features/$TAG/<seg_id>/<slide>/<region_id>/<draw>/"
echo "  <draw> is the line the build printed; readers take it as --draw."
echo ""
echo "  Every tile carries why it is there: white_frac, bucket, origin"
echo "  (grid / jitter / inherit), parent_x/parent_y, inherit_id, valid_frac."
echo "  inherit_id is the same number at every level for one physical location,"
echo "  so cross-level correspondence is an index lookup rather than a search."
echo ""
echo "  Inspect with:  python utilities/cli/inspect_cache_store/inspect_feature_store.py \\"
echo "                        result/cache/RefStore_features/$TAG"
