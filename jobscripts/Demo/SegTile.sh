#!/bin/bash
#SBATCH --job-name=SegTile                 # -> log/%x, result/%x/
#SBATCH --partition=normal                 # Partition
#SBATCH --time=00:30:00                    # a few tiles; +uni2 fit if used (3.5-6 min/slide)
#SBATCH --account=MST114560                # Account
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gpus-per-node=1                  # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                  # uni2 fit's tile loader workers
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH -o /work/u26130998/log/%x          # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x          # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  utilities/cli/demo/seg_tile.py -- one (or more) tile in, ONE COMBINED
#  quick-look figure per tile out (one panel per METHOD). No MaskStore write,
#  no whole-slide read except for a uni2 fit -- see seg_tile.py's own module
#  docstring for why that method alone needs a slide, and how it recovers
#  which slide a tile came from automatically from the tile's own folder
#  name, per distinct slide in the batch (WSI_NAME below is only a fallback).
# =============================================================================

# One or more tile image FILEs, SPACE-separated, e.g.
# IMAGE="a.png b.png c.png" -- required.
IMAGE="${IMAGE:?set IMAGE, e.g. IMAGE=/path/to/tile.png}"

# One or more of 'uni2' 'hest' 'hsv' 'otsu', SPACE-separated, e.g.
# METHOD="uni2 hest hsv otsu" -- see seg_tile.py's own "METHOD CHOICES". Each
# --image gets ONE figure with a panel per method listed here.
METHOD="${METHOD:-uni2 hest hsv otsu}"

# FALLBACK slide for a uni2 fit, used only when a tile's own folder name does
# not encode one (seg_tile.py's own `_recover_slide`) -- most batches from
# the standard tile-cache layout do not need this set at all.
WSI_NAME="${WSI_NAME:-}"
FIT_TILES="${FIT_TILES:-1000}"
COMPONENTS="${COMPONENTS:-16}"
BACKGROUND_THRESHOLD="${BACKGROUND_THRESHOLD:-0.5}"
LARGER_PCA_AS_FG="${LARGER_PCA_AS_FG:-1}"
LARGER_PCA_AS_FG_FLAG="--larger-pca-as-fg"
[ "$LARGER_PCA_AS_FG" = 0 ] && LARGER_PCA_AS_FG_FLAG="--no-larger-pca-as-fg"
WORKERS="${WORKERS:-8}"

DEVICE="${DEVICE:-cuda}"

# Unset -> result/SegTile/seg_<method>__<tile stem>.png
OUT="${OUT:-}"

echo "======== SegTile ========"
echo "  methods $METHOD"
echo "  images  $IMAGE"
echo "  wsi     ${WSI_NAME:-<none -- fallback only, per-tile auto-recovery tried first>}"

python utilities/cli/demo/seg_tile.py \
  --image $IMAGE \
  --method $METHOD \
  ${WSI_NAME:+--wsi-name "$WSI_NAME"} \
  --fit-tiles "$FIT_TILES" \
  --components "$COMPONENTS" \
  --background-threshold "$BACKGROUND_THRESHOLD" \
  $LARGER_PCA_AS_FG_FLAG \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  ${OUT:+--out "$OUT"}
status=$?

echo ""
echo "======== done (exit $status) ========"
echo "  figures -> result/SegTile/seg_<method1>-<method2>-...__<tile stem>.png (unless OUT was set)"

exit $status
