#!/bin/bash
#SBATCH --job-name=TestTissueMask         # Job name
#SBATCH --partition=dev               # Partition
#SBATCH --time=02:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (do not set 0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x           # STDOUT
#SBATCH -e /work/u26130998/log/%x           # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# =============================================================================
# utilities/test_modules/test_uni2_pca_seg.py without a slide (config,
# helpers), then utilities/test_modules/test_tissue_mask.py: the synthetic
# tests first (they must pass before any figure is drawn), then one figure of
# what each --seg recipe makes of $WSI.
#
#   SEG          recipes, one mask + backdrop each (MASK_RECIPES). uni2_pca
#                fits a PCA on --pca-fit-tiles tiles (PCA_FIT_TILES, 1000 =
#                the production mask; a different value is a different
#                seg_id) and costs minutes.
#   OPS          raw -> filtered -> merged -> patchable, each on its own view,
#                for every recipe -- min_ratio is a fraction of the mask AREA,
#                so the same cutoff keeps different regions at mask ds 14 and 4
#   SWEEP        the first plane recipe at each SWEEP_DS
#   TILING       the first plane recipe read whole against read tiled at each
#                budget, with the tile grid drawn
# =============================================================================

# WSI_NAME is resolved through AccessDatasets.locate, which lists every known
# name if it is not one. Ki67 names contain commas, so do NOT put WSI_NAME=...
# inside --export=... (comma splits). Export in the shell instead:
#   WSI_NAME='S1104043,G7E,110207' SEG='hsv hest' \
#     sbatch --export=ALL,WSI_NAME,SEG jobscripts/TestTissueMask.sh
WSI_NAME="${WSI_NAME:-S1104233,G7E,110208}"
WSI=$(python -c 'import sys; sys.path.insert(0, "utilities"); from AccessDatasets import locate; print(locate(sys.argv[1]).path)' "$WSI_NAME") \
  || { echo "could not resolve $WSI_NAME"; exit 1; }
SEG="${SEG:-hsv hest uni2_pca}"
# 1000 = Uni2PcaSegConfig's own fit_tiles, i.e. the production mask. This used
# to be 200 (2026-09-24 and earlier), and 200 is NOT the production mask: on the
# default Ki67 slide it drew a 33 per cent tissue mask in tile-sized blocks
# (Ki67 is 3.5-9.2 per cent; hest reads 6.5), so a figure made at 200 says
# nothing about what the cache holds. 1000 costs ~6 min more, once.
PCA_FIT_TILES="${PCA_FIT_TILES:-1000}"

OPS="${OPS:---ops}"                       # alternative: --no-ops
OPS_PATCH_TILE="${OPS_PATCH_TILE:-256}"
OPS_PATCH_DS="${OPS_PATCH_DS:-1.0}"
SWEEP="${SWEEP:---sweep}"                 # alternative: --no-sweep
SWEEP_DS="${SWEEP_DS:-4,16,32,64}"
TILING="${TILING:---tiling}"              # alternative: --no-tiling
TILING_DS="${TILING_DS:-64}"
SEG_CHUNK_PX_SWEEP="${SEG_CHUNK_PX_SWEEP:-16M,4M,1M}"
TILING_OVERLAP="${TILING_OVERLAP:-128}"

PER_ROW="${PER_ROW:-4}"
DPI="${DPI:-600}"
FIGURE_SCALE="${FIGURE_SCALE:-7,5}"
BBOX_LW="${BBOX_LW:-0.5}"
REGION_IDX="${REGION_IDX:---no-region-index}"   # alternative: --region-index

# ---------------- Run ----------------
# test_uni2_pca_seg's slide-free tier first: config, helpers. Seconds.
echo "======== test_uni2_pca_seg ========"
python utilities/test_modules/test_uni2_pca_seg.py
status=$?
echo ""
echo "======== test_tissue_mask ========"
python utilities/test_modules/test_tissue_mask.py \
  --wsi "$WSI" \
  --seg $SEG --pca-fit-tiles $PCA_FIT_TILES \
  $OPS    --ops-patch-tile $OPS_PATCH_TILE --ops-patch-ds $OPS_PATCH_DS \
  $SWEEP  --sweep-ds "$SWEEP_DS" \
  $TILING --tiling-ds $TILING_DS --seg-chunk-px-sweep "$SEG_CHUNK_PX_SWEEP" \
          --tiling-overlap $TILING_OVERLAP \
  --per-row $PER_ROW --dpi $DPI --figure-scale "$FIGURE_SCALE" \
  $REGION_IDX --bbox-lw $BBOX_LW
rc=$?
[ $rc -ne 0 ] && status=$rc
echo "======== done (exit $status) ========"
exit $status
