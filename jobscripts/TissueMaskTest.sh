#!/bin/bash
#SBATCH --job-name=TissueMaskTest         # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (do not set 0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/TissueMaskTest           # STDOUT
#SBATCH -e /work/u26130998/log/TissueMaskTest           # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# =============================================================================
# CLI knob inventory
# =============================================================================
#
# +---------------------------+-----------------+----------------------------------------+-----------------------+
# | Where                     | Default         | Meaning                                | CLI arg               |
# +---------------------------+-----------------+----------------------------------------+-----------------------+
# | otsu   mask ds            | 32              | Otsu baseline mask ds                  | --otsu-ds             |
# | sweep  ds list            | 4,16,32,64,128  | mask ds for --sweep matrix             | --sweep-ds            |
# | sweep  level list         | -1,-2,-3        | levels for --sweep matrix              | --sweep-level         |
# | ops    mask ds            | 32              | ops pipeline baseline mask ds          | --ops-ds              |
# | ops    min_ratio          | 0.05            | filter_regions cutoff                  | --ops-min-ratio       |
# | ops    patch tile         | 256             | filter_patchable tile_size             | --ops-patch-tile      |
# | ops    patch ds           | 4.0             | filter_patchable target level ds       | --ops-patch-ds        |
# | tiling mask ds            | 32              | tiling seam/grid mask ds               | --tiling-ds           |
# | tiling seg_chunk_px sweep   | 16M,4M,1M       | tiling budget sweep list               | --seg-chunk-px-sweep   |
# | tiling overlap            | 128             | per-tile margin px                     | --tiling-overlap      |
# | pca    fit tiles          | 200             | Uni2PcaSeg fit sample (level 0, no ds) | --pca-fit-tiles       |
# | hest   mask ds            | 64              | HEST-only mask ds                      | --hest-ds             |
# | hest   seg_chunk_px         | 4M              | HEST tile budget (hest/ops/sweep)      | --seg-chunk-px     |
# | vis    per row            | 4               | panels per figure row                  | --per-row             |
# | vis    dpi                | 600             | savefig dpi                            | --dpi                 |
# | vis    figure scale       | 7,5             | (col-scale,row-scale) for figsize      | --figure-scale        |
# | vis    bbox linewidth     | 1.5             | region bbox line width                 | --bbox-lw             |
# +---------------------------+-----------------+----------------------------------------+-----------------------+
#
# Boolean toggles (argparse BooleanOptionalAction):
#   --sweep / --no-sweep      : ds/level matrix panel
#   --ops / --no-ops          : filter/filter/merge pipeline
#   --tiling / --no-tiling    : tiled-inference test (seam + grid + budget sweep)
#   --pca  / --no-pca         : Uni2PcaSegFunc, the segmenter the pre-tile
#                               corpus was cut with. Adds a mask+thumb pair to
#                               row 0 AND a second set of ops rows -- the same
#                               filter_regions / merge_overlapping /
#                               filter_patchable, run on the PCA mask, beside
#                               the ones run on the HSV or HEST mask.
#
#                               It does NOT feed sweep or tiling. Those two vary
#                               a `method=` handed to from_wsi (ds, level,
#                               seg_chunk_px), and this segmenter has no such
#                               shape: it fits a PCA across the WHOLE slide
#                               first, reads at level 0, and derives its own
#                               mask ds. There is nothing there to sweep.
#
#                               The ops comparison is the reason to look:
#                               min_ratio is a fraction of the mask AREA, and a
#                               mask_ds 14 PCA plane has about five times the
#                               pixels of a mask_ds 32 HSV one, so the same
#                               cutoff keeps a different set of regions.
#   --hest / --no-hest        : HEST DL seg used by hest-only / ops / sweep / tiling
#   --region-index / --no-region-index : show region-idx labels on bboxes
#
# =============================================================================

# ---------------- Parameters ----------------
WSI=/work/u26130998/datasets/Ki67/S1103037,G7E,110122.mrxs

# --- Otsu baseline ---
OTSU_DS=32                          # Otsu mask ds

# --- Sweep matrix ---
SWEEP_DS="4,16,32,64,128"           # HSV can handle all; for HEST prune low values
SWEEP_LEVEL="-1,-2,-3"

# --- Ops pipeline (filter_regions / filter_patchable / merge_overlapping) ---
OPS_DS=4                            # ops baseline mask ds
OPS_MIN_RATIO=0.01                  # filter_regions cutoff
OPS_PATCH_TILE=256                  # filter_patchable tile_size
OPS_PATCH_DS=1.0                    # filter_patchable target level ds

# --- Tiling (adaptive halving) ---
TILING_DS=64                        # tiling seam/grid mask ds
SEG_CHUNK_PX_SWEEP="16M,4M,1M"       # tiling sweep budgets
TILING_OVERLAP=128                  # per-tile margin px

# --- PCA (aiNNModel/Uni2PcaSegFunc.py) ---
#
# The segmenter the pre-tile corpus was actually cut with, and the only one
# here that does NOT go through `from_wsi(method=...)`: it fits a PCA across
# the whole slide before it can threshold any part of it, so `mask_wsi(wsi)`
# is its entry point and `from_mask` is the half that follows. Every other
# method in this script is an image-in-mask-out callable.
#
# It reads at level 0 and derives its own mask ds from the patch grid
# (Uni2PcaSegFunc.LEVEL), so there is no --pca-ds to pass. There was a plane_ds
# and then a level argument; both are gone, and passing one is what made the
# 2026-08-26 run exit 1.
#
# 200 rather than the config default of 1000 keeps the fit to minutes. It is a
# hashed field, so a run at 200 and a production run at 1000 correctly get
# different identity ids -- this figure is not the mask in result/cache/masks/.
PCA="--pca"                       # alternative: --no-pca
PCA_FIT_TILES=200

# --- HEST ---
HEST_DS=4                           # HEST-only mask ds (bigger ds -> smaller image, safer)
SEG_CHUNK_PX=4M                  # HEST tile budget for hest-only / ops / sweep

# --- Visualization ---
PER_ROW=4                           # panels per figure row
DPI=600                             # savefig dpi
FIGURE_SCALE="7,5"                  # (col-scale, row-scale) for figsize
BBOX_LW=0.5

# Boolean sub-test toggles (argparse BooleanOptionalAction).
# Set each to the enable form; comment shows the alternative.
SWEEP="--sweep"                   # alternative: --no-sweep      (skip ds/level matrix)
OPS="--ops"                       # alternative: --no-ops        (skip filter/merge pipeline)
TILING="--tiling"                 # alternative: --no-tiling     (skip tiled inference test)
HEST="--hest"                     # alternative: --no-hest       (HSV only, no GPU seg)
REGION_IDX="--no-region-index"    # alternative: --region-index  (show region index labels)

# ---------------- Run ----------------
python utilities/test_modules/test_tissues_regions_mask.py \
  --wsi                "$WSI" \
  --otsu-ds            $OTSU_DS \
  $SWEEP  --sweep-ds   "$SWEEP_DS"       --sweep-level="$SWEEP_LEVEL" \
  $OPS    --ops-ds     $OPS_DS           --ops-min-ratio $OPS_MIN_RATIO \
          --ops-patch-tile $OPS_PATCH_TILE --ops-patch-ds $OPS_PATCH_DS \
  $TILING --tiling-ds  $TILING_DS        --seg-chunk-px-sweep "$SEG_CHUNK_PX_SWEEP" \
          --tiling-overlap $TILING_OVERLAP \
  $PCA    --pca-fit-tiles $PCA_FIT_TILES \
  $HEST   --hest-ds    $HEST_DS          --seg-chunk-px $SEG_CHUNK_PX \
  --per-row            $PER_ROW \
  --dpi                $DPI \
  --figure-scale       "$FIGURE_SCALE" \
  $REGION_IDX \
  --bbox-lw            $BBOX_LW
