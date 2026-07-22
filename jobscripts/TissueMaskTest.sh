#!/bin/bash
#SBATCH --job-name=TissueMaskTest         # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (do not set 0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/TissueMaskTest           # STDOUT
#SBATCH -e ./log/TissueMaskTest           # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

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
# | tiling max_pixels sweep   | 16M,4M,1M       | tiling budget sweep list               | --tiling-max-pixels   |
# | tiling overlap            | 128             | per-tile margin px                     | --tiling-overlap      |
# | hest   mask ds            | 64              | HEST-only mask ds                      | --hest-ds             |
# | hest   max_pixels         | 4M              | HEST tile budget (hest/ops/sweep)      | --hest-max-pixels     |
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
#   --hest / --no-hest        : HEST DL seg used by hest-only / ops / sweep / tiling
#   --region-index / --no-region-index : show region-idx labels on bboxes
#
# =============================================================================

# ---------------- Parameters ----------------
WSI=/work/u26130998/datasets/Ki67/S1151088,G7E,111220.mrxs

# --- Otsu baseline ---
OTSU_DS=32                          # Otsu mask ds

# --- Sweep matrix ---
SWEEP_DS="4,16,32,64,128"           # HSV can handle all; for HEST prune low values
SWEEP_LEVEL="-1,-2,-3"

# --- Ops pipeline (filter_regions / filter_patchable / merge_overlapping) ---
OPS_DS=4                            # ops baseline mask ds
OPS_MIN_RATIO=0.01                  # filter_regions cutoff
OPS_PATCH_TILE=256                  # filter_patchable tile_size
OPS_PATCH_DS=64.0                   # filter_patchable target level ds

# --- Tiling (adaptive halving) ---
TILING_DS=64                        # tiling seam/grid mask ds
TILING_MAX_PIXELS="16M,4M,1M"       # tiling sweep budgets
TILING_OVERLAP=128                  # per-tile margin px

# --- HEST ---
HEST_DS=64                          # HEST-only mask ds (bigger ds -> smaller image, safer)
HEST_MAX_PIXELS=4M                  # HEST tile budget for hest-only / ops / sweep

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
  $TILING --tiling-ds  $TILING_DS        --tiling-max-pixels "$TILING_MAX_PIXELS" \
          --tiling-overlap $TILING_OVERLAP \
  $HEST   --hest-ds    $HEST_DS          --hest-max-pixels $HEST_MAX_PIXELS \
  --per-row            $PER_ROW \
  --dpi                $DPI \
  --figure-scale       "$FIGURE_SCALE" \
  $REGION_IDX \
  --bbox-lw            $BBOX_LW
