#!/bin/bash
#SBATCH --job-name=RealTest              # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                # CPU cores per task
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH --array=0-9                      # One task per slide (10 slides)
#SBATCH -o ./log/RealTest_%a             # STDOUT
#SBATCH -e ./log/RealTest_%a             # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Locate real microscope photos in their own WSI ----------------
# No ground-truth position exists for these photos -- the sidecar .json files
# hold Ki-67 nucleus polygons in photo coordinates, not slide positions. Each
# photo is judged on
#   1. SIFT+RANSAC inlier count  (hundreds = correct, single digits = wrong)
#   2. the warped / blend / checkerboard panels, which overlap the WSI back
#      onto the photo and make a few px of misalignment visible
#
# ONE ARRAY TASK PER SLIDE, not per photo. LocaScopePipeline.build() segments
# the whole slide and encodes a KNN reference bank, and the first shot routed to
# a level makes that level's retriever encode every tile of the slide. All of it
# is per-WSI. locate_photo.py therefore takes the whole photo FOLDER and loops
# inside one process, so that work happens once instead of 170-478 times.
#
# Layout: photos live in <slide-stem>_ki67/ next to the matching .mrxs, e.g.
#   S1104360_ki67/1.bmp   <->   S1104360,G7E,110208.mrxs
# The pairing is derived from the folder name, so adding a slide to the dataset
# needs no edit here beyond widening --array.
#
# To run one slide only:   sbatch --array=5 realtest.sh
# To list the mapping:     ls -d /work/u26130998/datasets/Ki67/*_ki67 | sort | cat -n

DATA=/work/u26130998/datasets/Ki67
OUT=result/RealTest

# Photos per slide vary 71..478; every task fits well inside the 24h limit.
# 'all' draws both figures for every photo, sorted into <out>/success/ and
# <out>/fail/ by the SIFT verdict. Measured at ~2.3 MB per photo, so the whole
# 2296-photo set is roughly 5 GB. FIGURE_LIMIT applies to 'sample' only.
FIGURES=all           # none | fail | sample | all
FIGURE_LIMIT=5
LIMIT=0               # 0 = every photo in the folder
STRIDE=1

# Figures need the per-photo pipeline result, which only exists while the photo
# is being run -- they cannot be reconstructed from predictions.csv. So a run
# whose point is the figures has to redo every photo, and resume (which skips
# anything already in the CSV) would make it a no-op.
# COST: this also disables restart-where-it-stopped. A task killed at photo 400
# of 478 starts again at 1. Set to 0 once the figures exist.
NO_RESUME=1
RESUME_FLAG=""
[ "$NO_RESUME" = "1" ] && RESUME_FLAG="--no-resume"

mapfile -t DIRS < <(ls -d "$DATA"/*_ki67 | sort)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#DIRS[@]}" ]; then
  echo "[skip] array index $IDX >= ${#DIRS[@]} slides"
  exit 0
fi

PHOTO_DIR=${DIRS[$IDX]}
STEM=$(basename "$PHOTO_DIR")
STEM=${STEM%_ki67}

# The .mrxs carries scanner suffixes the photo folder does not, e.g.
# S1103037_ki67 -> "S1103037,G7E,110122.mrxs", so match on the stem prefix.
WSI=$(ls "$DATA/$STEM",*.mrxs 2>/dev/null | head -1)
if [ -z "$WSI" ]; then
  echo "[skip] no .mrxs matching $STEM in $DATA"
  exit 0
fi

echo "======== [$IDX] $(basename "$PHOTO_DIR")  ->  $(basename "$WSI") ========"
echo "photos: $(ls "$PHOTO_DIR"/*.bmp 2>/dev/null | wc -l)"

python utilities/cli/locate_photo.py \
  "$PHOTO_DIR" \
  "$WSI" \
  --out "$OUT/$(basename "$PHOTO_DIR")" \
  --figures $FIGURES --figure-limit $FIGURE_LIMIT \
  --limit $LIMIT --stride $STRIDE $RESUME_FLAG \
  --precision fp16 --batch-size 1024

echo ""
echo "======== done -> $OUT/$(basename "$PHOTO_DIR") ========"

# Merge every slide's CSV afterwards (run once, after the array finishes):
#   awk 'FNR==1 && NR!=1 {next} {print}' result/RealTest/*/predictions.csv \
#       > result/RealTest/all_predictions.csv
