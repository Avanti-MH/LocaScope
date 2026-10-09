#!/bin/bash
#SBATCH --job-name=ShowCameraEffects      # -> log/%x, result/%x/
#SBATCH --partition=dev                   # a few seconds of image operations
#SBATCH --time=00:20:00
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0) -- nothing here uses it
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh

# =============================================================================
#  utilities/cli/demo/show_camera_effects.py
#
#  Every effect of the camera's domain gap alone, then all of them together ("ALL, extreme values";
#  and one random draw), on a slide crop and on a checkerboard. Per input, in result/<job>/:
#    camera_effects_<input>.png         the labelled tiles
#    camera_effects_<input>_diff.png    |tile - none| x GAIN, with each tile's mean difference
#    camera_effects_<input>_input.png   what the effects were applied to
#
#    sbatch jobscripts/ShowCameraEffects.sh
#    WSI="" sbatch jobscripts/ShowCameraEffects.sh          # the checkerboard only
#    WSI=S1104233,G7E,110208 SEED=3 sbatch jobscripts/ShowCameraEffects.sh
# =============================================================================
WSI="${WSI-BRACS_1228}"      # a slide name (BRACS is SVS, 4x per level); empty: no slide
LEVEL="${LEVEL:-1}"          # the pyramid level the crop is read at
SEED="${SEED:-0}"            # the crop's position and the random draw
GAIN="${GAIN:-8}"            # the difference sheet is |d| x GAIN

ARGS=(--level "$LEVEL" --seed "$SEED" --gain "$GAIN")
[ -n "$WSI" ] && ARGS+=(--wsi "$WSI")

python -u utilities/cli/demo/show_camera_effects.py "${ARGS[@]}"
