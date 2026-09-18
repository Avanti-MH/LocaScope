#!/bin/bash
#SBATCH --job-name=CameraTest                      # Job name
#SBATCH --partition=normal2                        # Partition
#SBATCH --time=00:40:00                            # a handful of FoV reads per slide
#SBATCH --account=MST114560                        # Account
#SBATCH --nodes=1                                  # Number of nodes
#SBATCH --gpus-per-node=1                          # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                          # openslide decode, single threaded
#SBATCH --ntasks-per-node=1                        # Tasks per node
#SBATCH -o /work/u26130998/log/CameraTest          # STDOUT
#SBATCH -e /work/u26130998/log/CameraTest          # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  utilities/test_modules/test_camera_output_to_level0.py -- query_sim/camera.py
#  had NO jobscript until 2026-09-16, which is most of why two bugs lived in
#  its augment chain for as long as they did. The file tests two things:
#
#  1. THE MAP. `Camera.output_to_level0` answers "where on the slide did this
#     output pixel come from", and the pooling bench looks up every query
#     tile's answer through it. A wrong map does not raise -- it reports that
#     no pooling finds anything, which reads as a finding. So the test cuts a
#     tile, rotates it back, and asks whether the WSI at the computed place
#     looks like that tile MORE than the sign-flipped and one-tile-shifted
#     candidates do. The sign of the inverse rotation is the part most likely
#     to be wrong and is invisible at 0 and 180 degrees, which is why all four
#     of 0/90/180/270 are checked.
#
#  2. THE SEED. Two Cameras built from one seed must produce BIT-IDENTICAL
#     pixels, and one Camera given the same `capture(rng=...)` twice must too.
#     Nothing asserted this before, and that is exactly how `apply_stage_shift`
#     and `apply_noise` went on drawing from the process-global `np.random`:
#     `Camera.__init__` used to re-seed that global, so the ordinary
#     build-then-shoot order came out reproducible anyway and nothing looked
#     wrong until a caller built several Cameras before shooting any of them
#     (training/MppRoutingHead/Datasets.py's `_CameraBank`, one per rung).
#
#  TWO SLIDES, TWO FORMATS, ON PURPOSE. BRACS is SVS and steps 4x per pyramid
#  level; Ki67 is MIRAX and steps 2x (CLAUDE.md, "Pyramid spacing decides how
#  hard stage 1 is"). `QueryFromWSI` picks `chosen_level` by searching for the
#  finest level at or below the requested mpp, so the two pyramids send it down
#  different branches -- a test on one format alone leaves the other's
#  level-choice arithmetic unexercised.
# =============================================================================

# WSI NAMES, not paths: AccessDatasets.locate resolves them, and its own
# docstring names hardcoded dataset paths as the mistake that broke 23 files
# when /work/.../Ki67 became Ki67_with_photo. Override with
#   sbatch --export=ALL,SLIDES="BRACS_1476" jobscripts/CameraTest.sh
SLIDES="${SLIDES:-BRACS_1228 S1104233,G7E,110208}"
LEVEL="${LEVEL:-1}"
SEED="${SEED:-0}"

echo "======== CameraTest ========"
echo "  slides  $SLIDES"
echo "  level   $LEVEL   seed $SEED"

status=0
for name in $SLIDES; do
    # locate() raises KeyError listing every known name if this one is not
    # found, which is a better failure than the test reading the wrong tissue.
    path=$(python - "$name" <<'EOF'
import sys
sys.path.insert(0, 'utilities')
from AccessDatasets import locate
print(locate(sys.argv[1]).path)
EOF
    ) || { echo "  could not resolve $name"; status=1; continue; }

    echo ""
    echo "-------- $name --------"
    echo "  $path"
    python utilities/test_modules/test_camera_output_to_level0.py \
        --wsi "$path" \
        --level "$LEVEL" \
        --seed "$SEED"
    rc=$?
    [ $rc -ne 0 ] && status=$rc
done

echo ""
echo "======== done (exit $status) ========"
exit $status
