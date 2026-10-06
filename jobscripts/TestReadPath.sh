#!/bin/bash
#SBATCH --job-name=TestReadPath            # -> log/%x
#SBATCH --partition=dev                    # tests go to dev
#SBATCH --time=01:30:00                    # the slide reads and the renders dominate
#SBATCH --account=MST114560                # Account
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gpus-per-node=1                  # GPUs per node (不要設0)
#SBATCH --cpus-per-task=4                  # read_grid's workers
#SBATCH --mem=32G                          # one region read whole, for the reference
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH -o /work/u26130998/log/%x          # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x          # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  Everything that gets pixels off a slide, position to photo. One jobscript
#  since 2026-10-03 (CameraTest.sh merged in): the reader and the renderer
#  are now one path, so their tests run on the same slides together.
#
#  No slide, run once:
#    test_read_geometry   the level rule, the level px, the read rectangle; the
#                         sampler reserves exactly what is read (a decoy each)
#    test_tile_sampler    WHERE: lattice, richness, overlap, inherit, cache,
#                         and the rectangular-camera fov section
#
#  Per slide:
#    test_slide_reader    SlideReader.read against read_region_rgb, and
#                         read_grid against the whole-region read (WsiTissues-
#                         Container's cut), at each of LEVELS
#    test_camera          Render: output_to_level0 against pixels (map), two
#                         seeds bit-identical (seed), the augment rewrites
#                         against the legacy bodies (augment, writes a CSV)
#    test_fov_supply      FovSupply against the frozen old one (equiv), then
#                         reproducibility, a draw passed in, a ladder (supply)
#
#  Two slides, two formats, on purpose: BRACS is SVS and steps 4x per level
#  at non-integer ds (4.00003, 16.002 -- the one-read-per-region path), Ki67
#  is MIRAX and steps 2x. A test on one format leaves the other's level
#  arithmetic unexercised.
#
#    sbatch jobscripts/TestReadPath.sh
#    SLIDES="BRACS_1228" LEVELS=0 sbatch jobscripts/TestReadPath.sh
#    ONLY="map seed" sbatch jobscripts/TestReadPath.sh      # test_camera sections
#    NO_SLIDES=1 sbatch jobscripts/TestReadPath.sh          # the slide-free tests only
#
#  Output: the log; test_camera's augment section writes
#  result/<job>/augment_equivalence_<slide>.csv. Nothing goes to any cache.
# =============================================================================
SLIDES="${SLIDES:-BRACS_1228 S1104233,G7E,110208}"
LEVELS="${LEVELS:-0 1}"
CAMERA_LEVEL="${CAMERA_LEVEL:-1}"
SEED="${SEED:-0}"
SHOTS="${SHOTS:-5}"          # test_camera augment section
ONLY="${ONLY:-}"             # test_camera sections: "map seed augment"; empty = all
[ "${NO_SLIDES:-0}" = "1" ] && SLIDES=""
CAMERA_ARGS=()
[ -n "$ONLY" ] && CAMERA_ARGS+=(--only $ONLY)

echo "======== TestReadPath ========"
echo "  slides  $SLIDES"
echo "  levels  $LEVELS (slide reader)   $CAMERA_LEVEL (camera, FovSupply)"
echo "  seed $SEED   shots $SHOTS   camera sections ${ONLY:-all}"
status=0
run() {
    echo ""
    echo "-------- $* --------"
    python "$@"
    rc=$?
    [ $rc -ne 0 ] && status=$rc
}
run utilities/test_modules/test_read_geometry.py
run utilities/test_modules/test_tile_sampler.py
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
    echo "======== $name ========"
    echo "  $path"
    for lv in $LEVELS; do
        run utilities/test_modules/test_slide_reader.py --wsi "$path" --level "$lv"
    done
    run utilities/test_modules/test_camera.py --wsi "$path" --level "$CAMERA_LEVEL" \
        --seed "$SEED" --shots "$SHOTS" "${CAMERA_ARGS[@]}"
    run utilities/test_modules/test_fov_supply.py --wsi "$path" --level "$CAMERA_LEVEL"
done
echo ""
echo "======== done (exit $status) ========"
exit $status
