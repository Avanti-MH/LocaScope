#!/bin/bash
#SBATCH --job-name=SubspaceKnn             # Job name -> log/<name>, result/<name>
#SBATCH --partition=normal2                # Partition
#SBATCH --time=03:00:00                    # seven slides; SVD and cosine only
#SBATCH --account=MST114560                # Account
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gpus-per-node=1                  # GPUs per node (不要設0) -- unused here
#SBATCH --cpus-per-task=8                  # the SVD is the whole cost
#SBATCH --mem=64G                          # one token store in flight per level
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH -o /work/u26130998/log/SubspaceKnn               # STDOUT
#SBATCH -e /work/u26130998/log/SubspaceKnn               # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# Runs write outside the checkout; see utilities/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# ---------------- Step 5: is the scale subspace worth anything? ---------------
#
# Steps 1-2 said mpp lives on 2 to 10 of the 1536 axes. That is a correlation.
# This asks the only question with a consequence: does a nearest-neighbour
# search run in those axes classify better than the one production runs today?
#
# The first run on BRACS_1228 settled two of the three questions and left one:
#
#   ranking components by |corr with log mpp| is dead. `variance` beat `scale`
#     at every r, because after projecting and renormalising every kept
#     direction votes equally -- so the rule's second pick, variance ratio
#     0.038, carried the same weight as PC1 at 0.228, and its third pick
#     correlated 0.618 with BACKGROUND.
#
#   the premise was wrong. "1536 dimensions is badly conditioned for a
#     nearest-neighbour search" measures 0.990 on clean tiles.
#
#   the dimension cut was never tested fairly. On arm B the loss came from
#     CENTRING: all 1530 components kept, and it still fell 0.680 -> 0.400.
#
# So the settings now isolate that:
#
#   production   x                 uncentred, full dimension. Pinned to
#                                  KnnClassifier.predict by a gate.
#   centred      V^T (x - mu)      all components, mean removed.
#   variance     V_r^T (x - mu)    top r by eigenvalue, mean removed.
#   uncentred    V_r^T x           the SAME directions, mean KEPT.  <- the test
#   scale        top r by |corr|   kept for continuity, no longer a candidate.
#   random       random r-dim      the decoy, averaged over 10 draws. One draw
#                                  per r gave 0.681 at r=2 and 0.532 at r=3,
#                                  which is two rolls of a die, not a trend.
#
# variance and uncentred select the SAME directions, so any gap between them is
# the mean removal and can be nothing else.
#
# Two arms:
#
#   A  tile -> tile   no domain gap. Cheap, and a gate: if the subspace loses
#                     here there is nothing to carry into B.
#   B  photo -> tile  the query stores -- FoV renders with colour temperature,
#                     vignetting, distortion, noise and JPEG. Grouped by fov_id
#                     so production's median-of-medians is reproduced per FoV.
#                     This is the arm with the decision in it.
#
# Arm A is split as a contiguous band in x, NOT at random: the reference stores
# hold overlap positions half a tile off the main grid, so a random split puts
# a tile and its 50%-overlapping neighbour on opposite sides and the nearest
# neighbour of a test tile is a near copy of itself. The random split is run
# too -- the gap between them is the size of that inflation, and it is worth
# seeing rather than hiding.
#
# --white-max repeats BOTH arms with every reference tile at or above 15%
# background dropped, then rebalances the levels. The quota sampler's background
# fraction rises with level (median 0.000 at L0, 0.626 at L3 on S1151088), so a
# component can separate levels by detecting emptiness. Arm B was left out of
# this control in the first version because query stores hold no white_frac --
# true, and beside the point: the confound is on the REFERENCE side, which is
# exactly the side that can be filtered.
#
# S1151088 is kept deliberately. Its scale component correlates -0.494 with
# background where the other six are inside +-0.1, so if the method breaks
# anywhere it breaks there -- and "when does this not work" is an answer.

SLIDES=(
  BRACS_1228 BRACS_1476 BRACS_1936
  "S1104233,G7E,110208" "S1104360,G7E,110208"
  "S1137178,G7E,110926" "S1151088,G7E,111220"
)

# One slide, for the run whose only job is to find out whether the gates pass
# and the CSVs come out shaped right. Set it at submission:
#
#     sbatch --export=ALL,ONLY_WSI=BRACS_1228,OUT=result/SubspaceKnn/smoke \
#            jobscripts/SubspaceKnn.sh
#
# BRACS_1228 is the one to smoke: three levels, the scale axis is PC1 with a
# clean 0.018 correlation to background, so if anything reads oddly there it is
# the code and not the slide.
if [ -n "${ONLY_WSI}" ]; then
  SLIDES=("${ONLY_WSI}")
fi

python utilities/bench_modules/bench_subspace_knn.py "${SLIDES[@]}" \
  --stores "$RESULT_ROOT"/cache/reference_features/"${ENCODER:-gigapath}" \
  --pooling cls \
  --per-level "${PER_LEVEL:-1000}" \
  --white-max "${WHITE_MAX:-0.15}" \
  ${OUT:+--out "${OUT}"}

echo ""
echo "======== done ========"
echo "  result/SubspaceKnn/subspace_knn_scores.csv       every setting, every r"
echo "  result/SubspaceKnn/subspace_knn_arm_b_fovs.csv   per FoV: what each"
echo "                                                   setting predicted"
echo "  result/SubspaceKnn/subspace_knn_selected.csv     which components, and"
echo "                                                   what else they track"
echo "  result/SubspaceKnn/subspace_knn_gates.csv        pinned / full_rank /"
echo "                                                   shuffled"
echo "  result/SubspaceKnn/subspace_knn_definitions.csv  every name, and what"
echo "                                                   it computes"
echo ""
echo "  Read the gates FIRST. If any failed, the scores are not evidence:"
echo "    pinned     production must equal KnnClassifier.predict exactly"
echo "    full_rank  uncentred at full rank must equal production exactly --"
echo "               the only check on the projection arithmetic itself"
echo "    shuffled   permuted labels must fall to chance"
echo ""
echo "  Then the verdict, in subspace_knn_accuracy__<slide>.png, right panel:"
echo "    does the uncentred curve reach the production line at small r?"
echo "      yes -> the dimension cut is fine and centring was the whole problem"
echo "      no  -> the cut fails under the domain gap; step 5 closes negative"
echo ""
echo "  subspace_knn_confusion__<slide>.png turns the error direction from an"
echo "  inference into something visible: one column = collapse, a leaning"
echo "  diagonal = a bias."
echo ""
echo "  subspace_knn_vs_baseline__<slide>.png ranks every recipe against"
echo "  production. Read the +w/-l counts next to each dot, not only where the"
echo "  dot sits: +8/-7 and +8/-1 land in the same place on the axis and are a"
echo "  coin flip and a real margin respectively."
