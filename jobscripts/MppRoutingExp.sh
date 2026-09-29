#!/bin/bash
#SBATCH --job-name=MppRoutingExp              # -> log/%x, result/%x/
#SBATCH --partition=normal2                   # Partition
#SBATCH --time=48:00:00                       # sampler_routing reads a live WSI + GPU encode
#SBATCH --account=MST114560                   # Account
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gpus-per-node=1                     # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                     # SVD / eigh / openslide reads
#SBATCH --mem=400G                            # one token store in flight per level
#SBATCH --ntasks-per-node=1                   # Tasks per node
#SBATCH -o /work/u26130998/log/%x             # STDOUT
#SBATCH -e /work/u26130998/log/%x             # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# Runs write outside the checkout; see utilities/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# =============================================================================
#  utilities/bench_modules/bench_mpp_feature_decomposition.py
#  Where does scale live in the feature space, and does routing get better if
#  a candidate uses that instead of the raw 1536-D space?
#
#  Merge of Benchmarks/MppFeatureDecomposition.sh (minus its stage1_compare
#  part, which is Benchmarks/Stage1MppBench.sh now), SubspaceKnn.sh and, in
#  the python, FeatureAxes' analysis. 2026-09-29.
# =============================================================================
#
# PARTS -- one or more of, or 'all':
#   axes             feature_axes_analysis. Purely descriptive: where does mpp
#                   live in the 1536-D feature space. Never runs a KNN.
#                   (jobscripts/FeatureAxes.sh runs the standalone
#                   bench_feature_axes.py, a different file.)
#   subspace_knn     was SubspaceKnn.sh. Does a KNN run in the subspace `axes`
#                   found actually route better? Reads cached FeatureStore
#                   reference/query stores (arm B carries the synthetic camera
#                   domain gap). Rationale below.
#   sampler_routing  Samples its OWN fresh, UNCACHED reference tiles straight
#                   off a live WSI (TileSampler, SuperPoint stageA's own
#                   recipe: DsLadder rungs, n=100/rung), renders a DISJOINT set
#                   of query positions as an actual photo (QueryFromWSI +
#                   simulate_microscope_photo, 1.475 MPixels at 45:32), and
#                   runs the CURRENT production baseline (KnnEstMpp) alongside
#                   SubspaceKnn projected candidates on the same draw, per
#                   rung -- the arena any future routing method drops into
#                   next to the baseline it has to beat.
#
# axes / subspace_knn read TileRetrievalBench.sh's reference stores; the
# default PARTS is sampler_routing only, which needs no store. Pass
# PARTS="all" (or "axes subspace_knn") with DRAW once a store exists.
PARTS="${PARTS:-sampler_routing}"
# DRAW is the `reference draw` line that dump prints (<sampler_id>_<plan>).
DRAW="${DRAW:-}"

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

# ── axes / subspace_knn (read result/cache/<job>_features/, need wsi_stem) ──
# All seven -- each analysed on its own; nothing averages across slides.
SLIDES=(
  BRACS_1228 BRACS_1476 BRACS_1936
  "S1104233,G7E,110208" "S1104360,G7E,110208"
  "S1137178,G7E,110926" "S1151088,G7E,111220"
)
PER_LEVEL="${PER_LEVEL:-1000}"
SEED="${SEED:-42}"
WHITE_MAX="${WHITE_MAX:-0.15}"

# One slide only, e.g. for a smoke run:
#   ONLY_WSI=BRACS_1228 PARTS=subspace_knn sbatch jobscripts/MppRoutingExp.sh
# BRACS_1228 is the one to smoke: three levels, the scale axis is PC1 with a
# clean 0.018 correlation to background, so if anything reads oddly there it is
# the code and not the slide.
if [ -n "${ONLY_WSI:-}" ]; then
  SLIDES=("${ONLY_WSI}")
fi

# ── sampler_routing only (reads a LIVE WSI, not a cached store) ───────────
WSI_NAME="${WSI_NAME:-BRACS_1228}"
TILE="${TILE:-256}"
# SuperPoint stageA's own ladder (extract_pretiles.py's DEFAULT_RUNGS).
RUNGS="${RUNGS:-1 2 4 8 16 32}"
# SuperPoint stageA's own n (extract_pretiles.py's _RECIPES['stageA']) --
# NOT KnnEstMpp's own default of 40.
SAMPLER_N_PER_RUNG="${SAMPLER_N_PER_RUNG:-100}"
SAMPLER_QUERY_PER_RUNG="${SAMPLER_QUERY_PER_RUNG:-20}"
# 1.475 MPixels at 45:32 -- CLAUDE.md's real-photo spec (1440x1024), not
# query_sim's 4:3/12MP default.
MPIXELS="${MPIXELS:-1.475}"
K="${K:-5}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SEG="${SEG:-hest}"

echo "======== MppRoutingExp ========"
echo "  parts  $PARTS"
echo "  slides (axes/subspace_knn)  ${SLIDES[@]}"
echo "  wsi_name (sampler_routing)  $WSI_NAME"

python utilities/bench_modules/bench_mpp_feature_decomposition.py \
  "${SLIDES[@]}" \
  --parts $PARTS \
  --stores "$RESULT_ROOT/cache/TileRetrievalBench_features/${ENCODER:-gigapath}" \
  ${DRAW:+--draw "$DRAW"} \
  --pooling cls \
  --per-level "$PER_LEVEL" \
  --white-max "$WHITE_MAX" \
  --seed "$SEED" \
  --wsi-name "$WSI_NAME" \
  --tile "$TILE" \
  --rungs $RUNGS \
  --sampler-n-per-rung "$SAMPLER_N_PER_RUNG" \
  --sampler-query-per-rung "$SAMPLER_QUERY_PER_RUNG" \
  --mpixels "$MPIXELS" \
  --k "$K" \
  --batch-size "$BATCH_SIZE" \
  --seg "$SEG" \
  ${OUT:+--out "${OUT}"}
status=$?

echo ""
echo "======== done (exit $status) ========"
echo "  [axes]             result/MppRoutingExp/<encoder>/axes_*.csv, axes_*.png"
echo "  [subspace_knn]     result/MppRoutingExp/<encoder>/subspace_knn_*.csv, *.png"
echo "                       subspace_knn_scores.csv      every setting, every r"
echo "                       subspace_knn_arm_b_fovs.csv  per FoV: what each setting predicted"
echo "                       subspace_knn_selected.csv    which components, and what else they track"
echo "                       subspace_knn_gates.csv       pinned / full_rank / shuffled"
echo "                       subspace_knn_definitions.csv every name, and what it computes"
echo "  [sampler_routing]  result/MppRoutingExp/<encoder>/sampler_routing_scores.csv"
echo "                       one row per (method, ds-or-total): KnnEstMpp (baseline) +"
echo "                       SubspaceKnn[variance/uncentred @ 2/10], each broken down PER"
echo "                       RUNG plus a 'total' row; the printed report also shows, per"
echo "                       rung, which reference level the k-NN vote landed on"
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

exit $status
