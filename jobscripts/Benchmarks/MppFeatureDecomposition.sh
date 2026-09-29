#!/bin/bash
#SBATCH --job-name=MppFeatureDecomposition   # -> log/%x, result/%x/
#SBATCH --partition=normal2                   # Partition
#SBATCH --time=48:00:00                       # sampler_routing reads a live WSI + GPU encode
#SBATCH --account=MST114560                   # Account
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gpus-per-node=1                     # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                     # SVD / eigh / openslide reads
#SBATCH --mem=400G                             # one token store in flight per level
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

# ---------------- TEMPORARY: report checkpoints that fail to load ---------
# READ-ONLY -- does NOT delete anything. The 2026-09-17 KeyError: 'extra'
# looked like a stale pre-`extra`-format checkpoint at first, but every file
# under weights/ postdates the code that writes `extra`, which rules that
# out. More likely: `_best.pt` is REWRITTEN every time training finds a new
# best epoch (`torch.save` is not atomic), so a read landing mid-write from
# a still-running training job would load a truncated file missing whatever
# key happened to write last -- a checkpoint caught that way is fine, not
# corrupt, and deleting it would destroy real training progress. This block
# only REPORTS which files fail to load right now; _build_methods already
# skips one that fails rather than dying. DELETE THIS BLOCK once the
# 2026-09-17 failure is understood well enough not to need a standing check.
python3 -c "
import torch, glob, os
wdir = '${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result/MppRoutingHead/weights'
bad = []
for p in sorted(glob.glob(os.path.join(wdir, '*_best.pt'))):  # matches CLASSIFIER_WEIGHTS=all's own glob -- *_last.pt is never read by this job
    try:
        ckpt = torch.load(p, map_location='cpu')
        if 'extra' not in ckpt:
            bad.append((p, \"loads, but missing 'extra'\"))
    except Exception as exc:
        bad.append((p, f'{type(exc).__name__}: {exc}'))
if not bad:
    print('(every checkpoint loads and has extra)')
else:
    print(f'{len(bad)} checkpoint(s) with a problem right now (NOT deleted):')
    for p, why in bad:
        print(f'  {p}: {why}')
"

# =============================================================================
#  utilities/bench_modules/bench_mpp_feature_decomposition.py -- 2026-09-14
#  merge of FeatureAxes.sh + SubspaceKnn.sh, plus a new sampler_routing part.
# =============================================================================
#
# PARTS -- one or more of, or 'all' (default):
#   axes             feature_axes_analysis (was bench_feature_axes.py/
#                   FeatureAxes.sh). Purely descriptive: where does mpp live
#                   in the 1536-D feature space. Never runs a KNN.
#   subspace_knn     was bench_subspace_knn.py/SubspaceKnn.sh. Does a KNN run
#                   in the subspace `axes` found actually route better? Reads
#                   cached FeatureStore reference/query stores (arm B carries
#                   the synthetic camera domain gap).
#   sampler_routing  NEW. Samples its OWN fresh, UNCACHED reference tiles
#                   straight off a live WSI (TileSampler, SuperPoint stageA's
#                   own recipe: DsLadder rungs, n=100/rung, never written to
#                   result/cache/tiles/), renders a DISJOINT set of query
#                   positions as an actual photo (QueryFromWSI +
#                   simulate_microscope_photo, sized like a real photo --
#                   1.475 MPixels at 45:32), and runs the CURRENT production
#                   baseline (KnnEstMpp) alongside SubspaceKnn projected
#                   candidates on the same draw, per rung -- the arena any
#                   future routing method drops into next to the baseline it
#                   has to beat.
#   stage1_compare   NEW (2026-09-17). Compares whichever full StageInterface
#                   estimators are named (--knn-encoder / --classifier-weights)
#                   on --n-wsi slides from EACH of --datasets, drawn from the
#                   RECORDED test split (training/MppRoutingHead/Datasets.
#                   wsi_split) so no slide was seen by a checkpoint's own
#                   training/selection. Writes one per-shot row per (FoV,
#                   method) to <sampler_id>_<seg_id>.csv; read it with
#                   utilities/cli/metrics/analyze_stage1_metrics.py, which
#                   owns the CSV schema and the scoring -- this part's job is
#                   only to produce rows in the shape that file expects.
#
# 2026-09-14: axes/subspace_knn need result/cache/reference_features/, which
# was deleted -- default PARTS is sampler_routing only until that cache is
# rebuilt. Pass PARTS="all" (or "axes subspace_knn") once it is.
PARTS="${PARTS:-sampler_routing}"
# axes / subspace_knn read PoolingBench.sh's reference stores; DRAW is the
# `reference draw` line that dump prints (<sampler_id>_<plan>).
DRAW="${DRAW:-}"

# ── axes / subspace_knn (read result/cache/features/, need --wsi-stem) ────
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
#   sbatch --export=ALL,ONLY_WSI=BRACS_1228,PARTS=sampler_routing \
#          jobscripts/Benchmarks/MppFeatureDecomposition.sh
if [ -n "${ONLY_WSI}" ]; then
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

# ── stage1_compare only ────────────────────────────────────────────────────
DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
N_WSI="${N_WSI:-9}"
STAGE1_N_PER_RUNG="${STAGE1_N_PER_RUNG:-20}"
RATIO="${RATIO:-45:32}"
# SEG: hsv (free, no model) / hest (DeepLabV3+ResNet-50) / uni2 (fits a PCA
# across the whole scanned rectangle first, 3.5-6 GPU-min/slide -- see
# Uni2PcaSegConfig's own docstring). Built once, shared across every slide
# and every method regardless of which one this picks.
SEG="${SEG:-hest}"
KNN_SAMPLES="${KNN_SAMPLES:-40}"
KNN_K="${KNN_K:-5}"
# KNN_ENCODER: space-separated encoder names. Defaults to BOTH -- the same
# pair training/MppRoutingHead's own baseline 2 trains against
# (cli/train.py's --encoders default) -- not to "none", so a plain
# PARTS=stage1_compare run compares something without the caller having to
# name it. Pass KNN_ENCODER="" explicitly to skip KnnEstMpp entirely.
KNN_ENCODER="${KNN_ENCODER:-gigapath uni2}"
# CLASSIFIER_WEIGHTS: space-separated checkpoint paths, or "all" (default)
# to glob EVERY *_best.pt this repo has trained so far -- run all of them
# together rather than one at a time, so a new checkpoint from a fresh
# training run needs no path typed in here to be included next time this
# jobscript runs. Pass CLASSIFIER_WEIGHTS="" explicitly to skip
# ClassifierEstMpp entirely.
#
# ONE shared directory (2026-09-22, back from a hardcoded .../ord_b/weights
# that this "all" default pointed at while ord_a/ord_b lived in their own
# subdirectories -- that meant "all" never actually meant all: bal's own
# checkpoints, and any ord_a ones, were silently excluded). Now that every
# checkpoint's filename carries its own --loss as a segment when it is not
# 'bal' (see Checkpoints.weight_filename's own docstring), bal/ord_a/ord_b
# checkpoints of the same encoder+head coexist in this one directory without
# overwriting each other, so one glob genuinely finds all of them --
# analyze_stage1_metrics.py's method_of() is what then tells them apart in
# the report (loss appended to the label only when it is not 'bal').
CLASSIFIER_WEIGHTS="${CLASSIFIER_WEIGHTS:-all}"
if [ "$CLASSIFIER_WEIGHTS" = "all" ]; then
  WEIGHTS_DIR="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result/MppRoutingHead/weights"
  CLASSIFIER_WEIGHTS="$(ls "$WEIGHTS_DIR"/*_best.pt 2>/dev/null | tr '\n' ' ')"
  if [ -z "$CLASSIFIER_WEIGHTS" ]; then
    echo "[warn] CLASSIFIER_WEIGHTS=all found no *_best.pt under $WEIGHTS_DIR"
  fi
fi

STAGE1_ARGS=(--seg "$SEG")
[ "${NATIVE_ONLY:-0}" = "1" ] && STAGE1_ARGS+=(--native-only)
# avoids coarse rungs (huge footprint, little disjoint room) coming up short
[ "${OVERLAP:-0}" = "1" ] && STAGE1_ARGS+=(--overlap)
[ -n "$KNN_ENCODER" ] && STAGE1_ARGS+=(--knn-encoder $KNN_ENCODER)
[ -n "$CLASSIFIER_WEIGHTS" ] && STAGE1_ARGS+=(--classifier-weights $CLASSIFIER_WEIGHTS)

# Real numbers before the real run, not a guess: params memory (exact) +
# one measured forward pass's peak, per method -- "one at a time" is what
# stage1_compare's own method-outer loop needs now (2026-09-17's OOM fix);
# "all at once" is what the OLD build-everything-up-front design needed,
# and why job 346494 died against --mem=64G. Read-only, does not affect
# the run below either way.
case "$PARTS" in *stage1_compare*)
  echo ""
  echo "======== [stage1_compare memory estimate] ========"
  python utilities/cli/diagnostics/estimate_method_memory.py \
    --tile "$TILE" --knn-samples "$KNN_SAMPLES" --knn-k "$KNN_K" \
    --batch "$BATCH_SIZE" \
    $([ -n "$KNN_ENCODER" ] && echo --knn-encoder $KNN_ENCODER) \
    $([ -n "$CLASSIFIER_WEIGHTS" ] && echo --classifier-weights $CLASSIFIER_WEIGHTS)
  ;;
esac

echo "======== MppFeatureDecomposition ========"
echo "  parts  $PARTS"
echo "  slides (axes/subspace_knn)  ${SLIDES[@]}"
echo "  wsi_name (sampler_routing)  $WSI_NAME"
echo "  datasets (stage1_compare)  $DATASETS   n_wsi=$N_WSI   seg=$SEG   native_only=${NATIVE_ONLY:-0}"

# Piped through `tee` (not run plain) only so `stage1_compare`'s own
# printed CSV path can be recovered below without re-deriving
# _sampling_recipe_id()'s hash in bash -- everything still streams to this
# job's normal stdout/log exactly as before. `${PIPESTATUS[0]}` (not `$?`,
# which after a pipe would be `tee`'s own exit code) is what makes this
# still fail the same way a plain run would.
STAGE1_LOG_TEE="$(mktemp)"
python utilities/bench_modules/bench_mpp_feature_decomposition.py \
  "${SLIDES[@]}" \
  --parts $PARTS \
  --stores "${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result/cache/PoolingBench_features/${ENCODER:-gigapath}" \
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
  --datasets $DATASETS \
  --n-wsi "$N_WSI" \
  --n-per-rung "$STAGE1_N_PER_RUNG" \
  --knn-samples "$KNN_SAMPLES" \
  --knn-k "$KNN_K" \
  --ratio "$RATIO" \
  "${STAGE1_ARGS[@]}" \
  2>&1 | tee "$STAGE1_LOG_TEE"
status=${PIPESTATUS[0]}

# ---------------- stage1_compare: auto-analyze the CSV it just wrote -------
# READ-ONLY -- runs analyze_stage1_metrics.py against exactly the file this
# run produced (its path is recovered from this run's own stdout, not
# guessed by globbing result/MppFeatureDecomposition/*.csv, which could pick
# up a different run's file). No-op if stage1_compare was not in $PARTS, or
# if the run failed before it printed that line.
case "$PARTS" in *stage1_compare*)
  STAGE1_CSV=$(grep -F ' -- read with utilities/cli/metrics/analyze_stage1_metrics.py' \
    "$STAGE1_LOG_TEE" | tail -1 | awk '{print $1}')
  if [ -n "$STAGE1_CSV" ] && [ -f "$STAGE1_CSV" ]; then
    echo ""
    echo "======== [stage1_compare analysis] ========"
    python utilities/cli/metrics/analyze_stage1_metrics.py "$STAGE1_CSV"
  else
    echo ""
    echo "[warn] stage1_compare: could not recover the CSV path from this run's own output -- skipping auto-analysis"
  fi
  ;;
esac
rm -f "$STAGE1_LOG_TEE"

echo ""
echo "======== done (exit $status) ========"
echo "  [axes]             result/MppFeatureDecomposition/<encoder>/axes_*.csv, axes_*.png"
echo "  [subspace_knn]      result/MppFeatureDecomposition/<encoder>/subspace_knn_*.csv, *.png"
echo "  [sampler_routing]   result/MppFeatureDecomposition/<encoder>/sampler_routing_scores.csv"
echo "                       one row per (method, ds-or-total): KnnEstMpp"
echo "                       (baseline) + SubspaceKnn[variance/uncentred @ 2/10],"
echo "                       each broken down PER RUNG plus a 'total' row; the printed"
echo "                       report also shows, per rung, which reference level the"
echo "                       k-NN vote actually landed on"
echo "  [stage1_compare]    result/MppFeatureDecomposition/<sampler_id>_<seg_id>.csv  (no <encoder> -- spans several)"
echo "                       python utilities/cli/metrics/analyze_stage1_metrics.py <that csv>"
echo ""
echo "  stage1_compare: runs EVERYTHING by default -- KnnEstMpp(gigapath),"
echo "  KnnEstMpp(uni2), and every *_best.pt checkpoint already trained under"
echo "  result/MppRoutingHead/weights/, all on the same drawn FoVs:"
echo "    PARTS=stage1_compare sbatch jobscripts/Benchmarks/MppFeatureDecomposition.sh"
echo "  Narrow it with KNN_ENCODER=/CLASSIFIER_WEIGHTS=, e.g. one checkpoint only:"
echo "    PARTS=stage1_compare KNN_ENCODER='' \\"
echo "      CLASSIFIER_WEIGHTS=/work/u26130998/result/MppRoutingHead/weights/gigapath_frozen_arcface_best.pt \\"
echo "      sbatch jobscripts/Benchmarks/MppFeatureDecomposition.sh"

exit $status
