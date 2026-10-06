#!/bin/bash
#SBATCH --job-name=Stage1MppBench             # -> log/%x, result/%x/
#SBATCH --partition=normal2                   # Partition
#SBATCH --time=48:00:00                       # 13+ methods, one encoder resident at a time
#SBATCH --account=MST114560                   # Account
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gpus-per-node=1                     # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                     # openslide reads for the FoV crops
#SBATCH --mem=400G                             # host RSS peaks while an encoder + masks are resident
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
#  utilities/bench_modules/bench_stage1_mpp.py
#  (was the `stage1_compare` part of Benchmarks/MppFeatureDecomposition.sh /
#  bench_mpp_feature_decomposition.py; split out 2026-09-29. The other parts
#  are jobscripts/MppRoutingExp.sh now.)
# =============================================================================
#
# Compares whichever full StageInterface estimators are named (--knn-encoder /
# --classifier-weights) on --n-wsi slides from EACH of --datasets, drawn from
# the RECORDED test split (utilities/cli/build_cache/make_split.py) so no slide
# was seen by a checkpoint's own training/selection. Writes one per-shot row
# per (FoV, method) to result/Stage1MppBench/<sampler_id>_<seg_id>_<region_id>.csv;
# this script then runs utilities/cli/metrics/analyze_stage1_metrics.py on it,
# which owns the CSV schema and the scoring.
#
# CACHES: masks are read from MppRoutingHead's cache; the FoV draw is
# Stage1MppBench's whatever the job name, so a smoke, a timing run and the
# stages test (TestLocaScopeStages) read the FoVs the full run scores -- one
# draw, one key, as long as STAGE1_N_PER_RUNG, SEED and OVERLAP are left alone.
# A smoke scores a subset of it with FOV_PER_RUNG rather than drawing others.
MASK_CACHE_JOB="${MASK_CACHE_JOB-MppRoutingHead}"   # its masks cover every val slide and the first test slides; "" = this job's own
SAMPLER_CACHE_JOB="${SAMPLER_CACHE_JOB-Stage1MppBench}"   # "" = this job's own
TILE="${TILE:-256}"
MPIXELS="${MPIXELS:-1.475}"
BATCH_SIZE="${BATCH_SIZE:-4096}"

DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
N_WSI="${N_WSI:-5}"   # per dataset; val and test must match, n_wsi is in the file name the test run finds the val thresholds by
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
# run compares something without the caller having to
# name it. Pass KNN_ENCODER="" explicitly to skip KnnEstMpp entirely.
KNN_ENCODER="${KNN_ENCODER-gigapath uni2}"   # no colon: an explicit "" skips
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
CLASSIFIER_WEIGHTS="${CLASSIFIER_WEIGHTS-all}"   # no colon: an explicit "" skips
if [ "$CLASSIFIER_WEIGHTS" = "all" ]; then
  WEIGHTS_DIR="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result/MppRoutingHead/weights"
  CLASSIFIER_WEIGHTS="$(ls "$WEIGHTS_DIR"/*_best.pt 2>/dev/null | tr '\n' ' ')"
  if [ -z "$CLASSIFIER_WEIGHTS" ]; then
    echo "[warn] CLASSIFIER_WEIGHTS=all found no *_best.pt under $WEIGHTS_DIR"
  fi
fi

# PROTOTYPE_WEIGHTS: the same rule for PrototypicalRoutingHead -- "all"
# (default) globs every *_best.pt under its weights directory, "" skips
# PrototypeEstMpp. _6rung.pt / _native.pt are other selections of the same
# runs and are not read.
PROTOTYPE_WEIGHTS="${PROTOTYPE_WEIGHTS-all}"   # no colon: an explicit "" skips
if [ "$PROTOTYPE_WEIGHTS" = "all" ]; then
  PROTO_DIR="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result/PrototypicalRoutingHead/weights"
  PROTOTYPE_WEIGHTS="$(ls "$PROTO_DIR"/*_best.pt 2>/dev/null | tr '\n' ' ')"
  if [ -z "$PROTOTYPE_WEIGHTS" ]; then
    echo "[warn] PROTOTYPE_WEIGHTS=all found no *_best.pt under $PROTO_DIR"
  fi
fi
# CLASSIC=1 (default) runs the fingerprint baseline ClassicEstMpp; 0 skips it.
CLASSIC="${CLASSIC:-1}"
CLASSIC_K="${CLASSIC_K:-3}"
# RULES: vote rules (bench_stage1_mpp.RULES) every classifier and prototype
# method is scored under, all from one forward pass. Empty (default) = all.
# SPLIT: val or test (default). FoV_Vote.md fixes every risk threshold on val
# before test is looked at, so run SPLIT=val first: its analysis writes the
# thresholds, and the test run of the same recipe picks them up by name.
RULES="${RULES:-}"
SPLIT="${SPLIT:-test}"

STAGE1_ARGS=(--seg "$SEG")
[ -n "$MASK_CACHE_JOB" ] && STAGE1_ARGS+=(--mask-cache-job "$MASK_CACHE_JOB")
[ -n "$SAMPLER_CACHE_JOB" ] && STAGE1_ARGS+=(--sampler-cache-job "$SAMPLER_CACHE_JOB")
[ "${NATIVE_ONLY:-0}" = "1" ] && STAGE1_ARGS+=(--native-only)
# avoids coarse rungs (huge footprint, little disjoint room) coming up short;
# on by default since the val and test runs both use it (OVERLAP=0 turns it off)
[ "${OVERLAP:-1}" = "1" ] && STAGE1_ARGS+=(--overlap)
[ -n "${FOV_PER_RUNG:-}" ] && STAGE1_ARGS+=(--fov-per-rung "$FOV_PER_RUNG")
[ -n "$KNN_ENCODER" ] && STAGE1_ARGS+=(--knn-encoder $KNN_ENCODER)
[ -n "$CLASSIFIER_WEIGHTS" ] && STAGE1_ARGS+=(--classifier-weights $CLASSIFIER_WEIGHTS)
[ -n "$PROTOTYPE_WEIGHTS" ] && STAGE1_ARGS+=(--prototype-weights $PROTOTYPE_WEIGHTS)
[ "$CLASSIC" = "1" ] && STAGE1_ARGS+=(--classic --classic-k "$CLASSIC_K")
[ -n "$RULES" ] && STAGE1_ARGS+=(--rules $RULES)
STAGE1_ARGS+=(--split "$SPLIT")

# Real numbers before the real run, not a guess: params memory (exact) +
# one measured forward pass's peak, per method -- "one at a time" is what
# stage1_compare's own method-outer loop needs now (2026-09-17's OOM fix);
# "all at once" is what the OLD build-everything-up-front design needed,
# and why job 346494 died against --mem=64G. Read-only, does not affect
# the run below either way.
echo ""
echo "======== [memory estimate] ========"
python utilities/cli/diagnostics/estimate_method_memory.py \
  --tile "$TILE" --knn-samples "$KNN_SAMPLES" --knn-k "$KNN_K" \
  --batch "$BATCH_SIZE" \
  $([ -n "$KNN_ENCODER" ] && echo --knn-encoder $KNN_ENCODER) \
  $([ -n "$CLASSIFIER_WEIGHTS" ] && echo --classifier-weights $CLASSIFIER_WEIGHTS)

echo "======== Stage1MppBench ========"
echo "  datasets  $DATASETS   n_wsi=$N_WSI   seg=$SEG   native_only=${NATIVE_ONLY:-0}"

# Piped through `tee` only so the CSV path the python prints can be recovered
# below without re-deriving _sampling_recipe_id()'s hash in bash -- everything
# still streams to this job's stdout/log. `${PIPESTATUS[0]}` (not `$?`, which
# after a pipe is `tee`'s) keeps the exit status honest.
STAGE1_LOG_TEE="$(mktemp)"
python -u utilities/bench_modules/bench_stage1_mpp.py \
  --tile "$TILE" \
  --mpixels "$MPIXELS" \
  --seed "${SEED:-42}" \
  --datasets $DATASETS \
  --n-wsi "$N_WSI" \
  --n-per-rung "$STAGE1_N_PER_RUNG" \
  --knn-samples "$KNN_SAMPLES" \
  --knn-k "$KNN_K" \
  --ratio "$RATIO" \
  "${STAGE1_ARGS[@]}" \
  2>&1 | tee "$STAGE1_LOG_TEE"
status=${PIPESTATUS[0]}

# ---------------- auto-analyze the CSV it just wrote -----------------------
# READ-ONLY -- runs analyze_stage1_metrics.py against exactly the file this
# run produced (its path is recovered from this run's own stdout, not
# guessed by globbing result/Stage1MppBench/*.csv, which could pick
# up a different run's file). Skipped with a warning if the run failed before
# it printed that line.
STAGE1_CSV=$(grep -F ' -- read with utilities/cli/metrics/analyze_stage1_metrics.py' \
  "$STAGE1_LOG_TEE" | tail -1 | awk '{print $1}')
if [ -n "$STAGE1_CSV" ] && [ -f "$STAGE1_CSV" ]; then
  echo ""
  echo "======== [analysis] ========"
  if [ "$SPLIT" = "val" ]; then
    python utilities/cli/metrics/analyze_stage1_metrics.py "$STAGE1_CSV" --fit-thresholds
  else
    python utilities/cli/metrics/analyze_stage1_metrics.py "$STAGE1_CSV" --thresholds auto
  fi
else
  echo ""
  echo "[warn] could not recover the CSV path from this run's own output -- skipping auto-analysis"
fi
rm -f "$STAGE1_LOG_TEE"

echo ""
echo "======== done (exit $status) ========"
echo "  result/Stage1MppBench/<sampler_id>_<seg_id>_<region_id>.csv  (no <encoder> -- spans several)"
echo "    python utilities/cli/metrics/analyze_stage1_metrics.py <that csv>"
echo ""
echo "  Runs EVERYTHING by default -- KnnEstMpp(gigapath), KnnEstMpp(uni2), ClassicEstMpp,"
echo "  and every *_best.pt under result/MppRoutingHead/weights/ and"
echo "  result/PrototypicalRoutingHead/weights/ under every vote rule, on the same FoVs."
echo "  Val first (fixes the risk thresholds), then test:"
echo "    SPLIT=val sbatch jobscripts/Benchmarks/Stage1MppBench.sh"
echo "    sbatch jobscripts/Benchmarks/Stage1MppBench.sh"
echo "  Narrow it with KNN_ENCODER= CLASSIFIER_WEIGHTS= PROTOTYPE_WEIGHTS= CLASSIC=0 RULES=, e.g.:"
echo "    KNN_ENCODER= PROTOTYPE_WEIGHTS= CLASSIC=0 \\"
echo "      CLASSIFIER_WEIGHTS=/work/u26130998/result/MppRoutingHead/weights/gigapath_frozen_arcface_best.pt \\"
echo "      sbatch jobscripts/Benchmarks/Stage1MppBench.sh"

exit $status
