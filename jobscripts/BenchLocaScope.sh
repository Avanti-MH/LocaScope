#!/bin/bash
#SBATCH --job-name=BenchLocaScope        # Job name -> log/<name> and result/<name>/
#SBATCH --partition=normal2              # Partition
#SBATCH --time=48:00:00                  # partition normal caps at 2 days
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=4                # >1 so --multi-gpu has cards to use
#SBATCH --cpus-per-task=8                # DataParallel feeds every card from
                                         # one CPU-side transform loop
#SBATCH --mem=600G                       # DataParallel feeds every card from
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH -o /work/u26130998/log/%x          # STDOUT
#SBATCH -e /work/u26130998/log/%x          # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# Runs write outside the checkout; see utilities/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# ---------------- End-to-end bench, with the stage-2 ranking exposed ---------
#
# Renders synthetic shots with known positions (bench_locascope.split_shots: the
# first N_WSI slides of each dataset's recorded SPLIT, PER_LEVEL FoVs at every
# native level, the window bench's FovSupply + Render, nothing stored) and runs
# every one through LocaScopePipeline, recording
# per-stage error. Two extra measurements answer a question the winner-only
# metrics cannot: when retrieval picks the wrong window, was the right one
# further down the list, or was it never proposed at all? Those two call for
# opposite fixes -- a verification pass, versus repairing the features.
#
#   --topk       free. Reads similarity scores compute_sim_maps already
#                produced and find_best discards. Records retr_hit_rank, the
#                rank at which the truth first appears.
#   --sift-topk  NOT free: one SIFT pass per candidate per shot. Records both
#                sift_hit_rank (correct, judged against ground truth) and
#                sift_verified_rank (accepted, judged only on SIFT's own inlier
#                count). The gap between them is the whole question, because
#                only the second one is available on real photographs, which
#                have no ground truth at all.
#
# Read summary.txt bottom-up: `accepted-and-correct` says how often the inlier
# count picked the right candidate. If that is near 100%, a retrieval-proposes /
# SIFT-verifies loop is worth building for the real photos. If it is not, the
# loop would lock onto a wrong position with confidence, which is worse than
# the current single-guess failure.
#
# SIZE THE RUN BEFORE COMMITTING TO IT. SIFT cost per crop varies by more than
# an order of magnitude -- some crops blow past the BFMatcher descriptor cap of
# 262144 (see utilities/cli/diagnostics/analyze_sift_keypoints.py). Set LIMIT to 30 first
# and read t_verify_s out of metrics.csv, rather than estimating.

DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
SPLIT="${SPLIT:-test}"
N_WSI="${N_WSI:-5}"          # per dataset; BRACS test's first 5 are the masked ones
PER_LEVEL="${PER_LEVEL:-50}"  # FoVs per native level per slide

TOPK="${TOPK:-20}"               # candidates enumerated per shot (free)
SIFT_TOPK="${SIFT_TOPK:-5}"           # candidates SIFT actually verifies (K passes per shot)
LIMIT="${LIMIT:-0}"               # 0 = every shot; set 30 for a costing run first
RESUME=1              # 1 = keep the existing metrics.csv and skip what is in it
                      # RESUME=0 DELETES an existing metrics.csv, it does not
                      # append to it. A resumed run is the only way to keep the
                      # rows a walltime kill left behind.
DRAW_FIGURES="${DRAW_FIGURES:-0}"        # 4-panel diagnostics for the first N shots, -1 = all.
                      # -1 on a few thousand shots is several GB of png; prefer
                      # DRAW_FAILURES below, which draws only what went wrong.
DRAW_FAILURES="confident-wrong wrong no-recall"
                      # "" = off. Files land in figures/<category>/ :
                      #   confident_wrong  SIFT claimed success and was past
                      #                    tolerance -- the only class a
                      #                    deployment cannot detect by itself
                      #   wrong_abstained  past tolerance, SIFT abstained
                      #   no_recall        truth never proposed; gets the
                      #                    RETRIEVAL figure, not the SIFT one
FAIL_TOL_UM=100       # centre error above which a shot counts as wrong

LIMIT_FLAG=""
[ "$LIMIT" -gt 0 ] && LIMIT_FLAG="--limit $LIMIT"
RESUME_FLAG=""
[ "$RESUME" -eq 1 ] && RESUME_FLAG="--resume"
FAIL_FLAG=""
[ -n "$DRAW_FAILURES" ] && FAIL_FLAG="--draw-failures $DRAW_FAILURES --fail-tol-um $FAIL_TOL_UM"

echo "======== $DATASETS #$SPLIT  n_wsi=$N_WSI  per_level=$PER_LEVEL  topk=$TOPK  sift-topk=$SIFT_TOPK ========"
echo

# ---------------- WSI feature cache ----------------
#
# Each (slide, level) feature map is written here and reused by the next run.
# result/cache/ rather than result/<job>/ because it is shared across jobs and
# `make clean-job JOB=cache` is then the one obvious way to purge it.
#
# A HIT SKIPS THE ENCODE AND THE READ: stage 3 reads its own crop on demand
# (SiftRansacLocalizer.read_wsi_crop), so no region is held in memory and a
# cached level costs no slide read at all. Before 2026-10-05 the whole region
# was read either way, 278 s of a 563 s BRACS_1228 L0 build.
#
# Addressed by the mask recipe (seg_id / region_id) and the grid, under
# result/cache/<job>_features/<encoder>/; the encoder's full identity (config +
# a sha256 of the loaded weights) is checked on every read, and the geometry is
# rechecked against the mask in hand before anything is trusted.
#
# MODE=w on the first run. Otherwise it has to trust the write and the read at
# once, and a failure cannot say which. Every miss prints which field differed,
# so a permanently cold cache does not read like a correctly invalidated one.
FEATURES_CACHE_JOB="${FEATURES_CACHE_JOB:-BenchLocaScope}"
FEATURE_STORE_MODE="${FEATURE_STORE_MODE:-rw}"
echo "feature cache: result/cache/${FEATURES_CACHE_JOB}_features/  (mode=$FEATURE_STORE_MODE)"
echo ""

# PROFILE=1 runs the bench under cProfile, writes result/<job>/profile.prof
# and prints it into this log at the end: by cumulative time (which call a
# shot's time goes through) and by own time (where it is actually spent).
# Use it with a small LIMIT -- the profile is of the whole run.
PY=(python)
if [ "${PROFILE:-0}" = "1" ]; then
  PROF_DIR="$RESULT_ROOT/${SLURM_JOB_NAME:-BenchLocaScope}"
  mkdir -p "$PROF_DIR"
  PY=(python -m cProfile -o "$PROF_DIR/profile.prof")
  echo "profile -> $PROF_DIR/profile.prof"
fi

# --out is omitted on purpose: bench_locascope falls back to
# result/<SLURM_JOB_NAME>/, keeping the run beside its own log.
"${PY[@]}" utilities/bench_modules/bench_locascope.py \
  --datasets $DATASETS --split "$SPLIT" --n-wsi "$N_WSI" \
  --sampler-n-per-rung "$PER_LEVEL" \
  --topk       $TOPK \
  --sift-topk  $SIFT_TOPK \
  --draw-figures $DRAW_FIGURES \
  --multi-gpu \
  --precision fp16 --batch-size 8192 \
  --seg none \
  --features-cache-job "$FEATURES_CACHE_JOB" \
  --feature-store-mode "$FEATURE_STORE_MODE" \
  $LIMIT_FLAG $RESUME_FLAG $FAIL_FLAG

if [ "${PROFILE:-0}" = "1" ]; then
  echo ""
  echo "======== profile: cumulative ========"
  python -c "import pstats; pstats.Stats('$PROF_DIR/profile.prof').sort_stats('cumulative').print_stats(40)"
  echo "======== profile: own time ========"
  python -c "import pstats; pstats.Stats('$PROF_DIR/profile.prof').sort_stats('tottime').print_stats(30)"
fi

echo ""
echo "======== done -> result/BenchLocaScope/ ========"
echo "  summary.txt      recall@K, SIFT-over-top-K, accepted-and-correct"
echo "  recall_at_k.png  recall vs K, overall and per routed level"
echo "  stage2_retr_cdf.png  now marks what percentile one tile is"
