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
#SBATCH -o ./log/BenchLocaScope          # STDOUT
#SBATCH -e ./log/BenchLocaScope          # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- End-to-end bench, with the stage-2 ranking exposed ---------
#
# Runs every shot of a query_sim corpus through LocaScopePipeline and records
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
# 262144 (see utilities/cli/analyze_sift_keypoints.py). Set LIMIT to 30 first
# and read t_verify_s out of metrics.csv, rather than estimating.

GT_CSV=result/MultiBatch1440/gt.csv
IMAGES=result/MultiBatch1440/images

TOPK=20               # candidates enumerated per shot (free)
SIFT_TOPK=5           # candidates SIFT actually verifies (K passes per shot)
LIMIT=0               # 0 = every shot; set 30 for a costing run first
RESUME=1              # 1 = keep the existing metrics.csv and skip what is in it
                      # RESUME=0 DELETES an existing metrics.csv, it does not
                      # append to it. A resumed run is the only way to keep the
                      # rows a walltime kill left behind.
DRAW_FIGURES=0        # 4-panel diagnostics for the first N shots, -1 = all.
                      # -1 on a 2500-shot corpus is ~6 GB of png; prefer
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

if [ ! -f "$GT_CSV" ]; then
  echo "[abort] $GT_CSV not found -- run jobscripts/MultiBatch1440.sh first"
  exit 1
fi

LIMIT_FLAG=""
[ "$LIMIT" -gt 0 ] && LIMIT_FLAG="--limit $LIMIT"
RESUME_FLAG=""
[ "$RESUME" -eq 1 ] && RESUME_FLAG="--resume"
FAIL_FLAG=""
[ -n "$DRAW_FAILURES" ] && FAIL_FLAG="--draw-failures $DRAW_FAILURES --fail-tol-um $FAIL_TOL_UM"

echo "======== gt=$GT_CSV  topk=$TOPK  sift-topk=$SIFT_TOPK ========"
echo "shots: $(( $(wc -l < "$GT_CSV") - 1 ))"
echo

# --out is omitted on purpose: bench_locascope falls back to
# result/<SLURM_JOB_NAME>/, keeping the run beside its own log.
python utilities/test_modules/bench_locascope.py \
  --gt-csv     "$GT_CSV" \
  --images-dir "$IMAGES" \
  --topk       $TOPK \
  --sift-topk  $SIFT_TOPK \
  --draw-figures $DRAW_FIGURES \
  --multi-gpu \
  --precision fp16 --batch-size 8192 \
  --mask-all \
  $LIMIT_FLAG $RESUME_FLAG $FAIL_FLAG

echo ""
echo "======== done -> result/BenchLocaScope/ ========"
echo "  summary.txt      recall@K, SIFT-over-top-K, accepted-and-correct"
echo "  recall_at_k.png  recall vs K, overall and per routed level"
echo "  stage2_retr_cdf.png  now marks what percentile one tile is"
