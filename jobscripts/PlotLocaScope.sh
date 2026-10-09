#!/bin/bash
#SBATCH --job-name=PlotLocaScope         # Job name -> log/<name> and result/<name>/
#SBATCH --partition=dev                  # no GPU work: reads the cache and the slides
#SBATCH --time=02:00:00
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # the partition asks for one
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH -o /work/u26130998/log/%x          # STDOUT
#SBATCH -e /work/u26130998/log/%x          # STDERR

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate locascope
source jobscripts/_env.sh

# ---------------- Offline figures from a cached run ----------------
#
# Reads the cache of a BenchLocaScope run (synthetic FoVs, with truth), or with
# REAL=1 of a realtest.sh run (real photos, no truth). The stage flags MUST be
# the run's: they give the addresses of its entries, and an entry the flags do
# not name is a miss ("stage N: miss"). Writes <out>/joined.csv (every photo),
# selected.csv, figures/stats/, figures/fov/ and, with EXPORT=demo, demo/.
#
# VARIABLES, by what they belong to:
#
#   which cache
#     REAL               1: the real photos of a realtest.sh run (job RealTest);
#                        no truth, so `gt`, the truth footprint, PLOT and EXPORT
#                        are left out and the answer is marked on `result`
#     STAGE_JOB          the run's job name (BenchLocaScope; RealTest with REAL=1)
#     DATASETS SPLIT N_WSI PER_LEVEL ROUTE     bench only: its slides and FoVs
#     SLIDES             REAL only: slide names; empty = every slide
#     LIMIT              the run's LIMIT (0 = a full run)
#
#   which stages -- the run's, exactly
#     STAGE1  STAGE2  STAGE3  SEG
#     STAGE2_K           the run's --stage2-k (how many candidates stage 2 kept);
#                        empty = the recipe's
#     STAGE3_TOPK        the run's --stage3-topk (how many of stage 2's
#                        candidates stage 3 verified); empty = the recipe's (100).
#                        realtest.sh runs 100 unless told otherwise and prints the
#                        value to copy when it ends.
#
#   what to draw
#     SELECT   pandas query on joined.csv, e.g. "level == 0 and s2_hit_rank > 1",
#              "s3_rank1_err_um < 5", "index in [3, 17]"; REAL: "answer_confidence
#              < 0.2" (a poor fit), "answer_retrieval_only == 1" (no fit at all)
#     SAMPLE   at most this many of the selected rows are drawn
#     PANELS   photo,gt,candidates,stage1,crop,matches,located,result,windows
#              ("" = none)
#     PLOT     recall cdf:s3_rank1_err_um confusion scatter:x,y ("" = none)
#     BY       group the statistics by this column
#     EXPORT   demo: the demo page in <out>/demo/ (open index.html)
#
#   anything else
#     EXTRA    flags passed to plot_locascope.py as they are, e.g. the output
#              directory:  EXTRA="--out /work/u26130998/result/PlotReal"

# which cache
REAL="${REAL:-}"
STAGE_JOB="${STAGE_JOB:-$([ -n "$REAL" ] && echo RealTest || echo BenchLocaScope)}"
DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
SPLIT="${SPLIT:-val}"
N_WSI="${N_WSI:-10}"
PER_LEVEL="${PER_LEVEL:-50}"
ROUTE="${ROUTE:-both}"
SLIDES="${SLIDES:-}"
LIMIT="${LIMIT:-0}"

# which stages
STAGE1="${STAGE1:-knn:gigapath}"
STAGE2="${STAGE2:-slidewin:gigapath}"
STAGE3="${STAGE3:-sift:default}"
SEG="${SEG:-hest}"
STAGE2_K="${STAGE2_K:-}"
STAGE3_TOPK="${STAGE3_TOPK:-}"

# what to draw
SELECT="${SELECT:-all}"
SAMPLE="${SAMPLE:-20}"
PANELS="${PANELS:-photo,candidates,matches,located,result,windows}"
if [ -n "$REAL" ]; then PLOT=""; else PLOT="${PLOT:-recall cdf:s3_rank1_err_um cdf:s1_mpp_err_rel confusion}"; fi
BY="${BY:-level}"
EXPORT="${EXPORT:-}"

EXTRA="${EXTRA:-}"

echo "cache job $STAGE_JOB   real=${REAL:-0}   stages $STAGE1 | $STAGE2 | $STAGE3   seg $SEG"
echo "stage 2: k=${STAGE2_K:-recipe}   stage 3: topk=${STAGE3_TOPK:-recipe}   limit $LIMIT"

python utilities/cli/plot/plot_locascope.py \
  --datasets $DATASETS --split "$SPLIT" --n-wsi "$N_WSI" \
  --sampler-n-per-rung "$PER_LEVEL" \
  --stage1 "$STAGE1" --stage2 "$STAGE2" --stage3 "$STAGE3" --route "$ROUTE" \
  --seg "$SEG" --limit "$LIMIT" --stage-cache-job "$STAGE_JOB" \
  --select "$SELECT" --sample "$SAMPLE" --panels "$PANELS" --by "$BY" \
  $([ -n "$PLOT" ] && echo --plot $PLOT) \
  $([ -n "$EXPORT" ] && echo --export $EXPORT) \
  $([ -n "$REAL" ] && echo --real) \
  $([ -n "$SLIDES" ] && echo --slides $SLIDES) \
  $([ -n "$STAGE2_K" ] && echo --stage2-k $STAGE2_K) \
  $([ -n "$STAGE3_TOPK" ] && echo --stage3-topk $STAGE3_TOPK) \
  $EXTRA || exit $?
