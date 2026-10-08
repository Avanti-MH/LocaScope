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
ml load miniconda3/24.11.1
ml load cuda/12.6
conda activate gigapath
source jobscripts/_env.sh

# ---------------- Offline figures from a BenchLocaScope run ----------------
#
# The run flags MUST be the bench's (same DATASETS ... ROUTE, LIMIT and EXTRA
# stage flags): they give the addresses of its entries. STAGE_JOB is the
# bench's job name. Writes result/<job>/joined.csv (every FoV x route),
# selected.csv, figures/stats/, figures/fov/ and, with EXPORT=demo, demo/.
#
#   SELECT   pandas query on joined.csv, e.g. "level == 0 and s2_hit_rank > 1",
#            "s3_rank1_err_um < 5", "index in [3, 17]"; "all" = every row
#   PANELS   photo,gt,candidates,stage1,crop,matches,result,windows ("" = none)
#   PLOT     recall cdf:s3_rank1_err_um confusion scatter:x,y ("" = none)

DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
SPLIT="${SPLIT:-test}"
N_WSI="${N_WSI:-5}"
PER_LEVEL="${PER_LEVEL:-50}"
SEG="${SEG:-hest}"
STAGE1="${STAGE1:-knn:gigapath}"
STAGE2="${STAGE2:-slidewin:gigapath}"
STAGE3="${STAGE3:-sift:default}"
ROUTE="${ROUTE:-both}"
LIMIT="${LIMIT:-0}"
EXTRA="${EXTRA:-}"
STAGE_JOB="${STAGE_JOB:-BenchLocaScope}"

SELECT="${SELECT:-all}"
SAMPLE="${SAMPLE:-20}"
PANELS="${PANELS:-photo,candidates,matches,result,windows}"
PLOT="${PLOT:-recall cdf:s3_rank1_err_um cdf:s1_mpp_err_rel confusion}"
BY="${BY:-level}"
EXPORT="${EXPORT:-}"            # demo: the demo page in result/<job>/demo/ (open index.html)

python utilities/cli/plot/plot_locascope.py \
  --datasets $DATASETS --split "$SPLIT" --n-wsi "$N_WSI" \
  --sampler-n-per-rung "$PER_LEVEL" \
  --stage1 "$STAGE1" --stage2 "$STAGE2" --stage3 "$STAGE3" --route "$ROUTE" \
  --seg "$SEG" --limit "$LIMIT" --stage-cache-job "$STAGE_JOB" \
  --select "$SELECT" --sample "$SAMPLE" --panels "$PANELS" --by "$BY" \
  $([ -n "$PLOT" ] && echo --plot $PLOT) \
  $([ -n "$EXPORT" ] && echo --export $EXPORT) \
  $EXTRA || exit $?
