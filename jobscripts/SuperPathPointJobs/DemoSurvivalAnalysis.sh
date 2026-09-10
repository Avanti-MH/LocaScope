#!/bin/bash
#SBATCH --job-name=DemoSurvivalAnalysis    # -> log/%x, result/%x/
#SBATCH --partition=normal                 # Partition
#SBATCH --time=00:30:00                    # both parts: no training, chains_
                                            # stack reads a whole real tree
                                            # (~425 tiles), merge_grid is
                                            # synthetic unless CHECKPOINT/
                                            # REAL_FLOW is set (real detections)
#SBATCH --account=MST114560                # Account
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gpus-per-node=1                  # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                  # openslide reads + PNG + matplotlib
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH -o /work/u26130998/log/%x          # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x          # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  cli/demo_survival_analysis.py -- merged 2026-09-11 from two standalone
#  jobscripts (DemoChainsStack.sh, DemoMergeGrid.sh) whose underlying CLIs
#  had grown into one file each with almost nothing shared beyond WSI_NAME/
#  TILE/C_RUNGS/CACHE_ROOT/OUT. PARTS selects which demo(s) actually run --
#  both by default, matching what running both scripts used to do.
# =============================================================================
#
# PARTS -- one or both of:
#   chains_stack   smoke-test all three axes' OWN/REUSE-F paths against a
#                 REAL slide (spec.md 3.2, plan.md 2.1②) -- needs
#                 `stageB-fOwn` to hold at least one complete chain for
#                 WSI_NAME (run `PrepareChainStack.sh AXES=F` first if not).
#   merge_grid     `_merge_within_radius`'s three implementations compared
#                 (original / B retired / A adopted) -- synthetic-only by
#                 default, CHECKPOINT+WSI_NAME add a real-data comparison,
#                 REAL_FLOW=1 adds a real-pipeline comparison.
PARTS="${PARTS:-chains_stack merge_grid}"

# ── shared ───────────────────────────────────────────────────────────────
# Path/stem resolved from WSI_NAME via `utilities/AccessDatasets.py`.
# chains_stack defaults to BRACS_1228 if unset; merge_grid only needs it
# with CHECKPOINT or REAL_FLOW.
WSI_NAME="${WSI_NAME:-}"
TILE="${TILE:-256}"
C_RUNGS="${C_RUNGS:-1.0 2.0 4.0 8.0 16.0}"
CACHE_ROOT="${CACHE_ROOT:-}"
TILES_ROOT="${TILES_ROOT:-}"

# ── chains_stack only ────────────────────────────────────────────────────
RUNGS="${RUNGS:-1.0 2.0 4.0 8.0 16.0}"
LINEAGE_INDEX="${LINEAGE_INDEX:-0}"

# ── merge_grid only ──────────────────────────────────────────────────────
N_CLUSTERS="${N_CLUSTERS:-40}"
MAX_PER_CLUSTER="${MAX_PER_CLUSTER:-6}"
JITTER="${JITTER:-1.5}"
EXTENT="${EXTENT:-100.0}"
RADIUS="${RADIUS:-6.0}"
SEED="${SEED:-0}"

TIMING_N_CLUSTERS="${TIMING_N_CLUSTERS:-800}"
TIMING_EXTENT="${TIMING_EXTENT:-400.0}"
SKIP_ORIGINAL_ABOVE="${SKIP_ORIGINAL_ABOVE:-20000}"
REPEATS="${REPEATS:-3}"

# Real-data comparison -- optional. Leave CHECKPOINT unset to stay
# synthetic-only (the default).
CHECKPOINT="${CHECKPOINT:-}"
TREE_INDEX="${TREE_INDEX:-0}"
REAL_RUNG="${REAL_RUNG:-}"
PLOT_REAL="${PLOT_REAL:-0}"
PLOT_REAL_FLAG=""
[ "$PLOT_REAL" = 1 ] && PLOT_REAL_FLAG="--plot-real"

# Real-flow comparison -- optional, needs CHECKPOINT + WSI_NAME too.
REAL_FLOW="${REAL_FLOW:-0}"
REAL_FLOW_N_TREES="${REAL_FLOW_N_TREES:-2}"
REAL_FLOW_FLAG=""
[ "$REAL_FLOW" = 1 ] && REAL_FLOW_FLAG="--real-flow"

# Unset -> job_result_dir('DemoSurvivalAnalysis') (a DIRECTORY shared by
# both parts -- each part keeps its own filenames underneath it).
OUT="${OUT:-}"

echo "======== DemoSurvivalAnalysis ========"
echo "  parts  $PARTS"
echo "  slide  ${WSI_NAME:-<part default>}   tile $TILE   c-rungs $C_RUNGS"
python training/SuperPathPoint/cli/demo_survival_analysis.py \
  --parts $PARTS \
  ${WSI_NAME:+--wsi-name "$WSI_NAME"} \
  --tile "$TILE" \
  --c-rungs $C_RUNGS \
  ${TILES_ROOT:+--tiles-root "$TILES_ROOT"} \
  ${CACHE_ROOT:+--cache-root "$CACHE_ROOT"} \
  --rungs $RUNGS \
  --lineage-index "$LINEAGE_INDEX" \
  --n-clusters "$N_CLUSTERS" \
  --max-per-cluster "$MAX_PER_CLUSTER" \
  --jitter "$JITTER" \
  --extent "$EXTENT" \
  --radius "$RADIUS" \
  --seed "$SEED" \
  --timing-n-clusters "$TIMING_N_CLUSTERS" \
  --timing-extent "$TIMING_EXTENT" \
  --skip-original-above "$SKIP_ORIGINAL_ABOVE" \
  --repeats "$REPEATS" \
  ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
  ${CHECKPOINT:+--tree-index "$TREE_INDEX"} \
  ${REAL_RUNG:+--real-rung "$REAL_RUNG"} \
  $PLOT_REAL_FLAG \
  $REAL_FLOW_FLAG \
  --real-flow-n-trees "$REAL_FLOW_N_TREES" \
  ${OUT:+--out "$OUT"}
status=$?

echo ""
echo "======== done (exit $status) ========"
echo "  result -> result/\${SLURM_JOB_NAME}/"
echo "    [chains_stack] figures/r_stack_{own,reuseF}.png"
echo "                   figures/pyramid_{lineage,overview}_{own,reuseF}.png"
echo "    [merge_grid]   merge_grid_demo.png"
if [ -n "$CHECKPOINT" ]; then
  echo "                   merge_grid_demo_real.png   only if PLOT_REAL=1"
fi
if [ "$REAL_FLOW" = 1 ]; then
  echo "                   merge_grid_demo_real_flow.png"
fi

exit $status
