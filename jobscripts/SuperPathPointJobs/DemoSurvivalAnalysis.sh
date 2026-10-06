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
#  TILE/C_RUNGS/CHAINSTACK_CACHE_JOB/OUT, plus a third part (visualize) added the same
#  day. PARTS selects which demo(s) actually run -- all three by default.
# =============================================================================
#
# PARTS -- one or more of:
#   chains_stack   smoke-test all three axes' OWN/REUSE-F paths against a
#                 REAL slide (spec.md 3.2, plan.md 2.1②) -- needs
#                 `stageB-fOwn` to hold at least one complete chain for
#                 WSI_NAME (run `PrepareChainStack.sh AXES=F` first if not).
#   merge_grid     `_merge_within_radius`'s three implementations compared
#                 (original / B retired / A adopted) -- synthetic-only by
#                 default, CHECKPOINT+WSI_NAME add a real-data comparison,
#                 REAL_FLOW=1 adds a real-pipeline comparison.
#   visualize      synthetic, step-by-step GIFs for a PPT demo of this
#                 stage: keypoints -> anchor merge -> alive. Synthetic and
#                 torch-free by default; CHECKPOINT+WSI_NAME add one real-
#                 data closing figure (pass C_RUNGS="4 8 16" etc. to keep
#                 that figure to the coarse rungs only).
#   scale_diagnostic  does RStack's own resampling filter matter, or does
#                 the six-pattern classification track pure SCALE? Runs
#                 the real detector on N_SCALE_TILES real F-chain tiles,
#                 twice each (RStack's shrink+grow vs a plain Gaussian
#                 blur at sigma=ds/2, same real pixels) -- ALWAYS needs
#                 CHECKPOINT, never in the default PARTS list. Also pools
#                 decay-rate-vs-survival-breadth across every anchor.
#   scale_synthetic_control  is decay-rate a working ruler at all? Known-
#                 radius synthetic blobs through the SAME decay-rate probe
#                 -- the sanity check for scale_diagnostic's own real-data
#                 correlation. No checkpoint, no slide, no torch, fast.
PARTS="${PARTS:-chains_stack merge_grid visualize}"

# ── shared ───────────────────────────────────────────────────────────────
# Path/stem resolved from WSI_NAME via `utilities/AccessDatasets.py`.
# chains_stack defaults to BRACS_1228 if unset; merge_grid only needs it
# with CHECKPOINT or REAL_FLOW.
WSI_NAME="${WSI_NAME:-}"
TILE="${TILE:-256}"
C_RUNGS="${C_RUNGS:-1.0 2.0 4.0 8.0 16.0}"
# CHAINSTACK_CACHE_JOB: whose result/cache/<job>_chainstack/ tile cache to use
# (default: this job's own). Was CACHE_ROOT, a bare path, until 2026-10-06.
CHAINSTACK_CACHE_JOB="${CHAINSTACK_CACHE_JOB:-}"
PRETILE_CACHE_JOB="${PRETILE_CACHE_JOB:-}"   # empty = ExtractPreTiles

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

# Real-data comparison -- optional for chains_stack/merge_grid/visualize
# (leave unset to stay synthetic-only there), REQUIRED for scale_diagnostic.
# Defaults to the gray_pre arm's last checkpoint (2026-09-14) so
# scale_diagnostic can be launched with just PARTS=scale_diagnostic
# WSI_NAME=... -- override to point at a different arm/epoch.
CHECKPOINT="${CHECKPOINT:-/work/u26130998/result/TrainSuperPathPoint/model_256_gray_pre/superpathpoint_last.pt}"
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

# ── visualize only ───────────────────────────────────────────────────────
# VIZ_ALPHA left UNSET by default on purpose -- the python CLI's own
# default (_VIZ_ALPHA_DEFAULT) is DERIVED from the actual synthetic point
# layout, so leaving this unset here means it can never drift out of sync
# with that scene the way a hand-typed default in this script once did
# (2026-09-11). Set VIZ_ALPHA explicitly only to override that computed
# default, e.g. to test the safety check itself.
VIZ_ALPHA="${VIZ_ALPHA:-}"
VIZ_FPS="${VIZ_FPS:-1.2}"
VIZ_REAL_CROP="${VIZ_REAL_CROP:-256}"

# ── scale_diagnostic only ────────────────────────────────────────────────
# ALWAYS needs CHECKPOINT (above) -- it runs the real detector, there is no
# synthetic-only mode here.
N_SCALE_TILES="${N_SCALE_TILES:-10}"
# 3.0 is plan.md 2.2's settled production value (tau = max(SCALE_TAU_FLOOR,
# SCALE_ALPHA * ds)), not a default invented for this jobscript.
SCALE_ALPHA="${SCALE_ALPHA:-3.0}"
SCALE_TAU_FLOOR="${SCALE_TAU_FLOOR:-0.0}"
# Shuffles for the decay-rate correlation's permutation test (SEED, shared
# with merge_grid above, reseeds this too) -- used by BOTH scale_diagnostic
# and scale_synthetic_control.
DECAY_PERMUTATIONS="${DECAY_PERMUTATIONS:-500}"

# ── scale_synthetic_control only ─────────────────────────────────────────
# 2917 matches the pooled anchor count scale_diagnostic reported on
# BRACS_1228 with the default N_SCALE_TILES=10 -- override to match
# whatever a different scale_diagnostic run actually produced (see its own
# printed "[decay vs survival] n=..." line).
N_SYNTHETIC_POINTS="${N_SYNTHETIC_POINTS:-2917}"

# Unset -> job_result_dir('DemoSurvivalAnalysis') (a DIRECTORY shared by
# every part -- each part keeps its own filenames underneath it).
OUT="${OUT:-}"

echo "======== DemoSurvivalAnalysis ========"
echo "  parts  $PARTS"
echo "  slide  ${WSI_NAME:-<part default>}   tile $TILE   c-rungs $C_RUNGS"
python training/SuperPathPoint/cli/demo_survival_analysis.py \
  --parts $PARTS \
  ${WSI_NAME:+--wsi-name "$WSI_NAME"} \
  --tile "$TILE" \
  --c-rungs $C_RUNGS \
  ${PRETILE_CACHE_JOB:+--pretile-cache-job "$PRETILE_CACHE_JOB"} \
  ${CHAINSTACK_CACHE_JOB:+--chainstack-cache-job "$CHAINSTACK_CACHE_JOB"} \
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
  ${VIZ_ALPHA:+--viz-alpha "$VIZ_ALPHA"} \
  --viz-fps "$VIZ_FPS" \
  --viz-real-crop "$VIZ_REAL_CROP" \
  --n-scale-tiles "$N_SCALE_TILES" \
  --scale-alpha "$SCALE_ALPHA" \
  --scale-tau-floor "$SCALE_TAU_FLOOR" \
  --decay-permutations "$DECAY_PERMUTATIONS" \
  --n-synthetic-points "$N_SYNTHETIC_POINTS" \
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
echo "    [visualize]    visualize_{keypoints,anchor_merge,alive}.gif"
echo "                   visualize_{keypoints,anchor_merge,alive}_frames/frame_NN.png"
if [ -n "$CHECKPOINT" ]; then
  echo "                   visualize_real_example.png"
fi
if [[ " $PARTS " == *" scale_diagnostic "* ]]; then
  echo "    [scale_diagnostic]  figures/scale_diagnostic_tile*.png"
  echo "                       figures/scale_diagnostic_decay_vs_survival.png"
fi
if [[ " $PARTS " == *" scale_synthetic_control "* ]]; then
  echo "    [scale_synthetic_control]  figures/scale_synthetic_control.png"
  echo "                       figures/scale_synthetic_control_examples.png"
fi

exit $status
