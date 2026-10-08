#!/bin/bash
#SBATCH --job-name=SurvivalAlphaAnalysis   # -> log/%x, result/%x/
#SBATCH --partition=normal                 # Partition
#SBATCH --time=04:00:00                    # detector forward pass per tile, no training
#SBATCH --account=MST114560                # Account
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --gpus-per-node=1                  # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                  # openslide reads, not a model
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH -o /work/u26130998/log/%x          # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x          # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  plan.md 2.2① -- calibrate alpha, one bucket (F/R/C) at a time
# =============================================================================
#
# C first: R's own tiles need stageA's already-extracted store (see
# prepare_chain_stack.py), which is real data this run reads, not something
# it samples. F is left off by default -- run it explicitly with AXES=F once
# a real chain is worth spending GPU time on.
#
# ONE decoy_shift KIND PER RUN. `DECOY_KIND=fixed` (the default) is the
# cheapest of the three (`survival_alpha_analysis.py`'s
# decoy_shift_fixed/_random/_rotate) -- comparing kinds against each other is
# a second run with a different `DECOY_KIND`, not something this script does
# in one pass.

# FROM_CSV=<dir> -- redraw figures from a previous run's CSVs (that run's
# own result/<job>/ directory) without recomputing anything: no checkpoint,
# no slide, no GPU. Does not need CHECKPOINT/WSI_NAME set at all.
FROM_CSV="${FROM_CSV:-}"
if [ -n "$FROM_CSV" ]; then
  AXES="${AXES:-C R}"
  QUANTILES="${QUANTILES:-0.5 0.9 0.99}"
  TAU_FLOOR="${TAU_FLOOR:-0.0}"
  echo "======== SurvivalAlphaAnalysis (--from-csv) ========"
  echo "  reading  $FROM_CSV"
  python training/SuperPathPoint/cli/survival_alpha_analysis.py \
    --from-csv "$FROM_CSV" \
    --axes $AXES \
    --quantiles $QUANTILES \
    --tau-floor "$TAU_FLOOR" \
    ${OUT:+--out "$OUT"}
  status=$?
  echo ""
  echo "======== done (exit $status) ========"
  exit $status
fi

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=<path to a trained KeypointNet .pt, e.g. /work/u26130998/result/TrainSuperPathPoint/model_256_gray_pre/<checkpoint>.pt>}"

# SMOKE_TEST=1 -- a cheap check before the real (possibly hours-long) run:
# does the grid-hash merge now in production (`SurvivalProcess.
# _merge_within_radius`) still agree with its two retired predecessors, on a
# small, fast, always-registered slide (BRACS_1228 -- NOT whatever WSI_NAME
# the real run below targets)? Runs `DemoSurvivalAnalysis.sh`'s merge_grid
# part itself rather than re-deriving its logic, with these parameters fixed
# (this is the exact invocation that first caught the real-flow skip-guard
# bug this round), and exits right after instead of going on to the real
# analysis. Does not need WSI_NAME to be set at all.
SMOKE_TEST="${SMOKE_TEST:-0}"
if [ "$SMOKE_TEST" = 1 ]; then
  echo "======== SurvivalAlphaAnalysis smoke test (via DemoSurvivalAnalysis.sh, merge_grid) ========"
  PARTS=merge_grid \
  CHECKPOINT="$CHECKPOINT" \
  WSI_NAME=BRACS_1228 \
  SKIP_ORIGINAL_ABOVE=50000 \
  PLOT_REAL=1 \
  REAL_FLOW=1 \
  REAL_FLOW_N_TREES=2 \
  bash jobscripts/SuperPathPointJobs/DemoSurvivalAnalysis.sh
  exit $?
fi

# Path/stem are resolved from WSI_NAME via `utilities/AccessDatasets.py` --
# see that file for the registered names (`AccessDatasets.list_names()`).
WSI_NAME="${WSI_NAME:?set WSI_NAME=<a name AccessDatasets.py registers, e.g. BRACS_1228>}"

TILE="${TILE:-256}"
RUNGS="${RUNGS:-1.0 2.0 4.0 8.0 16.0}"
C_RUNGS="${C_RUNGS:-1.0 2.0 4.0 8.0 16.0}"

# ALIVE_METHOD=baseline (Patterns.alive_from, production), exp_decay
# (AliveCandidates.alive_exp_decay_joint_score, candidate 3 -- needs
# COMBINED_THRESHOLD, a different scale from SCORE_THRESHOLD, see that
# candidate's own docstring), probability_map (candidate 1) or
# scale_extremum (candidate 2) -- 2026-09-11: all three axes (F/R/C)
# support every alive-method now, so AXES's default no longer depends on
# ALIVE_METHOD. Candidate 4 is not runnable yet.
ALIVE_METHOD="${ALIVE_METHOD:-baseline}"
COMBINED_THRESHOLD="${COMBINED_THRESHOLD:-}"
# probability_map/scale_extremum only -- AliveCandidates.assemble_
# generation_map's own main/overlap fold ('C' axis only, F/R have one tile
# per rung, nothing to fold), the scale-extremum margin, and
# probe_via_probability_map's own sub-pixel search grid spacing (that
# function's own docstring: "a real accuracy/speed knob, not yet tuned").
MAP_OVERLAP_MODE="${MAP_OVERLAP_MODE:-union}"
EXTREMUM_MARGIN="${EXTREMUM_MARGIN:-0.0}"
SAMPLE_STEP="${SAMPLE_STEP:-0.5}"

AXES="${AXES:-C R}"

# Unset by default -- the python side resolves this from the checkpoint's
# own cfg.detection_threshold, the value its labels were cut at.
SCORE_THRESHOLD="${SCORE_THRESHOLD:-}"
ALPHA_MIN="${ALPHA_MIN:-0.5}"
ALPHA_STEP="${ALPHA_STEP:-1.0}"
ALPHA_MAX="${ALPHA_MAX:-8.0}"
TAU_FLOOR="${TAU_FLOOR:-0.0}"

MERGE_RADIUS_2ND="${MERGE_RADIUS_2ND:-0.0}"

DECOY_KIND="${DECOY_KIND:-fixed}"
DECOY_MAGNITUDE="${DECOY_MAGNITUDE:-8.0}"
DECOY_SEED="${DECOY_SEED:-0}"

QUANTILES="${QUANTILES:-0.5 0.9 0.99}"

# Which pre-tile cache the three axes' corpora are read from
# (prepare_chain_stack.py's addresses; nothing is extracted here).
PRETILE_CACHE_JOB="${PRETILE_CACHE_JOB:-ExtractPreTiles}"
# CHAINSTACK_CACHE_JOB: whose tile cache (result/cache/<job>/slide=<s>/chainstack/) to use
# (default: this job's own). Was CACHE_ROOT, a bare path, until 2026-10-06.
CHAINSTACK_CACHE_JOB="${CHAINSTACK_CACHE_JOB:-}"

# Unset -> job_result_dir('SurvivalAlphaAnalysis') (the single-run default).
# A sweep (many decoy/alive/WSI combinations under one SLURM_JOB_NAME) MUST
# set this to a distinct directory per combination, or every task collides
# into the same result/<job>/ and overwrites the others' CSVs.
OUT="${OUT:-}"

echo "======== SurvivalAlphaAnalysis ========"
echo "  slide  $WSI_NAME"
echo "  axes   $AXES"
echo "  tile   $TILE   rungs ${RUNGS}   c-rungs ${C_RUNGS}"
echo "  alphas $ALPHA_MIN:$ALPHA_STEP:$ALPHA_MAX   tau-floor $TAU_FLOOR"
echo "  decoy  $DECOY_KIND   magnitude $DECOY_MAGNITUDE (x rung_shrink)   seed $DECOY_SEED"
echo "  merge-radius-2nd $MERGE_RADIUS_2ND"
echo "  score-threshold ${SCORE_THRESHOLD:-<from the checkpoint cfg.detection_threshold>}"
echo "  alive-method $ALIVE_METHOD${COMBINED_THRESHOLD:+  combined-threshold $COMBINED_THRESHOLD}"
if [ "$ALIVE_METHOD" = probability_map ] || [ "$ALIVE_METHOD" = scale_extremum ]; then
  echo "  map-overlap-mode $MAP_OVERLAP_MODE   sample-step $SAMPLE_STEP"
fi
if [ "$ALIVE_METHOD" = scale_extremum ]; then
  echo "  extremum-margin $EXTREMUM_MARGIN"
fi
echo ""

python training/SuperPathPoint/cli/survival_alpha_analysis.py \
  --checkpoint "$CHECKPOINT" \
  --wsi-name "$WSI_NAME" \
  --pretile-cache-job "$PRETILE_CACHE_JOB" \
  --tile "$TILE" \
  --rungs $RUNGS \
  --c-rungs $C_RUNGS \
  --axes $AXES \
  ${SCORE_THRESHOLD:+--score-threshold "$SCORE_THRESHOLD"} \
  --alive-method "$ALIVE_METHOD" \
  ${COMBINED_THRESHOLD:+--combined-threshold "$COMBINED_THRESHOLD"} \
  --map-overlap-mode "$MAP_OVERLAP_MODE" \
  --extremum-margin "$EXTREMUM_MARGIN" \
  --sample-step "$SAMPLE_STEP" \
  --alphas "$ALPHA_MIN" "$ALPHA_STEP" "$ALPHA_MAX" \
  --tau-floor "$TAU_FLOOR" \
  --merge-radius-2nd "$MERGE_RADIUS_2ND" \
  --decoy-kind "$DECOY_KIND" \
  --decoy-magnitude "$DECOY_MAGNITUDE" \
  --decoy-seed "$DECOY_SEED" \
  --quantiles $QUANTILES \
  ${CHAINSTACK_CACHE_JOB:+--chainstack-cache-job "$CHAINSTACK_CACHE_JOB"} \
  ${OUT:+--out "$OUT"}
status=$?

echo ""
echo "======== done (exit $status) ========"
echo "  figures -> result/\${SLURM_JOB_NAME}/figures/alpha_curves_<axis>.png"
echo "             result/\${SLURM_JOB_NAME}/figures/heatmaps_<axis>.png"
echo "             result/\${SLURM_JOB_NAME}/figures/patterns_<axis>.png"
echo "  tables  -> result/\${SLURM_JOB_NAME}/alpha_curve_<axis>.csv"
echo "             result/\${SLURM_JOB_NAME}/offset_quantiles_<axis>.csv"
echo "             result/\${SLURM_JOB_NAME}/pattern_curve_<axis>.csv"

exit $status
