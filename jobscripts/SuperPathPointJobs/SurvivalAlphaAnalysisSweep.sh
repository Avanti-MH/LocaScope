#!/bin/bash
#SBATCH --job-name=SurvivalAlphaAnalysisSweep   # -> log/%x_%a, result/%x/...
#SBATCH --partition=8gpus                       # Partition
#SBATCH --time=48:00:00                         # per task -- same budget as a single run
#SBATCH --account=MST114560                     # Account
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --gpus-per-node=1                       # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                       # openslide reads, not a model
#SBATCH --ntasks-per-node=1                     # Tasks per node
#SBATCH --array=0-15                            # 1 decoy (random) x 4 alive x 4 WSI
#SBATCH -o /work/u26130998/log/%x_%a            # STDOUT, one file per array task
#SBATCH -e /work/u26130998/log/%x_%a            # STDERR

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

conda activate locascope
source jobscripts/_env.sh

# =============================================================================
#  2026-09-11: decoy_kind fixed to 'random' (not swept -- the user's own
#  choice; 'fixed'/'rotate' were the earlier 24-combination sweep, see git
#  history if that comparison is wanted again) x 4 alive_method (candidates
#  0/1/2/3 -- 4/mixture skipped on purpose, it does not depend on alpha at
#  all, AlphaSelectionNotes.md/AliveCandidates.py's own docstring, so
#  sweeping it across alpha would just repeat one flat answer) x 4 WSI (2
#  HE/BRACS, 2 Ki67) -- one SLURM array task per combination, each a full
#  call to SurvivalAlphaAnalysis.sh with its own --out so the 16 runs do not
#  collide into one result/ directory.
#
#  WSI choice: BRACS_1228/BRACS_1476 and S1104233/S1151088 (Ki67) were
#  picked because this session's earlier real-data/real-flow checks already
#  touched BRACS_1228 and S1104233 -- their ChainStack cache may already
#  exist, making those two cheaper to (re)run than a cold slide. Each WSI's
#  own 'C' axis happens to hold ~10 ChainStacks (observed, not forced by any
#  --limit flag -- none exists), matching the "四片各10顆tree" the sweep
#  was asked for without needing one.
#
#  AXES="C" is forced for every task, not just probability_map/
#  scale_extremum (which need it -- 'C'-axis only,
#  survival_alpha_analysis.py refuses any other --axes for them): this
#  sweep is C-only across the board so all 16 tasks are directly
#  comparable on the same axis.
#
#  INDEX -> COMBINATION (wsi fastest, then alive; decoy_kind is not a sweep
#  axis this time, always 'random'):
#    idx = alive_i*4 + wsi_i
#     0  baseline          BRACS_1228             8  probability_map  BRACS_1228
#     1  baseline          BRACS_1476             9  probability_map  BRACS_1476
#     2  baseline          S1104233,G7E,110208   10  probability_map  S1104233,G7E,110208
#     3  baseline          S1151088,G7E,111220   11  probability_map  S1151088,G7E,111220
#     4  exp_decay         BRACS_1228            12  scale_extremum   BRACS_1228
#     5  exp_decay         BRACS_1476            13  scale_extremum   BRACS_1476
#     6  exp_decay         S1104233,G7E,110208   14  scale_extremum   S1104233,G7E,110208
#     7  exp_decay         S1151088,G7E,111220   15  scale_extremum   S1151088,G7E,111220
#
#  COMBINED_THRESHOLD is NOT calibrated (plan.md/AlphaSelectionNotes.md:
#  candidate 3 is runnable, not tuned) -- every exp_decay task in this sweep
#  uses the SAME value so the four WSI results are at least comparable to
#  each other; re-running with a different COMBINED_THRESHOLD is a second
#  sweep, not something this script does in one pass. Same reasoning for
#  EXTREMUM_MARGIN (candidate 2, not calibrated either -- defaults to 0.0,
#  the AliveCandidates.py default, unless overridden).

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=<path to a trained KeypointNet .pt>}"
COMBINED_THRESHOLD="${COMBINED_THRESHOLD:?set COMBINED_THRESHOLD=<value for the exp_decay tasks -- not yet calibrated, pick one and record it>}"

# Absolute, matching every other jobscript's OUT (MultiBatch1440.sh,
# WsiHealthCheck.sh all build off this). A bare "result/..." resolves
# relative to sbatch's cwd, which is the repo checkout -- landing the sweep's
# output INSIDE the repo instead of beside it, the exact split _paths.py and
# CLAUDE.md's repo-layout section exist to prevent.
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

DECOY_KINDS=(random)
ALIVE_METHODS=(baseline exp_decay probability_map scale_extremum)
WSI_NAMES=("BRACS_1228" "BRACS_1476" "S1104233,G7E,110208" "S1151088,G7E,111220")

n_wsi=${#WSI_NAMES[@]}
n_alive=${#ALIVE_METHODS[@]}

idx=$SLURM_ARRAY_TASK_ID
wsi_i=$(( idx % n_wsi )); idx=$(( idx / n_wsi ))
alive_i=$(( idx % n_alive )); idx=$(( idx / n_alive ))
decoy_i=$idx

DECOY_KIND="${DECOY_KINDS[$decoy_i]}"
ALIVE_METHOD="${ALIVE_METHODS[$alive_i]}"
WSI_NAME="${WSI_NAMES[$wsi_i]}"
WSI_SLUG="$(echo "$WSI_NAME" | tr ',' '_')"          # commas -> filesystem-safe

OUT="$RESULT_ROOT/${SLURM_JOB_NAME}/${DECOY_KIND}/${ALIVE_METHOD}/${WSI_SLUG}"
mkdir -p "$OUT"

echo "======== sweep task $SLURM_ARRAY_TASK_ID / $(( n_wsi * n_alive * ${#DECOY_KINDS[@]} - 1 )) ========"
echo "  decoy=$DECOY_KIND  alive=$ALIVE_METHOD  wsi=$WSI_NAME"
echo "  out=$OUT"
echo ""

CHECKPOINT="$CHECKPOINT" \
WSI_NAME="$WSI_NAME" \
AXES="C" \
DECOY_KIND="$DECOY_KIND" \
ALIVE_METHOD="$ALIVE_METHOD" \
COMBINED_THRESHOLD="$([ "$ALIVE_METHOD" = exp_decay ] && echo "$COMBINED_THRESHOLD")" \
OUT="$OUT" \
bash jobscripts/SuperPathPointJobs/SurvivalAlphaAnalysis.sh
exit $?
