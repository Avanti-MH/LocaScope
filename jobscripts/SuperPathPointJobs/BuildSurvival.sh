#!/bin/bash
#SBATCH --job-name=BuildSurvival          # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=08:00:00                   # chains x 6 rungs x 2 axes, no training
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                 # PNG reads
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  spec.md 3.2 Stage B: the survival table, both axes
# =============================================================================
#
# RUN ONE IS A CALIBRATION RUN, NOT A RESULT (ClaudeRules 8). `tau_alpha` is
# 1.5 and nobody has measured it. What run one produces is the
# match-rate-against-tau curve and its decoy; the value comes from the knee.
# An attribution split quoted from run one is quoting a guess.
#
# WHAT IT NEEDS THAT DOES NOT EXIST YET: chains. A chain is one level-0 centre
# with a tile at EVERY rung, and the 2026-08-27 extraction was run with
# inheritance off -- 6,388 tiles, 0 chains, `inherit_id` is -1 on every row. So
# `ExtractPreTiles` has to be re-run with `InheritConfig` on before this script
# has anything to read. That re-extraction also invalidates the HA labels
# (`LabelMeta.pretile_id` is the extraction's hash), so MakeHaLabels follows it.
#
#   ExtractPreTiles (inherit on)  ->  MakeHaLabels  ->  BuildSurvival
#
# ONE PASS, TWO AXES. The 'R' stack is DERIVED from each chain's ds 1 tile
# rather than extracted: an 'R' rung IS `tile` level-0 px shrunk by ds and grown
# back, and the ds 1 tile IS `tile` level-0 px. That is not a saving, it is the
# requirement -- 新生歸因 asks whether a point born late on 'F' is also born
# late on 'R', and that has an answer only if the two axes are about the same
# physical point. Two extractions would choose centres by their own
# admissibility and the axes would have to be joined spatially afterwards, on
# exactly the quantity the analysis is about.
#
# `max_keypoints` IS OFF, by construction -- `SurvivalProcess` never passes one.
# A cap is global competition that we imposed, and a point dropping out of the
# top N is indistinguishable from one killed by its neighbourhood.
#
# THE STORE HOLDS `score` AND `dist`, NOT `alive`. `alive` is score-over-a-
# threshold AND dist-within-tau, and storing it freezes both into a file that
# costs hours to rebuild. The threshold sweep that decides whether a finding
# survives is a re-read (`Patterns.alive_from`).
#
#   sbatch jobscripts/SuperPathPointJobs/BuildSurvival.sh
#   CHECKPOINT=... LIMIT=20 sbatch .../BuildSurvival.sh      # smoke run
#
# The chain corpus, not `cache/tiles` -- see build_survival.DEFAULT_TILE_ROOT.
TILES_ROOT="${TILES_ROOT:-/work/u26130998/result/cache/tiles_chains}"

CHECKPOINT="${CHECKPOINT:-/work/u26130998/result/TrainSuperPathPoint/model_256_gray_pre/superpathpoint_last.pt}"

TILE=256

# The permissive cut the store is written at. Every later question re-cuts at
# some higher value on these arrays; a question below it needs a re-run. Same
# idiom as `make_ha_labels.THRESHOLD_LADDER`.
SCORE_THRESHOLD=0.001

# tau = alpha * ds level-0 px. CALIBRATION. A fixed level-0 tau would make the
# coarse rungs unable to match BY DEFINITION -- one of their pixels is `ds`
# level-0 px -- and the output would read as "keypoints all die at coarse
# resolution", a wrong answer that looks like a discovery.
TAU_ALPHA=1.5


# HOW FAR THE DECOY PROBE SITS FROM THE ANCHOR, as a multiple of ds. The decoy
# is a SECOND PROBE at a place the point is not, stored beside the real one, so
# that "beats the decoy" is a measurement rather than an identity: deriving it
# as `dist + shift <= tau` is the match rate at a shifted alpha -- the same
# curve under another name, and it read `margin 1.1` at every rung on
# 2026-09-01 before this existed.
DECOY_ALPHA="${DECOY_ALPHA:-8.0}"

# 0 = every chain. A small number makes this a smoke run.
LIMIT="${LIMIT:-0}"

# Empty = DsLadder's default six. Set it when the corpus does not have all six
# -- `MppStack.chains` measures completeness against THIS list, and a chain
# missing a rung it was never asked for is not incomplete. The smoke corpus is
# five rungs, so reading it with the default would find zero chains.
DS="${DS:-}"

echo "======== BuildSurvival ========"
echo "  detector   $CHECKPOINT"
echo "  tiles      $TILES_ROOT"
echo "  tile $TILE   score_threshold $SCORE_THRESHOLD   tau_alpha $TAU_ALPHA"
echo "  chains     ${LIMIT:-all}   rungs ${DS:-DsLadder default}"
echo ""

python training/SuperPathPoint/cli/build_survival.py \
  --checkpoint "$CHECKPOINT" \
  --tiles-root "$TILES_ROOT" \
  --tile "$TILE" \
  --score-threshold "$SCORE_THRESHOLD" \
  --tau-alpha "$TAU_ALPHA" \
  --decoy-alpha "$DECOY_ALPHA" \
  ${DS:+--ds $DS} \
  --limit-chains "$LIMIT"
status=$?

echo ""
echo "======== done  (exit $status) ========"
echo "  tables -> result/\${SLURM_JOB_NAME}/<slide>__<F|R>__t${TILE}.safetensors"
echo ""
echo "  Read the tau curve first, and read it as a CALIBRATION. The six-pattern"
echo "  split and the attribution split are both downstream of a tau nobody has"
echo "  measured; quoting them from this run is quoting 1.5."

exit $status
