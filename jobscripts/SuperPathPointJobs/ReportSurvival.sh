#!/bin/bash
#SBATCH --job-name=ReportSurvival         # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=02:00:00                   # reads tables, draws figures
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                 # PNG reads for the example strips
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  Stage B, runs one and two. spec.md 3.2, plan.md P1.
# =============================================================================
#
# TWO RUNS AND THE ORDER IS FORCED. `inspect` chooses tau; `report` reports the
# three numbers, all of which are downstream of tau. Running `report` first
# produces the same table with `alpha = 1.5` behind it -- a value nobody has
# measured (ClaudeRules 8).
#
#   STAGE=inspect  sbatch .../ReportSurvival.sh     # run one: the tau curve
#   STAGE=report TAU_ALPHA=2.0 sbatch .../ReportSurvival.sh    # run two
#
# WHAT RUN ONE PRODUCES, and it is a curve rather than an answer: match rate
# against tau per rung, with a shifted decoy, plus where the probed peak
# actually sits. READ THE KNEE, NOT THE PEAK -- the widest tau always matches
# most, and the useful value is where the real rate stops gaining on the decoy.
#
# WHAT RUN TWO PRODUCES:
#   A  the band fraction        can Stage C's head be two outputs (spec.md 3.3)
#   B  the late-born fraction   is there anything for it to learn
#   C  the one-rung-only share  can stage 1 use a scale signature
#
# NONE OF THE THREE IS DONE WITHOUT ITS SWEEP AND ITS NULL. A fraction at one
# threshold is not a finding, and a fraction with no null is not one either:
# bands are common on any corpus where most points survive most rungs, so the
# null holds the MEASURED per-rung rates fixed and the excess over it is the
# part that is about structure.
#
# AND READ THE STRIPS. Every number in run two is equally consistent with
# "晚生型 means a large structure came into scale" and with "晚生型 means the
# detector fires on a blur artefact". The strips -- one point, its tile at
# every rung -- are the only part that can tell those apart, and the points are
# chosen by RULE (score percentiles, fixed seed) because an author's picks and
# a rule's picks look identical on the page.
#
STAGE="${STAGE:-inspect}"

TABLES="${TABLES:-/work/u26130998/result/BuildSurvival}"
TILES_ROOT="${TILES_ROOT:-/work/u26130998/result/cache/tiles_chains}"

# Run two only. Pass what run one's knee said; the default is the calibration
# value and quoting a result built on it is quoting the default.
TAU_ALPHA="${TAU_ALPHA:-}"

# Three, and they have to straddle something. Pick them off run one's
# scores.png: a sweep whose points all sit in the flat part of the score
# distribution moves nothing and proves nothing.
THRESHOLDS="${THRESHOLDS:-0.005 0.015 0.030}"

echo "======== ReportSurvival  stage: $STAGE ========"
echo "  tables $TABLES"
echo ""

case "$STAGE" in
  inspect)
    python training/SuperPathPoint/cli/inspect_survival.py \
      --tables "$TABLES"
    ;;
  report)
    if [ -z "$TAU_ALPHA" ]; then
      echo "  TAU_ALPHA is unset, so the store's own value is used -- which is"
      echo "  the calibration default unless it was rebuilt. Run STAGE=inspect"
      echo "  first and pass its knee, or this reports 1.5 under another name."
      echo ""
    fi
    python training/SuperPathPoint/cli/report_survival.py \
      --tables "$TABLES" \
      --tiles-root "$TILES_ROOT" \
      --thresholds $THRESHOLDS \
      ${TAU_ALPHA:+--tau-alpha "$TAU_ALPHA"}
    ;;
  *)
    echo "unknown stage: $STAGE   (known: inspect report)"
    exit 1
    ;;
esac
status=$?

echo ""
echo "======== done  (exit $status) ========"
echo "  -> result/\${SLURM_JOB_NAME}/"
echo ""
echo "  inspect: read the KNEE off figures/tau_curve.png, cross-check it"
echo "           against the p90 line in figures/offsets.png, and pick the"
echo "           three sweep thresholds off figures/scores.png."
echo "  report : read figures/threshold_sweep.png BEFORE any number in the"
echo "           CSVs -- a fraction that moves across that range is a number"
echo "           with no finding behind it. Then the strips."

exit $status
