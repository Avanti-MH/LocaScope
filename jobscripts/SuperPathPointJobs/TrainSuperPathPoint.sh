#!/bin/bash
#SBATCH --job-name=TrainSuperPathPoint    # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                   # four arms, 5344 pairs, 50 epochs
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                 # DataLoader workers read PNGs
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# wandb writes to the project's own output root, not to $HOME. Offline is the
# safe default on a compute node with no outbound network; `wandb sync` uploads
# afterwards from the login node.
export WANDB_DIR="/work/u26130998/result/${SLURM_JOB_NAME:-TrainSuperPathPoint}"
mkdir -p "$WANDB_DIR"


# =============================================================================
#  spec.md 12 step 6: four arms on one set of labels
# =============================================================================
#
# ONE SET OF LABELS, FOUR ARMS. Homographic Adaptation produces COORDINATES,
# and a coordinate does not care how many channels the image had, so all four
# train against the identical label store (spec.md 13). What differs is one
# number in the backbone and where the weights started.
#
# WHAT THE PAIR CAN AND CANNOT SETTLE, because the difference matters for what
# goes in the conclusion:
#
#   CAN   repeatability on the two held-out SLIDES, reported per slide -- the
#         trainer emits `val/<stem>/...` columns and the run prints them as a
#         table. One slide winning and the other losing means the difference is
#         a property of the slide, and two slides is not enough to decide --
#         record it as undecided (spec.md 1, fourth row).
#   CANNOT  cross-stain generalisation. Both stains are in the training split,
#         so the held-out number says "an unseen slide", not "an unseen stain".
#         That needs the leave-one-stain-out arm (spec.md 6.5), which reuses
#         these same tiles and is not in v1. Until then the words "cross-stain"
#         do not belong in a conclusion drawn from this run.
#
# EVERY CRITERION IS A MARGIN OVER A DECOY, AND THE DECOY IS ITS CEILING. There
# is no keypoint ground truth on a WSI, so an absolute repeatability of 0.62
# would need a reference to mean anything. The decoy supplies it: the same
# comparison against points shifted past the NMS radius, so whatever matches is
# what point DENSITY alone buys.
#
# Since `margin = repeatability / decoy` and `repeatability <= 1`, the decoy is
# literally the ceiling: `margin <= 1/decoy`. The 2026-08-28 run came back with
# decoy 0.92 and margin 1.03 against a ceiling of 1.10 -- a third of a ten per
# cent range, which is why epoch 0 and epoch 9 were indistinguishable. That is
# what `max_keypoints = 420` in KeypointNetConfig is for, and it is why
# `val/<stem>/points_per_view` is logged beside every margin: a margin at an
# unrecorded density cannot be compared with the next epoch's.
#
#   sbatch jobscripts/SuperPathPointJobs/TrainSuperPathPoint.sh      # all four
#   ARMS="gray gray_pre" sbatch .../TrainSuperPathPoint.sh            # two
#
# FOUR ARMS, TWO AXES: channels x initialisation. `_pre` starts from upstream
# SuperPoint v6 instead of at random.
#
#   THE RGB TRANSFER IS EXACT, not approximate. The released weights are 1
#   channel; conv1 is repeated three times and divided by three, which makes
#   the 3-channel network's response to a luma image identical to the
#   1-channel network's. So the two axes stay orthogonal -- without it, "RGB is
#   worse" would be inseparable from "RGB's transfer was weaker".
#
#   `gray_pre` IS SELF-DISTILLATION. The HA labels were produced by those same
#   upstream weights (spec.md 13), so this arm asks "does a second round on its
#   own labels beat learning them from scratch" -- SuperPoint's round 2. It
#   does NOT ask "does pretraining help"; that needs a different pretraining
#   source. Write the conclusion with the first sentence.
#
#   TWO HELD-OUT SLIDES CANNOT SETTLE SIX PAIRWISE COMPARISONS. The goal of
#   this round is to RULE OUT bad combinations, not to declare a winner.

# TWO ARMS, AND THEY ARE NOT THE SAME EXPERIMENT (plan.md P0-c).
#
#   gray_pre   250 epochs, batch 64  -> 20,750 steps. The real run.
#   gray       50 epochs,  batch 128 -> 2,050 steps. AN ABLATION, and the step
#              count is deliberately unchanged from the 2026-08-31 run so that
#              augmentation is the ONLY variable that moved. It answers "does
#              unfreezing help", NOT "can this train from scratch" -- 2,050
#              steps from random init did not reach `ln(65)`-minus-much last
#              time and will not this time. Do not read it as a verdict.
#
# rgb and rgb_pre are OFF, and the reason is a correction. They were dropped as
# "no difference"; the matched-density re-eval says otherwise -- RGB is ahead in
# all ten comparisons (five budgets x two inits) by 2 to 5 per cent, and wins
# BOTH held-out slides at top320 and top420. Small, but not noise. They are off
# because four arms x 250 epochs is 40 hours against a 24 hour limit, not
# because the question is closed. Revisit once Stage A converges.
ARMS="${ARMS:-gray_pre gray}"

TILE=256

# Per arm, because the two are different experiments. `EPOCHS`/`BATCH` override
# both if set.
EPOCHS_gray_pre="${EPOCHS:-250}"
BATCH_gray_pre="${BATCH:-64}"
EPOCHS_gray="${EPOCHS:-50}"
BATCH_gray="${BATCH:-128}"

# ROTATION ONLY (plan.md P0-b). The query is a microscope photograph of a flat
# slide at an arbitrary angle; stage 1 has already fixed the scale and
# perspective on a coverslip is small. It is NARROWER than the thirteen-option
# sampler the labels were voted on, and narrower is the safe direction -- the
# student is asked to be invariant to less than its teacher was, never more.
#
# PRE_TILE_FACTOR = 3 still holds: the 2000-draw calibration needed 2.702 for
# the full sampler and rotation alone needs sqrt(2)/0.85 = 1.66.
HOMOGRAPHY=rotation

# Points per view the repeatability is measured at, top-N by score with the
# threshold at zero. A BUDGET AND NOT A THRESHOLD: `margin = repeat / decoy`,
# the decoy rises with density, so two models cut by one threshold sit on two
# different scales. The 2026-08-31 run reported 1.50 and 3.59 at 420 and 159
# points per view with nothing saying they were incomparable.
#
# 200 is inside the label corpus's own range (per-rung `n_kp` means 3 to 527,
# overall 146). The ladder over several budgets is `cli/reeval_density.py`, run
# once on the finished checkpoints.
VAL_BUDGET=200

# Upstream's SuperPoint LR, constant, no schedule (`base_model.py:212`). The
# 1e-3 in the configs is MagicPoint's, which is the stage this project skips.
LR=1e-4

# RE-SETTLED BY THE 3b PROBE OF 2026-08-27, AND IT WENT THE OTHER WAY.
#
# The rule spec.md 6.5 states in advance: align-min if the worst rung still has
# hundreds of tiles, loss-weight if it has forty. The 2026-08-26 run answered
# "hundreds" -- ds 32 came back with 1784 of 2000 over four slides, 91 per cent
# kept for an exactly flat ladder, and align-min was the obvious pick.
#
# THAT NUMBER WAS THE TISSUE GATE, NOT THE SLIDES. The gate admitted only
# background <= 50 per cent and the sampler then asked for 500 positions it had
# already been handed; what it measured was how fast the budget ran out, not how
# many DISJOINT positions a rung holds. The 12-slide probe measures the second
# thing -- `n_admissible` is the whole candidate pool, and at the coarse rungs
# the sampler now takes essentially all of it (ds 32: 578 taken of 583 there):
#
#   tile 256, per slide, admissible positions
#              ds 1     ds 2    ds 4   ds 8   ds 16   ds 32
#   best      69,011   18,853   4,908  1,257    317      80
#   worst     17,807    4,718   1,262    318     86      18
#   12 slides 461,141  123,624 32,474  8,465  2,242     583
#
# align-min truncates every rung to the WORST CELL, and the worst cell is 18.
# Six rungs x 12 slides x 18 is 1,296 tiles -- against 8,465 available at ds 8
# alone. That is not balancing a ladder, it is deleting one.
#
# SO: NOT align-min, decided 2026-08-27. Not yet loss-weight either -- the rung
# weights are a second decision and the corpus has to exist first. For now the
# trainer uses every tile each rung supplies, unbalanced, and the imbalance is
# a known and recorded property of the v1 run rather than a discovery.
#
# What would change it: more slides. ds 32 needs about 25 of them to reach the
# 1,200 that 12 slides reach at ds 16. That is the lever, not the switch.
BALANCE=none

# online. Compute nodes on normal2 DO reach api.wandb.ai -- GMR-Conv and
# gigapath_retrieval have online runs from this same partition and account,
# and neither sets WANDB_MODE at all. Credentials come from ~/.netrc.
#
# This said "compute nodes here have no outbound network" until 2026-08-28.
# That was an assumption written as a fact, and it cost one run: the job was
# submitted offline, and SLURM takes its copy of this script AT SUBMIT TIME,
# so editing the file does not reach a job already running -- only a resubmit
# does.
#
# Set `offline` and `wandb sync $WANDB_DIR` from a login node afterwards only
# if a node turns out to be firewalled.
WANDB_MODE=online

status=0
run () {   # run <label> <command...>
  echo ""
  echo "======== $1 ========"
  shift
  "$@" || status=1
}

echo "======== TrainSuperPathPoint  arms: $ARMS ========"
echo "  tile $TILE   lr $LR   balance $BALANCE   homography $HOMOGRAPHY"
echo "  gray_pre ${EPOCHS_gray_pre}ep x b${BATCH_gray_pre}   gray ${EPOCHS_gray}ep x b${BATCH_gray}   val budget $VAL_BUDGET"
echo ""

for model in $ARMS; do
  PRETRAINED=""
  case "$model" in
    gray)     CHANNELS=1 ; EPOCHS_N=$EPOCHS_gray     ; BATCH_N=$BATCH_gray ;;
    rgb)      CHANNELS=3 ; EPOCHS_N=$EPOCHS_gray     ; BATCH_N=$BATCH_gray ;;
    gray_pre) CHANNELS=1 ; EPOCHS_N=$EPOCHS_gray_pre ; BATCH_N=$BATCH_gray_pre
              PRETRAINED="--pretrained" ;;
    rgb_pre)  CHANNELS=3 ; EPOCHS_N=$EPOCHS_gray_pre ; BATCH_N=$BATCH_gray_pre
              PRETRAINED="--pretrained" ;;
    *)
      echo "unknown arm: $model   (known: gray rgb gray_pre rgb_pre)"
      status=1
      continue
      ;;
  esac

  run "model_${TILE}_${model}  ($CHANNELS ch${PRETRAINED:+, upstream init}, ${EPOCHS_N}ep x b${BATCH_N})" \
    python training/SuperPathPoint/cli/train_superpathpoint.py \
      --tile "$TILE" --channels "$CHANNELS" $PRETRAINED \
      --epochs "$EPOCHS_N" --batch-size "$BATCH_N" --lr "$LR" \
      --balance "$BALANCE" \
      --homography "$HOMOGRAPHY" \
      --val-budget "$VAL_BUDGET" \
      --workers "${SLURM_CPUS_PER_TASK:-4}" \
      --wandb-mode "$WANDB_MODE"
done

echo ""
echo "======== done  (exit $status) ========"
echo "  checkpoints -> result/\${SLURM_JOB_NAME}/model_${TILE}_<arm>/"
echo "  history     -> the same directory, train_history.csv"
echo "  wandb       -> $WANDB_DIR   (wandb sync it from a login node)"
echo ""
echo "  Read the three loss magnitudes after the first epoch before anything"
echo "  else. lambda_loss=10000 compensates for the descriptor term's double"
echo "  normalisation, and whether that lands at the detector term's scale on"
echo "  THIS data is a fact about the data, not about upstream. If the"
echo "  descriptor term is orders of magnitude off, the two halves of that"
echo "  compensation have come apart -- see SuperPoint/Losses.py."
echo ""
echo "  Then the repeatability MARGIN, per held-out slide. The absolute value"
echo "  has no reference on a WSI; the ratio over the shifted decoy does."

exit $status
