#!/bin/bash
#SBATCH --job-name=ReEvalSuperPathPoint   # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=02:00:00                   # 4 arms x 1044 pairs x 6 rules, no training
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

# =============================================================================
#  Re-score the four trained arms at a MATCHED point budget
# =============================================================================
#
# No training. This reads the checkpoints TrainSuperPathPoint.sh wrote, runs the
# same held-out pairs through each one once, and cuts the resulting probability
# maps several ways.
#
# WHY. The 2026-08-31 run produced two groups of numbers that look like a
# result and are not:
#
#   arm        pts/view   decoy   margin   ceiling
#   gray            420   0.442     1.50      2.26
#   rgb             420   0.420     1.57      2.38
#   gray_pre        159   0.221     3.59      4.52
#   rgb_pre         161   0.222     3.56      4.51
#
# `margin = repeatability / decoy`, and the decoy is what point DENSITY alone
# buys -- so the ceiling `1/decoy` moves with the density. The `_pre` arms were
# not 2.4x better; they were scored at a third of the density, where the ceiling
# is twice as high. Nothing in that table compares the four models.
#
# AND THE 420 IS NOT A POINT COUNT. It is `max_keypoints`, hit exactly, on every
# tile. A count that lands on the cap to the integer is the cap selecting, not
# the model. The cause is arithmetic and it is worth writing down:
#
#     the detector is a 65-way softmax per cell, so a model that has learnt
#     nothing puts 1/65 = 0.015385 on every class
#     `detection_threshold` was 0.015
#
# The threshold sits BELOW the value of total ignorance. For an undertrained
# detector every cell passes it, NMS thins the field to a few thousand, and the
# cap takes the top 420. Meanwhile the `_pre` arms, whose detector CE is 0.19,
# emit a sharp map and land on their own density -- 254 on BRACS_1598 against a
# label density of about 295, and 62 on the Ki67 slide against about 40. Those
# arms were reproducing their teacher, which is what self-distillation from the
# weights that MADE the labels does.
#
# WHAT THIS RUN DOES INSTEAD. `--budgets` cuts each view to exactly the top N by
# score, threshold at zero, so the density is pinned by construction and the
# decoy becomes the same quantity for every arm. The ladder is the LABEL's own
# densities (per (slide, rung) `n_kp` means run 3 to 527 over the 72 stores;
# the two held-out slides average about 295 and 40), plus 420 so the old
# numbers still have something to be compared against.
#
# WHAT IT STILL CANNOT SETTLE, unchanged by any of this:
#
#   * cross-stain generalisation -- both stains are in TRAIN, so held-out means
#     an unseen SLIDE (spec.md 6.5)
#   * `gray_pre` vs `gray` as a question about pretraining -- the labels came
#     from those weights, so the arm asks "does a second round on its own
#     labels beat learning them from scratch" (spec.md 13)
#   * six pairwise comparisons on two held-out slides. One arm ahead on one
#     slide and behind on the other is UNDECIDED, not a small win
#
#   sbatch jobscripts/SuperPathPointJobs/ReEvalSuperPathPoint.sh
#   ARMS="gray gray_pre" sbatch .../ReEvalSuperPathPoint.sh
#   EPOCH_TAG=epoch021 sbatch .../ReEvalSuperPathPoint.sh
#
ARMS="${ARMS:-gray rgb gray_pre rgb_pre}"

TILE=256

# Which checkpoint. `last` is epoch 49 of the 2026-08-31 run. It is the default
# rather than best-by-margin because the margin those arms were ranked by is
# the unmatched one this run exists to replace -- picking a checkpoint with a
# broken instrument and then re-measuring it with a fixed one would carry the
# break forward. Once the matched table exists, a checkpoint rule can be an
# argument about numbers instead of a guess.
EPOCH_TAG="${EPOCH_TAG:-last}"

# The matched budgets. Every view is cut to exactly this many points by score,
# so `points_per_view` in the output MUST come back equal to the budget -- the
# CLI prints a WARNING for any row where it did not, which is NMS having left
# fewer survivors than the budget asked for.
BUDGETS="${BUDGETS:-40 80 160 320 420}"

# Where the training run put them. A literal, not `result/$SLURM_JOB_NAME`:
# this job's name is ReEvalSuperPathPoint and the checkpoints are not in its
# own output directory.
MODELS_ROOT="${MODELS_ROOT:-/work/u26130998/result/TrainSuperPathPoint}"

echo "======== ReEvalSuperPathPoint ========"
echo "  arms $ARMS   tile $TILE   checkpoint $EPOCH_TAG"
echo "  budgets $BUDGETS"
echo "  models  $MODELS_ROOT"
echo ""

python training/SuperPathPoint/cli/reeval_density.py \
  --models-root "$MODELS_ROOT" \
  --arms $ARMS \
  --tile "$TILE" \
  --epoch-tag "$EPOCH_TAG" \
  --budgets $BUDGETS \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
status=$?

echo ""
echo "======== done  (exit $status) ========"
echo "  table -> result/\${SLURM_JOB_NAME}/matched_density.csv"
echo ""
echo "  Read DOWN one rule block, never across two. The ceiling is 1/decoy and"
echo "  the decoy is a function of the density, so a margin at top40 and one at"
echo "  top420 are on two different scales."
echo ""
echo "  The crossing is the interesting case: an arm that wins at top40 and"
echo "  loses at top320 found a few good points and nothing else, which is a"
echo "  different object from one that is uniformly better. That is what the"
echo "  ladder is for, and one budget could not have shown it."

exit $status
