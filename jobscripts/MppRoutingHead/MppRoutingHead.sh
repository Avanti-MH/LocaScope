#!/bin/bash
#SBATCH --job-name=MppRoutingHead                  # Job name
#SBATCH --partition=normal2                        # Partition
#SBATCH --time=48:00:00                            # Runtime (hh:mm:ss)
#SBATCH --account=MST114560                        # Account
#SBATCH --nodes=1                                  # Number of nodes
#SBATCH --gpus-per-node=2                          # GPUs per node (不要設0)
#SBATCH --cpus-per-task=12                         # DataLoader workers + main
#SBATCH --ntasks-per-node=1                        # Tasks per node
#SBATCH --mem=600G                                 # host RAM
#SBATCH -o /work/u26130998/log/MppRoutingHead      # STDOUT
#SBATCH -e /work/u26130998/log/MppRoutingHead      # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  training/MppRoutingHead -- ONE JOB, THREE TRAINING PASSES, then test:
#
#    [1a] bal     baseline 2 (gigapath, uni2) + baseline 3 (convnext_v2),
#                 every head
#    [1b] ord_a   baseline 2, uni2 only, every head
#    [1c] ord_b   baseline 2, uni2 only, every head
#    [2]  evaluate every *_best.pt on the test split
#
#  RESUMABLE. RESUME_DIR (default $OUT/resume) receives every model's full
#  state every epoch; re-submitting THIS SAME COMMAND after a walltime kill
#  continues each model from its last finished epoch and skips the finished
#  ones (--epochs is the total). RESUME_DIR= (empty) turns it off.
#
#  --merge is always on: the three passes write into the same val_scores.csv,
#  keyed by (baseline, encoder, head, loss), so none erases another's rows.
#
#  Train and evaluate are TWO STEPS IN ONE JOB because they
#  are two halves of one question and the second one is minutes next to the
#  first one's hours; keeping them in separate submissions only creates a gap
#  in which the weights sit unscored.
#
#  They stay two PROGRAMS, though, and that is the point of the split:
#  `cli/train.py` never opens the test manifest. Test lives behind
#  `cli/evaluate.py` alone, so "has anyone looked at test" is answerable by
#  which files exist.
#
#  SMOKE FIRST. Set SMOKE=1 (below) to run 2 WSIs, 1 epoch, one arm, one
#  encoder -- every line of plumbing, none of the cost. The full run is hours
#  of Camera renders and there is no cheap failure in the middle of it: a
#  missing checkpoint or a wrong dtype surfaces after the manifest build, which
#  is itself tens of minutes.
#
#  PARALLEL=1 runs baseline 2 and baseline 3 as TWO SEPARATE `train.py`
#  processes, one per GPU, at the same time -- see the note at the bottom for
#  what this does and does not buy, and why it might buy nothing.
# =============================================================================

# ---------------- knobs ----------------
SMOKE="${SMOKE:-0}"
PARALLEL="${PARALLEL:-0}"
BASELINE="${BASELINE:-all}"
ENCODERS="${ENCODERS:-gigapath uni2}"
ARMS="${ARMS:-arcface attn_linear linear mlp mlp_deep mlp_deep_residual mlp_wide mlp_deep_wide mlp_narrow}"
# The ordinal passes: which losses, and on which baseline-2 encoders.
# ORD_LOSSES= (empty) skips both.
ORD_LOSSES="${ORD_LOSSES:-ord_a ord_b}"
ORD_ENCODERS="${ORD_ENCODERS:-uni2}"
EPOCHS="${EPOCHS:-20}"
SEG="${SEG:-hest}"         # tissue-mask recipe (TissueMaskConfig.MASK_RECIPES)
N_PER_RUNG="${N_PER_RUNG:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WSI_GROUP="${WSI_GROUP:-8}"
MAX_WSI="${MAX_WSI:-}"
# Set below, once $OUT is known: default $OUT/resume. RESUME_DIR= (set but
# empty) disables resume -- every model trains from scratch, nothing written.
# 0 (default) = off for both. CLIP_GRAD_NORM caps every optimizer's L2
# gradient norm before each step; WARMUP_EPOCHS linearly ramps LR from 0
# over that many epochs. Both per --case, not project-wide defaults -- e.g.
# the 2026-09-17 convnext_v2/attn_linear collapse (epoch 6, joint trunk+head
# fine-tune with no protection) is the case that wants them; uni2's frozen
# mlp variants do not. See cli/train.py's --clip-grad-norm/--warmup-epochs
# help.
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
# ord_a/ord_b = spec.md's "Ordinal-aware loss" section, formulas 3a/3b -- see
# cli/train.py's --loss help. ORDINAL_WEIGHT/ORDINAL_SIGMA only matter under
# ord_a/ord_b respectively.
ORDINAL_WEIGHT="${ORDINAL_WEIGHT:-1.0}"
ORDINAL_SIGMA="${ORDINAL_SIGMA:-1.0}"
# 1: skip step 1 entirely and go straight to step 2 -- rescores whatever is
# already sitting in $OUT/weights (nothing trained this run). For testing
# evaluate.py itself (e.g. its print_and_plot output) without paying for a
# training run -- $OUT still has to already have weights in it, normally
# from an earlier plain run of this same script.
EVAL_ONLY="${EVAL_ONLY:-0}"
TAG="${TAG:-best}"
WANDB_PROJECT="${WANDB_PROJECT:-mpp-routing-head}"
# Defaults to the SLURM job id, not empty -- an empty RUN_NAME still gets a
# real wandb run name (b2-<encoder> / b3-<arm>, see cli/train.py), but with
# no prefix two submissions of this same script are two runs with the
# IDENTICAL name, distinguishable only by timestamp in the wandb UI.
# $SLURM_JOB_ID does not exist yet at `sbatch` time on the submitting shell --
# it is only set once this script is actually running on the allocated node,
# which is exactly why this default lives HERE and not in a `sbatch
# --export=...` example.
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-}}"
# WANDB_MODE itself is NOT set here -- jobscripts/_env.sh (sourced above)
# already exports it project-wide, and train.py's own --wandb-mode default
# reads it. This file only needs to pass the project/run-name through.
RUN_NAME_ARG=""
[ -n "$RUN_NAME" ] && RUN_NAME_ARG="--run-name $RUN_NAME"

# CPUS actually granted, not just what this file asks for -- SLURM sets
# SLURM_CPUS_PER_TASK, and a resubmit with different --cpus-per-task should
# not need this file edited to match.
CPUS="${SLURM_CPUS_PER_TASK:-12}"

# Sequential: leave 2 cores for the main process + OS/CUDA overhead, same
# ratio as before (8 cpus -> 6 workers). Parallel: TWO main processes now
# share the allocation, so each gets its own slice -- (CPUS-2)/2 workers,
# leaving 1 core per process. At CPUS=12 that is 5 workers each side.
NUM_WORKERS="${NUM_WORKERS:-$((CPUS > 2 ? CPUS - 2 : 1))}"
NUM_WORKERS_PAR="${NUM_WORKERS_PAR:-$(( (CPUS - 2) / 2 > 0 ? (CPUS - 2) / 2 : 1 ))}"

if [ "$SMOKE" = "1" ]; then
    # Everything small enough to finish in minutes, and still exercises:
    # manifest build, the val/test WSI split, Camera render in workers,
    # a frozen encoder forward, one backward, checkpoint write, checkpoint
    # RELOAD in a separate process, and both CSV writers.
    #
    # PARALLEL is IGNORED under smoke, not merely unhelpful: smoke forces
    # --baseline 2 (BASELINE_SMOKE), so there is only one baseline to split
    # across two GPUs, and the merge step below has nothing to merge.
    BASELINE="${BASELINE_SMOKE:-2}"
    ENCODERS="${ENCODERS_SMOKE:-gigapath}"
    ARMS="${ARMS_SMOKE:-linear}"
    ORD_LOSSES="${ORD_LOSSES_SMOKE:-}"
    EPOCHS=1
    N_PER_RUNG=4
    MAX_WSI=2
    PARALLEL=0
    echo "======== SMOKE ========"
    echo "  (pass WANDB_MODE=offline in --export if wandb is configured and"
    echo "   you do not want a smoke run showing up on the server)"
fi

if [ "$PARALLEL" = "1" ] && [ "$BASELINE" != "all" ]; then
    echo "PARALLEL=1 needs --baseline all (only one baseline was requested; running it on one GPU)"
    PARALLEL=0
fi

# Resolved to a CONCRETE path here, in bash, rather than left for train.py's
# own default -- PARALLEL mode needs $OUT/b2 and $OUT/b3 as real strings to
# pass as --out to the two processes and to merge afterward, so this file
# has to know it, not just train.py.
if [ -z "$OUT" ]; then
    OUT=$(python -c "
import sys
sys.path.insert(0, 'utilities')
import _paths
print(_paths.job_result_dir('MppRoutingHead'))
")
fi
mkdir -p "$OUT"
# `${RESUME_DIR-...}` (no colon): unset -> the default, set-but-empty -> off.
RESUME_DIR="${RESUME_DIR-$OUT/resume}"
MAX_WSI_ARG=""
[ -n "$MAX_WSI" ] && MAX_WSI_ARG="--max-wsi $MAX_WSI"
RESUME_DIR_ARG=""
[ -n "$RESUME_DIR" ] && RESUME_DIR_ARG="--resume-dir $RESUME_DIR"
# Always --merge: three passes share val_scores.csv (see the header).
MERGE_ARG="--merge"
TRAIN_EXTRA_ARGS=""
[ "$CLIP_GRAD_NORM" != "0" ] && TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS --clip-grad-norm $CLIP_GRAD_NORM"
[ "$WARMUP_EPOCHS" != "0" ] && TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS --warmup-epochs $WARMUP_EPOCHS"

echo "======== MppRoutingHead ========"
echo "  out       $OUT"
echo "  baseline  $BASELINE   parallel=$PARALLEL"
echo "  encoders  $ENCODERS"
echo "  arms      $ARMS"
echo "  bal on    baseline $BASELINE   ordinal passes: ${ORD_LOSSES:-(none)} on ${ORD_ENCODERS}"
echo "  epochs    $EPOCHS   n_per_rung $N_PER_RUNG   cpus $CPUS"
echo "  wandb     project=$WANDB_PROJECT   mode=${WANDB_MODE:-online, unset here -- train.py falls back to the same}"
echo "  run_name  prefix=${RUN_NAME:-(none)}   (defaults to \$SLURM_JOB_ID; runs are still b2-<encoder> / b3-<arm> even with no prefix)"
[ -n "$MAX_WSI" ] && echo "  max_wsi   $MAX_WSI"
echo "  resume    ${RESUME_DIR:-off (RESUME_DIR set empty)}"
[ "$CLIP_GRAD_NORM" != "0" ] && echo "  clip_grad_norm  $CLIP_GRAD_NORM"
[ "$WARMUP_EPOCHS" != "0" ] && echo "  warmup_epochs   $WARMUP_EPOCHS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# =============================================================================
#  Step 1: train
# =============================================================================
echo ""
# The val/test split is written once, by make_split.py, and only READ by
# train/evaluate (and by PrototypicalRoutingHead and the stage-1 bench). A
# split that already exists is kept -- deleting it is how you ask for a new one.
# --cache-job MakeSplit: this runs under SLURM_JOB_NAME=MppRoutingHead, and
# every reader defaults to the split make_split.py writes under its OWN name.
echo "======== [0] make_split ========"
python utilities/cli/build_cache/make_split.py --cache-job MakeSplit || exit $?

echo "======== [1a] train, loss bal ========"

if [ "$EVAL_ONLY" = "1" ]; then
    echo "  EVAL_ONLY=1 -- skipping training, scoring whatever is already in $OUT/weights"
    if [ ! -d "$OUT/weights" ]; then
        echo "  [abort] $OUT/weights does not exist -- EVAL_ONLY needs an earlier plain run's output to score"
        exit 1
    fi
elif [ "$PARALLEL" = "1" ]; then
    # ---------------- two processes, one GPU each ----------------
    # `--out` is DIFFERENT for each ($OUT/b2, $OUT/b3), not the shared $OUT --
    # cli/train.py's own main() writes val_scores.csv with `open(path, 'w')`,
    # a full truncate-then-write. Two processes finishing around the same
    # moment and pointed at the SAME file would have the second one's write
    # clobber the first's, silently -- one baseline's rows just vanish. The
    # mask and sampler caches under result/cache/<job>_mask/ and <job>_sampler/
    # ARE shared on purpose (that is the whole point of caching), and their
    # writes are atomic
    # (utilities/Cache.py) for exactly this: two processes racing to draw the
    # SAME slide for the first time compute the same deterministic content and
    # the first rename wins, rather than two writers interleaving into one
    # corrupt entry.
    #
    # Logs go to separate files -- interleaved stdout from two training loops
    # printing per-epoch lines at their own pace is not "still readable, just
    # ugly", it is two numbers from two different epochs landing on one line.
    mkdir -p "$OUT/b2" "$OUT/b3"
    LOG2="/work/u26130998/log/MppRoutingHead_b2.log"
    LOG3="/work/u26130998/log/MppRoutingHead_b3.log"
    echo "  baseline 2 -> GPU 0, $NUM_WORKERS_PAR workers, log: $LOG2"
    echo "  baseline 3 -> GPU 1, $NUM_WORKERS_PAR workers, log: $LOG3"

    CUDA_VISIBLE_DEVICES=0 python training/MppRoutingHead/cli/train.py \
        --baseline 2 \
        --encoders $ENCODERS \
        --heads $ARMS \
        --epochs "$EPOCHS" \
        --seg "$SEG" \
        --n-per-rung "$N_PER_RUNG" \
        --batch-size "$BATCH_SIZE" \
        --wsi-group-size "$WSI_GROUP" \
        --num-workers "$NUM_WORKERS_PAR" \
        --device cuda:0 \
        --wandb-project "$WANDB_PROJECT" \
        $RUN_NAME_ARG \
        $MAX_WSI_ARG $RESUME_DIR_ARG $MERGE_ARG $TRAIN_EXTRA_ARGS --out "$OUT/b2" \
        > "$LOG2" 2>&1 &
    pid2=$!

    CUDA_VISIBLE_DEVICES=1 python training/MppRoutingHead/cli/train.py \
        --baseline 3 \
        --heads $ARMS \
        --epochs "$EPOCHS" \
        --seg "$SEG" \
        --n-per-rung "$N_PER_RUNG" \
        --batch-size "$BATCH_SIZE" \
        --wsi-group-size "$WSI_GROUP" \
        --num-workers "$NUM_WORKERS_PAR" \
        --device cuda:1 \
        --wandb-project "$WANDB_PROJECT" \
        $RUN_NAME_ARG \
        $MAX_WSI_ARG $RESUME_DIR_ARG $MERGE_ARG $TRAIN_EXTRA_ARGS --out "$OUT/b3" \
        > "$LOG3" 2>&1 &
    pid3=$!

    # `&` starts both without waiting -- see the two `wait`s below for the
    # part that actually blocks until they are done. Two separate `wait`
    # calls (not `wait $pid2 $pid3`) so each exit code is captured under its
    # OWN name -- `wait` on several PIDs at once returns the LAST one's
    # status, which would silently drop a failure in the other.
    wait "$pid2"; rc2=$?
    wait "$pid3"; rc3=$?
    echo ""
    echo "  baseline 2 exit $rc2   (see $LOG2)"
    echo "  baseline 3 exit $rc3   (see $LOG3)"
    tail -n 25 "$LOG2" | sed 's/^/  [b2] /'
    tail -n 25 "$LOG3" | sed 's/^/  [b3] /'

    if [ $rc2 -ne 0 ] || [ $rc3 -ne 0 ]; then
        echo "at least one parallel run failed -- not merging, not scoring test"
        exit 1
    fi

    # ---------------- merge b2 + b3 into one $OUT ----------------
    # weights/ never collides: `Runtime.weight_filename` names every file
    # `<encoder>_<frozen|finetuned>_<arm>_<tag>.pt`, and b2's encoders
    # (gigapath, uni2, frozen) and b3's (convnext_v2, finetuned) cannot
    # produce the same string. val_scores.csv DOES need a real merge -- both
    # files share the same columns (both come out of the same
    # `val_report`/`score` code), so this is concat-with-one-header, not a
    # join.
    mkdir -p "$OUT/weights"
    cp "$OUT"/b2/weights/*.pt "$OUT/weights/" 2>/dev/null
    cp "$OUT"/b3/weights/*.pt "$OUT/weights/" 2>/dev/null
    python - "$OUT/b2/val_scores.csv" "$OUT/b3/val_scores.csv" "$OUT/val_scores.csv" <<'EOF'
import csv, sys
a, b, out = sys.argv[1:4]
rows = []
header = None
for path in (a, b):
    with open(path, newline='') as fh:
        r = csv.DictReader(fh)
        header = header or r.fieldnames
        rows += list(r)
with open(out, 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=header)
    wr.writeheader()
    wr.writerows(rows)
print(f'{out}  ({len(rows)} rows, merged from {a} + {b})')
EOF
else
    # ---------------- sequential: one process, everything ----------------
    python training/MppRoutingHead/cli/train.py \
        --baseline "$BASELINE" \
        --encoders $ENCODERS \
        --heads $ARMS \
        --epochs "$EPOCHS" \
        --seg "$SEG" \
        --n-per-rung "$N_PER_RUNG" \
        --batch-size "$BATCH_SIZE" \
        --wsi-group-size "$WSI_GROUP" \
        --num-workers "$NUM_WORKERS" \
        --wandb-project "$WANDB_PROJECT" \
        $RUN_NAME_ARG \
        $MAX_WSI_ARG $RESUME_DIR_ARG $MERGE_ARG $TRAIN_EXTRA_ARGS --out "$OUT"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "train.py exited $rc -- not scoring test on weights that may not exist"
        exit $rc
    fi
fi

# ---------------- [1b]/[1c] the ordinal passes ----------------
# Baseline 2 only, on ORD_ENCODERS only (uni2 by default): an ordinal loss is
# a question about the head on the encoder whose bal numbers are the ones
# worth moving, not a sweep over every trunk. Same heads, same data, same
# resume directory; the loss is in every file name, so nothing collides.
if [ "$EVAL_ONLY" != "1" ]; then
    for L in $ORD_LOSSES; do
        echo ""
        echo "======== [1] train, loss $L  (baseline 2: $ORD_ENCODERS) ========"
        python training/MppRoutingHead/cli/train.py \
            --baseline 2 \
            --encoders $ORD_ENCODERS \
            --heads $ARMS \
            --loss "$L" \
            --ordinal-weight "$ORDINAL_WEIGHT" \
            --ordinal-sigma "$ORDINAL_SIGMA" \
            --epochs "$EPOCHS" \
            --seg "$SEG" \
            --n-per-rung "$N_PER_RUNG" \
            --batch-size "$BATCH_SIZE" \
            --wsi-group-size "$WSI_GROUP" \
            --num-workers "$NUM_WORKERS" \
            --wandb-project "$WANDB_PROJECT" \
            $RUN_NAME_ARG \
            $MAX_WSI_ARG $RESUME_DIR_ARG $MERGE_ARG $TRAIN_EXTRA_ARGS --out "$OUT"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "train.py ($L) exited $rc -- not scoring test"
            exit $rc
        fi
    done
fi

# =============================================================================
#  Step 2: evaluate
# =============================================================================
# Rebuilds each checkpoint from the file alone (no --encoder, no --arm: those
# would be a second source of truth able to disagree with the weights) and
# scores it on 5 WSIs x 6 rungs x 50 positions per dataset. Writes
# test_scores_<tag>.csv and test_predictions_<tag>.csv, the second with one row
# per tile. Sequential regardless of PARALLEL: both GPUs are free again by
# this point, and evaluate.py has no baseline split to parallelise across --
# it loops over CHECKPOINTS, which is a few minutes next to training's hours.
echo ""
echo "======== [2] evaluate ========"
python training/MppRoutingHead/cli/evaluate.py \
    --tag "$TAG" \
    --seg "$SEG" \
    --num-workers "$NUM_WORKERS" \
    --out "$OUT"
rc=$?

echo ""
echo "======== done (exit $rc) ========"
exit $rc

# =============================================================================
#  ONE NODE, TWO GPUS -- what PARALLEL=1 buys and what it might not
#
#  Sequential (PARALLEL=0, the default): one process, one GPU, baseline 2 then
#  baseline 3. The second GPU sits idle -- requested anyway (see below) as
#  headroom against an OOM at hour six of a forty-eight hour allocation, not
#  because the code uses it.
#
#  Parallel (PARALLEL=1): baseline 2 and baseline 3 are independent -- neither
#  reads the other's output -- so running them as two processes on two GPUs is
#  correct, not just fast. What it is NOT proven to help: the render (Camera
#  capture, single-threaded openslide decode per worker) is done by CPU
#  workers, not the GPU, and PARALLEL mode HALVES how many workers each
#  baseline gets ($NUM_WORKERS_PAR vs $NUM_WORKERS). If render time dominates
#  wall-clock, as the smoke run's own timing suggested it might, two GPUs at
#  half the render throughput each can land close to one GPU at full
#  throughput -- a wash, not a 2x speedup. This is exactly why SMOKE forces
#  PARALLEL off: the answer needs a real run's numbers, not a guess, and the
#  first full run's two log files (MppRoutingHead_b2.log / _b3.log) are where
#  to look for it -- specifically how much of each epoch is the render loop
#  vs the forward/backward.
#
#  A genuinely different way to use two GPUs -- one encoder per GPU for
#  baseline 2's two encoders (gigapath, uni2), sharing ONE render pass between
#  them instead of running it twice -- would actually cut render work rather
#  than split it, but is a code change to `run_baseline2`, not a jobscript
#  knob, and is not worth designing before a render-vs-forward number exists.
# =============================================================================
