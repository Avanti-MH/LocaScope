#!/bin/bash
#SBATCH --job-name=ClassifyingRoutingHead                  # Job name
#SBATCH --partition=8gpus                          # Partition
#SBATCH --time=48:00:00                            # Runtime (hh:mm:ss)
#SBATCH --account=MST114560                        # Account
#SBATCH --nodes=1                                  # Number of nodes
#SBATCH --gpus-per-node=3                          # one per queue: gigapath, uni2, convnext_v2 (不要設0)
#SBATCH --cpus-per-task=36                         # DataLoader workers + main; 12 per GPU is the cap
#SBATCH --ntasks-per-node=1                        # Tasks per node
#SBATCH --mem=600G                                 # host RAM, 200G per GPU, the H200 cap
#SBATCH -o /work/u26130998/log/%x      # STDOUT
#SBATCH -e /work/u26130998/log/%x      # STDERR

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  training/ClassifyingRoutingHead -- ONE JOB, ONE QUEUE PER ENCODER, then test:
#
#    [1]  gigapath     baseline 2, every ARMS head:   bal -> ord_a -> ord_b   GPU 0
#         uni2         baseline 2, every ARMS head:   bal -> ord_a -> ord_b   GPU 1
#         convnext_v2  baseline 3, B3_ARMS (mlp_deep): bal -> ord_a -> ord_b  GPU 2
#         the queues run at once and never wait on one another
#    [2]  evaluate every *--best.pt on the test set (TEST_DATASET, @pure-half-test)
#
#  RESUMABLE. RESUME_DIR (default $OUT/resume) receives every model's full
#  state every epoch; re-submitting THIS SAME COMMAND after a walltime kill
#  continues each model from its last finished epoch and skips the finished
#  ones (--epochs is the total). RESUME_DIR= (empty) turns it off.
#
#  --merge is always on: a queue's losses write into the same val_scores.csv,
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
#  PARALLEL=1 runs the queues as separate `train.py` processes, one per GPU, at
#  the same time -- see "the queues" below for what that does and does not buy.
# =============================================================================

# ---------------- knobs ----------------
SMOKE="${SMOKE:-0}"
PARALLEL="${PARALLEL:-1}"
BASELINE="${BASELINE:-all}"
ENCODERS="${ENCODERS:-gigapath uni2}"
# Baseline 2's heads -- every one off the same encoder pass, so a head costs little:
# one mechanism each, paired against the head it differs from by that one thing
# (mlp_deep is the baseline). Baseline 3 trains one run per head, so it gets its own
# short list.
ARMS="${ARMS:-mlp_deep linear mlp mlp_wide mlp_narrow mlp_deep_wide mlp_deep_residual arcface attn_mlp_deep clsattn_mlp_deep mix_attn_mlp_deep mix_clsattn_mlp_deep attn_linear mix_attn_linear}"
B3_ARMS="${B3_ARMS:-mlp_deep}"
# The ordinal passes: which losses, and on which encoders -- baseline 2's frozen ones
# (every ARMS head, one pass each) and convnext_v2, baseline 3's fine-tuned trunk
# (B3_ARMS only, one run per head).
# ORD_LOSSES= (empty) skips both.
ORD_LOSSES="${ORD_LOSSES:-ord_a ord_b}"
ORD_ENCODERS="${ORD_ENCODERS:-gigapath uni2 convnext_v2}"
# Which slides: train is --train-dataset's --train-stain slides (@pure-train's
# Ki67 half), val and test are virtual datasets (make_dataset.py).
TRAIN_DATASET="${TRAIN_DATASET:-@pure-train}"
TRAIN_STAIN="${TRAIN_STAIN:-Ki67}"
VAL_DATASET="${VAL_DATASET:-@pure-half-val}"
TEST_DATASET="${TEST_DATASET:-@pure-half-test}"
EVAL_N_WSI="${EVAL_N_WSI:-10}"   # test slides of each dataset TEST_DATASET draws from
EPOCHS="${EPOCHS:-20}"
SEG="${SEG:-hest}"         # tissue-mask recipe (TissueMaskConfig.MASK_RECIPES)
N_PER_RUNG="${N_PER_RUNG:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WSI_GROUP="${WSI_GROUP:-8}"
MAX_WSI="${MAX_WSI-121}"
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
WANDB_PROJECT="${WANDB_PROJECT:-classifying-routing-head}"
# Empty by default: cli/train.py then names the wandb run after the job
# (SLURM_JOB_NAME) -- <job name>-b2-<encoder> / <job name>-b3-<arm> -- and keeps
# the job id and name in the run's config. Set RUN_NAME to replace the job name.
RUN_NAME="${RUN_NAME:-}"
# WANDB_MODE itself is NOT set here -- jobscripts/_env.sh (sourced above)
# already exports it project-wide, and train.py's own --wandb-mode default
# reads it. This file only needs to pass the project/run-name through.
RUN_NAME_ARG=""
[ -n "$RUN_NAME" ] && RUN_NAME_ARG="--run-name $RUN_NAME"

# CPUS actually granted, not just what this file asks for -- SLURM sets
# SLURM_CPUS_PER_TASK, and a resubmit with different --cpus-per-task should
# not need this file edited to match.
CPUS="${SLURM_CPUS_PER_TASK:-24}"

# Sequential: leave 2 cores for the main process + OS/CUDA overhead. Parallel:
# each queue gets its own slice (NUM_WORKERS_Q, set where the queues start).
NUM_WORKERS="${NUM_WORKERS:-$((CPUS > 2 ? CPUS - 2 : 1))}"

if [ "$SMOKE" = "1" ]; then
    # Everything small enough to finish in minutes, and still exercises:
    # manifest build, the val/test WSI sets, Camera render in workers,
    # a frozen encoder forward, one backward, checkpoint write, checkpoint
    # RELOAD in a separate process, and both CSV writers.
    #
    # PARALLEL is off under smoke: smoke forces --baseline 2 and one encoder
    # (BASELINE_SMOKE, ENCODERS_SMOKE), one queue, so there is nothing to spread.
    BASELINE="${BASELINE_SMOKE:-2}"
    ENCODERS="${ENCODERS_SMOKE:-gigapath}"
    ARMS="${ARMS_SMOKE:-linear}"
    ORD_LOSSES="${ORD_LOSSES_SMOKE:-}"
    EPOCHS=1
    N_PER_RUNG=4
    MAX_WSI=2
    EVAL_N_WSI=2
    PARALLEL=0
    echo "======== SMOKE ========"
    echo "  (pass WANDB_MODE=offline in --export if wandb is configured and"
    echo "   you do not want a smoke run showing up on the server)"
fi

# Resolved to a CONCRETE path here, in bash, rather than left for train.py's
# own default -- PARALLEL mode needs $OUT/<encoder> as real strings to
# pass as --out to the queues and to merge afterward, so this file
# has to know it, not just train.py.
if [ -z "$OUT" ]; then
    OUT=$(python -c "
import sys
sys.path.insert(0, 'utilities')
import _paths
print(_paths.job_result_dir('ClassifyingRoutingHead'))
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
# Training read mode (cli/train.py --read-level and its three companions,
# spec.md 詞彙). Unset: train.py's defaults, i.e. 'pyramid', the only mode
# before these existed -- same file names, same resume files.
[ -n "${READ_LEVEL:-}" ] && TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS --read-level $READ_LEVEL"
[ -n "${RESAMPLE_FROM:-}" ] && TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS --resample-from $RESAMPLE_FROM"
[ -n "${MAX_RESAMPLE_FACTOR:-}" ] && TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS --max-resample-factor $MAX_RESAMPLE_FACTOR"
[ -n "${RESAMPLED_SHARE:-}" ] && TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS --resampled-share $RESAMPLED_SHARE"

# WHOSE CACHES TO READ -- train.py's and evaluate.py's --mask-cache-job /
# --draw-cache-job. Without them a cache is named after
# THIS job (SLURM_JOB_NAME), so a run under another --job-name (a smoke, a side
# experiment) finds an empty result/cache/<that name>/ and segments every
# slide again. So both are named, the same in PrototypicalRoutingHead.sh:
#
#     MASK_CACHE_JOB        the masks: Masks, the cache every trainer and bench
#                           shares
#     DRAW_CACHE_JOB        the draws: MppRoutingHead, where both trainers' draws
#                           still are under this package's old name (the masks
#                           moved to Masks on 2026-10-20)
#
# The val and test sets are VAL_DATASET and TEST_DATASET above, virtual datasets
# written by jobscripts/MakeDataset.sh.
MASK_CACHE_JOB="${MASK_CACHE_JOB:-Masks}"
# A smoke draws with its own numbers (N_PER_RUNG 4), which no real run can reuse:
# its draws go to its own cache (Smoke*, what make clean-smoke removes). Its masks
# are the real ones, so they stay shared.
if [ "$SMOKE" = "1" ]; then DRAW_DEFAULT="${SLURM_JOB_NAME:-SmokeClassifyingRoutingHead}"; else DRAW_DEFAULT=MppRoutingHead; fi
DRAW_CACHE_JOB="${DRAW_CACHE_JOB:-$DRAW_DEFAULT}"
CACHE_ARGS="--mask-cache-job $MASK_CACHE_JOB --draw-cache-job $DRAW_CACHE_JOB"
TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS $CACHE_ARGS"
TRAIN_EXTRA_ARGS="$TRAIN_EXTRA_ARGS --train-dataset $TRAIN_DATASET --train-stain $TRAIN_STAIN --val-dataset $VAL_DATASET"

echo "======== ClassifyingRoutingHead ========"
echo "  out       $OUT"
echo "  baseline  $BASELINE   parallel=$PARALLEL"
echo "  read      ${READ_LEVEL:-pyramid}${RESAMPLE_FROM:+ from $RESAMPLE_FROM}${MAX_RESAMPLE_FACTOR:+ max x$MAX_RESAMPLE_FACTOR}${RESAMPLED_SHARE:+ share $RESAMPLED_SHARE}"
echo "  caches    mask from $MASK_CACHE_JOB, draws from $DRAW_CACHE_JOB"
echo "  data      train $TRAIN_DATASET ($TRAIN_STAIN)   val $VAL_DATASET   test $TEST_DATASET"
echo "  encoders  $ENCODERS"
echo "  arms      baseline 2: $ARMS"
echo "            baseline 3: $B3_ARMS"
echo "  bal on    baseline $BASELINE   ordinal passes: ${ORD_LOSSES:-(none)} on ${ORD_ENCODERS}"
echo "  epochs    $EPOCHS   n_per_rung $N_PER_RUNG   cpus $CPUS"
echo "  wandb     project=$WANDB_PROJECT   mode=${WANDB_MODE:-online, unset here -- train.py falls back to the same}"
echo "  run_name  prefix=${RUN_NAME:-(job name)}   (runs are <prefix>-b2-<encoder> / <prefix>-b3-<arm>)"
[ -n "$MAX_WSI" ] && echo "  max_wsi   $MAX_WSI"
echo "  resume    ${RESUME_DIR:-off (RESUME_DIR set empty)}"
[ "$CLIP_GRAD_NORM" != "0" ] && echo "  clip_grad_norm  $CLIP_GRAD_NORM"
[ "$WARMUP_EPOCHS" != "0" ] && echo "  warmup_epochs   $WARMUP_EPOCHS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---------------- the queues: one encoder, one GPU ----------------
# One queue per encoder -- ENCODERS (baseline 2, frozen, every ARMS head) and
# convnext_v2 (baseline 3, fine-tuned, B3_ARMS) -- trains bal and then, for an
# encoder in ORD_ENCODERS, each of ORD_LOSSES, one after another. The losses are
# independent, so no queue ever waits on another; each writes its own --out
# ($OUT/<encoder>) and its own log, because cli/train.py rewrites its val_scores
# files whole and two processes on one file would drop one's rows. The mask and
# draw caches ARE shared on purpose: their writes are atomic (utilities/Cache.py).
#
# PARALLEL=1 (default): every queue at once, one GPU each (ask for that many).
# PARALLEL=0: one queue after another on one GPU, writing $OUT itself.
#
# A cost to know: ONE baseline-2 process with several encoders renders each batch
# once and encodes it with each (run_baseline2); a queue per encoder renders it
# once PER encoder. Faster only where the GPU, not the render, is the bottleneck --
# every epoch line now prints `time  render wait ... train ...` to say which.
QUEUES=""
case "$BASELINE" in all|2) QUEUES="$ENCODERS" ;; esac
case "$BASELINE" in all|3) QUEUES="$QUEUES convnext_v2" ;; esac

# losses_of <encoder>: bal, then ORD_LOSSES when the encoder is in ORD_ENCODERS
losses_of () {
    local out="bal" o
    for o in $ORD_ENCODERS; do [ "$o" = "$1" ] && out="bal $ORD_LOSSES"; done
    echo $out
}

# train_unit <encoder> <loss> <out> [train.py args...]: one training of one encoder
train_unit () {
    local E="$1" L="$2" O="$3" B_ARGS
    shift 3
    if [ "$E" = "convnext_v2" ]; then B_ARGS="--baseline 3 --heads $B3_ARMS"
    else B_ARGS="--baseline 2 --encoders $E --heads $ARMS"; fi
    python training/ClassifyingRoutingHead/cli/train.py \
        $B_ARGS \
        --loss "$L" \
        --ordinal-weight "$ORDINAL_WEIGHT" \
        --ordinal-sigma "$ORDINAL_SIGMA" \
        --epochs "$EPOCHS" \
        --seg "$SEG" \
        --n-per-rung "$N_PER_RUNG" \
        --batch-size "$BATCH_SIZE" \
        --wsi-group-size "$WSI_GROUP" \
        --wandb-project "$WANDB_PROJECT" \
        $RUN_NAME_ARG \
        $MAX_WSI_ARG $RESUME_DIR_ARG $MERGE_ARG $TRAIN_EXTRA_ARGS --out "$O" "$@"
}

# run_queue <encoder> <out> [train.py args...]: the encoder's losses in turn
run_queue () {
    local E="$1" O="$2" L
    shift 2
    for L in $(losses_of "$E"); do
        echo ""
        echo "======== [1] train $E, loss $L ========"
        train_unit "$E" "$L" "$O" "$@" || { echo "train.py ($E, $L) failed"; return 1; }
    done
}

# merge_queues: every $OUT/<encoder> into the one $OUT evaluate.py reads. weights/
# never collides: the encoder is a segment of every name. The val_scores files
# share their columns (the same `val_report` code), so this is
# concat-with-one-header, not a join.
merge_queues () {
    local e f
    mkdir -p "$OUT/weights"
    for e in $QUEUES; do cp "$OUT/$e"/weights/*.pt "$OUT/weights/" 2>/dev/null; done
    for f in val_scores.csv val_scores_per_rung.csv; do
        python - $(for e in $QUEUES; do echo "$OUT/$e/$f"; done) "$OUT/$f" <<'PY'
import csv, os, sys
*parts, out = sys.argv[1:]
rows = []
header = []
# the UNION of the headers, in first-seen order: one file can carry a column
# another predates (read_level), and DictWriter refuses a key its header lacks
for path in parts:
    if not os.path.exists(path):
        continue
    with open(path, newline='') as fh:
        r = csv.DictReader(fh)
        header += [f for f in r.fieldnames if f not in header]
        rows += list(r)
with open(out, 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=header)
    wr.writeheader()
    wr.writerows(rows)
print(f'{out}  ({len(rows)} rows, merged from {" + ".join(p for p in parts if os.path.exists(p))})')
PY
    done
}

# =============================================================================
#  Step 1: train
# =============================================================================
echo ""
echo "======== [1] train: one queue per encoder ($QUEUES ), parallel=$PARALLEL ========"
for e in $QUEUES; do echo "  $e: $(losses_of "$e")"; done

if [ "$EVAL_ONLY" = "1" ]; then
    echo "  EVAL_ONLY=1 -- skipping training, scoring whatever is already in $OUT/weights"
    if [ ! -d "$OUT/weights" ]; then
        echo "  [abort] $OUT/weights does not exist -- EVAL_ONLY needs an earlier plain run's output to score"
        exit 1
    fi
elif [ "$PARALLEL" = "1" ]; then
    NQ=$(wc -w <<< "$QUEUES")
    CARDS=$(nvidia-smi -L | wc -l)
    if [ "$CARDS" -lt "$NQ" ]; then
        echo "PARALLEL=1 runs $NQ queues, one GPU each, but this job sees $CARDS card(s):"
        echo "  sbatch --gpus-per-node=$NQ --cpus-per-task=$((12 * NQ)) --mem=$((200 * NQ))G ..."
        exit 2
    fi
    # each queue's share of the cpus, one left for its main process (CpuBudget)
    NUM_WORKERS_Q="${NUM_WORKERS_Q:-$(( (CPUS - NQ) / NQ > 0 ? (CPUS - NQ) / NQ : 1 ))}"
    pids=(); i=0
    for e in $QUEUES; do
        LOGQ="/work/u26130998/log/${SLURM_JOB_NAME:-ClassifyingRoutingHead}.$e"
        mkdir -p "$OUT/$e"
        CUDA_VISIBLE_DEVICES=$i run_queue "$e" "$OUT/$e" \
            --num-workers "$NUM_WORKERS_Q" --cpu-processes "$NQ" --device cuda:0 > "$LOGQ" 2>&1 &
        pids+=($!)
        echo "  $e -> GPU $i, $NUM_WORKERS_Q workers, log: $LOGQ"
        i=$((i + 1))
    done
    # one `wait` per pid, so every queue's own exit code is kept
    failed=0; i=0
    for e in $QUEUES; do
        wait "${pids[$i]}"; rc=$?
        echo "  $e exit $rc"
        [ $rc -ne 0 ] && failed=1
        i=$((i + 1))
    done
    if [ $failed -ne 0 ]; then
        echo "at least one queue failed (see log/${SLURM_JOB_NAME:-ClassifyingRoutingHead}.<encoder>) -- not merging, not scoring test"
        exit 1
    fi
    merge_queues
else
    for e in $QUEUES; do
        run_queue "$e" "$OUT" --num-workers "$NUM_WORKERS" || exit 1
    done
fi

# =============================================================================
#  Step 2: evaluate
# =============================================================================
# Rebuilds each checkpoint from the file alone (no --encoder, no --arm: those
# would be a second source of truth able to disagree with the weights) and
# scores it on EVAL_N_WSI (10) WSIs x 6 rungs x 50 positions per dataset. Writes
# test_scores_<tag>.csv and test_predictions_<tag>.csv, the second with one row
# per tile. Sequential regardless of PARALLEL: every GPU is free again by this
# point, and evaluate.py loops over CHECKPOINTS, minutes next to training's hours.
echo ""
echo "======== [2] evaluate ========"
python training/ClassifyingRoutingHead/cli/evaluate.py \
    --tag "$TAG" \
    --test-dataset "$TEST_DATASET" \
    --n-wsi "$EVAL_N_WSI" \
    --seg "$SEG" \
    $CACHE_ARGS \
    --num-workers "$NUM_WORKERS" \
    --out "$OUT"
rc=$?

echo ""
echo "======== done (exit $rc) ========"
exit $rc

