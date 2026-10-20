#!/bin/bash
#SBATCH --job-name=PrototypicalRoutingHead        # Job name
#SBATCH --partition=8gpus                         # Partition
#SBATCH --time=48:00:00                           # Runtime (hh:mm:ss) -- resubmit to continue, see RESUME
#SBATCH --account=MST114560                       # Account
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --gpus-per-node=4                         # one per PARALLEL queue (不要設0)
#SBATCH --cpus-per-task=48                        # 12 per GPU, the H200 cap: render threads + main
#SBATCH --ntasks-per-node=1                       # Tasks per node
#SBATCH --mem=800G                                # host RAM, 200G per GPU, the H200 cap
#SBATCH -o /work/u26130998/log/%x   # STDOUT
#SBATCH -e /work/u26130998/log/%x   # STDERR

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  training/PrototypicalRoutingHead -- ONE JOB, 13 TRAINING RUNS, then test:
#
#    [1]  the BASELINE and one run per mechanism, each the baseline with that
#         ONE thing changed (RUNS below), so each run answers one question   13
#    [2]  cli/evaluate.py on the test set: the original table AND the K x K
#
#  ONLY="1 8" runs those runs of RUNS only (1-based, the [n/N] of the log).
#
#  K AND EPISODES. REUSE_K / VAL_REUSE_K / EPISODES_PER_EPOCH default to
#  `auto`: K = the largest the pool can actually draw, episodes = the 39
#  training combinations once per epoch (x K under `none`). Every run prints
#  "given -> used" for all three. Set SUPPLY_ONLY=1 to build the manifests,
#  print the per-rung supply and what `auto` resolves to, and stop -- that is
#  the run to look at before choosing a K by hand.
#
#  RESUME. RESUME_DIR (default $OUT/resume) receives each run's full state
#  every epoch. Re-submitting THIS SAME COMMAND after a walltime kill continues
#  each run from its last finished epoch; a finished run resumes into nothing
#  (--epochs is the total), so the loop walks straight past it. RESUME_DIR=
#  (set but empty) turns it off.
#
#  SAMPLER. Positions come from the shared Masks cache and ClassifyingRoutingHead's
#  draws (--mask-cache-job Masks / --draw-cache-job MppRoutingHead, the draws' old
#  cache name): the same function,
#  the same config and the same files, so both packages train on the same
#  positions rather than on two draws that should agree.
#
#  FOUR CARDS. PARALLEL=4 (the default) deals the 13 runs out to four queues,
#  one per card, running at once; each gets a quarter of the job's cpus
#  (CpuBudget) and renders in that many threads. Training is one process per
#  run, so no run is split across cards -- four runs are simply in flight
#  together. evaluate.py then runs once, after every queue. Fewer cards:
#
#      PARALLEL=2 sbatch --gpus-per-node=2 --cpus-per-task=24 --mem=400G <this script>
#      PARALLEL=1 sbatch --gpus-per-node=1 --cpus-per-task=12 --mem=200G <this script>
# =============================================================================

# ---------------- knobs ----------------
SMOKE="${SMOKE:-0}"
PARALLEL="${PARALLEL:-4}"       # N: run the runs in N queues at once, one card each (see the run loop)
SUPPLY_ONLY="${SUPPLY_ONLY:-0}"
TRAIN_DATASET="${TRAIN_DATASET:-@pure-train}"
TRAIN_STAIN="${TRAIN_STAIN:-Ki67}"   # trained on; the cross-domain pool is the train set's other stains
ENCODER="${ENCODER:-gigapath}"   # the baseline encoder
DTYPE="${DTYPE:-fp32}"
POOLING="${POOLING:-cls}"        # the baseline pooling
TILE="${TILE:-256}"
N_PER_RUNG="${N_PER_RUNG:-100}"
N_SUPPORT="${N_SUPPORT:-5}"
N_QUERY="${N_QUERY:-10}"
N_CHOICES="${N_CHOICES:-3 4 5}"
BASE_REUSE="${BASE_REUSE:-hold_q}"   # the baseline episode reuse; the reuse run trains none
REUSE_K_GIVEN="${REUSE_K-__unset__}"
REUSE_K="${REUSE_K:-5}"
VAL_REUSE_K="${VAL_REUSE_K:-4}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-auto}"
MAX_OVERLAP="${MAX_OVERLAP:-0.0}"
CROSS_DOMAIN="${CROSS_DOMAIN-on}"   # the cross-domain run only; CROSS_DOMAIN= drops it
VAL_DATASET="${VAL_DATASET:-@pure-half-val}"     # train.py --val-dataset
TEST_DATASET="${TEST_DATASET:-@pure-half-test}"  # evaluate.py --test-dataset
SEG="${SEG:-hest}"   # tissue-mask recipe (TissueMaskConfig.MASK_RECIPES)
MASK_CACHE_JOB="${MASK_CACHE_JOB:-Masks}"            # the masks every trainer and bench shares
DRAW_CACHE_JOB_GIVEN="${DRAW_CACHE_JOB-__unset__}"
DRAW_CACHE_JOB="${DRAW_CACHE_JOB:-MppRoutingHead}"   # both trainers' draws, under the classifier's old name
VAL_N_PER_RUNG="${VAL_N_PER_RUNG:-50}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
LR_PATIENCE="${LR_PATIENCE:-3}"
LR_FACTOR="${LR_FACTOR:-0.5}"
LOSS="${LOSS:-bal}"           # the baseline's loss
ORD_LOSS="${ORD_LOSS:-ord_a}" # the ordinal-loss run's
ORDINAL_WEIGHT="${ORDINAL_WEIGHT:-1.0}"
ORDINAL_SIGMA="${ORDINAL_SIGMA:-1.0}"
ENCODE_BATCH="${ENCODE_BATCH:-256}"
SEED="${SEED:-42}"
# Cap on the WSIs each training manifest draws from (--max-wsi): N chosen at
# random with --seed, the same N every time, of each dataset the train and
# cross-domain pools draw from. 121 is ki67_pure's whole size, and @pure-train holds 100 of
# each of its datasets, so by default the cap changes nothing (it once capped
# the cross-domain bracs/train to the 121 slides 366274 trained on). Unset uses
# 121; set but empty (MAX_WSI=) drops the cap.
MAX_WSI_GIVEN="${MAX_WSI-__unset__}"
MAX_WSI="${MAX_WSI-121}"
OUT="${OUT:-}"
WANDB_PROJECT="${WANDB_PROJECT:-prototypical-routing-head}"
RUN_NAME="${RUN_NAME:-}"          # empty: the wandb run is named after the job (SLURM_JOB_NAME)
# cli/evaluate.py (test set): 10 slides of each dataset x 50 per rung. The
# original table draws EVAL_N_EPISODES full 6-way episodes; the K x K table
# takes EVAL_KXK_K (auto = the largest the test set supplies).
EVAL_N_WSI="${EVAL_N_WSI:-10}"
EVAL_N_PER_RUNG="${EVAL_N_PER_RUNG:-50}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-100}"
EVAL_KXK_K="${EVAL_KXK_K:-auto}"
# How many times a draw that cannot be drawn or rendered is tried again
# (--feasibility-tries). FEASIBILITY_TRIES reaches train.py only when set, so a
# run that leaves it alone keeps train.py's own 100 and its resume identity.
# EVAL_FEASIBILITY_TRIES defaults to 100 HERE, not to evaluate.py's own 5: with
# 5, test K x K on bracs scored every combination holding rung 16 or 32 as
# missing, while train and val had 100.
FEASIBILITY_TRIES="${FEASIBILITY_TRIES:-}"
EVAL_FEASIBILITY_TRIES="${EVAL_FEASIBILITY_TRIES:-100}"
# EVAL_ONLY=1 skips training and scores the checkpoints already in $OUT/weights.
EVAL_ONLY="${EVAL_ONLY:-0}"
# How evaluate.py renders the SUPPORT side (--support-native): checkpoint (each
# as it trained), on (routing-support-native for all), off (routing-query for all).
# Off the default its CSVs are test_scores[_kxk]_support-<on|off>_*.
EVAL_SUPPORT_NATIVE="${EVAL_SUPPORT_NATIVE:-checkpoint}"
# SUPPORT_NATIVE (training): the probability, per episode, that the SUPPORT is rendered
# routing-support-native (rotation only) rather than routing-query. Unset: train.py's default, 0.5
# (a coin per support batch); 1: always; 0: never (the runs before it existed). Val then uses native support
# whenever it is above 0. A run with another value writes the SAME weight files, so a run that
# should keep the old ones goes to another OUT.

if [ "$SMOKE" = "1" ]; then
    # Rung 32 is ~1 position per ki67 slide, so a few slides only supply a
    # few positions there: 1 support + 2 query per rung keeps K >= 1 drawable
    # on 10 slides. `auto` then resolves whatever that pool allows.
    # 10 unless MAX_WSI was given on the command line, not the 121 above
    [ "$MAX_WSI_GIVEN" = "__unset__" ] && MAX_WSI=10
    N_PER_RUNG=20
    N_SUPPORT=5
    N_QUERY=10
    # K support x K query batches per draw. Left at `auto`, hold_q resolves to
    # the largest K the pool allows (177 on a 10-slide smoke), and one epoch
    # renders that many batches 39 times -- hours with nothing printed. 2 keeps
    # every code path and finishes in minutes; REUSE_K on the command line wins.
    [ "$REUSE_K_GIVEN" = "__unset__" ] && REUSE_K=2
    # Its draws (N_PER_RUNG 20) no real run can reuse: they go to its own cache
    # (Smoke*, what make clean-smoke removes). Its masks are the real ones and stay shared.
    [ "$DRAW_CACHE_JOB_GIVEN" = "__unset__" ] && DRAW_CACHE_JOB="${SLURM_JOB_NAME:-SmokePrototypicalRoutingHead}"
    VAL_N_PER_RUNG=20
    EPOCHS=2
    EVAL_N_WSI=2
    # 2 slides x 20 a rung: K x K needs n_support + n_query (15) of every rung for K=1, which 10 a
    # rung could not give on ki67_pure (K=0, evaluate refused)
    EVAL_N_PER_RUNG=20
    EVAL_N_EPISODES=3
    echo "======== SMOKE ========"
fi

if [ -z "$OUT" ]; then
    OUT=$(python -c "
import sys
sys.path.insert(0, 'utilities')
import _paths
print(_paths.job_result_dir('PrototypicalRoutingHead'))
")
fi
mkdir -p "$OUT"
# `${RESUME_DIR-...}` (no colon): unset -> the default, set-but-empty -> off.
RESUME_DIR="${RESUME_DIR-$OUT/resume}"

MAX_WSI_ARG=""
[ -n "$MAX_WSI" ] && MAX_WSI_ARG="--max-wsi $MAX_WSI"
RESUME_ARG=""
[ -n "$RESUME_DIR" ] && RESUME_ARG="--resume-dir $RESUME_DIR"

# The runs: "label|loss|reuse|cross|pooling|collapse:support_context:query_context:routing_head[|encoder]",
# the encoder ENCODER (gigapath) when left out. encoder-uni2 is the baseline on uni2: whether the
# encoder matters at all, so a large gap says the mechanisms should be checked on uni2 too.
# The first is the BASELINE; every other one is the baseline with ONE thing changed, the label
# naming what, so the pair (baseline, run) answers one question. `attn+mix` is --pooling attn
# --mix. A mechanism found to help is then trained combined in a later job, not crossed here.
BASE_SPEC="set_transformer:identity:identity:cosine_tau"
B="$LOSS|$BASE_REUSE||$POOLING"
RUNS=(
    "baseline|$B|$BASE_SPEC"
    "collapse-mean|$B|mean:identity:identity:cosine_tau"
    "collapse-shared_mlp|$B|shared_mlp:identity:identity:cosine_tau"
    "collapse-attn_pool|$B|attn_pool:identity:identity:cosine_tau"
    "route-attn_score|$B|set_transformer:identity:identity:attn_score"
    "pool-avg|$LOSS|$BASE_REUSE||avg|$BASE_SPEC"
    "pool-attn|$LOSS|$BASE_REUSE||attn|$BASE_SPEC"
    "pool-attn+mix|$LOSS|$BASE_REUSE||attn+mix|$BASE_SPEC"
    "reuse-none|$LOSS|none||$POOLING|$BASE_SPEC"
    "cross-on|$LOSS|$BASE_REUSE|$CROSS_DOMAIN|$POOLING|$BASE_SPEC"
    "loss-$ORD_LOSS|$ORD_LOSS|$BASE_REUSE||$POOLING|$BASE_SPEC"
    "matching-net|$B|off:bilstm:attnlstm:cosine_logsumexp"
    "encoder-uni2|$B|$BASE_SPEC|uni2"
)
# CROSS_DOMAIN= drops the cross-domain run rather than training a second baseline
if [ -z "$CROSS_DOMAIN" ]; then
    KEPT=()
    for r in "${RUNS[@]}"; do [ "${r%%|*}" = "cross-on" ] || KEPT+=("$r"); done
    RUNS=("${KEPT[@]}")
fi
if [ -n "${ONLY:-}" ]; then
    PICKED=()
    for n in $ONLY; do PICKED+=("${RUNS[$((n - 1))]}"); done
    RUNS=("${PICKED[@]}")
fi

echo "======== PrototypicalRoutingHead ========"
echo "  train_dataset=$TRAIN_DATASET ($TRAIN_STAIN)  val=$VAL_DATASET  test=$TEST_DATASET  encoder=$ENCODER"
echo "  baseline: pooling $POOLING  $BASE_SPEC  reuse $BASE_REUSE  cross off  loss $LOSS"
echo "  runs=${#RUNS[@]}: $(printf '%s ' "${RUNS[@]%%|*}")"
echo "  reuse-k=$REUSE_K  val-reuse-k=$VAL_REUSE_K  episodes-per-epoch=$EPISODES_PER_EPOCH  (given; each run prints what it used)"
echo "  n_support=$N_SUPPORT  n_query=$N_QUERY  max_overlap=$MAX_OVERLAP  epochs=$EPOCHS"
echo "  caches: masks from $MASK_CACHE_JOB, draws from $DRAW_CACHE_JOB   seg=$SEG"
echo "  resume: ${RESUME_DIR:-off (RESUME_DIR set empty)}"
echo "  retries: train ${FEASIBILITY_TRIES:-100 (train.py default)}   eval $EVAL_FEASIBILITY_TRIES   eval_only=$EVAL_ONLY   eval support=$EVAL_SUPPORT_NATIVE"
echo "  out=$OUT"

# RUN_I counts the runs for the title line; run_one is a function in THIS shell, not
# a subshell, so the counter survives between calls.
RUN_I=0
RUN_N=${#RUNS[@]}

# run_one <one RUNS entry> [extra args...]
run_one () {
    local LABEL LOSS_NOW REUSE CROSS POOL SPEC ENC
    IFS='|' read -r LABEL LOSS_NOW REUSE CROSS POOL SPEC ENC <<< "$1"
    ENC="${ENC:-$ENCODER}"
    shift
    IFS=':' read -r COLLAPSE SUPPORT_CONTEXT QUERY_CONTEXT ROUTING_HEAD <<< "$SPEC"
    local POOL_KIND="${POOL%+mix}" MIX_ARG=""
    [ "$POOL_KIND" != "$POOL" ] && MIX_ARG="--mix"
    # train.py names the wandb run <prefix>-<encoder>-<collapse>-<head>-<reuse>: the label goes
    # into the prefix, or the pooling, cross-domain and loss runs would share the baseline's name
    local RUN_NAME_ARG="--run-name ${RUN_NAME:-${SLURM_JOB_NAME:-PrototypicalRoutingHead}}-$LABEL"
    echo ""
    RUN_I=$((RUN_I + 1))
    echo "-------- [$RUN_I/$RUN_N] $LABEL: $ENC | $LOSS_NOW | $REUSE | $POOL | $COLLAPSE | $ROUTING_HEAD | ctx $SUPPORT_CONTEXT/$QUERY_CONTEXT | cross ${CROSS:-off} --------"
    python training/PrototypicalRoutingHead/cli/train.py \
        --train-dataset "$TRAIN_DATASET" \
        --train-stain "$TRAIN_STAIN" \
        --encoder "$ENC" \
        --dtype "$DTYPE" \
        --pooling "$POOL_KIND" $MIX_ARG \
        --support-context "$SUPPORT_CONTEXT" \
        --query-context "$QUERY_CONTEXT" \
        --collapse "$COLLAPSE" \
        --routing-head "$ROUTING_HEAD" \
        ${CROSS:+--cross-domain} \
        --episode-reuse "$REUSE" \
        --reuse-k "$REUSE_K" \
        --val-reuse-k "$VAL_REUSE_K" \
        --episodes-per-epoch "$EPISODES_PER_EPOCH" \
        --max-overlap "$MAX_OVERLAP" \
        --tile "$TILE" \
        --n-per-rung "$N_PER_RUNG" \
        --n-support "$N_SUPPORT" \
        --n-query "$N_QUERY" \
        --n-choices $N_CHOICES \
        --val-dataset "$VAL_DATASET" \
        --seg "$SEG" \
        --mask-cache-job "$MASK_CACHE_JOB" \
        --draw-cache-job "$DRAW_CACHE_JOB" \
        --val-n-per-rung "$VAL_N_PER_RUNG" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --lr-patience "$LR_PATIENCE" \
        --lr-factor "$LR_FACTOR" \
        --loss "$LOSS_NOW" \
        --ordinal-weight "$ORDINAL_WEIGHT" \
        --ordinal-sigma "$ORDINAL_SIGMA" \
        --encode-batch "$ENCODE_BATCH" \
        --seed "$SEED" \
        --cpu-processes "$PARALLEL" \
        --wandb-project "$WANDB_PROJECT" \
        --merge \
        $RESUME_ARG \
        $RUN_NAME_ARG \
        $MAX_WSI_ARG \
        ${FEASIBILITY_TRIES:+--feasibility-tries $FEASIBILITY_TRIES} \
        ${SUPPORT_NATIVE:+--support-native $SUPPORT_NATIVE} \
        --out "$OUT" "$@"
}

# The baseline with the cross-domain pool: what the supply check and the cache warm-up run as,
# so the slides of every run, the cross-domain ones too, are in them.
WARM="warm|$LOSS|$BASE_REUSE|$CROSS_DOMAIN|$POOLING|$BASE_SPEC"

if [ "$SUPPLY_ONLY" = "1" ]; then
    # The pool does not depend on the model or the reuse mode, only on the
    # data and N_SUPPORT/N_QUERY -- one call answers for every run below. The
    # cross-domain pool is included, since the cross-on run draws from it.
    run_one "$WARM" --supply-only
    exit $?
fi

FAILED=()
FAIL_DIR="$OUT/.failed"
if [ "$EVAL_ONLY" != "1" ]; then
if [ "$PARALLEL" -le 1 ]; then
    for entry in "${RUNS[@]}"; do
        run_one "$entry" || FAILED+=("$entry")
    done
else
    # PARALLEL=N: N queues at once, one card each, sharing this job's cpus
    # (CpuBudget gives each a share). The runs are dealt out alternately, so
    # each queue gets a mix of the cheap and the dear. Every run keeps its own
    # weights, resume file and wandb run; the two val_scores files are shared
    # and written under a lock (train.py). Ask for the cards and the cpus on
    # the command line:
    #
    #     PARALLEL=2 sbatch --gpus-per-node=2 --cpus-per-task=24 --mem=400G <this script>
    #
    # Queue i logs to <log>.p<i>; this file gets the summary.
    CARDS=$(nvidia-smi -L | wc -l)
    if [ "$CARDS" -lt "$PARALLEL" ]; then
        echo "PARALLEL=$PARALLEL but this job sees $CARDS card(s): ask for them with"
        echo "  sbatch --gpus-per-node=$PARALLEL ..."
        exit 2
    fi
    LOG="/work/u26130998/log/${SLURM_JOB_NAME:-PrototypicalRoutingHead}"
    # Warm the caches before the queues start, split N ways: shard i
    # (--cache-shard i/N, card i) segments and draws every N-th slide of the
    # train, cross-domain and val lists, then exits. Without it the N queues
    # start together and each segments the SAME uncached slides on its own
    # card. A slide already cached is a quick hit. Shard i logs to <log>.w<i>.
    echo "======== cache warm-up: $PARALLEL shards, one per card ========"
    wpids=()
    for ((i = 0; i < PARALLEL; i++)); do
        CUDA_VISIBLE_DEVICES=$i run_one "$WARM" \
            --cache-shard "$i/$PARALLEL" > "$LOG.w$i" 2>&1 &
        wpids+=($!)
    done
    WARM_RC=0
    for pid in "${wpids[@]}"; do wait "$pid" || WARM_RC=1; done
    [ "$WARM_RC" -ne 0 ] && { echo "cache warm-up failed: see $LOG.w*"; exit 1; }
    RUN_I=0
    rm -rf "$FAIL_DIR"; mkdir -p "$FAIL_DIR"
    pids=()
    for ((i = 0; i < PARALLEL; i++)); do
        (
            export CUDA_VISIBLE_DEVICES=$i
            for ((j = i; j < ${#RUNS[@]}; j += PARALLEL)); do
                RUN_I=$j        # run_one counts up from here: [j+1/N]
                run_one "${RUNS[$j]}" || echo "${RUNS[$j]}" >> "$FAIL_DIR/p$i"
            done
        ) > "$LOG.p$i" 2>&1 &
        pids+=($!)
        echo "queue $i/$PARALLEL: pid $! on card $i, log $LOG.p$i"
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    for f in "$FAIL_DIR"/p*; do
        [ -s "$f" ] && while IFS= read -r line; do FAILED+=("$line"); done < "$f"
    done
fi

echo ""
echo "======== training done ========"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  all runs finished"
else
    echo "  ${#FAILED[@]} run(s) failed:"
    for f in "${FAILED[@]}"; do echo "    $f"; done
fi
echo "  $OUT/weights/  (*--best.pt / *--6rung.pt / *--native.pt per run)"
echo "  $OUT/val_scores_per_combo.csv, val_scores_per_rung.csv"
else
    echo ""
    echo "======== EVAL_ONLY: training skipped, scoring $OUT/weights ========"
fi

# ---------------- evaluate.py: TEST set, every checkpoint ----------------
echo ""
echo "======== evaluate.py (test set $TEST_DATASET) ========"
python training/PrototypicalRoutingHead/cli/evaluate.py \
    --test-dataset "$TEST_DATASET" \
    --seg "$SEG" \
    --mask-cache-job "$MASK_CACHE_JOB" \
    --draw-cache-job "$DRAW_CACHE_JOB" \
    --tile "$TILE" \
    --n-wsi "$EVAL_N_WSI" \
    --n-per-rung "$EVAL_N_PER_RUNG" \
    --n-support "$N_SUPPORT" \
    --n-query "$N_QUERY" \
    --n-episodes "$EVAL_N_EPISODES" \
    --kxk-k "$EVAL_KXK_K" \
    --max-overlap "$MAX_OVERLAP" \
    --encode-batch "$ENCODE_BATCH" \
    --seed "$SEED" \
    --feasibility-tries "$EVAL_FEASIBILITY_TRIES" \
    --support-native "$EVAL_SUPPORT_NATIVE" \
    --out "$OUT"
eval_rc=$?

echo ""
echo "======== done (${#FAILED[@]} failed run(s), eval rc $eval_rc) ========"
echo "  $OUT/test_scores_per_combo.csv, test_scores_per_rung.csv        (original)"
echo "  $OUT/test_scores_kxk_per_combo.csv, test_scores_kxk_per_rung.csv  (K x K)"
[ ${#FAILED[@]} -ne 0 ] && exit 1
exit $eval_rc
