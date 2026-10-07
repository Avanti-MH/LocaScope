#!/bin/bash
#SBATCH --job-name=PrototypicalRoutingHead        # Job name
#SBATCH --partition=normal2                       # Partition
#SBATCH --time=48:00:00                           # Runtime (hh:mm:ss) -- resubmit to continue, see RESUME
#SBATCH --account=MST114560                       # Account
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --gpus-per-node=1                         # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                         # encode_batch + main
#SBATCH --ntasks-per-node=1                       # Tasks per node
#SBATCH --mem=200G                                # host RAM
#SBATCH -o /work/u26130998/log/%x   # STDOUT
#SBATCH -e /work/u26130998/log/%x   # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  training/PrototypicalRoutingHead -- ONE JOB, 11 TRAINING RUNS, then test:
#
#    [1]  every model combination in MODELS x every EPISODE_REUSE mode
#         (none / hold_q), --cross-domain-dataset off               5 x 2 = 10
#    [2]  the Matching-Net combination, episode reuse none,
#         --cross-domain-dataset $CROSS_DOMAIN_DATASET                    1
#    [3]  cli/evaluate.py on the test split: the original table AND the K x K
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
#  SAMPLER. Positions come from MppRoutingHead's own mask and sampler caches
#  (--mask-cache-job / --sampler-cache-job MppRoutingHead): the same function,
#  the same config and the same files, so both packages train on the same
#  positions rather than on two draws that should agree.
#
#  TWO CARDS. PARALLEL=2 deals the 11 runs out to two queues, one per card,
#  running at once; each gets half the job's cpus (CpuBudget). Training is
#  one process per run, so no run is split across cards -- two runs are
#  simply in flight together. evaluate.py then runs once, after both queues.
#
#      PARALLEL=2 sbatch --gpus-per-node=2 --cpus-per-task=16 <this script>
# =============================================================================

# ---------------- knobs ----------------
SMOKE="${SMOKE:-0}"
PARALLEL="${PARALLEL:-1}"       # N: run the runs in N queues at once, one card each (see the run loop)
SUPPLY_ONLY="${SUPPLY_ONLY:-0}"
TRAIN_DATASET="${TRAIN_DATASET:-ki67_pure}"
ENCODER="${ENCODER:-uni2}"
DTYPE="${DTYPE:-fp16}"
POOLING="${POOLING:-cls}"
TILE="${TILE:-256}"
N_PER_RUNG="${N_PER_RUNG:-100}"
N_SUPPORT="${N_SUPPORT:-5}"
N_QUERY="${N_QUERY:-10}"
N_CHOICES="${N_CHOICES:-3 4 5}"
EPISODE_REUSES="${EPISODE_REUSES:-none hold_q}"   # hold_s: same steps per epoch as hold_q, the held side differs; not run by default
REUSE_K_GIVEN="${REUSE_K-__unset__}"
REUSE_K="${REUSE_K:-5}"
VAL_REUSE_K="${VAL_REUSE_K:-4}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-auto}"
MAX_OVERLAP="${MAX_OVERLAP:-0.0}"
CROSS_DOMAIN_DATASET="${CROSS_DOMAIN_DATASET:-bracs/train}"   # [2] only
EVAL_DATASETS="${EVAL_DATASETS:-bracs/test ki67_with_photo}"
SEG="${SEG:-hest}"   # tissue-mask recipe (TissueMaskConfig.MASK_RECIPES)
CACHE_JOB="${CACHE_JOB:-MppRoutingHead}"
VAL_N_PER_RUNG="${VAL_N_PER_RUNG:-50}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-4}"
LR_PATIENCE="${LR_PATIENCE:-3}"
LR_FACTOR="${LR_FACTOR:-0.5}"
LOSS="${LOSS:-bal}"
ORDINAL_WEIGHT="${ORDINAL_WEIGHT:-1.0}"
ORDINAL_SIGMA="${ORDINAL_SIGMA:-1.0}"
ENCODE_BATCH="${ENCODE_BATCH:-64}"
SEED="${SEED:-42}"
# Cap on the WSIs each training manifest draws from (--max-wsi): N chosen at
# random with --seed, the same N every time. 121 is ki67_pure's whole size, so
# it caps only the cross-domain bracs/train -- to the 121 slides 366274 trained
# on, whose masks are already in the MppRoutingHead mask cache. Unset uses 121;
# set but empty (MAX_WSI=) drops the cap, which segments every native
# bracs/train slide that is not cached yet.
MAX_WSI_GIVEN="${MAX_WSI-__unset__}"
MAX_WSI="${MAX_WSI-121}"
OUT="${OUT:-}"
WANDB_PROJECT="${WANDB_PROJECT:-prototypical-routing-head}"
RUN_NAME="${RUN_NAME:-}"          # empty: the wandb run is named after the job (SLURM_JOB_NAME)
# cli/evaluate.py (test split): 5 slides x 50 per rung, deeper than val. The
# original table draws EVAL_N_EPISODES full 6-way episodes; the K x K table
# takes EVAL_KXK_K (auto = the largest the test split supplies).
EVAL_N_WSI="${EVAL_N_WSI:-5}"
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

if [ "$SMOKE" = "1" ]; then
    # Rung 32 is ~1 position per ki67 slide, so a few slides only supply a
    # few positions there: 1 support + 2 query per rung keeps K >= 1 drawable
    # on 10 slides. `auto` then resolves whatever that pool allows.
    # 10 unless MAX_WSI was given on the command line, not the 121 above
    [ "$MAX_WSI_GIVEN" = "__unset__" ] && MAX_WSI=10
    N_PER_RUNG=20
    N_SUPPORT=5
    N_QUERY=10
    EPISODE_REUSES="${EPISODE_REUSES_SMOKE:-hold_q}"
    # K support x K query batches per draw. Left at `auto`, hold_q resolves to
    # the largest K the pool allows (177 on a 10-slide smoke), and one epoch
    # renders that many batches 39 times -- hours with nothing printed. 2 keeps
    # every code path and finishes in minutes; REUSE_K on the command line wins.
    [ "$REUSE_K_GIVEN" = "__unset__" ] && REUSE_K=2
    VAL_N_PER_RUNG=20
    EPOCHS=2
    EVAL_N_WSI=2
    EVAL_N_PER_RUNG=10
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

# collapse:support_context:query_context:routing_head
# 5 models (was 8 + the matching net). mean is the parameter-free baseline; set_transformer is
# attention over the support set, attn_pool attention without interaction; the two routing heads
# are compared on the two ends (mean, set_transformer) rather than crossed with every collapse.
# shared_mlp and the other two crossings were dropped 2026-10-03; the matching net is run once,
# as step [2] (MATCHING_NET below), not here.
MODELS=(
    "set_transformer:identity:identity:cosine_tau"
    "mean:identity:identity:cosine_tau"
    "attn_pool:identity:identity:cosine_tau"
    "set_transformer:identity:identity:attn_score"
    "mean:identity:identity:attn_score"
)
MATCHING_NET="off:bilstm:attnlstm:cosine_logsumexp"

echo "======== PrototypicalRoutingHead ========"
echo "  train_dataset=$TRAIN_DATASET  encoder=$ENCODER  pooling=$POOLING  loss=$LOSS"
echo "  models=${#MODELS[@]}  episode reuse: $EPISODE_REUSES  + matching net x $CROSS_DOMAIN_DATASET"
echo "  reuse-k=$REUSE_K  val-reuse-k=$VAL_REUSE_K  episodes-per-epoch=$EPISODES_PER_EPOCH  (given; each run prints what it used)"
echo "  n_support=$N_SUPPORT  n_query=$N_QUERY  max_overlap=$MAX_OVERLAP  epochs=$EPOCHS"
echo "  caches: mask/sampler from ${CACHE_JOB}   seg=$SEG"
echo "  resume: ${RESUME_DIR:-off (RESUME_DIR set empty)}"
echo "  retries: train ${FEASIBILITY_TRIES:-100 (train.py default)}   eval $EVAL_FEASIBILITY_TRIES   eval_only=$EVAL_ONLY   eval support=$EVAL_SUPPORT_NATIVE"
echo "  out=$OUT"

# Runs this job makes: every model under every reuse mode, then the matching net.
# RUN_I counts them for the title line; run_one is a function in THIS shell, not
# a subshell, so the counter survives between calls.
RUN_I=0
RUN_N=$(( ${#MODELS[@]} * $(wc -w <<< "$EPISODE_REUSES") + 1 ))

# run_one <episode_reuse> <cross_domain_dataset> <model spec> [extra args...]
run_one () {
    local REUSE="$1" CROSS="$2" SPEC="$3"
    shift 3
    IFS=':' read -r COLLAPSE SUPPORT_CONTEXT QUERY_CONTEXT ROUTING_HEAD <<< "$SPEC"
    local RUN_NAME_ARG=""
    [ -n "$RUN_NAME" ] && RUN_NAME_ARG="--run-name $RUN_NAME"
    echo ""
    RUN_I=$((RUN_I + 1))
    echo "-------- [$RUN_I/$RUN_N] $REUSE | $COLLAPSE | $ROUTING_HEAD | ctx $SUPPORT_CONTEXT/$QUERY_CONTEXT | cross ${CROSS:-off} --------"
    python training/PrototypicalRoutingHead/cli/train.py \
        --train-dataset "$TRAIN_DATASET" \
        --encoder "$ENCODER" \
        --dtype "$DTYPE" \
        --pooling "$POOLING" \
        --support-context "$SUPPORT_CONTEXT" \
        --query-context "$QUERY_CONTEXT" \
        --collapse "$COLLAPSE" \
        --routing-head "$ROUTING_HEAD" \
        --cross-domain-dataset "$CROSS" \
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
        --eval-datasets $EVAL_DATASETS \
        --seg "$SEG" \
        --mask-cache-job "$CACHE_JOB" \
        --sampler-cache-job "$CACHE_JOB" \
        --val-n-per-rung "$VAL_N_PER_RUNG" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --lr-patience "$LR_PATIENCE" \
        --lr-factor "$LR_FACTOR" \
        --loss "$LOSS" \
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
        --out "$OUT" "$@"
}

if [ "$SUPPLY_ONLY" = "1" ]; then
    # The pool does not depend on the model or the reuse mode, only on the
    # data and N_SUPPORT/N_QUERY -- one call answers for every run below. The
    # cross-domain pool is included, since run [2] draws from it.
    run_one none "$CROSS_DOMAIN_DATASET" "${MODELS[0]}" --supply-only
    exit $?
fi

FAILED=()
FAIL_DIR="$OUT/.failed"
if [ "$EVAL_ONLY" != "1" ]; then
# The runs of this job as "reuse|cross|spec" lines: every model under every
# reuse mode (cross-domain off), then the matching net (run [2]).
RUNS=()
for REUSE in $EPISODE_REUSES; do
    for spec in "${MODELS[@]}"; do
        RUNS+=("$REUSE||$spec")
    done
done
RUNS+=("none|$CROSS_DOMAIN_DATASET|$MATCHING_NET")

if [ "$PARALLEL" -le 1 ]; then
    for entry in "${RUNS[@]}"; do
        IFS='|' read -r REUSE CROSS spec <<< "$entry"
        run_one "$REUSE" "$CROSS" "$spec" || FAILED+=("$REUSE $spec cross=${CROSS:-off}")
    done
else
    # PARALLEL=N: N queues at once, one card each, sharing this job's cpus
    # (CpuBudget gives each a share). The runs are dealt out alternately, so
    # each queue gets a mix of the cheap and the dear. Every run keeps its own
    # weights, resume file and wandb run; the two val_scores files are shared
    # and written under a lock (train.py). Ask for the cards and the cpus on
    # the command line:
    #
    #     PARALLEL=2 sbatch --gpus-per-node=2 --cpus-per-task=16 <this script>
    #
    # Queue i logs to <log>.p<i>; this file gets the summary.
    CARDS=$(nvidia-smi -L | wc -l)
    if [ "$CARDS" -lt "$PARALLEL" ]; then
        echo "PARALLEL=$PARALLEL but this job sees $CARDS card(s): ask for them with"
        echo "  sbatch --gpus-per-node=$PARALLEL ..."
        exit 2
    fi
    LOG="/work/u26130998/log/${SLURM_JOB_NAME:-PrototypicalRoutingHead}"
    rm -rf "$FAIL_DIR"; mkdir -p "$FAIL_DIR"
    pids=()
    for ((i = 0; i < PARALLEL; i++)); do
        (
            export CUDA_VISIBLE_DEVICES=$i
            for ((j = i; j < ${#RUNS[@]}; j += PARALLEL)); do
                IFS='|' read -r REUSE CROSS spec <<< "${RUNS[$j]}"
                RUN_I=$j        # run_one counts up from here: [j+1/N]
                run_one "$REUSE" "$CROSS" "$spec" \
                    || echo "$REUSE $spec cross=${CROSS:-off}" >> "$FAIL_DIR/p$i"
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
echo "  $OUT/weights/  (*_best.pt / *_6rung.pt / *_native.pt per run)"
echo "  $OUT/val_scores_per_combo.csv, val_scores_per_rung.csv"
else
    echo ""
    echo "======== EVAL_ONLY: training skipped, scoring $OUT/weights ========"
fi

# ---------------- evaluate.py: TEST split, every checkpoint ----------------
echo ""
echo "======== evaluate.py (test split) ========"
python training/PrototypicalRoutingHead/cli/evaluate.py \
    --eval-datasets $EVAL_DATASETS \
    --seg "$SEG" \
    --mask-cache-job "$CACHE_JOB" \
    --sampler-cache-job "$CACHE_JOB" \
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
