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
#SBATCH -o /work/u26130998/log/PrototypicalRoutingHead   # STDOUT
#SBATCH -e /work/u26130998/log/PrototypicalRoutingHead   # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  training/PrototypicalRoutingHead -- ONE JOB, 28 TRAINING RUNS, then test:
#
#    [1]  every model combination in MODELS x every EPISODE_REUSE mode
#         (none / hold_s / hold_q), --cross-domain-dataset off     9 x 3 = 27
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
# =============================================================================

# ---------------- knobs ----------------
SMOKE="${SMOKE:-0}"
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
EPISODE_REUSES="${EPISODE_REUSES:-none hold_s hold_q}"
REUSE_K="${REUSE_K:-auto}"
VAL_REUSE_K="${VAL_REUSE_K:-auto}"
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
MAX_WSI="${MAX_WSI:-}"
OUT="${OUT:-}"
WANDB_PROJECT="${WANDB_PROJECT:-prototypical-routing-head}"
RUN_NAME="${RUN_NAME:-${SLURM_JOB_ID:-}}"
# cli/evaluate.py (test split): 5 slides x 50 per rung, deeper than val. The
# original table draws EVAL_N_EPISODES full 6-way episodes; the K x K table
# takes EVAL_KXK_K (auto = the largest the test split supplies).
EVAL_N_WSI="${EVAL_N_WSI:-5}"
EVAL_N_PER_RUNG="${EVAL_N_PER_RUNG:-50}"
EVAL_N_EPISODES="${EVAL_N_EPISODES:-100}"
EVAL_KXK_K="${EVAL_KXK_K:-auto}"

if [ "$SMOKE" = "1" ]; then
    # Rung 32 is ~1 position per ki67 slide, so a few slides only supply a
    # few positions there: 1 support + 2 query per rung keeps K >= 1 drawable
    # on 10 slides. `auto` then resolves whatever that pool allows.
    MAX_WSI="${MAX_WSI:-10}"
    N_PER_RUNG=20
    N_SUPPORT=5
    N_QUERY=10
    EPISODE_REUSES="${EPISODE_REUSES_SMOKE:-hold_q}"
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
MODELS=(
    "set_transformer:identity:identity:cosine_tau"
    "mean:identity:identity:cosine_tau"
    "shared_mlp:identity:identity:cosine_tau"
    "attn_pool:identity:identity:cosine_tau"
    "set_transformer:identity:identity:attn_score"
    "mean:identity:identity:attn_score"
    "shared_mlp:identity:identity:attn_score"
    "attn_pool:identity:identity:attn_score"
    "off:bilstm:attnlstm:cosine_logsumexp"
)
MATCHING_NET="off:bilstm:attnlstm:cosine_logsumexp"

echo "======== PrototypicalRoutingHead ========"
echo "  train_dataset=$TRAIN_DATASET  encoder=$ENCODER  pooling=$POOLING  loss=$LOSS"
echo "  models=${#MODELS[@]}  episode reuse: $EPISODE_REUSES  + matching net x $CROSS_DOMAIN_DATASET"
echo "  reuse-k=$REUSE_K  val-reuse-k=$VAL_REUSE_K  episodes-per-epoch=$EPISODES_PER_EPOCH  (given; each run prints what it used)"
echo "  n_support=$N_SUPPORT  n_query=$N_QUERY  max_overlap=$MAX_OVERLAP  epochs=$EPOCHS"
echo "  caches: mask/sampler from ${CACHE_JOB}   seg=$SEG"
echo "  resume: ${RESUME_DIR:-off (RESUME_DIR set empty)}"
echo "  out=$OUT"

# run_one <episode_reuse> <cross_domain_dataset> <model spec> [extra args...]
run_one () {
    local REUSE="$1" CROSS="$2" SPEC="$3"
    shift 3
    IFS=':' read -r COLLAPSE SUPPORT_CONTEXT QUERY_CONTEXT ROUTING_HEAD <<< "$SPEC"
    local RUN_NAME_ARG=""
    [ -n "$RUN_NAME" ] && RUN_NAME_ARG="--run-name ${RUN_NAME}_${SUPPORT_CONTEXT}_${QUERY_CONTEXT}_${COLLAPSE}_${ROUTING_HEAD}"
    echo ""
    echo "-------- $REUSE  collapse=$COLLAPSE support_context=$SUPPORT_CONTEXT query_context=$QUERY_CONTEXT routing_head=$ROUTING_HEAD  cross_domain=${CROSS:-off} --------"
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
        --wandb-project "$WANDB_PROJECT" \
        --merge \
        $RESUME_ARG \
        $RUN_NAME_ARG \
        $MAX_WSI_ARG \
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
for REUSE in $EPISODE_REUSES; do
    for spec in "${MODELS[@]}"; do
        run_one "$REUSE" "" "$spec" || FAILED+=("$REUSE $spec")
    done
done
run_one none "$CROSS_DOMAIN_DATASET" "$MATCHING_NET" \
    || FAILED+=("none $MATCHING_NET cross=$CROSS_DOMAIN_DATASET")

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
    --out "$OUT"
eval_rc=$?

echo ""
echo "======== done (${#FAILED[@]} failed run(s), eval rc $eval_rc) ========"
echo "  $OUT/test_scores_per_combo.csv, test_scores_per_rung.csv        (original)"
echo "  $OUT/test_scores_kxk_per_combo.csv, test_scores_kxk_per_rung.csv  (K x K)"
[ ${#FAILED[@]} -ne 0 ] && exit 1
exit $eval_rc
