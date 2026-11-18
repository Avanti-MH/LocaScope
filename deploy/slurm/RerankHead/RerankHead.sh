#!/bin/bash
#SBATCH --job-name=RerankHead             # Job name -> log/<name>, result/<name>/
#SBATCH --partition=8gpus                 # Partition
#SBATCH --time=48:00:00                   # partition 8gpus caps at 2 days
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=8                 # one process per card, training one model together; fewer: see below
#SBATCH --cpus-per-task=96                # 12 per GPU, the H200 cap: slide reads and photo renders
#SBATCH --mem=1600G                       # 200G per GPU, the H200 cap
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source deploy/slurm/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  locascope/training/RerankHead -- the second phase of stage 2 (locascope/core/stage2_retrieval/spec.md)
#
#  Several heads are trained in ONE process on the SAME FoVs: a level of a slide
#  is encoded once and every head trains on it, and nothing of it is kept on disk
#  (no feature store, no candidate lists: they are made when used).
#
#    sbatch deploy/slurm/RerankHead/RerankHead.sh                       # tilegt, uni2 cls_avg (K = 500 of its list)
#    ENCODER=gigapath EXTRA="--k 100" sbatch deploy/slurm/RerankHead/RerankHead.sh
#    HEADS="tilegt corrnet winembed" TAG=run1 sbatch deploy/slurm/RerankHead/RerankHead.sh
#    HEADS=winembed ENCODER=GigaCoAE+RN3 POOLING=latent sbatch deploy/slurm/RerankHead/RerankHead.sh   # a CoAE network's usage as the tile feature (winembed only)
#    REAL_VAL=gigapath-winembed_found50_missed50 HEADS=winembed sbatch deploy/slurm/RerankHead/RerankHead.sh   # + the real photos of real_val/<name>.tsv scored every epoch, observed only; no list yet: made first, once (make_list.py)
#    REAL_VAL=gigapath-winembed_found50_missed50 REAL_VAL_BUILD_ONLY=1 sbatch --partition=dev --gpus-per-node=1 --cpus-per-task=4 --mem=16G --time=00:30:00 deploy/slurm/RerankHead/RerankHead.sh   # only the list, no training
#    DRY_RUN=1 bash deploy/slurm/RerankHead/RerankHead.sh                # the plan; opens nothing
#    SMOKE=1 sbatch deploy/slurm/RerankHead/RerankHead.sh                # 2 slides a dataset, 3 visits, 1 epoch
#
#  The weights are written to <results>/RerankHead/weights/ and read from there by
#  a recipe (RerankHead/weights/<file>.pt). One checkpoint per head: `_best` (the
#  epoch best at SELECT_K on the held-out FoVs) and `_last`.
#
#  Every flag of cli/train.py can be given in EXTRA, e.g.
#    EXTRA="--sampler-n-per-rung 100 --fov-per-visit 96" sbatch ...
# =============================================================================
HEADS="${HEADS:-tilegt}"
ENCODER="${ENCODER:-uni2}"              # the sliding window's encoder: uni2 | gigapath | GigaCoAE+<arm> (POOLING a usage; winembed only)
POOLING="${POOLING:-cls_avg}"
EPOCHS="${EPOCHS:-30}"
TAG="${TAG:-}"
SELECT_K="${SELECT_K:-30}"
N_TRAIN_WSI="${N_TRAIN_WSI:-0}"          # slides per training dataset; 0 = all
N_VAL_WSI="${N_VAL_WSI:-10}"            # slides of each dataset @pure-half-val draws from
READ_WORKERS="${READ_WORKERS:-5}"        # slide-read workers a process; 0 = all its CPUs. Job 605745 (4 nodes x 4 GPUs x 11 workers): CPUs
                                         # peaked at 26-51% of 48, and /dev/shm (the blocks the workers hand over) ran out on a node
REAL_VAL="${REAL_VAL:-}"                # a list of real photos (real_val/<name>.tsv) scored every epoch beside the validation; observed only, '' = off; not there yet: made first (once)
REAL_VAL_BUILD_ONLY="${REAL_VAL_BUILD_ONLY:-0}"   # 1 = only make the list of REAL_VAL (when it is not there) and stop: no training
EXTRA="${EXTRA:-}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$SMOKE" = "1" ]; then
  EPOCHS=1; N_TRAIN_WSI=2; N_VAL_WSI=2
  # A smoke run's weights go under result/<job name>/ with the log (log/<job name>) and the caches, so that every trace
  # of it starts with `Smoke` and `make clean-smoke` takes it away. They used to go to result/RerankHead/weights/ under
  # the very names of a real run's (no tag): a smoke run would have written over them.
  if [ "$DRY_RUN" != "1" ]; then
    case "${SLURM_JOB_NAME:-}" in
      Smoke*) ;;
      *) echo "SMOKE=1 needs a job name that starts with Smoke: sbatch -J SmokeRerankHead ..."; exit 2 ;;
    esac
    EXTRA="--out /work/u26130998/result/$SLURM_JOB_NAME $EXTRA"
  fi
  EXTRA="--max-visits 3 --fov-per-visit 8 --val-fov-per-visit 4 --steps-per-visit 4 $EXTRA"
fi

# An experiment (job name Exp*) keeps its weights with its log, under result/<job name>/, as a smoke run does: they never
# sit in result/RerankHead/weights/, where an arm reads the real ones. There is no `make clean-exp`: rm -rf the directory.
case "${SLURM_JOB_NAME:-}" in
  Exp*) [ "$DRY_RUN" != "1" ] && EXTRA="--out /work/u26130998/result/$SLURM_JOB_NAME $EXTRA" ;;
esac
[ "$DRY_RUN" = "1" ] && EXTRA="--dry-run $EXTRA"

# The list of real photos is made once, by this job, before the training (one process; the training only reads it). One that is already
# there is read as it is, never made again. DRY_RUN opens nothing, so it makes nothing.
if [ -n "$REAL_VAL" ] && [ "$DRY_RUN" != "1" ]; then
  REAL_VAL_LIST="locascope/training/RerankHead/real_val/$REAL_VAL.tsv"
  if [ -f "$REAL_VAL_LIST" ]; then
    echo "  real-val list exists, read as it is: $REAL_VAL_LIST"
  else
    echo "  real-val list is not there, making it: $REAL_VAL_LIST"
    python -u -m locascope.training.RerankHead.real_val.make_list "$REAL_VAL" || { echo "making the list $REAL_VAL failed"; exit 3; }
  fi
fi
if [ "$REAL_VAL_BUILD_ONLY" = "1" ]; then
  [ -n "$REAL_VAL" ] || { echo "REAL_VAL_BUILD_ONLY=1 needs REAL_VAL=<list name>"; exit 2; }
  exit 0
fi

# One process per GPU the job was given (`torchrun`): each takes every Nth slide of an epoch, encodes its levels and
# trains on them, and after every step the gradients are averaged over the processes, so there is one model. With
# fewer GPUs, ask for them with their share of the CPUs and memory:
#   sbatch --gpus-per-node=4 --cpus-per-task=48 --mem=800G deploy/slurm/RerankHead/RerankHead.sh
# Two nodes (16 GPUs; the `16gpus` partition) are the same with a torchrun on each node, joined at the first:
#   sbatch --partition=16gpus --nodes=2 --gpus-per-node=8 --cpus-per-task=96 --mem=1600G deploy/slurm/RerankHead/RerankHead.sh
GPUS="${SLURM_GPUS_ON_NODE:-1}"
NODES="${SLURM_NNODES:-1}"
LAUNCH="python -u"
if [ "$NODES" -gt 1 ]; then
  HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  # NOT ACTIVE, candidates for job 592907's death (see the comment block above `split_output` in cli/train.py):
  #   (2) the step gets less memory than the job:   srun --mem=0 --ntasks=$NODES ...
  #   (3) too many memory maps per process:         export MALLOC_ARENA_MAX=2
  LAUNCH="srun --ntasks=$NODES --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$GPUS --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:29500 --rdzv_id=$SLURM_JOB_ID"
elif [ "$GPUS" -gt 1 ]; then
  LAUNCH="torchrun --standalone --nproc_per_node=$GPUS"
fi
echo "  $NODES node(s) x $GPUS GPU(s): $LAUNCH"
export PYTHONUNBUFFERED=1      # under torchrun a process's stdout is a pipe: without this the log fills in blocks
export OMP_NUM_THREADS=1       # what torchrun sets anyway; set, it does not print its warning
export PYTHONWARNINGS="ignore:Plan failed with a cudnnException:UserWarning"   # the tissue segmenter's, once a call

$LAUNCH -m locascope.training.RerankHead.train \
  --heads $HEADS --encoder "$ENCODER" --pooling "$POOLING" --epochs "$EPOCHS" --tag "$TAG" --select-k "$SELECT_K" \
  --n-train-wsi "$N_TRAIN_WSI" --n-val-wsi "$N_VAL_WSI" --read-workers "$READ_WORKERS" \
  --real-val "$REAL_VAL" $EXTRA
status=$?

# With several processes each wrote its own log, <log>.r<rank>; this file has the settings, the epochs and the
# validation. The pieces put together, by rank, are <log>.ranks (the pieces stay).
LOG="/work/u26130998/log/${SLURM_JOB_NAME:-RerankHead}"
if [ "$GPUS" -gt 1 ] || [ "$NODES" -gt 1 ]; then
  { for f in $(ls "$LOG".r[0-9]* 2>/dev/null | sort -V); do echo "######## ${f##*/}"; cat "$f"; echo; done; } > "$LOG.ranks"
  echo "  the processes' logs: $LOG.r<rank>, together $LOG.ranks"
fi
exit $status
