#!/bin/bash
#SBATCH --job-name=GloRet                 # Job name -> log/<name>, result/<name>/
#SBATCH --partition=8gpus                 # Partition
#SBATCH --time=48:00:00                   # partition 8gpus caps at 2 days
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=8                 # one process per card, training one model together; fewer: see below
#SBATCH --cpus-per-task=96                # 12 per GPU, the H200 cap: slide reads and photo renders
#SBATCH --mem=1600G                       # 200G per GPU, the H200 cap
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083
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
#  locascope/training/GloRet -- GloRet (Arch 1, a bi-encoder) and LogLoRet (Arch 2, a pair scorer) of
#  locascope/core/stage2_retrieval/proposal.md, on synthetic FoVs made as they are used.
#
#  Several heads are trained in ONE process on the SAME FoVs: a level of a slide is encoded once and every head
#  trains on it. A head is the proposal's name: GloRet-{ContextEn}-{GlobalEn}, or
#  LogLoRet-{ContextEn}-{GlobalEn}[+{dustbin|sinkhorn|sigmoid}] (the no-match option is trained in).
#
#    HEADS="GloRet-SelfAttn-AttnPool LogLoRet-SelfAttn-AttnPool" sbatch deploy/slurm/GloRet/GloRet.sh
#    HEADS="GloRet-SelfAttn-AttnPool" EXTRA="--epochs 5" sbatch deploy/slurm/GloRet/GloRet.sh
#    DRY_RUN=1 bash deploy/slurm/GloRet/GloRet.sh                 # the plan; opens nothing
#    SMOKE=1 sbatch -J SmokeGloRet deploy/slurm/GloRet/GloRet.sh   # 2 slides a dataset, 3 visits, 1 epoch; weights in result/SmokeGloRet/
#
#  A LogLoRet trains on the list a sliding window makes at the size of a retrieval (--raw-topk 10000 windows; the list the
#  pooling cannot tell from the answer; POOLING=cls for the sliding window's cls list) and a step scores a pool of it
#  (TOP_P: an integer, the entries of the list), validation re-orders the first 100. Change them with
#    EXTRA="--raw-topk 1000 --val-keep 100" TOP_P="1.0 1000" sbatch deploy/slurm/GloRet/GloRet.sh
#
#  EVERY HEAD IS TRAINED THE SAME WAY, what differs is its scorer (GloRet: a vector a window; LogLoRet: a pair at a time). What
#  its softmax is over, and what it is validated as, are the head's to say, after its name:
#      Name@items=level|list    the windows of the level (a GloRet head's default) or the entries of the list (a LogLoRet's). It is
#                               how the head is trained, so what it is: over the level a RETRIEVER (validated listing the level's
#                               windows alone, R@500), over the list a RERANKER (re-ordering the list's first --val-keep, R@10),
#                               whichever architecture
#      Name@top=<value>         its own --top-p (a share 0.05 / 1.0 or a count 2000), else the job's
#  A head given other than the defaults is another head: its name, its log rows and its weights (`..._items-list`).
#  A list that holds no positive gets the answer ADDED to its items (log: `injected`), the FoV is not left out. Over a level a
#  LogLoRet head takes '--top-p' 0.5 or less (pools chosen once per episode: the first phase's scores in the warmup epochs, its own after). Example, a GloRet head as a reranker:
#      HEADS="GloRet-SelfAttn-AttnPool@items=list@top=2000 LogLoRet-SelfAttn-AttnPool" TOP_P="1.0 2000" ...
#
#  TilePool attnpool, in three stages, each a run of its own (--tag keeps their weights apart; the list is the sliding
#  window's `cls` one in all three):
#    A  the coarse levels (ds >= 4), whose tiles are few: their patch tokens (0.6 MB a tile) are kept in host memory
#       between epochs; 16 GPUs hold the training set (about 83 GB a process, estimated):
#         POOLING=attnpool HEADS="GloRet-SelfAttn-AttnPool LogLoRet-SelfAttn-AttnPool" \
#           EXTRA="--min-ds 4 --tag stageA" sbatch --partition=16gpus --nodes=4 --gpus-per-node=4 ... deploy/slurm/GloRet/GloRet.sh
#    B  the fine levels (ds <= 2), the LogLoRet re-trained from stage A on the list of 10000, the pool and all; only the
#       tiles under the windows a visit scores are read and encoded (--on-demand), so a level need not fit the card:
#         POOLING=attnpool HEADS="LogLoRet-SelfAttn-AttnPool" \
#           EXTRA="--max-ds 2 --on-demand --init-tag stageA --tag stageB --fov-per-visit 16 --prefetch 1" TOP_P=128 sbatch ...
#    C  the pool frozen (stage B's): a tile is one vector again, every level is encoded once and kept like a cls_avg
#       one, and the heads are trained on all levels as before:
#         POOLING=attnpool_frozen HEADS="GloRet-SelfAttn-AttnPool LogLoRet-SelfAttn-AttnPool" \
#           EXTRA="--tag stageC --pool-weights /work/u26130998/result/GloRet/weights/pure-train_stageB--gigapath--attnpool--LogLoRet-SelfAttn-AttnPool--dustbin--best.pt" sbatch ...
#
#  The weights are written to <results>/GloRet/weights/ (ArmSpec.learned_weights), where an arm reads them: `_best`
#  (GloRet: the epoch best at R@500 on the held-out FoVs; LogLoRet: at R@10) and `_last`. A smoke run's carry the
#  tag `smoke` in their data segment, so they overwrite nothing.
#
#  Every flag of cli/train.py can be given in EXTRA, e.g.
#    EXTRA="--sampler-n-per-rung 100 --fov-per-visit 96" sbatch ...
#
#  AN ARRAY: every element is a job of its own with the resources above (8 GPUs, 96 CPUs, 1600G each), and takes
#  one GROUP of heads, the groups given in HEAD_GROUPS separated by `;`. The levels are NOT shared between
#  elements: each encodes them again (a group trains its heads on one encoding), so make groups of several heads,
#  not one head each. The elements' logs must be told apart on the command line (-o/-e override the ones above):
#
#    HEAD_GROUPS="GloRet-SelfAttn-AttnPool GloRet-GNN-GraphPool; LogLoRet-SelfAttn-AttnPool LogLoRet-GNN-GraphPool" \
#      sbatch --array=0-1 -o '/work/u26130998/log/%x.%a' -e '/work/u26130998/log/%x.%a' deploy/slurm/GloRet/GloRet.sh
#
#  Element i writes log/<job>.<i> and, with several GPUs, log/<job>.<i>.r<rank> and log/<job>.<i>.ranks.
#  `--array=0-1%1` runs them one after the other; the project's 32-GPU cap (and the running-jobs-per-user cap of the
#  partition) decide how many run at once.
#
#  THE FORMAL RUNS (the basic combinations, to see which architecture and which option helps). Each is one job of its
#  own (log/<job>), all heads of a job trained on the same FoVs; the weights are named by encoder, pooling and head
#  (a head is trained by one job only). The defaults hold: TOP_P is set by the run (1.0: a GloRet head's softmax over every
#  window; 2000: a LogLoRet's over 2000 of the list), --episodes 3, --warmup-epochs 2, --min-candidates 256,
#  ignore zone 0.5, wrong rotations ignored, no soft targets, no decoder loss.
#
#   1. TilePool attnpool, gigapath, on its own (three stages; GloRet-SelfAttn-AttnPool and the LogLoRet that learns the pool):
#        bash deploy/slurm/GloRet/GloRetAttnPool.sh
#   2. gigapath cls, GloRet and LogLoRet together (15 heads; the first of each is also a cell of 3):
#        TOP_P="1.0 2000" ENCODER=gigapath POOLING=cls HEADS="GloRet-SelfAttn-AttnPool GloRet-SelfAttn-MaskedMean GloRet-SelfAttn-CNN \
#          GloRet-SelfAttn-Align GloRet-SptSelfAttn-AttnPool GloRet-SptSelfAttn-CNN GloRet-SptSelfAttn-Align \
#          GloRet-Identity-AttnPool GloRet-Identity-CNN GloRet-Identity-Align GloRet-GNN-GraphPool GloRet-CLSTrans-Identity \
#          LogLoRet-SelfAttn-AttnPool+dustbin LogLoRet-SelfAttn-AttnPool+sinkhorn LogLoRet-SelfAttn-AttnPool+sigmoid" \
#          sbatch -J GloRetGigaCls deploy/slurm/GloRet/GloRet.sh
#   3. {uni2, gigapath} x {cls, cls_avg, grid2x2}: the other five cells, two heads each (the same as in 2 for gigapath cls):
#        TOP_P="1.0 2000" ENCODER=uni2 POOLING=cls HEADS="GloRet-SelfAttn-AttnPool LogLoRet-SelfAttn-AttnPool+dustbin" sbatch -J GloRetUni2Cls deploy/slurm/GloRet/GloRet.sh
#        (-J GloRetUni2ClsAvg / GloRetUni2Grid with POOLING=cls_avg / grid2x2; -J GloRetGigaClsAvg / GloRetGigaGrid with ENCODER=gigapath)
#      The search of GloRet (exact | hnsw | ivf) is the retriever's, not trained: hnsw and ivf are TODO in
#      locascope/core/stage2_retrieval/RetrieverGloRet.py, the weights are the same.
#   4. d (--dim, the width after W_in) and d_r (--embed-dim, the retrieval vector) of gigapath cls, GloRet-SelfAttn-AttnPool:
#      1024 and no change (1536: nothing is compressed, so a decoder loss would not exist). The weights' names do not hold
#      the width: --tag keeps them apart (arm names for them: when the ablation is done).
#        TOP_P=1.0 ENCODER=gigapath POOLING=cls HEADS=GloRet-SelfAttn-AttnPool EXTRA="--dim 1024 --embed-dim 1024 --tag d1024" sbatch -J GloRetDim1024 deploy/slurm/GloRet/GloRet.sh
#        TOP_P=1.0 ENCODER=gigapath POOLING=cls HEADS=GloRet-SelfAttn-AttnPool EXTRA="--dim 1536 --embed-dim 1536 --tag d1536" sbatch -J GloRetDim1536 deploy/slurm/GloRet/GloRet.sh
#      Each of these takes 8 GPUs (and the project 32 at most): send them as the GPUs free up, not all at once.
# =============================================================================
HEADS="${HEADS:-GloRet-SelfAttn-AttnPool}"
if [ -n "$HEAD_GROUPS" ]; then
  if [ -z "$SLURM_ARRAY_TASK_ID" ] && [ "$DRY_RUN" != "1" ]; then
    echo "HEAD_GROUPS is for an array: sbatch --array=0-N ..."; exit 2
  fi
  IFS=';' read -r -a GROUPS_OF_HEADS <<< "$HEAD_GROUPS"
  HEADS="${GROUPS_OF_HEADS[${SLURM_ARRAY_TASK_ID:-0}]}"
  if [ -z "${HEADS// /}" ]; then
    echo "array element ${SLURM_ARRAY_TASK_ID:-0} has no group: HEAD_GROUPS holds ${#GROUPS_OF_HEADS[@]}"; exit 2
  fi
fi
echo "  heads: $HEADS"
ENCODER="${ENCODER:-gigapath}"          # the sliding window's encoder and the features the heads read: uni2 | gigapath
POOLING="${POOLING:-cls_avg}"
EPOCHS="${EPOCHS:-20}"
N_TRAIN_WSI="${N_TRAIN_WSI:-0}"          # slides per training dataset; 0 = all
N_VAL_WSI="${N_VAL_WSI:-10}"            # slides of each dataset @pure-half-val draws from
EXTRA="${EXTRA:-}"
# TOP_P is --top-p: how many of a FoV's items its softmax is over, a share for a GloRet head (a number with a point: 0.05,
# 1.0 is every window of the level) and a count for a LogLoRet head (an integer: 2000 of the list). A job with heads of both
# kinds gives one of each ("1.0 2000"). There is no default, as a run must say what it is (a smoke run: 1.0 and 256).
TOP_P="${TOP_P:-}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$SMOKE" = "1" ]; then
  EPOCHS=1; N_TRAIN_WSI=2; N_VAL_WSI=2
  # A smoke run's weights go under result/<job name>/ with the log (log/<job name>) and the caches, so that every trace
  # of it starts with `Smoke` and `make clean-smoke` takes it away; they never sit in result/GloRet/weights/, where an
  # arm reads the real ones. (A chain of stages reads the one before from there: result/<job>/GloRet/weights/.)
  if [ "$DRY_RUN" != "1" ]; then
    case "${SLURM_JOB_NAME:-}" in
      Smoke*) ;;
      *) echo "SMOKE=1 needs a job name that starts with Smoke: sbatch -J SmokeGloRet ..."; exit 2 ;;
    esac
    EXTRA="--out /work/u26130998/result/$SLURM_JOB_NAME $EXTRA"
  fi
  EXTRA="--max-visits 3 --fov-per-visit 8 --val-fov-per-visit 4 --steps-per-visit 4 --tag smoke $EXTRA"
fi

# An experiment (job name Exp*) keeps its weights with its log, under result/<job name>/, as a smoke run does: they never
# sit in result/GloRet/weights/, where an arm reads the real ones. There is no `make clean-exp`: rm -rf the directory.
case "${SLURM_JOB_NAME:-}" in
  Exp*) [ "$DRY_RUN" != "1" ] && EXTRA="--out /work/u26130998/result/$SLURM_JOB_NAME $EXTRA" ;;
esac
[ "$DRY_RUN" = "1" ] && EXTRA="--dry-run $EXTRA"
if [ -z "$TOP_P" ]; then
  if [ "$SMOKE" = "1" ] || [ "$DRY_RUN" = "1" ]; then TOP_P="1.0 256"
  else echo "TOP_P is not set: the share of the level's windows a GloRet head's softmax is over (1.0: all) and/or the count of the list a LogLoRet head's is over (2000)"; exit 2
  fi
fi

# One process per GPU the job was given (`torchrun`): each takes every Nth slide of an epoch, encodes its levels and
# trains on them, and after every step the gradients are averaged over the processes, so there is one model per head.
# With fewer GPUs, ask for them with their share of the CPUs and memory:
#   sbatch --gpus-per-node=1 --cpus-per-task=12 --mem=200G deploy/slurm/GloRet/GloRet.sh
# Two nodes (16 GPUs; the `16gpus` partition) are the same with a torchrun on each node, joined at the first:
#   sbatch --partition=16gpus --nodes=2 --gpus-per-node=8 --cpus-per-task=96 --mem=1600G deploy/slurm/GloRet/GloRet.sh
GPUS="${SLURM_GPUS_ON_NODE:-1}"
NODES="${SLURM_NNODES:-1}"
LAUNCH="python -u"
if [ "$NODES" -gt 1 ]; then
  HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  LAUNCH="srun --ntasks=$NODES --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$GPUS --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:$((29500 + ${SLURM_ARRAY_TASK_ID:-0})) --rdzv_id=$SLURM_JOB_ID"
elif [ "$GPUS" -gt 1 ]; then
  LAUNCH="torchrun --standalone --nproc_per_node=$GPUS"
fi
echo "  $NODES node(s) x $GPUS GPU(s): $LAUNCH"
export PYTHONUNBUFFERED=1      # under torchrun a process's stdout is a pipe: without this the log fills in blocks
export OMP_NUM_THREADS=1       # what torchrun sets anyway; set, it does not print its warning
export PYTHONWARNINGS="ignore:Plan failed with a cudnnException:UserWarning"   # the tissue segmenter's, once a call

$LAUNCH -m locascope.training.GloRet.train \
  --heads $HEADS --encoder "$ENCODER" --pooling "$POOLING" --epochs "$EPOCHS" \
  --n-train-wsi "$N_TRAIN_WSI" --n-val-wsi "$N_VAL_WSI" --top-p $TOP_P $EXTRA
status=$?

# With several processes each wrote its own log, <log>.r<rank>; this file has the settings, the epochs and the
# validation. The pieces put together, by rank, are <log>.ranks (the pieces stay).
LOG="/work/u26130998/log/${SLURM_JOB_NAME:-GloRet}${SLURM_ARRAY_TASK_ID:+.$SLURM_ARRAY_TASK_ID}"
if [ "$GPUS" -gt 1 ] || [ "$NODES" -gt 1 ]; then
  { for f in $(ls "$LOG".r[0-9]* 2>/dev/null | sort -V); do echo "######## ${f##*/}"; cat "$f"; echo; done; } > "$LOG.ranks"
  echo "  the processes' logs: $LOG.r<rank>, together $LOG.ranks"
fi
exit $status
