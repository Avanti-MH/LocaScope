#!/bin/bash
# =============================================================================
#  The arm with TilePool `attnpool` (locascope/core/stage2_retrieval/proposal.md §3.2), on its own: its training is not that of the
#  other arms (cls, cls_avg: GloRet.sh), so it is a job chain of its own, with its own job name, log and weights.
#
#  Three stages, each a GloRet.sh job that starts when the one before has FINISHED WELL (afterok):
#    A  --min-ds 4          the coarse levels, whose patch tokens (0.6 MB a tile) are kept in host memory between
#                           epochs; GloRet and LogLoRet heads                                   weights tag stageA
#    B  --max-ds 2 --on-demand  the fine levels, the heads trained further on the list of 10000, from A's weights (the
#                           LogLoRet head, and the GloRet heads as rerankers of the list); the tiles under the windows a
#                           visit scores are read when they are asked for, never a whole level  weights tag stageB
#    C  attnpool_frozen     stage B's attention pool frozen: a tile is one vector again, every level is encoded once
#                           and kept like a cls_avg one; the heads are trained on all levels    weights tag stageC
#
#  This file only SUBMITS the three (run it with bash, not sbatch): they are queued, and the log of all three is
#  log/$JOB (A writes it, B and C append). The weights: <results>/GloRet/weights/pure-train_stage{A,B,C}--...
#
#    bash deploy/slurm/GloRet/GloRetAttnPool.sh                              # 1 node x 8 GPUs a stage, partition 8gpus
#    NODES=2 GPUS=4 PARTITION=16gpus bash deploy/slurm/GloRet/GloRetAttnPool.sh   # 2 nodes x 4 GPUs (less /dev/shm a node)
#    HEADS_A="GloRet-GNN-AttnPool LogLoRet-GNN-AttnPool" HEAD_B=LogLoRet-GNN-AttnPool bash ...
#    SMOKE=1 bash deploy/slurm/GloRet/GloRetAttnPool.sh                      # 2 slides, 3 visits, 1 epoch; JOB=SmokeGloRetAttnPool
#    DRY_RUN=1 bash deploy/slurm/GloRet/GloRetAttnPool.sh                    # the three sbatch lines; submits nothing
#    COMMON_EXTRA="--steps-per-visit 4 --fovs-per-step 16 --episodes 2 --val-fov-per-visit 32" bash ...   # the steps of another run
#    A_JOB=<job id> AFTER=<job id> bash ...    # stage A is that job already; B and C start after it and after the AFTER job
#    AFTER=<job id> bash ...                                               # stage A starts only if that job (a test) ended well
#
#  HEAD_B is the LogLoRet head whose attention pool stage C freezes for all its heads (HEADS_C, by default the heads of stage A);
#  stage B trains it and, by default, each GloRet head of HEADS_A as a reranker of the list (HEADS_B to say otherwise).
# =============================================================================
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
JOB="${JOB:-GloRetAttnPool}"
[ "$SMOKE" = "1" ] && JOB="${JOB/#GloRet/SmokeGloRet}"
case "$JOB" in Smoke*) ;; *) [ "$SMOKE" = "1" ] && { echo "SMOKE=1: the job name must start with Smoke ($JOB)"; exit 2; } ;; esac

NODES="${NODES:-1}"                      # nodes a stage uses
GPUS="${GPUS:-8}"                        # GPUs on each (12 CPUs and 200G a GPU, the cap)
PARTITION="${PARTITION:-$([ "$SMOKE" = "1" ] && echo dev || echo 8gpus)}"
TIME="${TIME:-$([ "$SMOKE" = "1" ] && echo 02:00:00 || echo 48:00:00)}"
HEADS_A="${HEADS_A:-GloRet-SelfAttn-AttnPool LogLoRet-SelfAttn-AttnPool}"
HEAD_B="${HEAD_B:-LogLoRet-SelfAttn-AttnPool}"
HEADS_C="${HEADS_C:-$HEADS_A}"
TOP_P_A="${TOP_P_A:-1.0 2000}"           # --top-p of a stage: a share (1.0: every window) for a GloRet head, a count for a LogLoRet
TOP_P_B="${TOP_P_B:-128}"                  # the tokens of only the pool are read (--on-demand): a count of the list, a GloRet head's too
if [ -z "${HEADS_B:-}" ]; then
  HEADS_B="$HEAD_B"
  for h in $HEADS_A; do case "$h" in GloRet-*) HEADS_B="$HEADS_B $h@items=list@top=$TOP_P_B" ;; esac; done
fi
TOP_P_C="${TOP_P_C:-1.0 2000}"
EXTRA_B="${EXTRA_B:-}"                         # more flags for stage B only (e.g. --warmup-epochs 1: the first epoch's pools from the teacher, the rest from the network)
COMMON_EXTRA="${COMMON_EXTRA:-}"                 # flags of all three stages, before the stage's own (which win): the steps, episodes and validation size of the run it is compared with
EPOCHS_A="${EPOCHS_A:-20}"
EPOCHS_B="${EPOCHS_B:-5}"
EPOCHS_C="${EPOCHS_C:-20}"
EXCLUDE="25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083"
RESULTS="/work/u26130998/result"
case "$JOB" in Exp*) RESULTS="$RESULTS/$JOB" ;; esac                   # an experiment's too (GloRet.sh sends its weights there)
[ "$SMOKE" = "1" ] && RESULTS="$RESULTS/$JOB"          # a smoke run's weights sit with its log, under its name

case " $HEADS_A " in *" $HEAD_B "*) ;; *) echo "HEAD_B ($HEAD_B) must be one of HEADS_A ($HEADS_A)"; exit 2 ;; esac
case "$HEAD_B" in LogLoRet-*) ;; *) echo "HEAD_B must be a LogLoRet (stage B re-ranks the list)"; exit 2 ;; esac

# the weights stage B writes for HEAD_B (GloRet.sh names them: <data>_<tag>--<encoder>--<pooling>--<method>[--<no-match>]--best)
METHOD_B="${HEAD_B%%+*}"; OP_B="${HEAD_B#*+}"; [ "$OP_B" = "$HEAD_B" ] && OP_B="dustbin"
POOL_WEIGHTS="$RESULTS/GloRet/weights/pure-train_stageB--gigapath--attnpool--${METHOD_B}--${OP_B}--best.pt"

SBATCH=(sbatch --parsable -J "$JOB" --partition="$PARTITION" --time="$TIME" --nodes="$NODES" --gpus-per-node="$GPUS"
        --cpus-per-task=$((12 * GPUS)) --mem="$((200 * GPUS))G" --exclude="$EXCLUDE")
export SMOKE

submit() {   # submit <dependency or ""> <open-mode> <heads> <pooling> <extra flags> <--top-p values>
  local dep="$1" mode="$2" heads="$3" pooling="$4" extra="$5" top="$6"
  local cmd=("${SBATCH[@]}" "--open-mode=$mode")
  [ -n "$dep" ] && cmd+=("--dependency=afterok:$dep")
  cmd+=(deploy/slurm/GloRet/GloRet.sh)
  if [ "$DRY_RUN" = "1" ]; then
    echo "HEADS=\"$heads\" POOLING=$pooling TOP_P=\"$top\" EXTRA=\"$COMMON_EXTRA $extra\" ${cmd[*]}" >&2
    echo "dry$RANDOM"
  else
    HEADS="$heads" POOLING="$pooling" TOP_P="$top" EXTRA="$COMMON_EXTRA $extra" "${cmd[@]}"
  fi
}

# a smoke run is one epoch of three visits (GloRet.sh): no --epochs, which would override it, and small visits
if [ "$SMOKE" = "1" ]; then EA=""; EB=""; EC=""; VB="--fov-per-visit 4"; else EA="--epochs $EPOCHS_A"; EB="--epochs $EPOCHS_B"; EC="--epochs $EPOCHS_C"; VB="--fov-per-visit 16"; fi
A="${A_JOB:-}"                                  # A_JOB=<job id>: stage A is already submitted (or ran), start the chain at B
[ -z "$A" ] && A=$(submit "${AFTER:-}" truncate "$HEADS_A" attnpool "--min-ds 4 --tag stageA $EA" "$TOP_P_A")
B=$(submit "$A${A_JOB:+${AFTER:+:$AFTER}}" append "$HEADS_B" attnpool "--max-ds 2 --on-demand --init-tag stageA --tag stageB $EB $VB --prefetch 1 $EXTRA_B" "$TOP_P_B")
C=$(submit "$B" append "$HEADS_C" attnpool_frozen "--pool-weights $POOL_WEIGHTS --tag stageC $EC" "$TOP_P_C")
echo "$JOB: A=$A  B=$B  C=$C   (log/$JOB; weights under $RESULTS/GloRet/weights/)"
