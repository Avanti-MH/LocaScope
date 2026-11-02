#!/bin/bash
#SBATCH --job-name=GloRetMain             # Job name -> log/<name>, result/<name>/ (a smoke run: -J SmokeGloRetMain)
#SBATCH --partition=8gpus                 # Partition
#SBATCH --time=48:00:00                   # partition 8gpus caps at 2 days
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=8                 # one process per card, training one model together
#SBATCH --cpus-per-task=96                # 12 per GPU, the H200 cap
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
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  THE RECIPE of this job, the only one in this file; to run other settings write over it.
#  GloRet (a bi-encoder) and LogLoRet (a pair scorer) both as RETRIEVERS: trained over the level's windows, validated listing the
#  level (LogLoRet: every window of the level scored, a pair each), selected on R@500; trained in ONE process on the SAME FoVs.
#  Trained on `bench` FoVs; validated, and its weights selected, on `bench-real` FoVs of the held-out slides; the real photos
#  (confirmed places of ki67_with_photo) are read every epoch for monitoring only. The job prints every term of the loss as it is
#  before its weight (log: `training losses`), so that the weights can be set to make the terms of one size.
#  A smoke run (`sbatch -J SmokeGloRetMain --partition=dev --time=02:00:00 --gpus-per-node=1 --cpus-per-task=12 --mem=200G
#  jobscripts/GloRet/GloRetMain.sh`) is the same recipe on 2 slides, 3 visits, 1 epoch, 40 real photos; its weights, log and caches
#  are under Smoke*.
# =============================================================================
TOP_P="0.1"                             # a share of the level's windows each FoV's softmax is over, for BOTH heads (<= 0.5: episodes); to try: 0.5, 1.0
HEADS="GloRet-SelfAttn-AttnPool LogLoRet-SelfAttn-AttnPool@items=level"
ENCODER="gigapath"
POOLING="grid2x2"                       # tile pool; to try: cls, cls_avg
FOV="bench"                             # training FoVs
VAL_FOV="bench-real"                    # validation FoVs: where the weights are selected
REAL_PHOTOS="1000"                      # confirmed real photos read every epoch (monitoring only)
EPOCHS="10"                            # LogLoRet's pairs make an epoch hours long (50 slides: about 3 h in the steady state, a guess); best and last weights are saved every epoch
N_TRAIN_WSI="25"                        # slides of EACH training source (2 sources: 50 slides; 0 = all 200, which a 48 h job cannot get through)
N_VAL_WSI="10"
VAL_LOGLORET_SHARE="1.0"                # LogLoRet as a retriever scores EVERY window of the level (a pair each); the share of validation FoVs and real photos it is ranked on: all
EPISODES="3"                            # to try: 1, 9
MAX_FOV_USES="2"                      # a FoV is trained on at most about this many times in a visit: with fewer FoVs than steps x FoVs-per-step / this, the steps take fewer FoVs each (0: no limit)
WARMUP_EPOCHS="3"                       # to try: 0, 5
SOFT_TARGET="gaussian"                  # round 2 of ExpGloRetScreen (622591): sigma 0.5 with w-geo 1.0 was the one clear gain for GloRet; exponential, near-alpha 0.1, ignore-expand 1.0, wrong-rotation negative gained little
W_GEO="1.0"                             # both heads are retrievers here, so the settings that helped GloRet as a retriever in round 2 of ExpGloRetScreen (622591) are the job's for both
SIGMA="0.5"
NEAR_ALPHA="0.3"
IGNORE_EXPAND="0.5"
WRONG_ROTATION="ignore"
W_CORR="0.1"                            # LogLoRet's correspondence loss
W_CYCLE="0.1"                           # LogLoRet's cycle loss
RECON="vae"
W_RECON="0.1"
W_KL="1e-3"
PAIR_CHUNK="4096"                       # LogLoRet: pairs scored at a time (memory against speed, not the result); measured on one GPU: 256 -> 1024 made the steps and the validation ranking about twice as fast
ENCODER_BATCH="2048"                    # tiles the tile encoder runs at a time (was 128: slow at the first epoch's encoding of the levels)
GPU_LEVELS_GB="40"                       # GB of each card the encoded levels stay on between epochs (no copy back from the host); the rest stays on the host. 0: all on the host
# =============================================================================

EXTRA=""
case "${SLURM_JOB_NAME:-}" in
  Smoke*)
    WORLD=$(( ${SLURM_NNODES:-1} * ${SLURM_GPUS_ON_NODE:-1} ))          # a process takes every WORLD-th slide: each gets one (at least 2 a source)
    WARMUP_EPOCHS=1                                                       # a smoke has two epochs: the first pools are the teacher's (the first phase's scores), the second the heads' own, so that the change between them runs
    EPOCHS=2; N_TRAIN_WSI=$(( WORLD / 2 > 2 ? WORLD / 2 : 2 )); N_VAL_WSI=$N_TRAIN_WSI; REAL_PHOTOS=$(( 40 * (WORLD > 1 ? WORLD / 2 : 1) ))
    EXTRA="--out /work/u26130998/result/$SLURM_JOB_NAME --max-visits 3 --fov-per-visit 8 --val-fov-per-visit 4 --steps-per-visit 4 --tag smoke" ;;
  *) ;;
esac
case "${SLURM_JOB_NAME:-}" in
  *Profile) EXTRA="$EXTRA --profile" ;;     # -J SmokeGloRetMainProfile: also run the first pool choice and the first step of each head under torch.profiler
esac
case "${SLURM_JOB_NAME:-}" in
  Exp*) EXTRA="--out /work/u26130998/result/$SLURM_JOB_NAME" ;;
esac

echo "================================================================================"
echo " GloRetMain   job $SLURM_JOB_ID on $(hostname)   partition $SLURM_JOB_PARTITION   $(date '+%F %T')"
echo "================================================================================"
echo " resources     ${SLURM_NNODES:-1} node(s) x ${SLURM_GPUS_ON_NODE:-1} GPUs, $SLURM_CPUS_PER_TASK CPUs a node, one process per GPU"
echo " heads         (both trained as retrievers over the level's windows, on the same FoVs)"
for h in $HEADS; do echo "                 $h"; done
echo " features      encoder $ENCODER, tile pool $POOLING"
echo " data          train: $N_TRAIN_WSI slides of EACH source (0 = all 200), FoVs '$FOV'"
echo "               validation (weights are selected here): $N_VAL_WSI slides of each source, FoVs '$VAL_FOV'"
echo "               real photos (monitoring only): $REAL_PHOTOS    epochs: $EPOCHS"
echo "               LogLoRet is ranked on $VAL_LOGLORET_SHARE of the validation FoVs and real photos"
echo " sampling      top-p $TOP_P of the level's windows per FoV,  episodes $EPISODES,  warmup epochs $WARMUP_EPOCHS,  max uses of a FoV $MAX_FOV_USES"
echo " labels        soft target $SOFT_TARGET (sigma $SIGMA, near-alpha $NEAR_ALPHA),  ignore-expand $IGNORE_EXPAND,  wrong rotation: $WRONG_ROTATION"
echo " loss weights  geo $W_GEO,  corr $W_CORR,  cycle $W_CYCLE,  recon $RECON $W_RECON,  kl $W_KL"
echo " pair chunk    $PAIR_CHUNK pairs of LogLoRet scored at a time;  tile encoder batch $ENCODER_BATCH tiles;  encoded levels kept on the card up to $GPU_LEVELS_GB GB"
echo " extra         ${EXTRA:-none}"
echo " host limits   max_map_count $(cat /proc/sys/vm/max_map_count),  virtual memory $(ulimit -v),  open files $(ulimit -n),  processes $(ulimit -u),  locked memory $(ulimit -l)"
echo "               $(free -g | awk '/Mem:/ {print "node memory " $2 " GB, free " $4 " GB"}')"
echo "================================================================================"

GPUS="${SLURM_GPUS_ON_NODE:-1}"
NODES="${SLURM_NNODES:-1}"
LAUNCH="python -u"
if [ "$NODES" -gt 1 ]; then       # several nodes (sbatch --nodes=2 --gpus-per-node=8 ...): a torchrun on each node, joined at the first (as GloRet.sh)
  HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  LAUNCH="srun --ntasks=$NODES --ntasks-per-node=1 torchrun --nnodes=$NODES --nproc_per_node=$GPUS --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:29500 --rdzv_id=$SLURM_JOB_ID"
elif [ "$GPUS" -gt 1 ]; then
  LAUNCH="torchrun --standalone --nproc_per_node=$GPUS"
fi
export PYTHONUNBUFFERED=1      # under torchrun a process's stdout is a pipe: without this the log fills in blocks
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=8      # job 635195 failed on a 17 MiB array on a node with 1.3 TB free, so a limit on mappings, not memory, is the suspect (cause not proven):
                               # glibc gives each thread of the FoV renderers an arena of its own (8 x 96 CPUs at most), each a set of mappings; 8 caps them
export PYTHONWARNINGS="ignore:Plan failed with a cudnnException:UserWarning"

$LAUNCH training/GloRet/cli/train.py \
  --heads $HEADS --encoder "$ENCODER" --pooling "$POOLING" --fov "$FOV" --val-fov "$VAL_FOV" --real-photos "$REAL_PHOTOS" \
  --epochs "$EPOCHS" --n-train-wsi "$N_TRAIN_WSI" --n-val-wsi "$N_VAL_WSI" --val-logloret-share "$VAL_LOGLORET_SHARE" \
  --top-p "$TOP_P" --episodes "$EPISODES" --max-fov-uses "$MAX_FOV_USES" --warmup-epochs "$WARMUP_EPOCHS" \
  --soft-target "$SOFT_TARGET" --w-geo "$W_GEO" --sigma "$SIGMA" --near-alpha "$NEAR_ALPHA" \
  --ignore-expand "$IGNORE_EXPAND" --wrong-rotation "$WRONG_ROTATION" \
  --w-corr "$W_CORR" --w-cycle "$W_CYCLE" --recon "$RECON" --w-recon "$W_RECON" --w-kl "$W_KL" --gpu-levels-gb "$GPU_LEVELS_GB" --pair-chunk "$PAIR_CHUNK" --encoder-batch "$ENCODER_BATCH" $EXTRA
status=$?

LOG="/work/u26130998/log/${SLURM_JOB_NAME:-GloRetMain}"
if [ "$GPUS" -gt 1 ] || [ "$NODES" -gt 1 ]; then
  { for f in $(ls "$LOG".r[0-9]* 2>/dev/null | sort -V); do echo "######## ${f##*/}"; cat "$f"; echo; done; } > "$LOG.ranks"
  echo "  the processes' logs: $LOG.r<rank>, together $LOG.ranks"
fi
# a smoke run shares its job name, so its log is written over by the next: keep a copy by job id
case "${SLURM_JOB_NAME:-}" in
  Smoke*) KEEP=/work/u26130998/scratch/gloret/smoke_logs; mkdir -p "$KEEP"
          cp -p "$LOG" "$KEEP/$SLURM_JOB_NAME.$SLURM_JOB_ID"
          [ -f "$LOG.ranks" ] && cp -p "$LOG.ranks" "$KEEP/$SLURM_JOB_NAME.$SLURM_JOB_ID.ranks" ;;
esac
exit $status
