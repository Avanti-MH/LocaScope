#!/bin/bash
#SBATCH --job-name=CoaeTrain              # -> log/<name>, result/<name>/<dir>/ ; a smoke run: -J SmokeCoaeTrain
#SBATCH --partition=8gpus
#SBATCH --time=12:00:00                   # 1 GPU: 12 CPUs and 200G, the H200 cap a GPU; 12 h is the upper bound of an arm (resume: COAE_RESUME)
#SBATCH --account=MST114560
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083,25a-hgpn062,25a-hgpn149
#SBATCH -o /work/u26130998/log/%x
#SBATCH -e /work/u26130998/log/%x

# =============================================================================
#  CoAE (locascope/training/CoAE/spec.md), one arm a job. The settings come in as COAE_* variables and go to train.py as flags; the
#  names are not made here: `train.py --print-arm` (the arm code, CoaeConfig.arm) and `--print-dir` (the directory, CoaeConfig.dir_name) make them from the flags. The code is
#  `<local><global><sources>[.<modifier>...]`: R (reconstruction) | D | L (twin contrast on the decoder / the latent) | x, then P
#  (descriptor) | N (descriptor + target contrast), H, V (view contrasts) | x, then the simulated sources' count, then what is off its default.
#    COAE_LEVEL=<0|light|large>      default large   the perturbation level of the perturbed group (0 = no perturbed group)  --level      .noperturb|.lightperturb
#    COAE_MECH=<dep|ind|mix>         default mix     how the views of a place share their targets                             --mechanism  .dep|.ind
#    COAE_VARIANT=<vref|vavg|vdist>  default vref    what the decoder's target is made of                                     --variant    .vavg|.vdist
#    COAE_SIMS=<a,b|none>            (the three)     the simulated sources (none = ref only)                                  --sims       the count (3, 2, 0); .sims-<a+b> off the fixed sets
#    COAE_NOISE=<sigma>              (config 0.5)    the bottleneck noise                                                     --noise      .noise<sigma>
#    COAE_POOL=<c>                   (config 4)      the side of cls''s place descriptor (cells)                              --pool-cells .desc<c>
#    COAE_ENCODER=<name>             (gigapath)      the FM                                                                   --encoder    （代號不變；目錄名加 `-<name>`；stage 2 用 `<Fm>CoAE+<arm>` 的前綴）
#    COAE_LORA=1                     0               the FM with LoRA                                                         --lora       .lora
#    COAE_FMCLS=1                    0               cls' keeps the FM's own CLS as its residual                              --cls-residual .fmresidual
#    COAE_WPOOL=<w>                  (config 1)      the weight of cls''s descriptor loss (0 = the ablation: no P)            --w-pool     P, .weightP<w>
#    COAE_NCE=1                      0               the target contrast on the linear head (w_nce = W_NCE)                   --w-nce      N
#    COAE_VNCE=<1|only>              0               EXPERIMENT ARMS ONLY (spec.md §5): the view-to-view contrast on cls' (w_vnce = W_VNCE);
#                                                    only: w_vnce 1 and no content loss (--w-recon 0 --w-pool 0); not with COAE_WPOOL,
#                                                    COAE_WRECON or COAE_NCE=1                                                --w-vnce     V, .weightV<w>
#    COAE_WRECON=<w>                 (config 1)      the weight of the decoder's reconstruction (0: an experiment arm that has a contrast)  --w-recon  R, .weightR<w>
#    COAE_DNCE=1                     0               EXPERIMENT ARMS ONLY (spec.md §5): the twin contrast on the decoder's output blocks (w_dnce = W_TWIN)  --w-dnce  D
#    COAE_LNCE=1                     0               EXPERIMENT ARMS ONLY: the same twin contrast on the latent (w_lnce = W_TWIN); not with COAE_DNCE=1  --w-lnce  L
#    COAE_HNCE=1                     0               EXPERIMENT ARMS ONLY: the view-to-view contrast on the linear head's output (w_hnce = W_HNCE)  --w-hnce  H
#    COAE_SEED=<n>                   (config 0)      the training seed (a repeat of an arm)                                   --seed       .seed<n>
#    COAE_CAP=<n>                    128             the pool: places a (slide, rung) cell keeps for the whole training       --pool-cap   .pool<n>  (the formal arm 512)
#    COAE_SLIDES=<n>                 (all)           train on the first n slides only                                         --slides     .slides<n>
#    COAE_EPOCHS=<n>                 (config 3)      TrainConfig.epochs; 0 = build the probe, measure once, stop; the formal arm 12   --epochs  .epochs<n>
#    COAE_FM_DTYPE=<fp32|bf16>       (see right)     (no name part) the FM's weight dtype  --fm-dtype: not given = bf16, but fp32 when
#                                                    COAE_CAP is given (the formal arm) or COAE_LORA=1 (LoRA needs fp32)
#    COAE_COMPILE=0                  1               (no name part) no torch.compile on the FM's forward                       --no-compile
#    COAE_WARMUP=<x>                 (config 1.0)    (no name part) the linear warm-up, in epochs (a fraction is fine)         --warmup-epochs
#    COAE_RESUME=<k>                                 (no name part) continue the same arm after epoch k, from weights/epoch<k>.pt   --resume
#    COAE_ARGS                                       anything else for train.py, appended last (a --places here overrides; a flag that is a setting of the arm is in its name)
#  COAE_POOL (desc<c>) is the side of cls''s descriptor; COAE_CAP (pool<n>) is the pool of places: two things.
#  Every GPU takes 128 places a step, whatever the views a place has: places = 128 x GPUs. An epoch lists about 128 places of each cell
#  (cell_parts = cap / 128), so its length does not follow the pool.
#  One process a GPU (torchrun). Out: result/<job>/<dir>/ (log.tsv, val.tsv, weights/), log/<job>_<dir>; examples: code -> directory.
#    sbatch -J ExpCoae deploy/slurm/CoAE/CoaeTrain.sh                                        -> RP3 -> large-mix-vref (pool 128, 3 epochs)
#    COAE_CAP=512 COAE_EPOCHS=12 sbatch deploy/slurm/CoAE/CoaeTrain.sh                       -> RP3.pool512.epochs12 -> large-mix-vref-cap512-epochs12 (the formal arm)
#    COAE_CAP=512 COAE_EPOCHS=12 COAE_RESUME=3 sbatch deploy/slurm/CoAE/CoaeTrain.sh        -> the same arm, on from weights/epoch3.pt
#    COAE_SIMS=none sbatch -J ExpCoae deploy/slurm/CoAE/CoaeTrain.sh                         -> RP0 -> large-mix-vref-refonly
#    COAE_EPOCHS=0 sbatch -J ExpCoaeAudit -p dev -t 00:40:00 deploy/slurm/CoAE/CoaeTrain.sh   -> RP3.epochs0 -> large-mix-vref-epochs0 (the probe once;
#        the audit job is deploy/slurm/CoAE/CoaeAudit.sh). COAE_POOL must divide the Co-figure's cells (gigapath 28: 2, 4, 7)
#    sbatch -J SmokeCoaeTrain -p dev -t 01:00:00 --gpus-per-node=2 --cpus-per-task=24 --mem=400G \
#           --export=ALL,COAE_ARGS="--epochs 1 --cell-epoch 2 --places 16 --log-every 1" deploy/slurm/CoAE/CoaeTrain.sh
#        (the last one: --epochs 1 is in the name too, RP3.epochs1 -> large-mix-vref-epochs1)
# =============================================================================
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate locascope
cd /work/u26130998/LocaScope || exit 2
source deploy/slurm/_env.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore:Plan failed with a cudnnException:UserWarning"   # cudnn's plan notices (as GloRet's jobs)

JOB="${SLURM_JOB_NAME:-CoaeTrain}"
LEVEL="${COAE_LEVEL:-large}" MECH="${COAE_MECH:-mix}" VARIANT="${COAE_VARIANT:-vref}" SIMS="${COAE_SIMS:-}"
NOISE="${COAE_NOISE:-}" POOL="${COAE_POOL:-}" ENCODER="${COAE_ENCODER:-}" LORA="${COAE_LORA:-0}" FMCLS="${COAE_FMCLS:-0}" WPOOL="${COAE_WPOOL:-}"
NCE="${COAE_NCE:-0}" VNCE="${COAE_VNCE:-0}" HNCE="${COAE_HNCE:-0}" DNCE="${COAE_DNCE:-0}" LNCE="${COAE_LNCE:-0}" WRECON="${COAE_WRECON:-}" SEED="${COAE_SEED:-}"
W_NCE=0.5 W_VNCE=0.5 W_TWIN=0.5 W_HNCE=0.5                                  # the contrasts' weights (tau: the config's 0.1)
EPOCHS="${COAE_EPOCHS:-}" WARMUP="${COAE_WARMUP:-}" CAP="${COAE_CAP:-}" SLIDES="${COAE_SLIDES:-}" RESUME="${COAE_RESUME:-}"
COMPILE="${COAE_COMPILE:-1}"
FM_DTYPE="${COAE_FM_DTYPE:-}"
if [[ -z $FM_DTYPE ]]; then                                                    # the formal arm (and LoRA) in full precision, the experiments in bf16
    if [[ -n $CAP || $LORA == 1 ]]; then FM_DTYPE=fp32; else FM_DTYPE=bf16; fi
fi
if [[ $VNCE != 0 && $VNCE != 1 && $VNCE != only ]]; then echo "COAE_VNCE must be 0, 1 or only, got $VNCE"; exit 2; fi
if [[ $VNCE == only && ( -n $WPOOL || $NCE == 1 ) ]]; then echo "COAE_VNCE=only sets w_pool 0 and has no target contrast: not with COAE_WPOOL or COAE_NCE=1"; exit 2; fi
for v in NCE HNCE DNCE LNCE; do                                                # the 0/1 switches
    if [[ ${!v} != 0 && ${!v} != 1 ]]; then echo "COAE_$v must be 0 or 1, got ${!v}"; exit 2; fi
done
if [[ $VNCE == only && -n $WRECON ]]; then echo "COAE_VNCE=only sets w_recon 0: not with COAE_WRECON"; exit 2; fi
if [[ $DNCE == 1 && $LNCE == 1 ]]; then echo "COAE_DNCE=1 and COAE_LNCE=1 are two ways to the same place (the twin contrast on the decoder's blocks or on the latent): give one"; exit 2; fi
NGPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
# 128 places a GPU a step, whatever the views a place has; an epoch lists about 128 places of each cell (the pool goes round in cap / 128 epochs)
PLACES=$(( 128 * NGPU ))
CAPV=${CAP:-128}
CELL_PARTS=$(( CAPV > 128 ? CAPV / 128 : 1 ))
FLAGS=" --level $LEVEL --mechanism $MECH --variant $VARIANT --places $PLACES --pool-cap $CAPV --cell-parts $CELL_PARTS"
[[ -n $SIMS ]]    && FLAGS+=" --sims $SIMS"
[[ -n $NOISE ]]   && FLAGS+=" --noise $NOISE"
[[ -n $POOL ]]    && FLAGS+=" --pool-cells $POOL"
[[ -n $ENCODER ]] && FLAGS+=" --encoder $ENCODER"
[[ $LORA == 1 ]]  && FLAGS+=" --lora"
[[ $FMCLS == 1 ]] && FLAGS+=" --cls-residual"
[[ -n $WRECON ]]  && FLAGS+=" --w-recon $WRECON"
[[ -n $WPOOL ]]   && FLAGS+=" --w-pool $WPOOL"
[[ $NCE == 1 ]]   && FLAGS+=" --w-nce $W_NCE"
[[ $VNCE == 1 ]]  && FLAGS+=" --w-vnce $W_VNCE"
[[ $VNCE == only ]] && FLAGS+=" --w-vnce 1 --w-recon 0 --w-pool 0"
[[ $DNCE == 1 ]]  && FLAGS+=" --w-dnce $W_TWIN"
[[ $LNCE == 1 ]]  && FLAGS+=" --w-lnce $W_TWIN"
[[ $HNCE == 1 ]]  && FLAGS+=" --w-hnce $W_HNCE"
[[ -n $SEED ]]    && FLAGS+=" --seed $SEED"
[[ -n $SLIDES ]]  && FLAGS+=" --slides $SLIDES"
FLAGS+=" --fm-dtype $FM_DTYPE"
[[ $COMPILE == 0 ]] && FLAGS+=" --no-compile"
[[ -n $EPOCHS ]]  && FLAGS+=" --epochs $EPOCHS"
[[ -n $WARMUP ]]  && FLAGS+=" --warmup-epochs $WARMUP"
[[ -n $RESUME ]]  && FLAGS+=" --resume $RESUME"
FLAGS+=" --workers 4 ${COAE_ARGS:-}"                  # reader threads of the probe's Supply only (at start); the places are read by the worker processes
# the names: only train.py knows them (CoaeConfig.arm, the code; CoaeConfig.dir_name, the directory); flags that make no arm stop the job here
ARM=$(python -m locascope.training.CoAE.train --print-arm --out /dev/null $FLAGS) || { echo "the flags do not make an arm"; exit 2; }
DIR=$(python -m locascope.training.CoAE.train --print-dir --out /dev/null $FLAGS) || { echo "the flags do not make a directory name"; exit 2; }
OUT="/work/u26130998/result/$JOB/$DIR"
exec >> "/work/u26130998/log/${JOB}_${DIR}" 2>&1      # a log a run (jobs of one name run side by side; log/<job> keeps only the start)
echo "======== the recipe of this job ========"
echo "  job $SLURM_JOB_ID name $JOB partition $SLURM_JOB_PARTITION gpus $CUDA_VISIBLE_DEVICES ($NGPU) cpus $SLURM_CPUS_PER_TASK"
echo "  cpus: nproc $(nproc), SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK, allowed $(taskset -cp $$ | sed 's/.*: //')"
echo "  arm $ARM (directory $DIR) -> $OUT"
if [[ $VNCE == only ]]; then SHOW_RECON=0 SHOW_POOL=0; else SHOW_RECON=${WRECON:-config 1} SHOW_POOL=${WPOOL:-config 1}; fi
echo "  content: w_recon $SHOW_RECON, w_pool $SHOW_POOL"
echo "  contrasts: nce $NCE (w_nce $W_NCE), vnce $VNCE (w_vnce $W_VNCE; only: 1), hnce $HNCE (w_hnce $W_HNCE), dnce $DNCE and lnce $LNCE (w_twin $W_TWIN), seed ${SEED:-config}"
echo "  FM weights $FM_DTYPE, compile $([[ $COMPILE == 0 ]] && echo off || echo on)"
echo "  train.py$FLAGS"
shm() { echo "  /dev/shm $1: $(find /dev/shm -maxdepth 1 -name 'psm_*' -user "$USER" -printf '%s\n' 2>/dev/null | awk '{n++; b+=$1} END{printf "%d psm_ blocks of %s, %.2f GB", n, ENVIRON["USER"], b/1e9}'); node $(df -h /dev/shm | awk 'NR==2{print $3" used of "$2}')"; }   # the Supply's boards (ours only; other users' blocks share the node): listed, never removed here
shm "at the start"
echo "========================================"
torchrun --standalone --nproc_per_node "$NGPU" -m locascope.training.CoAE.train --out "$OUT" $FLAGS
status=$?
shm "at the end (0 psm_ blocks of this job: nothing leaked)"
exit $status
