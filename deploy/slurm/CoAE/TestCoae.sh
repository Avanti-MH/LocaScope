#!/bin/bash
#SBATCH --job-name=TestCoae               # -> log/%x, result/%x/
#SBATCH --partition=dev
#SBATCH --time=00:30:00
#SBATCH --account=MST114560
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                 # the partitions take at least 1; only 1b and 2 use it (no training here)
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083,25a-hgpn062,25a-hgpn149
#SBATCH -o /work/u26130998/log/%x
#SBATCH -e /work/u26130998/log/%x

# =============================================================================
#  Every test of CoAE (locascope/training/CoAE), one after the other; each part of the flow is run once:
#    1  tests/TestCoAE/test_coae.py         no slide, no FM: the layout, the levels, the views, the Co-figure's fields and moments, the masks, the losses
#    1b tests/TestCoAE/test_sim_gpu.py      the GPU simulation of the views (GpuSims) against the CPU's (the lines also in result/TestCoae/sim_gpu.txt)
#    2  tests/TestCoAE/test_coae_slide.py   on 1 slide: the supply (two shared Supplies), the targets, the network frozen and with the residual
#                                       cls', which loss reaches what, the probe (1 slide)
#  No training here (user, 2026-11-16): train.py is covered by the Smoke runs (-J SmokeCoaeTrain, deploy/slurm/CoAE/CoaeTrain.sh).
#    sbatch deploy/slurm/CoAE/TestCoae.sh
# =============================================================================
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate locascope
cd /work/u26130998/LocaScope || exit 2
source deploy/slurm/_env.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore:Plan failed with a cudnnException:UserWarning"   # cudnn's plan notices (as GloRet's jobs)

OUT=/work/u26130998/result/$SLURM_JOB_NAME
echo "======== the recipe of this job ========"
echo "  job $SLURM_JOB_ID name $SLURM_JOB_NAME gpus $CUDA_VISIBLE_DEVICES -> $OUT"
echo "  cpus: nproc $(nproc), SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK, allowed $(taskset -cp $$ | sed 's/.*: //')"
shm() { echo "  /dev/shm $1: $(find /dev/shm -maxdepth 1 -name 'psm_*' -user "$USER" -printf '%s\n' 2>/dev/null | awk '{n++; b+=$1} END{printf "%d psm_ blocks of %s, %.2f GB", n, ENVIRON["USER"], b/1e9}'); node $(df -h /dev/shm | awk 'NR==2{print $3" used of "$2}')"; }   # the Supply's boards (ours only; other users' blocks share the node): listed, never removed here
shm "at the start"
echo "========================================"
status=0
run() {                                     # name, command...: runs it, keeps the first failure's exit code
  local name=$1; shift
  echo ""; echo "======== $name ========"
  "$@"; local rc=$?
  [[ $rc -ne 0 && $status -eq 0 ]] && status=$rc
  echo "======== $name: exit $rc ========"
}
run "1 test_coae" python -u tests/TestCoAE/test_coae.py
COAE_SIM_OUT=$OUT run "1b test_sim_gpu" python -u tests/TestCoAE/test_sim_gpu.py
run "2 test_coae_slide" python -u tests/TestCoAE/test_coae_slide.py
shm "at the end (0 psm_ blocks of this job: nothing leaked)"
echo ""
if [ $status -eq 0 ]; then echo "======== ALL PASS ========"; else echo "======== FAILED (exit $status) ========"; fi
exit $status
