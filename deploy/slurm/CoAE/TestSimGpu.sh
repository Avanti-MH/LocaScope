#!/bin/bash
#SBATCH --job-name=TestSimGpu             # -> log/%x, result/%x/
#SBATCH --partition=dev
#SBATCH --time=00:20:00
#SBATCH --account=MST114560
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083,25a-hgpn062,25a-hgpn149
#SBATCH -o /work/u26130998/log/%x
#SBATCH -e /work/u26130998/log/%x

# =============================================================================
#  The GPU simulation of CoAE's views against the CPU's (locascope/training/CoAE/GpuSims.py): tests/TestCoAE/test_sim_gpu.py on one GPU.
#  Every number line is also written to result/TestSimGpu/sim_gpu.txt. The last line of the log is ALL PASS or FAILED (exit N).
#    sbatch deploy/slurm/CoAE/TestSimGpu.sh
# =============================================================================
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate locascope
cd /work/u26130998/LocaScope || exit 2
source deploy/slurm/_env.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTHONWARNINGS="ignore:Plan failed with a cudnnException:UserWarning"   # cudnn's plan notices (as TestCoae.sh)

OUT=/work/u26130998/result/$SLURM_JOB_NAME
mkdir -p "$OUT"
echo "======== the recipe of this job ========"
echo "  job $SLURM_JOB_ID name $SLURM_JOB_NAME gpus $CUDA_VISIBLE_DEVICES -> $OUT"
echo "  cpus: nproc $(nproc), SLURM_CPUS_PER_TASK $SLURM_CPUS_PER_TASK"
echo "========================================"
COAE_SIM_OUT=$OUT python -u tests/TestCoAE/test_sim_gpu.py
status=$?
echo ""
if [ $status -eq 0 ]; then echo "======== ALL PASS ========"; else echo "======== FAILED (exit $status) ========"; fi
exit $status
