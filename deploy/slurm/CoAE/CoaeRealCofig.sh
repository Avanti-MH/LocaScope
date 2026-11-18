#!/bin/bash
#SBATCH --job-name=CoaeRealCofig         # -> log/%x, result/%x/real_cofig.json
#SBATCH --partition=dev
#SBATCH --time=00:20:00
#SBATCH --account=MST114560
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12               # 12 per GPU, the H200 cap
#SBATCH --mem=200G                       # 200G per GPU, the H200 cap
#SBATCH --ntasks-per-node=1
#SBATCH -o /work/u26130998/log/%x
#SBATCH -e /work/u26130998/log/%x

# =============================================================================
#  THE RECIPE: the predicted Co-figure of the chosen checkpoint (A1's last epoch) against the one computed from the real validation
#  photo itself (locascope/training/CoAE/final.py cofig; Validate.real_cofig): a signal, cos / scaled MSE. Reads real photos: send it
#  after the RealTest arms. Reported beside the probe's `map` rows, not judged.
#    sbatch deploy/slurm/CoAE/CoaeRealCofig.sh
# =============================================================================
CKPT=/work/u26130998/result/CoaeTrain/large-mix-vref/weights/epoch2.pt
FOVS=/work/u26130998/result/CoaeTrain/val_fovs.csv
OUT=/work/u26130998/result/$SLURM_JOB_NAME
echo "======== the recipe of this job ========"
echo "  job $SLURM_JOB_ID name $SLURM_JOB_NAME on $(hostname) gpus $CUDA_VISIBLE_DEVICES"
echo "  checkpoint $CKPT"
echo "  photos     $FOVS"
echo "  out        $OUT (real_cofig.json)"
echo "========================================"
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate locascope
cd /work/u26130998/LocaScope || exit 2
source deploy/slurm/_env.sh
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python -u -m locascope.training.CoAE.final cofig --checkpoint "$CKPT" --fovs "$FOVS" --out "$OUT"
status=$?
echo ""
if [ $status -eq 0 ]; then echo "======== ALL DONE ========"; else echo "======== FAILED (exit $status) ========"; fi
exit $status
