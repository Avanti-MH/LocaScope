#!/bin/bash
#SBATCH --job-name=AccuracyV1
#SBATCH --partition=normal2
#SBATCH --time=24:00:00
#SBATCH --account=MST114560
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH -o ./log/AccuracyV1
#SBATCH -e ./log/AccuracyV1

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Parameters ----------------
SVS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs
MRXS=/work/u26130998/datasets/Ki67/S1104043,G7E,110207.mrxs

TOTAL_PATCHES=4096
HEST_DS=4
TILE_SIZE=256
TISSUE_RATIO=0.5
SEED=42
BATCH_SIZE=128
TOME_R_SWEEP="0,1,2,3,4,8"

# --out-dir defaults to result/$SLURM_JOB_NAME (set by SLURM automatically)
# --tmp-dir defaults to log/tmp
# Override here only if you want a non-standard location.

# ---------------- Run ----------------
python utilities/test_modules/bench_gigapath_accuracy.py \
  --svs             "$SVS" \
  --mrxs            "$MRXS" \
  --total-patches   $TOTAL_PATCHES \
  --hest-ds         $HEST_DS \
  --tile-size       $TILE_SIZE \
  --tissue-ratio    $TISSUE_RATIO \
  --seed            $SEED \
  --batch-size      $BATCH_SIZE \
  --tome-r-sweep    "$TOME_R_SWEEP"

# Resume tiles (skip HEST + sampling if tiles already exist):
# python utilities/test_modules/bench_gigapath_accuracy.py --resume-tiles
