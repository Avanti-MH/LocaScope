#!/bin/bash
#SBATCH --job-name=BenchMark              # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=48:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/BenchMark                # STDOUT
#SBATCH -e ./log/BenchMark                # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Parameters ----------------
WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs
N_PATCHES=4096
WARMUP=2
REPEATS=3
BATCH_SIZES="8 16 32 64 128 256 512"
DTYPES="fp32 fp16"
LEVELS="0 1 2"
OVERLAPS="true false"

python utilities/test_modules/bench_gigapath_infer.py \
  --wsi $WSI \
  --n-patches $N_PATCHES \
  --warmup $WARMUP \
  --repeats $REPEATS \
  --batch-sizes $BATCH_SIZES \
  --dtypes $DTYPES \
  --levels $LEVELS \
  --overlaps $OVERLAPS
