#!/bin/bash
#SBATCH --job-name=BenchMarkV2              # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=48:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o ./log/BenchMarkV2                # STDOUT
#SBATCH -e ./log/BenchMarkV2                # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Parameters ----------------
WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs
WARMUP=2
REPEATS=3

# comparison mode
COMPARE_PATCHES=40960
COMPARE_BS="8 16 64 128 512 1024 4096"
WSI_COMPARE_BS="32 128 512"
WSI_COMPARE_LEVEL=0
WSI_COMPARE_OVERLAP=false

# standard mode
N_PATCHES=4096
BATCH_SIZES="8 16 32 64 128 256 512"
DTYPES="fp32 fp16"
LEVELS="0 1 2"
OVERLAPS="true false"

# ---------------- Run ----------------

# comparison mode: 7 configs × batch sweep (Part 1) + WSI comparison (Part 2)
# python utilities/test_modules/bench_gigapath_infer.py \
#   --wsi              $WSI \
#   --compare \
#   --compare-patches  $COMPARE_PATCHES \
#   --compare-bs       $COMPARE_BS \
#   --wsi-compare-bs   $WSI_COMPARE_BS \
#   --wsi-compare-level   $WSI_COMPARE_LEVEL \
#   --wsi-compare-overlap $WSI_COMPARE_OVERLAP \
#   --warmup $WARMUP

# comparison mode: Part 1 only, no-wsi
python utilities/test_modules/bench_gigapath_infer.py \
  --compare \
  --no-wsi \
  --compare-patches  $COMPARE_PATCHES \
  --compare-bs       $COMPARE_BS \
  --warmup $WARMUP

# standard mode: single model config, detailed level × overlap × batch × dtype sweep
# python utilities/test_modules/bench_gigapath_infer.py \
#   --wsi         $WSI \
#   --n-patches   $N_PATCHES \
#   --warmup      $WARMUP \
#   --repeats     $REPEATS \
#   --batch-sizes $BATCH_SIZES \
#   --dtypes      $DTYPES \
#   --levels      $LEVELS \
#   --overlaps    $OVERLAPS