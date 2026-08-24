#!/bin/bash
#SBATCH --job-name=EoMTest # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/EoMTest # STDOUT
#SBATCH -e /work/u26130998/log/EoMTest # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# Runs write outside the checkout; see utilities/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# CORPUS=result/MultiBatch
# BENCH=result/BenchLocaScope

# # ---------------- Step 1: generate the shot corpus ----------------
# echo "======== [1/2] multi_batch: generate synthetic shots ========"
# python query_sim/cli/multi_batch.py \
#   /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs \
#   /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_FEA/BRACS_1936.svs \
#   /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_MT/Type_DCIS/BRACS_1476.svs \
#   /work/u26130998/datasets/Ki67/S1151088,G7E,111220.mrxs \
#   /work/u26130998/datasets/Ki67/S1104233,G7E,110208.mrxs \
#   /work/u26130998/datasets/Ki67/S1104360,G7E,110208.mrxs \
#   /work/u26130998/datasets/Ki67/S1137178,G7E,110926.mrxs \
#   --per-camera 20 --jitter 0.05 \
#   --out $CORPUS

# # ---------------- Step 2: run the 3-stage pipeline over the whole corpus ----------------
# echo ""
# echo "======== [2/2] bench_locascope: 3-stage pipeline + metrics + plots ========"
# python utilities/bench_modules/bench_locascope.py \
#   --gt-csv     $CORPUS/gt.csv \
#   --images-dir $CORPUS/images \
#   --out        $BENCH \
#   --precision fp16 --batch-size 1024 \
#   --draw-figures -1


cd /work/u26130998/LocaScope
python utilities/test_modules/test_EoMT.py \
    --tile-figure /work/u26130998/prov-gigapath/images/01581x_25327y.png \
                  /work/u26130998/prov-gigapath/images/01581x_25583y.png \
    --wsi /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1003691.svs \
          /work/u26130998/datasets/Ki67/S1103520,G7E,110126.mrxs