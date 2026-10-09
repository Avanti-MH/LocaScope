#!/bin/bash
#SBATCH --job-name=RUN # Job name
#SBATCH --partition=8gpus                 # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/RUN # STDOUT
#SBATCH -e /work/u26130998/log/RUN # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
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
#   /work/u26130998/datasets/Ki67_with_photo/S1151088_G7E_111220_mrxs/S1151088,G7E,111220.mrxs \
#   /work/u26130998/datasets/Ki67_with_photo/S1104233_G7E_110208_mrxs/S1104233,G7E,110208.mrxs \
#   /work/u26130998/datasets/Ki67_with_photo/S1104360_G7E_110208_mrxs/S1104360,G7E,110208.mrxs \
#   /work/u26130998/datasets/Ki67_with_photo/S1137178_G7E_110926_mrxs/S1137178,G7E,110926.mrxs \
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


# cd /work/u26130998/LocaScope
# python utilities/test_modules/test_EoMT.py \
#     --tile-figure /work/u26130998/prov-gigapath/images/01581x_25327y.png \
#                   /work/u26130998/prov-gigapath/images/01581x_25583y.png \
#     --wsi /work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1003691.svs \
#           /work/u26130998/datasets/Ki67_with_photo/S1103520_G7E_110126_mrxs/S1103520,G7E,110126.mrxs

# python aiNNModel/models/common/migrate_checkpoint.py \
#   /work/u26130998/result/MppRoutingHead/weights/*mlp*.pt  --write

# python utilities/test_modules/test_cache.py
# python utilities/test_modules/test_tissue_mask.py
# python utilities/test_modules/test_tile_sampler.py
# python utilities/test_modules/test_uni2_pca_seg.py
# python utilities/test_modules/test_patching_lib.py
# python utilities/cli/build_cache/make_split.py --cache-job MakeSplit
# python utilities/test_modules/test_store.py
# python utilities/test_modules/TestSuperPathPoint/test_chain_stack.py
# python utilities/test_modules/TestSuperPathPoint/test_superpathpoint.py
# sbatch jobscripts/SuperPathPointJobs/TestSuperPathPoint.sh
# # once those pass, a small real extraction (it prints the corpus key at the top of the log):
# N=20 DS="1 2 4 8 16" WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_BT/Type_N/BRACS_1598.svs CACHE_JOB=ExtractPreTilesSmoke sbatch jobscripts/SuperPathPointJobs/ExtractPreTiles.sh
# sbatch jobscripts/PatchingLibTest.sh
# sbatch jobscripts/TissueMaskTest.sh