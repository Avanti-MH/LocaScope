#!/bin/bash
#SBATCH --job-name=GigapathEncoderConfigExp   # Job name
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                   # Runtime (hh:mm:ss)
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x      # STDOUT
#SBATCH -e /work/u26130998/log/%x      # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# Replaces AccuracyV1.sh and the encoder-throughput-sweep task
# bench_gigapath_infer.py used to run (that call had gone dead, commented
# out, inside a jobscript later renamed LocatePhotoTimeBreakdown.sh and since
# deleted). Both python files merged into
# utilities/bench_modules/bench_gigapath_configs.py on 2026-09-17; EXP picks
# which of its two modes this jobscript runs.
#
#   EXP=accuracy   (default) bench_gigapath_configs.py --mode accuracy
#   EXP=speed      bench_gigapath_configs.py --mode speed --compare

EXP="${EXP:-accuracy}"

case "$EXP" in
  accuracy)
    SVS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs
    MRXS=/work/u26130998/datasets/Ki67_with_photo/S1104043_G7E_110207_mrxs/S1104043,G7E,110207.mrxs
    TOTAL_PATCHES="${TOTAL_PATCHES:-4096}"
    HEST_DS="${HEST_DS:-4}"
    TILE_SIZE="${TILE_SIZE:-256}"
    SEED="${SEED:-42}"
    BATCH_SIZE="${BATCH_SIZE:-128}"

    python utilities/bench_modules/bench_gigapath_configs.py \
      --mode accuracy \
      --svs             "$SVS" \
      --mrxs            "$MRXS" \
      --total-patches   "$TOTAL_PATCHES" \
      --seg hest --mask-ds "$HEST_DS" \
      --tile-size       "$TILE_SIZE" \
      --seed            "$SEED" \
      --batch-size      "$BATCH_SIZE"
    ;;

  speed)
    # --no-wsi: Part 2 (real-WSI comparison) skipped -- this runs only the
    # synthetic-patch sweep, same as bench_gigapath_infer.py's own dead call
    # this replaces.
    WARMUP="${WARMUP:-2}"
    COMPARE_PATCHES="${COMPARE_PATCHES:-40960}"
    COMPARE_BS="${COMPARE_BS:-8 16 64 128 512 1024 4096}"

    python utilities/bench_modules/bench_gigapath_configs.py \
      --mode speed \
      --compare \
      --no-wsi \
      --compare-patches  "$COMPARE_PATCHES" \
      --compare-bs       $COMPARE_BS \
      --warmup "$WARMUP"
    ;;

  *)
    echo "[abort] EXP must be accuracy or speed, got '$EXP'"
    exit 1
    ;;
esac
