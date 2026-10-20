#!/bin/bash
#SBATCH --job-name=MakeDataset                     # Job name (the cache is AccessDatasets' own, not this name)
#SBATCH --partition=dev                            # Partition
#SBATCH --time=01:00:00                            # the catalog: one header per slide, ~620 slides
#SBATCH --account=MST114560                        # Account
#SBATCH --nodes=1                                  # Number of nodes
#SBATCH --gpus-per-node=1                          # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                          # headers only, no pixel
#SBATCH --ntasks-per-node=1                        # Tasks per node
#SBATCH -o /work/u26130998/log/%x                  # STDOUT
#SBATCH -e /work/u26130998/log/%x                  # STDERR

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

conda activate locascope
source jobscripts/_env.sh

# =============================================================================
#  utilities/cli/build_cache/make_dataset.py -- the catalog and the virtual
#  datasets (`@<name>`), in AccessDatasets' own cache, result/cache/Datasets/,
#  whatever this job is called.
#
#    sbatch jobscripts/MakeDataset.sh                              # the catalog
#    ARGS="--catalog --datasets bracs/test --refresh" sbatch jobscripts/MakeDataset.sh
#    ARGS="--catalog --recipe pure" sbatch jobscripts/MakeDataset.sh   # the catalog, then a recipe's @-datasets
#    ARGS="--list" sbatch jobscripts/MakeDataset.sh                # what is written, checked
# =============================================================================

ARGS="${ARGS:---catalog}"
echo "======== make_dataset.py $ARGS ========"
python utilities/cli/build_cache/make_dataset.py $ARGS
