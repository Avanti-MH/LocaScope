#!/bin/bash
#SBATCH --job-name=TestAccessDatasets              # Job name
#SBATCH --partition=dev                         # Partition
#SBATCH --time=00:10:00                            # registry checks + a disk stat per file
#SBATCH --account=MST114560                        # Account
#SBATCH --nodes=1                                  # Number of nodes
#SBATCH --gpus-per-node=1                          # GPUs per node (不要設0)
#SBATCH --cpus-per-task=1                          # no model, no slide read
#SBATCH --ntasks-per-node=1                         # Tasks per node
#SBATCH -o /work/u26130998/log/%x  # STDOUT
#SBATCH -e /work/u26130998/log/%x  # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  utilities/AccessDatasets.py -- one hand-written naming-convention rule
#  per DATASET (`_DATASETS`), not one hand-written entry per WSI. Used by
#  more than one package's cli/, so its test lives flat under
#  utilities/test_modules/, not any one package's Test<Package>/.
# =============================================================================
#
# 2026-09-11: `list_names()`/`locate()` are themselves directory reads now,
# not lookups in a static table, so almost nothing here is "pure" any more
# -- the `disk` section's name marks what still matters, though: whether a
# test needs a SPECIFIC real name to exist (not portable off this cluster)
# or just needs `list_names()` to return something self-consistent. This is
# what catches a broken naming rule or a moved root in seconds instead of a
# downstream job reading the wrong tissue -- a directory rename has already
# broken this file's paths twice before (the Ki67 -> Ki67_with_photo
# rename, then organize_mrxs.py --rename-existing's container-naming
# migration).

echo "======== TestAccessDatasets ========"
python utilities/test_modules/test_access_datasets.py
status=$?

echo ""
echo "======== test_wsi_split (the split's one writer) ========"
python utilities/test_modules/test_wsi_split.py
rc=$?
[ $rc -ne 0 ] && status=$rc

echo ""
echo "======== done (exit $status) ========"
exit $status
