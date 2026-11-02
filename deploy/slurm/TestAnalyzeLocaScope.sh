#!/bin/bash
#SBATCH --job-name=TestAnalyzeLocaScope             # -> log/%x
#SBATCH --partition=dev                   # Partition
#SBATCH --time=00:30:00
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=4                 # CPU cores per task
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh

# =============================================================================
#  The tests of the analyses: analyze_locascope (how a stage-2 candidate list is
#  judged, and that its tables, report and figures run) and the figures of
#  analyze_stage1_metrics on a table of made-up stage-1 rows. No slide, no cache,
#  no model.
#
#    sbatch jobscripts/TestAnalyzeLocaScope.sh
# =============================================================================
python utilities/test_modules/test_analyze_locascope.py
status=$?

echo ""
# the last line says it all: ALL PASS, or FAILED with the exit code
if [ $status -eq 0 ]; then echo "======== ALL PASS ========"; else echo "======== FAILED (exit $status) ========"; fi
exit $status
