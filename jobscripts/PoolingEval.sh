#!/bin/bash
#SBATCH --job-name=PoolingEval           # Job name -> log/<name>
#SBATCH --partition=normal2              # Partition
#SBATCH --time=01:00:00                  # minutes expected; slack for IO
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # GPUs per node
#SBATCH --cpus-per-task=8                # cosine matrices are the whole cost
#SBATCH --mem=64G                        # one (slide, level) pair at a time
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH -o ./log/PoolingEval             # STDOUT
#SBATCH -e ./log/PoolingEval             # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- Score existing stores; no GPU, no WSI, no model -------------
#
# Split out of PoolingBench.sh because the dump and the scoring answer to
# different resources. The dump needs a GPU and an hour; eval reads the stores
# that dump already wrote and does cosine matrices on the CPU, so asking for a
# card would idle it and queue longer for nothing.
#
# Use this whenever the QUESTION changes but the features do not: a new pooling
# added to POOLINGS, a different combine_slots, a re-cut of the delta bins.
# Re-dump only when the FEATURES change -- a different encoder, tile size, mask,
# or sampling.
#
# The pairing gate runs first for the same reason it does in PoolingBench.sh: a
# report written from stores that do not pair looks legitimate and is wrong.

OUT=result/cache/features
REPORT=result/cache/pooling_report.txt

echo "======== verifying answer indices ========"
python utilities/cli/inspect_feature_store.py "$OUT" --pairs
PAIR_RC=$?

if [ $PAIR_RC -ne 0 ]; then
  echo ""
  echo "!!!!!!!! pairing check FAILED -- skipping eval on purpose !!!!!!!!"
  exit $PAIR_RC
fi

echo ""
echo "======== eval ========"
python utilities/test_modules/bench_gigapath_pooling.py \
  --phase eval --out "$OUT" --report "$REPORT"

echo ""
echo "======== done ========"
echo "  report  $REPORT"
echo ""
echo "  Read the SUMMARY block at the end -- it aggregates across the 23"
echo "  (slide, level) combinations; the per-combination tables above it are for"
echo "  chasing a single case once the summary says which one."
echo ""
echo "  Two of those tables are new and answer questions the earlier run could"
echo "  not:"
echo "    by rotation   half the queries are turned 90 deg and there is no"
echo "                  rotation search here. Read the rot0 lead column: a"
echo "                  pooling that only leads at rot90 is rotation invariant,"
echo "                  not a better descriptor, and production already has"
echo "                  rotation search around the encoder."
echo "    whitening     closed form, no training, fitted on the reference pool"
echo "                  that build() already computes. It bounds from below what"
echo "                  a learned projection head could be worth."
