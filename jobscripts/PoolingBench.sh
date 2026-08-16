#!/bin/bash
#SBATCH --job-name=PoolingBench          # Job name -> log/<name> and result/<name>/
#SBATCH --partition=normal2              # Partition
#SBATCH --time=08:00:00                  # ~50 min expected; slack for MRXS masks
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # ONE on purpose -- see note below
#SBATCH --cpus-per-task=8                # openslide reads + the CPU transform
#SBATCH --mem=128G                       # per-tile reads, not whole regions
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH -o /work/u26130998/log/PoolingBench            # STDOUT
#SBATCH -e /work/u26130998/log/PoolingBench            # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath


# Runs write outside the checkout; see utilities/test_modules/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# ---------------- Does a different pooling find what CLS misses? -------------
#
# GigaPath computes 197 tokens per tile and keeps one: timm pools with
# global_pool='token', which is x[:, 0]. Retrieval's largest failure bucket is
# "the truth was never proposed" (32.3% of 1398 shots), which is a statement
# about the descriptor, so this asks whether keeping some of the other 196
# helps. It touches neither retrieval nor the pipeline -- it is a tile-level
# retrieval task built from stores.
#
# Design and the reasoning behind each choice: log/TODO.log, the 2026-08-08
# entry "197 個 token 只用了 1 個".
#
# ONE GPU is deliberate. gigapath_encode_tokens calls forward_features on the
# unwrapped module, so DataParallel would be bypassed anyway, and the whole run
# is under 100k tiles -- minutes on a single card. Asking for four would queue
# longer for no gain.
#
# MEMORY: 128G, not the 600G BenchLocaScope needs. That job reads a mask_all
# region whole (18.7 Gpx at L0, RSS 339 GB observed); this one computes grid
# coordinates from PatchGrid.from_size, which touches no pixels, and then reads
# 256x256 tiles individually.

OUT="$RESULT_ROOT"/cache/features
REPORT="$RESULT_ROOT"/cache/pooling_report.txt

K=2000                # reference tiles per (slide, level)
K_FLOOR=2000          # floor for coarse levels; they take min(available, this)
QUERIES=1000          # query tiles per (slide, level) -- 20 per FoV, 2 rotations
MASK_DS=4             # segmentation resolution
SEED=0

# Empty runs every slide in the gt csv. Set to a substring for one slide.
ONLY_WSI=""
# Empty runs every level present. Set e.g. "0 1" to restrict.
ONLY_LEVELS=""

# ---------------- background quota on the distractor pool --------------------
#
# The reference pool used to be a uniform draw over the retrieval grid, which
# sounds neutral and is not. On BRACS_1228 at level 0 the median grid position
# is 72% background and 46% of them are pure background (result/RefStore). Those
# distractors can never outrank an answer, so a nominal pool of 3000 was an
# effective pool of roughly half that -- and the share differs per level, so the
# per-level numbers were comparing descriptor difficulty and pool composition at
# the same time.
#
# QUOTA_FLOOR_LT15 is the main knob: the least of the pool that must be
# tissue-dense. Raising it makes the distractors harder, lowering it makes them
# emptier.
#
# It does NOT reach the old uniform draw. At 0 the allocation gives 10% to the
# middle band and pours the remaining 90% into the background buckets in order,
# which is emptier than uniform, not equal to it -- a uniform draw is
# proportional to what the grid happens to offer, and the quota shape in
# plan_level cannot express "proportional". Reproducing the old composition
# would need a mode that does not exist yet.
#
# The pool SIZE is unchanged -- the bench keeps its own k / ds**2 rule and feeds
# it to the sampler as the target. Only the composition moves.
QUOTA_FLOOR_LT15=0.85
QUOTA_JITTER_CAP=0.20

echo "======== pooling dump  k=$K  queries=$QUERIES per (slide, level) ========"
echo "out=$OUT"
echo

WSI_FLAG=""
[ -n "$ONLY_WSI" ] && WSI_FLAG="--wsi $ONLY_WSI"
LEVEL_FLAG=""
[ -n "$ONLY_LEVELS" ] && LEVEL_FLAG="--levels $ONLY_LEVELS"

python utilities/test_modules/bench_gigapath_pooling.py \
  --phase dump \
  --out "$OUT" \
  -k $K --k-floor $K_FLOOR --queries $QUERIES \
  --mask-ds $MASK_DS --seed $SEED \
  --quota-floor-lt15 $QUOTA_FLOOR_LT15 \
  --quota-jitter-cap $QUOTA_JITTER_CAP \
  $WSI_FLAG $LEVEL_FLAG
DUMP_RC=$?

if [ $DUMP_RC -ne 0 ]; then
  echo ""
  echo "======== dump failed (exit $DUMP_RC) -- not evaluating ========"
  exit $DUMP_RC
fi

# Gate, not decoration. If a query's answer indices do not land in the paired
# reference store, every question is unanswerable, every pooling scores the same
# nothing, and the report reads "no pooling improves recall" -- a finding rather
# than a bug. Cheap enough to always run.
echo ""
echo "======== verifying answer indices ========"
python utilities/cli/inspect_feature_store.py "$OUT" --pairs
PAIR_RC=$?

if [ $PAIR_RC -ne 0 ]; then
  echo ""
  echo "!!!!!!!! pairing check FAILED -- skipping eval on purpose !!!!!!!!"
  echo "  A report written from stores that do not pair would look legitimate"
  echo "  and be wrong. Read the output above, fix, and rerun."
  exit $PAIR_RC
fi

echo ""
echo "======== eval (no GPU; rerun on a login node any time) ========"
python utilities/test_modules/bench_gigapath_pooling.py \
  --phase eval --out "$OUT" --report "$REPORT"

echo ""
echo "======== done ========"
echo "  stores  $OUT/"
echo "  report  $REPORT"
echo ""
echo "  Read it for CONSISTENCY across the 25 (slide, level) combinations, not"
echo "  for a winner in any one of them. A pooling that leads on one slide and"
echo "  not the next has told you nothing -- that is how classify_region died"
echo "  (see the M4.2 entry in log/TODO.log)."
echo ""
echo "  Re-eval without re-dumping:"
echo "    python utilities/test_modules/bench_gigapath_pooling.py --phase eval"
echo "  Delta histograms:"
echo "    python utilities/cli/inspect_feature_store.py --pairs --hist"

# ---------------- reading a root that holds more than one sampling rule -------
#
# sampler_id is in cfg_hash, so a quota dump and the older uniform dump land on
# DIFFERENT filenames and neither overwrites the other. They coexist, which is
# the point: the comparison needs both.
#
#   --phase eval    needs no selector. It keys stores by cfg_hash and pairs a
#                   query store only with the reference of the same hash, so
#                   each batch is scored on its own and both appear in one
#                   report. The header line of every combination now prints the
#                   sampler, 'uniform (pre-quota)' for the old ones.
#
#   bench_mpp_estimate  DOES need one, because it looks a store up by
#                   (slide, level, pooling) and would find two:
#                       --sampler-id ''        the pre-quota uniform draws
#                       --sampler-id <hash>    one quota config
#                   Omitting it raises and names both files rather than picking
#                   one silently. The hash is printed by the dump, on the line
#                   under each level.
