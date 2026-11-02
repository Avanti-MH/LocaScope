#!/bin/bash
#SBATCH --job-name=TileRetrievalBench    # Job name -> log/<name> and result/<name>/
#SBATCH --partition=8gpus                # Partition
#SBATCH --time=08:00:00                  # ~50 min expected; slack for MRXS masks
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # ONE on purpose -- see note below
#SBATCH --cpus-per-task=8                # openslide reads + the CPU transform
#SBATCH --mem=128G                       # per-tile reads, not whole regions
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH -o /work/u26130998/log/%x      # STDOUT
#SBATCH -e /work/u26130998/log/%x      # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# ---------------- Tile-level retrieval: which pooling finds the tile ---------
#
# Every store is a cache entry in CACHE_JOB's tree (bench_tile_retrieval's
# docstring has the addresses): per (slide, level) a reference draw with its
# tokens, and the queries' render with the query tiles and the grid tiles they
# answer to. A rerun encodes only what is missing.
#
#   PHASE=dump   reads the slides, encodes, writes the stores (GPU)
#   PHASE=eval   reads the stores only -- no model; sbatch it alone any time
#   PHASE=all    dump, then eval  (default)

PHASE="${PHASE:-all}"
CACHE_JOB="${CACHE_JOB:-TileRetrievalBench}"
ENCODER="${ENCODER:-gigapath}"
HEAD="${HEAD:-}"          # conch_vit needs HEAD=trunk: this bench calls tokens()
K="${K:-2000}"            # reference tiles at L0; max(K/ds^2, K_FLOOR) per level
K_FLOOR="${K_FLOOR:-2000}"
QUERIES="${QUERIES:-500}" # query positions per (slide, level), one tile each
SEED="${SEED:-0}"
DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
N_WSI="${N_WSI:-5}"
ONLY_WSI="${ONLY_WSI:-}"
ONLY_LEVELS="${ONLY_LEVELS:-}"

ARGS=(--encoder "$ENCODER" --cache-job "$CACHE_JOB" -k "$K" --k-floor "$K_FLOOR"
      --queries "$QUERIES" --seg hest --seed "$SEED"
      --datasets $DATASETS --n-wsi "$N_WSI")
[ -n "$HEAD" ] && ARGS+=(--head "$HEAD")
[ -n "$ONLY_WSI" ] && ARGS+=(--wsi "$ONLY_WSI")
[ -n "$ONLY_LEVELS" ] && ARGS+=(--levels $ONLY_LEVELS)

if [ "$PHASE" = "dump" ] || [ "$PHASE" = "all" ]; then
  echo "======== dump  k=$K  queries=$QUERIES per (slide, level) ========"
  python -u utilities/bench_modules/bench_tile_retrieval.py --phase dump "${ARGS[@]}"
  rc=$?
  [ $rc -ne 0 ] && { echo "======== dump failed (exit $rc) ========"; exit $rc; }
fi

if [ "$PHASE" = "eval" ] || [ "$PHASE" = "all" ]; then
  echo ""
  echo "======== eval (no model) ========"
  python utilities/bench_modules/bench_tile_retrieval.py --phase eval "${ARGS[@]}" || exit $?
fi

echo ""
echo "======== done -> result/${SLURM_JOB_NAME:-TileRetrievalBench}/$ENCODER${HEAD:+_$HEAD}/reference_report.txt ========"
echo "  Read it for CONSISTENCY across the (slide, level) combinations, not for a"
echo "  winner in any one of them. A pooling that leads on one slide and not the"
echo "  next has told you nothing -- that is how classify_region died (M4.2)."
