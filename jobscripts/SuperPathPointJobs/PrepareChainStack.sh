#!/bin/bash
#SBATCH --job-name=PrepareChainStack      # -> log/%x, result/%x/
#SBATCH --partition=normal                # Partition
#SBATCH --time=24:00:00                   # 12 slides in one job now (2026-09-06);
                                          # resumable -- a timeout only costs the
                                          # slides not yet reached, re-submit to
                                          # pick up where it left off
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                 # openslide reads, not a model
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

conda activate gigapath
source jobscripts/_env.sh

# =============================================================================
#  plan.md 2.1 -- ONE jobscript.
# =============================================================================
#
# `prepare_chain_stack.py` reads each axis's corpus for each slide; a corpus
# whose draw of the slide is not cached yet is drawn then (the mask cache +
# `TileSampler.cached`), and the pixels are read from the slide on demand.
#
# THREE CORPORA, THREE SHAPES:
#   F   stageB-fOwn   inherited chains (share=1.0)
#   R   stageA        no chains, full ladder -- the training corpus
#   C   stageB-cOwn   no chains, single rung
#
# LEAVE WSI_NAME UNSET FOR ALL 12 SLIDES IN ONE JOB. Set it for one slide
# only (ad hoc / debugging -- a failure raises straight through instead of
# being caught and summarised, see prepare_chain_stack.py's module
# docstring). Path/stem are resolved from the name via
# `utilities/AccessDatasets.py`, not typed here -- see that file for the
# registered names.
WSI_NAME="${WSI_NAME:-}"

TILE="${TILE:-256}"

# Shared by F (chain completeness + own draw) and R (output rungs).
# No ds 32 -- too few admissible positions per slide (spec.md 13: 583 across
# all 12) to be worth the corpus it costs.
RUNGS="${RUNGS:-1.0 2.0 4.0 8.0 16.0}"

# C's mother is always the coarsest of these (2026-09-05).
C_RUNGS="${C_RUNGS:-1.0 2.0 4.0 8.0 16.0}"

# Empty = build all three. Set e.g. AXES="F R" for a partial run.
AXES="${AXES:-F R C}"

# R/C's own local ChainStack cache (descendants/derived rungs). Defaults off
# -- RStack.from_own's docstring: degrade is cheap, not worth the disk IO at
# scale. Set for a small/demo run where re-generating the same few tiles
# repeatedly is worth not recomputing at all.
# CHAINSTACK_CACHE_JOB: whose tile cache (result/cache/<job>/slide=<s>/chainstack/) to use
# (default: this job's own). Was CACHE_ROOT, a bare path, until 2026-10-06.
CHAINSTACK_CACHE_JOB="${CHAINSTACK_CACHE_JOB:-}"

# result/cache/<DRAW_CACHE_JOB>/: where every corpus's draws are, side by side.
DRAW_CACHE_JOB="${DRAW_CACHE_JOB:-SuperPathPointCorpus}"

# Which cached masks a corpus is drawn through, when it has to be drawn:
# build_mask_store.py's recipe and job.
SEG="${SEG:-uni2_pca}"
MASK_CACHE_JOB="${MASK_CACHE_JOB:-BuildMaskStore}"

echo "======== PrepareChainStack ========"
echo "  slide  ${WSI_NAME:-<all 12, no WSI_NAME given>}"
echo "  axes   $AXES"
echo "  tile   $TILE   rungs ${RUNGS}   c-rungs ${C_RUNGS}"
echo "  draws  result/cache/${DRAW_CACHE_JOB}/"
echo ""

python training/SuperPathPoint/cli/prepare_chain_stack.py \
  ${WSI_NAME:+--wsi-name "$WSI_NAME"} \
  --tile "$TILE" \
  --rungs $RUNGS \
  --c-rungs $C_RUNGS \
  --draw-cache-job "$DRAW_CACHE_JOB" \
  --seg "$SEG" --mask-cache-job "$MASK_CACHE_JOB" \
  --axes $AXES \
  ${CHAINSTACK_CACHE_JOB:+--chainstack-cache-job "$CHAINSTACK_CACHE_JOB"}
status=$?

echo ""
echo "======== done (exit $status) ========"
exit $status
