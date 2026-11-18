#!/bin/bash
#SBATCH --job-name=BuildMaskStore         # -> log/%x[_<task>], result/%x/
#SBATCH --partition=8gpus
#SBATCH --time=12:00:00
#SBATCH --account=MST114560
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1                 # one card per part (do not set 0)
#SBATCH --cpus-per-task=12                # the H200 cap per GPU
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083,25a-hgpn062,25a-hgpn149
#SBATCH -o /work/u26130998/log/%x
#SBATCH -e /work/u26130998/log/%x

# =============================================================================
#  Fill the mask store: one tissue mask per slide, for the recipe you name.
# =============================================================================
#
# Moved out of SuperPathPointJobs/ (where it was the UNI2-PCA mask of twelve fixed slides, then a tile-yield probe):
# the recipe, its ds and the slides are the recipe block below now, nothing is fixed in the script.
#
# What is written, per slide:  result/cache/$CACHE_JOB/slide=<slide>/mask/mask_<seg_id>.safetensors + record_<seg_id>.json
# The mask is stored PACKED (packbits + zlib, SlideMask.save): a HEST mask is some 1 MB, not hundreds. A recipe or a ds
# that differs is a different seg_id and so a different file beside the others: nothing is overwritten. A slide that
# already has the mask is read back and skipped, so the job can be run again after a failure.
#
#   all registered slides, HEST at ds 4 (its default), as it stands:
#       sbatch deploy/slurm/BuildMaskStore.sh
#   other recipe, another ds, a dataset or a virtual dataset (a name from AccessDatasets):
#       SEG=hsv MASK_DS=4 DATASETS="Ki67_with_photo @pure-train" sbatch deploy/slurm/BuildMaskStore.sh
#   in parts (one job each; a part that died is run again alone):
#       sbatch --array=0-3 deploy/slurm/BuildMaskStore.sh        # PARTS is the array's size, set it alike: PARTS=4
#
# ---------------- the recipe ----------------
SEG="${SEG:-hest}"                 # none | hsv | hest | uni2_pca   (TissueMaskConfig.MASK_RECIPES)
MASK_DS="${MASK_DS-}"              # the segmentation ds of a plane recipe; "" = 4 for hest (its training scale: at ds 1 it calls a third of the tissue, ExpCheckDsTissue 2026-11-10) and 1 for hsv / none. Not for uni2_pca*
[ -z "$MASK_DS" ] && case "$SEG" in hest) MASK_DS=4 ;; uni2_pca*) ;; *) MASK_DS=1 ;; esac
DATASETS="${DATASETS:-}"           # AccessDatasets ids, registered or virtual (@name), space separated; "" = every registered dataset
NAMES="${NAMES:-}"                 # or slide names (a smoke: NAMES="S1100333,G7E,110105"); with these and no DATASETS, only these are made
CACHE_JOB="${CACHE_JOB:-Masks}"    # whose cache the masks go to: result/cache/<CACHE_JOB>/
PARTS="${PARTS:-1}"                # the array's size, with --array=0-(PARTS-1)
WORKERS="${WORKERS:-8}"            # uni2_pca: tile readers
HEST_DTYPE="${HEST_DTYPE:-fp16}"   # hest only: fp16 or fp32 (the recipe's own); a different seg_id, beside the fp32 masks
SEG_BATCH="${SEG_BATCH:-16}"       # plane recipes: tiles of one shape per forward pass (speed only)
READ_WORKERS="${READ_WORKERS:-4}"  # plane recipes: threads reading ahead of the GPU (speed only)
EXTRA="${EXTRA:-}"                 # more build_mask_store flags, e.g. --overwrite (remake a mask that is cached)
# --------------------------------------------

ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6
conda activate locascope
cd /work/u26130998/LocaScope || exit 2
source deploy/slurm/_env.sh        # HF_HOME; must be exported before python starts
export PYTHONUNBUFFERED=1

JOB="${SLURM_JOB_NAME:-BuildMaskStore}"
TASK="${SLURM_ARRAY_TASK_ID:-}"
# One job, one log: the job's first call starts it afresh; a later call in the same job (ExpUni2PcaMask runs one a pair) appends.
LOG="/work/u26130998/log/${JOB}${TASK:+_$TASK}"
grep -q "  job ${SLURM_JOB_ID:-none} " "$LOG" 2>/dev/null || : > "$LOG"
exec >> "$LOG" 2>&1

FLAGS=(--seg "$SEG" --cache-job "$CACHE_JOB")
if [ -n "$DATASETS" ]; then FLAGS+=(--datasets $DATASETS); elif [ -z "$NAMES" ]; then FLAGS+=(--all); fi
[ -n "$NAMES" ] && FLAGS+=(--wsi-names "$NAMES")
case "$SEG" in uni2_pca*) UNI=1 ;; *) UNI=0 ;; esac
[ -n "$MASK_DS" ] && [ $UNI = 0 ] && FLAGS+=(--mask-ds "$MASK_DS")
[ $UNI = 1 ] && FLAGS+=(--workers "$WORKERS")
[ "$SEG" = "hest" ] && [ -n "$HEST_DTYPE" ] && FLAGS+=(--hest-dtype "$HEST_DTYPE")
[ $UNI = 0 ] && FLAGS+=(--seg-batch "$SEG_BATCH" --read-workers "$READ_WORKERS")
OUT="/work/u26130998/result/$JOB"
if [ -n "$TASK" ]; then FLAGS+=(--part "$TASK" --parts "$PARTS"); OUT="$OUT/part$TASK"; fi
FLAGS+=(--out "$OUT")

echo "======== the recipe of this job ========"
echo "  job $SLURM_JOB_ID name $JOB task ${TASK:--}   seg $SEG   cache job $CACHE_JOB"
# only what this recipe takes: the plane knobs mean nothing to uni2_pca (it reads the slide itself), and the other way
case "$SEG" in
  uni2_pca*) echo "  $SEG: its recipe's ds and fit; tile readers $WORKERS" ;;
  hest)     echo "  mask ds ${MASK_DS:-recipe default}   hest dtype ${HEST_DTYPE:-recipe default}   batch $SEG_BATCH   read workers $READ_WORKERS" ;;
  *)        echo "  mask ds ${MASK_DS:-recipe default}   batch $SEG_BATCH   read workers $READ_WORKERS" ;;
esac
echo "  slides: ${DATASETS:-${NAMES:-every registered dataset}}   part ${TASK:--} of $PARTS"
echo "  build_mask_store ${FLAGS[*]} $EXTRA"
echo "========================================"
python -u -m locascope.cli.build_cache.build_mask_store "${FLAGS[@]}" $EXTRA
status=$?
echo "======== done (exit $status) ========"
echo "  masks -> result/cache/$CACHE_JOB/slide=<slide>/mask/     table -> $OUT/build_mask_store.csv"
exit $status
