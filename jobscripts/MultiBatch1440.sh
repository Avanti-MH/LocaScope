#!/bin/bash
#SBATCH --job-name=MultiBatch1440        # Job name -> log/<name> and result/<name>/
#SBATCH --partition=normal               # Partition
#SBATCH --time=12:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                # CPU cores per task
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH --mem=700G                       # see the note on ds=1 memory below
#SBATCH -o ./log/MultiBatch1440          # STDOUT
#SBATCH -e ./log/MultiBatch1440          # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- A synthetic corpus shaped like the real photographs --------
#
# The real microscope photos are 1440x1024 32-bit BMP: 45:32, 1.475 MP. The
# query_sim default is 4:3 at 12 MP, and result/MultiBatch/ was generated with
# it -- every one of its 300 shots is 4000x3000. Tuning against that corpus is
# tuning against a materially easier problem than the one the photographs pose.
#
# QueryFromWSI derives the output size as
#     factor = (MPixels * 1e6 / (w_r * h_r)) ** 0.5
#     w = int(factor * w_r);  h = int(factor * h_r)
# (query_sim/source/wsi_query.py:50). 45:32 with 1.47456 MP lands on 1440.000 x
# 1024.000 exactly, so nothing depends on how int() rounds. 1.475 also yields
# 1440x1024, but by truncating 1440.215 -- prefer the exact value.
#
# WHAT THIS COSTS THE RETRIEVER. Area drops 8x, and the query tile grid with it:
# 15x11 = 165 tiles becomes 5x4 = 20. A window's score is the mean cosine over
# those tiles, so the same signal is now averaged over 8x fewer samples. The
# correct-vs-random separation measured on the real photos was already only
# 0.5687 vs 0.5408; expect the numbers here to get worse, and expect them NOT to
# be comparable with anything measured on result/MultiBatch/. That is the point
# of the run, not a defect in it.
#
# Same slides and same seed as the 12 MP corpus, so FoV shape is the only
# variable that moved. Both datasets on purpose: BRACS SVS steps 4x per pyramid
# level and Ki67 MRXS steps 2x, and the 2x spacing is what makes stage 1 confuse
# adjacent levels. A corpus of one or the other would hide that.
#
# MASK: --hest at --mask-ds 1.0, tiled at 4M pixels per forward pass. ds=1 is
# the setting that makes the tissue mask agree with the level-0 geometry the
# sampler works in, and HEST is the segmentation we are moving to. The mask is
# built once per WSI now, so that cost is paid level_count times less than it
# used to be.
#
# --mem=700G, and it is not padding. from_wsi at ds=1 materialises the whole
# level as one array, so memory scales with pixel count and nothing bounds it
# (--max-pixels bounds the GPU, not the host). Two runs measured the host cost
# at 16 bytes per level-0 pixel:
#
#   BRACS_1228   6.58 Gpx   105 GB   <- measured, the only slide that finished
#   BRACS_1936  12.34 Gpx   197 GB   <- OOM killed here at 200G, twice
#   S1104360    18.70 Gpx   299 GB   <- largest of the seven
#
# The MRXS canvases are 80 Gpx each; the numbers above are after limit_bounds
# crops to the scanned rectangle, which is the only reason they are affordable
# at all. Roughly 25 GB per slide also failed to come back in the first run, so
# 299 + 6 * 25 is the figure 700G is covering.

BRACS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test
KI67=/work/u26130998/datasets/Ki67

WSIS=(
  "$BRACS/Group_AT/Type_ADH/BRACS_1228.svs"
  "$BRACS/Group_AT/Type_FEA/BRACS_1936.svs"
  "$BRACS/Group_MT/Type_DCIS/BRACS_1476.svs"
  "$KI67/S1104233,G7E,110208.mrxs"
  "$KI67/S1104360,G7E,110208.mrxs"
  "$KI67/S1137178,G7E,110926.mrxs"
  "$KI67/S1151088,G7E,111220.mrxs"
)

WH_RATIO=45:32        # real photos are 1440x1024
MPIXELS=1.47456       # exactly 1440.000 x 1024.000, see the note above
PER_CAMERA=100         # shots per (WSI, pyramid level)
JITTER=0.05
SEED=0

echo "======== corpus: $WH_RATIO  ${MPIXELS}MP  (1440x1024) ========"
printf '%s\n' "${WSIS[@]}"
echo

# --out is omitted on purpose: multi_batch falls back to
# result/<SLURM_JOB_NAME>/, which keeps this run beside its own log and lets
# `make clean-job JOB=MultiBatch1440` remove the pair.
python query_sim/cli/multi_batch.py \
  "${WSIS[@]}" \
  --wh-ratio   $WH_RATIO \
  --MPixels    $MPIXELS \
  --per-camera $PER_CAMERA \
  --jitter     $JITTER \
  --mask-ds    1.0 \
  --hest \
  --max-pixels 4000000 \
  --seed       $SEED
RC=$?

# The exit code has to be looked at. An OOM kill leaves the shell running, so
# without this the script printed "done" and a command to check a gt.csv that
# did not exist -- the tail of the log claimed success for a job that died on
# its second slide.
echo ""
if [ $RC -ne 0 ]; then
  echo "======== FAILED (exit $RC) -> result/MultiBatch1440/ ========"
  echo "gt.csv holds whatever finished before the failure; it is written per"
  echo "camera, so the images already on disk are still usable. Count them:"
  echo "  wc -l result/MultiBatch1440/gt.csv"
  echo "For an OOM, the [mem ...] lines give the high-water mark to set --mem from."
  exit $RC
fi

echo "======== done -> result/MultiBatch1440/ ========"
echo "Check the shape actually landed before benching on it:"
echo "  python -c \"import csv,collections; print(collections.Counter((r['fov_width'],r['fov_height']) for r in csv.DictReader(open('result/MultiBatch1440/gt.csv'))))\""
