#!/bin/bash
#SBATCH --job-name=MultiBatch1440     # -> log/<name> and result/<name>/
#SBATCH --partition=normal               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                # CPU cores per task
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH -o ./log/MultiBatch1440       # STDOUT
#SBATCH -e ./log/MultiBatch1440       # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath

# ---------------- One process per slide, written out ------------------------
#
# Same corpus as MultiBatch1440.sh -- 45:32 at 1.47456 MP, the real
# photographs' 1440x1024 -- but seven python invocations instead of one process
# looping over seven slides.
#
# WHY. The single-process version segfaults, and the pattern points at
# accumulated process state rather than at any one slide:
#
#   run 1   BRACS_1228 ok, BRACS_1936 SIGSEGV                    (2nd slide)
#   run 2   BRACS_1228 ok, BRACS_1944 ok, BRACS_1476 SIGSEGV     (3rd slide)
#   diag    BRACS_1936 ALONE, read only    -> survived, 4096 tiles, exit 0
#   diag    BRACS_1936 ALONE, read + HEST  -> survived, 4096 tiles, exit 0
#
# The slide that killed run 1 completes cleanly on its own. Both crashes hit
# the first slide over about 12 Gpx, and neither was OOM: exit 139, no
# oom_kill, peak 46-60 GB against a 200 GB limit. So the trigger needs a large
# slide AND a process that has already done work. A fresh process per slide
# removes the second half.
#
# The calls are spelled out one per line rather than driven by a loop or an
# array so that a slide can be commented out, re-ordered or re-run by editing
# exactly one line, and so the log reads in the same order as the file.
#
# They run in sequence, so no two writers ever touch gt.csv and there is
# nothing to merge. Every call carries --append; the rm below is what defines
# where this corpus starts.
#
# A failing call does not stop the ones after it: there is no set -e, and
# gt.csv already holds every camera that finished before the crash.
#
# NOT A SUBSTITUTE FOR THE DIAGNOSIS. locate_photo runs a whole photo folder in
# one process and bench_locascope runs a whole corpus in one process, so the
# same accumulation will surface there and be harder to see. PYTHONFAULTHANDLER
# stays on; if a call still dies, the log carries the Python stack.
#
# BRACS_1936 IS BACK IN. It was swapped for BRACS_1944 in the single-process
# script because it crashed there; here it gets its own process, which is the
# condition under which it has twice been shown to complete. BRACS_1944 stays,
# so this corpus is three BRACS and four Ki67.

export PYTHONFAULTHANDLER=1

BRACS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test
KI67=/work/u26130998/datasets/Ki67

# Tile budget for the segmentation forward pass and, via --read-max-pixels, for
# the read as well. Lowered from the 4000000 the single-process script uses,
# because the crashes correlate with tile width once overlap is added:
#
#   BRACS_1228  1912x1680 -> expanded 2168x1936   ok
#   BRACS_1944  1080x1898 -> expanded 1336x2154   ok
#   BRACS_1936  1383x2178 -> expanded 1639x2434   crashed in batch, ok alone
#   BRACS_1476  1381x2458 -> expanded 1637x2714   crashed
#
# 2M halves the longer side once more, putting every expanded tile back under
# the widths that have completed.
#
# CONFOUND, on purpose: this changes two things at once, the process boundary
# and the tile size. If all seven survive it will not say which one mattered.
# Set this back to 4000000 to test the process boundary on its own -- that is
# the cleaner experiment, and the one to run if the goal is understanding the
# segfault rather than getting the corpus out.
MAX_PIXELS=2000000

# 45:32 with 1.47456 MP lands on 1440.000 x 1024.000 exactly, so nothing
# depends on how int() rounds in QueryFromWSI (query_sim/source/wsi_query.py).
# --read-max-pixels is not passed: its default of 256M already makes from_wsi
# read tile by tile, and the grid is min(read_max_pixels, max_pixels), so the
# tiles are MAX_PIXELS either way. It only earns a value of its own when there
# is no model to bound the grid -- an HSV mask leaves max_pixels None, and then
# read_max_pixels is the only thing standing between ds=1 and the whole level
# in memory.
ARGS="--wh-ratio 45:32 --MPixels 1.47456 --per-camera 100 --jitter 0.05"
ARGS="$ARGS --mask-ds 1.0 --hest --max-pixels $MAX_PIXELS --seed 0 --append"

OUT="result/$SLURM_JOB_NAME"
mkdir -p "$OUT"
rm -f "$OUT/gt.csv" "$OUT/skips.csv"

echo "======== corpus: 45:32  1.47456MP  (1440x1024) ========"
echo "max-pixels : $MAX_PIXELS"
echo "output     : $OUT"
echo ""

echo "################ 1/7  BRACS_1228 ################"
python query_sim/cli/multi_batch.py "$BRACS/Group_AT/Type_ADH/BRACS_1228.svs" $ARGS
echo "exit=$?"

echo "################ 2/7  BRACS_1936 ################"
python query_sim/cli/multi_batch.py "$BRACS/Group_AT/Type_FEA/BRACS_1936.svs" $ARGS
echo "exit=$?"

echo "################ 3/7  BRACS_1944 ################"
python query_sim/cli/multi_batch.py "$BRACS/Group_AT/Type_FEA/BRACS_1944.svs" $ARGS
echo "exit=$?"

echo "################ 4/7  S1104233 ################"
python query_sim/cli/multi_batch.py "$KI67/S1104233,G7E,110208.mrxs" $ARGS
echo "exit=$?"

echo "################ 5/7  S1104360 ################"
python query_sim/cli/multi_batch.py "$KI67/S1104360,G7E,110208.mrxs" $ARGS
echo "exit=$?"

echo "################ 6/7  S1137178 ################"
python query_sim/cli/multi_batch.py "$KI67/S1137178,G7E,110926.mrxs" $ARGS
echo "exit=$?"

echo "################ 7/7  S1151088 ################"
python query_sim/cli/multi_batch.py "$KI67/S1151088,G7E,111220.mrxs" $ARGS
echo "exit=$?"

echo ""
echo "======== done -> $OUT/ ========"
echo "Any slide that died shows a non-zero exit above; 139 = SIGSEGV,"
echo "137 = OOM kill. The others still ran, and gt.csv holds every camera"
echo "that finished. Re-run one slide by running its line on its own."
echo ""
echo "Check the shape actually landed before benching on it:"
echo "  python -c \"import csv,collections; print(collections.Counter((r['fov_width'],r['fov_height']) for r in csv.DictReader(open('$OUT/gt.csv'))))\""
echo "  wc -l $OUT/gt.csv"
