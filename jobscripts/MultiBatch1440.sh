#!/bin/bash
#SBATCH --job-name=MultiBatch1440     # -> log/<name> and result/<name>/
#SBATCH --partition=normal               # Partition
#SBATCH --time=24:00:00                  # Runtime (hh:mm:ss)
#SBATCH --account=MST114560              # Account
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus-per-node=1                # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                # CPU cores per task
#SBATCH --ntasks-per-node=1              # Tasks per node
#SBATCH -o /work/u26130998/log/MultiBatch1440       # STDOUT
#SBATCH -e /work/u26130998/log/MultiBatch1440       # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# Runs write outside the checkout; see utilities/_paths.py
RESULT_ROOT="${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/result"

# ---------------- One process per slide, written out ------------------------
#
# 45:32 at 1.47456 MP, the shape of the real photographs (1440x1024), built as
# seven python invocations rather than one process looping over seven slides.
#
# WHY ONE PROCESS PER SLIDE. It is no longer a workaround. Four runs died at
# exit 139 while this corpus was being built, and the cause turned out to have
# nothing to do with the process boundary: PYTHONFAULTHANDLER put the fault in
# cv2.connectedComponentsWithStats, inside _search_tissue_regions, which was
# being handed the whole mask in one call. It segfaults rather than raising
# somewhere above 2**33 pixels -- 8.40 Gpx survived, 12.34 Gpx did not -- and
# _search_tissue_regions now decimates until the input is safely under.
#
# Three hypotheses were wrong along the way, which is worth recording because
# each looked well supported: a cudnn tile-shape problem (the warning is
# printed at tile 1 and recovered from), accumulated process state (refuted
# when a fresh process died on the same slide), and tile width (the crash is
# after all the tiling, on the assembled mask).
#
# What survives is that a slide should not be able to take the run down with
# it. One process each means a crash costs one slide, any slide can be re-run
# on its own, and the failure is attributable at a glance.
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
# PYTHONFAULTHANDLER stays on. It is what turned three days of guessing into
# one stack trace, and locate_photo and bench_locascope have not been through
# a slide this large yet.
#
# THE SLIDES ARE THE ORIGINAL THREE. BRACS_1944 was only ever a stand-in for
# BRACS_1936 while 1936 was crashing; with the cause fixed it goes back out, so
# the BRACS side is one slide per type again -- ADH, FEA, DCIS -- and the same
# slides and seed as the 12 MP corpus, which is what makes FoV shape the only
# variable that moved between them.

export PYTHONFAULTHANDLER=1

BRACS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test
KI67=/work/u26130998/datasets/Ki67

# Tile budget for the segmentation forward pass, and for the read too since
# from_wsi takes the smaller of the two budgets. It was dropped to 2M while the
# segfault was thought to be about tile width; that turned out to be wrong, so
# nothing forces the smaller value now.
#
# It is kept at 2M anyway, for a different and better-supported reason: the
# budget changes the mask. Same slide, only the budget moved --
#
#   BRACS_1228 @ 4M   2048 tiles   tissue_frac 12.8%   regions 6
#   BRACS_1228 @ 2M   4096 tiles   tissue_frac 13.0%   regions 4
#
# -- so overlap=128 does not fully isolate HEST's receptive field, and the two
# corpora are not interchangeable. 2M is what the surviving slides of the last
# run used, and staying on it keeps this corpus internally consistent. Changing
# it is a decision about the mask, not about speed. The cost is real: the grid
# doubles and the overlap is a larger share of a smaller tile, so the read
# amplification goes from about 1.32x to 1.43x.
SEG_CHUNK_PX=2000000

# 45:32 with 1.47456 MP lands on 1440.000 x 1024.000 exactly, so nothing
# depends on how int() rounds in QueryFromWSI (query_sim/source/wsi_query.py).
# --read-chunk-px is not passed: its default of 256M already makes from_wsi
# read tile by tile, and the grid is min(read_chunk_px, seg_chunk_px), so the
# tiles are SEG_CHUNK_PX either way. It only earns a value of its own when there
# is no model to bound the grid -- an HSV mask leaves seg_chunk_px None, and then
# read_chunk_px is the only thing standing between ds=1 and the whole level
# in memory.
ARGS="--wh-ratio 45:32 --MPixels 1.47456 --per-camera 50 --jitter 0.05"
ARGS="$ARGS --mask-ds 1.0 --hest --seg-chunk-px $SEG_CHUNK_PX --seed 0 --append"

OUT="$RESULT_ROOT/$SLURM_JOB_NAME"
mkdir -p "$OUT"
rm -f "$OUT/gt.csv" "$OUT/skips.csv"

echo "======== corpus: 45:32  1.47456MP  (1440x1024) ========"
echo "seg-chunk-px: $SEG_CHUNK_PX"
echo "output     : $OUT"
echo ""

echo "################ 1/7  BRACS_1228 ################"
python query_sim/cli/multi_batch.py "$BRACS/Group_AT/Type_ADH/BRACS_1228.svs" $ARGS
echo "exit=$?"

echo "################ 2/7  BRACS_1936 ################"
python query_sim/cli/multi_batch.py "$BRACS/Group_AT/Type_FEA/BRACS_1936.svs" $ARGS
echo "exit=$?"

echo "################ 3/7  BRACS_1476 ################"
python query_sim/cli/multi_batch.py "$BRACS/Group_MT/Type_DCIS/BRACS_1476.svs" $ARGS
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
