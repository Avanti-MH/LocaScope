#!/bin/bash
#SBATCH --job-name=MakeHaLabels           # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=24:00:00                   # UNKNOWN until STAGE=measure has run
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --ntasks-per-node=1               # Tasks per node
#SBATCH -o /work/u26130998/log/%x         # STDOUT, named by --job-name
#SBATCH -e /work/u26130998/log/%x         # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/24.11.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate gigapath
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts


# =============================================================================
#  spec.md 12 steps 4 and 5: Homographic Adaptation, then the decision point
# =============================================================================
#
# FOUR STAGES, AND THE ONLY ONE THAT NEEDS NO PRE-TILE STORE IS `demo`.
#
#   demo      one pre-tile cut straight off the slide: three single views with
#             their own keypoints beside the HA aggregate at three thresholds,
#             one aggregate panel per DEMO_NUM. Minutes. Reads no store, so it
#             is the only stage that runs while step 3c is still going.
#   measure   one slide, one rung, 100 tiles at N=100. Ten minutes. Prints
#             seconds per tile and extrapolates the three models' GPU-hours.
#   labels    every rung of every slide. Hours to days -- and how many is
#             exactly what `measure` is for.
#   inspect   the decision point of spec.md 12 step 5: are these labels worth
#             training on, or is the answer a different teacher?
#
#   STAGE=demo sbatch jobscripts/MakeHaLabels.sh        # runs today
#   STAGE=measure sbatch jobscripts/MakeHaLabels.sh     # once 3c has finished
#   sbatch jobscripts/MakeHaLabels.sh                   # all four
#
# `demo` and `inspect` answer overlapping questions and are not the same tool.
# `demo` shows ONE tile in full and needs nothing but the slide; `inspect` is
# the gate, reads the finished store, and scores two independent HA runs against
# a shifted decoy. Look at the demo first because it costs minutes and can
# already say "the teacher finds nothing on H&E".
#
# WHY THE MEASUREMENT COMES FIRST. spec.md 13 lists the HA wall clock as
# undecided, and what it decides is whether Stage A is one job or a project:
# whether a second round (R > 1) is affordable, and whether model_512 and
# model_1024 are worth opening at all. The cost is
# `tiles x N x forward(tile_size)` and the forward goes with tile AREA, so ONE
# measurement extrapolates to all three -- linear in N and in tile count,
# quadratic in tile size. The --time above is a guess until it has run.
#
# WHAT THE DECISION POINT CAN CONCLUDE. Not just "continue". The teacher is
# COCO-trained on natural images and may simply not work on H&E, and the
# alternatives -- SIFT or Harris as the teacher, or doing upstream's stage 1
# after all -- are choosable only once this has produced the information. The
# four outcomes are in the header of cli/inspect_ha_labels.py.
#
# The one that needs a model is outcome D: two independent HA runs of the same
# tile agreeing no better than a shifted decoy, i.e. the points are view-specific
# noise that N=100 did not average away. That is why `inspect` runs
# --with-model.

STAGE="${STAGE:-demo measure labels inspect}"

# Upstream's N (`magic-point_coco_export.yaml:12`). The cost is strictly linear
# in it, and spec.md 13 says how to choose it properly when the time comes: run
# once at N=200, recompute the aggregate from the first k, plot the overlap
# against k, take the knee. One execution, not five.
NUM=100

TILE=256

# THE number that decides what a label IS. It is where the 100-view aggregate
# sits between "any view found it" and "every view found it", and it is a
# hashed field -- a store cut at one value cannot be re-cut at another, so
# changing it after the labels stage means redoing the labels stage.
#
# 0.015 is upstream's HA EXPORT value (`magic-point_coco_export.yaml:9`, which
# carries `# 0.001` beside it as the value it was raised FROM). Every other
# upstream config -- training, evaluation, repeatability -- uses 0.001, so the
# 15x is deliberate and specific to this step.
#
# It was 0.005 here until 2026-08-26, which is `superpoint_pytorch.py`'s
# INFERENCE default: upstream's number, off the wrong step, 3x too permissive.
# Rough arithmetic for why that matters: a pixel found by k of 100 views at
# probability p averages to k*p/100, so at 0.005 one confident view out of a
# hundred is enough and the label is essentially the UNION of the views.
#
# Every run prints the three-value ladder (0.001 / 0.005 / 0.015) with what
# each keeps, so the choice is visible in the log rather than assumed.
SCORE_THRESHOLD=0.015

# What gets REPORTED beside the cut, and it is not the same kind of thing: the
# ladder costs one extra NMS per tile and writes nothing, while SCORE_THRESHOLD
# above is hashed and decides the store. So a value one wants to LOOK at goes
# here and does not force a re-cut. 0.025 sits above the cut on purpose --
# without a rung above it there is no evidence that 0.015 is not already too
# strict, only that it is stricter than 0.005.
#
# The cut is folded in automatically whether or not it is listed, because a
# report that omits the value the store was cut at cannot say what that value
# did.
LADDER="0.001 0.015 0.025"

# Which corpus. result/cache/tiles/ holds TWO complete sets -- 36 stores cut at
# 0.75 from before the 3b probe settled it, and 36 cut at 0.5 after. The store
# design is why both survive: sampler_id is in the cfg hash, so neither
# overwrote the other. make_ha_labels REFUSES a mixed store rather than
# processing both, because both is this step run twice, half of it on a corpus
# that was rejected.
SAMPLER_ID=""   # empty = the store holds one corpus and it is unambiguous

# The demo cuts its own pre-tile off the slide rather than reading the store,
# so it needs a path and not a stem -- which is what lets it run while step 3c
# is still filling result/cache/tiles/. Same slide as MEASURE_SLIDE below.
WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs

# The measurement's slide and rung. ds 4 is native on both pyramid shapes, so
# the number is about HA rather than about a resample.
MEASURE_SLIDE=BRACS_1228
MEASURE_DS=4
MEASURE_TILES=100

# Tiles inspected twice for the agreement number. 2N forwards each.
INSPECT_TILES=8

# The demo's num values, one aggregate panel each. 10 against 100 is the
# comparison worth looking at: if they carry nearly the same points, the
# tenfold cost of N=100 bought nothing on this stain, and that is a finding
# rather than a figure. Cost is one HA run per value, so 110 forwards total.
DEMO_NUM="10 100"

status=0
run () {   # run <label> <command...>
  echo ""
  echo "======== $1 ========"
  shift
  "$@" || status=1
}

echo "======== MakeHaLabels  stages: $STAGE ========"

for stage in $STAGE; do
  case "$stage" in

    measure)
      run "measure  ($MEASURE_TILES tiles, $MEASURE_SLIDE ds $MEASURE_DS, N=$NUM)" \
        python training/SuperPathPoint/cli/make_ha_labels.py \
          --tile "$TILE" --num "$NUM" \
          --score-threshold "$SCORE_THRESHOLD" \
          --threshold-ladder $LADDER \
          ${SAMPLER_ID:+--sampler-id "$SAMPLER_ID"} \
          --wsi-stem "$MEASURE_SLIDE" --ds "$MEASURE_DS" \
          --limit "$MEASURE_TILES" \
          --out "/work/u26130998/result/${SLURM_JOB_NAME:-MakeHaLabels}/measure"
      echo ""
      echo "  Read the extrapolation before letting the labels stage run to the"
      echo "  end. If model_256 is already tens of GPU-hours, R > 1 is not"
      echo "  affordable and that is a finding, not a reason to lower N."
      ;;

    labels)
      # The store is keyed on the teacher AND the HA config, so re-running with
      # a changed config writes a second file rather than overwriting the first
      # -- and a reader that asks for one gets an error rather than the other.
      # Rungs already done are skipped; --overwrite is the way to redo them.
      run "labels  (every rung, every slide, N=$NUM)" \
        python training/SuperPathPoint/cli/make_ha_labels.py \
          --tile "$TILE" --num "$NUM" \
          --score-threshold "$SCORE_THRESHOLD" \
          --threshold-ladder $LADDER \
          ${SAMPLER_ID:+--sampler-id "$SAMPLER_ID"}
      ;;

    demo)
      # spec.md 14's cli/demo_ha.py, and the picture that makes the threshold
      # argument checkable rather than arithmetic. Cuts its own pre-tile off
      # the slide through the same DsLadder plan extract_pretiles uses, so it
      # runs while step 3c is still going.
      #
      # THREE VIEW PANELS FIRST. The teacher is not viewpoint-invariant, and
      # HA rests entirely on that: each view fires on a different subset, and
      # the aggregate is the vote. Nearly identical view panels mean HA is
      # averaging a hundred copies of one answer; disjoint ones mean the
      # aggregate is noise. Nobody has checked which of those H&E gives.
      #
      # Then the aggregate panels, carrying all three thresholds at once --
      # 0.001 (upstream training), 0.005 (pytorch inference), 0.015 (upstream
      # HA export, which is what make_ha_labels cuts at). They nest, so the
      # markers are drawn largest-set-first.
      run "demo  (HA aggregate against single views, num $DEMO_NUM)" \
        python training/SuperPathPoint/cli/demo_ha.py \
          --wsi "$WSI" --ds "$MEASURE_DS" --tile "$TILE" \
          --view-threshold $SCORE_THRESHOLD \
          --thresholds $LADDER --cut "$SCORE_THRESHOLD" \
          --num $DEMO_NUM \
          --out "/work/u26130998/result/${SLURM_JOB_NAME:-MakeHaLabels}/ha_demo__ds${MEASURE_DS}__num$(echo $DEMO_NUM | tr ' ' '-').png"
      ;;

    inspect)
      run "inspect  (the decision point, --with-model)" \
        python training/SuperPathPoint/cli/inspect_ha_labels.py \
          --with-model --tiles "$INSPECT_TILES" --num "$NUM"
      ;;

    *)
      echo "unknown stage: $stage   (known: demo measure labels inspect)"
      status=1
      ;;
  esac
done

echo ""
echo "======== done  (exit $status) ========"
echo "  labels  -> result/cache/keypoint_labels/<slide>__ds<d>__<cfg8>.safetensors"
echo "  tables  -> result/\${SLURM_JOB_NAME}/make_ha_labels.csv, ha_labels.csv"
echo "  figures -> result/\${SLURM_JOB_NAME}/ha_labels__<slide>_ds<d>.png"
echo "             result/\${SLURM_JOB_NAME}/ha_demo__ds<d>__num<...>.png"
echo ""
echo "  The figure decides step 5, and the four outcomes are printed under it."
echo "  Two of them are not 'continue':"
echo "    n_kp at the cap on every tile means the CAP selected, not the"
echo "      threshold -- raise --points-per-megapixel and redo, do not raise"
echo "      the threshold."
echo "    two runs agreeing no better than the decoy means the points are"
echo "      view-specific noise. More views will not fix that."

exit $status
