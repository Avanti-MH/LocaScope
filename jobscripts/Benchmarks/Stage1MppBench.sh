#!/bin/bash
#SBATCH --job-name=Stage1MppBench             # -> log/%x, result/%x/
#SBATCH --partition=8gpus                     # Partition
#SBATCH --time=48:00:00                       # 13+ methods, one encoder resident at a time
#   ran 2026-10-10 (val): sbatch --time=08:00:00 --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn006 ...
#SBATCH --account=MST114560                   # Account
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gpus-per-node=1                     # GPUs per node (不要設0)
#SBATCH --cpus-per-task=12                    # the H200 cap per GPU: renders and openslide reads for the FoVs
#SBATCH --mem=200G                            # the H200 cap per GPU (1 GPU here; 400G was refused)
#SBATCH --ntasks-per-node=1                   # Tasks per node
#SBATCH -o /work/u26130998/log/%x             # STDOUT
#SBATCH -e /work/u26130998/log/%x             # STDERR

# ---------------- Load modules ----------------
ml purge
ml load miniconda3/26.1.1
ml load cuda/12.6

# ---------------- Activate environment ----------------
conda activate locascope
source jobscripts/_env.sh    # HF_HOME; must be exported before python starts

# =============================================================================
#  utilities/bench_modules/bench_stage1_mpp.py, then its analysis
# =============================================================================
#
# Every STAGE1 method (<method>:<recipe>: KNN_RECIPES, CLASSIC_RECIPES) and every trained
# CHECKPOINT (prototype and classifier, built from what each file records about itself, not from a
# recipe) on the same FoVs -- the first N_WSI
# slides of each dataset's recorded SPLIT, the FOV recipe's draw -- one method
# resident at a time. Each (slide, method) is one stage-1 cache entry under
#   result/cache/<job>/slide=/seg=/region=/plan=/draw=/render=<gap>/stage1/
# so a rerun scores only what is missing, a new checkpoint costs only itself, and jobs running
# different methods at once never touch the same file.
# The reference banks are draw entries of the same job, their features beside
# them. The analysis takes the same flags, joins the ground truth and writes
# result/<job>/stage1_<split>.csv and its tables.
#
# FoV_Vote.md fixes every risk threshold on val before test is looked at:
#   SPLIT=val sbatch jobscripts/Benchmarks/Stage1MppBench.sh     fits them
#   sbatch jobscripts/Benchmarks/Stage1MppBench.sh               reads them back
# Smoke: LIMIT=5 N_WSI=1 (the first 5 FoVs of each slide; recorded in the
# entries, so a full run does not take them for its own).

DATASETS="${DATASETS:-bracs/test ki67_with_photo}"
SPLIT="${SPLIT:-test}"         # ran 2026-10-10: SPLIT=val
N_WSI="${N_WSI:-5}"            # per dataset; val and test must match.  ran 2026-10-10: N_WSI=10
FOV="${FOV:-bench}"            # query_sim/FovSupply.FOV_RECIPES
PER_LEVEL="${PER_LEVEL:-}"     # replaces the recipe's n_per_rung when set
SEG="${SEG:-hest}"             # the mask the reference banks are drawn on
STAGE1="${STAGE1:-knn:gigapath knn:uni2 classic:default}"    # recipe methods; STAGE1=none for no one of them
#   ran 2026-10-10: STAGE1="knn:gigapath knn:uni2 classifier:gigapath-arcface prototype:uni2-mean-cosine-tau classic:default"
# Trained checkpoints, each a method: directories (their *.pt), globs or files. All of them are: 33 prototype files (11 trainings x 6rung/best/native) and 132 classifier files (44 trainings x
# best/best_unweighted/last). A method takes ~10-14 min over the 20 slides, so one job for all of them
# would take more than a day: split them over jobs by a glob each (CHECKPOINTS=..., STAGE1=none), e.g.
#   R=/work/u26130998/result
#   CHECKPOINTS="$R/PrototypicalRoutingHead/weights"                       prototypes (33)
#   CHECKPOINTS="$R/MppRoutingHead/weights/gigapath_frozen_*.pt"           classifiers on gigapath (27)
#   CHECKPOINTS="$R/MppRoutingHead/weights/convnext_v2_finetuned_*.pt"     classifiers on convnext_v2 (24)
#   CHECKPOINTS="$R/MppRoutingHead/weights/uni2_frozen_*_best.pt"          uni2 classifiers, best (27)
#   ... _best_unweighted.pt, _last.pt                                     uni2 classifiers, the others
# Sharded: every shard BENCH=1 ANALYZE=0 (the analysis would write the one result/<job>/stage1_<split>.csv from
# only its own methods); once they are all done, one BENCH=0 ANALYZE=1 job with the CHECKPOINTS (or CKPT_SET=all) and STAGE1 of all of them
# scores them all together.
BENCH="${BENCH:-1}"
ANALYZE="${ANALYZE:-1}"
# SAVE_PHOTOS=1 keeps every FoV photo beside its render record (photos_<gap>/<i>.png, ~250 MB a slide), so each
# method after the first reads it instead of rendering it. Off by default; a render entry that has photos is read
# from either way. Not an analysis flag, so it is kept out of ARGS.
SAVE_PHOTOS="${SAVE_PHOTOS:-0}"
R=/work/u26130998/result
W=$R/MppRoutingHead/weights
P=$R/PrototypicalRoutingHead/weights
# CKPT_SET picks the default CHECKPOINTS (CHECKPOINTS=... overrides it):
#   shortlist (default)  29 checkpoints picked on VAL (not test): classifiers by their best-epoch val level_accuracy
#                        (result/MppRoutingHead/val_scores.csv, mean of the two datasets), prototypes by their best-epoch val
#                        accuracy (result/PrototypicalRoutingHead/val_scores_per_combo.csv, n_pairs-weighted, mean of the two
#                        datasets). ~6 h in one job.
#   all                  every file of both weights directories (165); too long for one job, shard it with CHECKPOINTS
CKPT_SET="${CKPT_SET:-shortlist}"
if [ "$CKPT_SET" = "all" ]; then
  DEFAULT_CHECKPOINTS="$P $W"
else
  # classifiers, `_best`: the 12 best uni2 trainings by val (mlp family; ord_a / ord_b / bal), then one convnext_v2 and one
  # gigapath to see whether the encoder is what matters
  DEFAULT_CHECKPOINTS=""
  for n in uni2_frozen_mlp_deep_wide_ord_b uni2_frozen_mlp_wide_ord_b uni2_frozen_mlp_deep_wide_ord_a \
           uni2_frozen_mlp_narrow_ord_b uni2_frozen_mlp_deep_ord_a uni2_frozen_mlp_deep_wide \
           uni2_frozen_mlp_deep_residual_ord_b uni2_frozen_mlp_deep_residual uni2_frozen_mlp_deep \
           uni2_frozen_mlp_deep_residual_ord_a uni2_frozen_mlp_deep_ord_b uni2_frozen_mlp_ord_b \
           convnext_v2_finetuned_mlp_deep_residual gigapath_frozen_mlp_deep; do
    DEFAULT_CHECKPOINTS="$DEFAULT_CHECKPOINTS $W/${n}_best.pt"
  done
  # prototypes: the 5 best trainings by val (the first is the matching net), each as `_best` (total over combinations),
  # `_6rung` (full 6-way) and `_native` (native-query combinations) -- which of them suits FoVs is not known beforehand
  for n in uni2_frozen_cls_bilstm_attnlstm_off_cosine_logsumexp_bracs_train_none-k5 \
           uni2_frozen_cls_identity_identity_set_transformer_cosine_tau_none_none-k5 \
           uni2_frozen_cls_identity_identity_set_transformer_attn_score_none_none-k5 \
           uni2_frozen_cls_identity_identity_attn_pool_cosine_tau_none_none-k5 \
           uni2_frozen_cls_identity_identity_set_transformer_cosine_tau_none_hold_q-k5; do
    for v in best 6rung native; do DEFAULT_CHECKPOINTS="$DEFAULT_CHECKPOINTS $P/${n}_${v}.pt"; done
  done
fi
CHECKPOINTS="${CHECKPOINTS:-$DEFAULT_CHECKPOINTS}"
LIMIT="${LIMIT:-0}"
MASK_CACHE_JOB="${MASK_CACHE_JOB:-MppRoutingHead}"
EXTRA="${EXTRA:-}"
#   ran 2026-10-10: EXTRA="--draw-cache-job BenchLocaScope --render-cache-job BenchLocaScope"
#   (the FoV draws and photos are the pipeline bench's cache job's, so the same photos; the stage-1
#   entries stay under this job, result/cache/Stage1MppBench/)

ARGS=(--datasets $DATASETS --split "$SPLIT" --n-wsi "$N_WSI" --fov "$FOV"
      --seg "$SEG" --mask-cache-job "$MASK_CACHE_JOB" --limit "$LIMIT"
      --stage1 $STAGE1 --checkpoints $CHECKPOINTS)
[ -n "$PER_LEVEL" ] && ARGS+=(--sampler-n-per-rung "$PER_LEVEL")
ARGS+=($EXTRA)

status=0
if [ "$BENCH" = "1" ]; then
if [ "$STAGE1" != "none" ]; then
  echo "======== [memory estimate] (the recipe methods; a checkpoint's is its file's size and the encoder's) ========"
  python utilities/cli/diagnostics/estimate_method_memory.py --stage1 $STAGE1
fi

echo ""
echo "======== Stage1MppBench: $STAGE1 + checkpoints $CHECKPOINTS ========"
echo "  $DATASETS #$SPLIT  n_wsi=$N_WSI  fov=$FOV  seg=$SEG  limit=$LIMIT"
BENCH_ARGS=()
[ "$SAVE_PHOTOS" = "1" ] && BENCH_ARGS+=(--save-photos)
python -u utilities/bench_modules/bench_stage1_mpp.py "${ARGS[@]}" "${BENCH_ARGS[@]}"
status=$?
fi

if [ "$ANALYZE" = "1" ]; then
echo ""
echo "======== [analysis] ========"
if [ "$SPLIT" = "val" ]; then
  python utilities/cli/metrics/analyze_stage1_metrics.py "${ARGS[@]}" --fit-thresholds
else
  python utilities/cli/metrics/analyze_stage1_metrics.py "${ARGS[@]}" --thresholds auto
fi
rc=$?
[ $status -eq 0 ] && status=$rc
fi

echo ""
echo "======== done (exit $status) -> result/${SLURM_JOB_NAME:-Stage1MppBench}/ ========"
exit $status
