#!/bin/bash
# =============================================================================
#  The experiments of the training mechanisms and of the arms (locascope/core/stage2_retrieval/proposal.md §9, §10), at a size to compare
#  them on: gigapath cls, 20 slides of each training dataset, 4 epochs, the validation FoVs 32 a visit (more, so that two
#  settings that differ by a few points can be told from noise).
#
#  A SETTING IS A HEAD. Every head of a job is trained on the SAME FoVs, the same levels (encoded once) and validated on
#  the same held-out FoVs, so two settings are compared on paired data; the flags after a head's name are its own
#  (`Name@w-geo=1.0@sigma=2`, train.py's HEAD_FLAGS, @items, @top). A job is a ROUND:
#
#    ROUND=1  the candidates     --top-p (1.0 all / 0.6 each step / 0.2, 0.05, 0.01 episodes), --episodes, --warmup-epochs,
#                                a GloRet head over the list (as a reranker), a LogLoRet over 128 / 512 / 2000 of the
#                                list, a LogLoRet over 1000 windows of the level (the teacher's pools)
#    ROUND=2  the labels         the ignore zone (--ignore-expand), the wrong rotations (negatives or ignored), the soft
#                                targets (--w-geo, --soft-target, --sigma, --near-alpha)
#    ROUND=3  the rest           the temperature, the learning rate, the widths, the decoder (--recon), dropout; a LogLoRet's
#                                correspondence and cycle losses
#    ROUND=4  the arms           the 12 GloRet and 3 LogLoRet heads of gigapath cls (GloRet.sh, THE FORMAL RUNS 2), one setting
#
#  Rounds 2 to 4 take the winner of round 1 for GloRet's and LogLoRet's --top-p: G_TOP (default 0.05) and L_TOP (2000).
#  TilePool attnpool is not a head of these jobs (its tokens do not fit them): its experiment is
#      JOB=ExpGloRetAttnPool N_TRAIN_WSI=20 N_VAL_WSI=10 NODES=2 GPUS=4 PARTITION=16gpus TIME=24:00:00 bash deploy/slurm/GloRet/GloRetAttnPool.sh
#
#    ROUND=1 bash deploy/slurm/GloRet/ExpGloRetScreen.sh      # one job name, re-sent over itself: its log is overwritten, its
#                                                           # weights (result/ExpGloRetScreen/) are named by round (--tag) and head
#    DRY_RUN=1 ROUND=2 bash deploy/slurm/GloRet/ExpGloRetScreen.sh   # the sbatch line; submits nothing
#    AFTER=<job id> ROUND=1 bash ...                       # starts only if that job (TestGloRet) ended well
#
#  A step with --top-p 1.0 costs every window of the level whatever the number of FoVs (locascope/training/GloRet/Full.py), so a visit
#  here is 6 steps of 16 FoVs, not the 16 steps of 8 of a formal run. There is no `make clean-exp`: rm -rf result/ExpGloRetScreen.
# =============================================================================
ROUND="${ROUND:-1}"
DRY_RUN="${DRY_RUN:-0}"
JOB=ExpGloRetScreen
G_TOP="${G_TOP:-0.05}"
L_TOP="${L_TOP:-2000}"
G=GloRet-SelfAttn-AttnPool
L=LogLoRet-SelfAttn-AttnPool
EXCLUDE="25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083"

case "$ROUND" in
  1) TOP_P="1.0 2000"
     HEADS="$G $G@top=0.6 $G@top=0.2 $G@top=0.05 $G@top=0.01 $G@top=0.05@episodes=1 $G@top=0.05@episodes=8 \
$G@top=0.05@warmup-epochs=0 $G@items=list@top=2000 \
$L@top=128 $L@top=512 $L $L@items=level@top=1000" ;;
  2) TOP_P="$G_TOP $L_TOP"
     HEADS="$G $G@ignore-expand=0 $G@ignore-expand=0.25 $G@ignore-expand=1.0 $G@wrong-rotation=negative \
$G@w-geo=1.0 $G@w-geo=1.0@sigma=0.5 $G@w-geo=1.0@sigma=2 $G@w-geo=1.0@soft-target=exponential \
$G@w-geo=1.0@near-alpha=0.1 $G@w-geo=1.0@near-alpha=0.6 \
$L $L@ignore-expand=0 $L@wrong-rotation=negative $L@w-geo=1.0" ;;
  3) TOP_P="$G_TOP $L_TOP"
     HEADS="$G $G@tau-init=0.02 $G@tau-init=0.1 $G@lr=1e-4 $G@lr=1e-3 $G@dim=256@embed-dim=256 $G@dim=1024@embed-dim=1024 \
$G@dropout=0 $G@dropout=0.3 $G@embed-dim=256 $G@embed-dim=256@recon=ae@w-recon=0.1 \
$L $L@w-corr=0@w-cycle=0 $L@w-corr=0.3@w-cycle=0.3 $L@lr=1e-4 $L@dropout=0.3" ;;
  4) TOP_P="$G_TOP $L_TOP"
     HEADS="$G GloRet-SelfAttn-MaskedMean GloRet-SelfAttn-CNN GloRet-SelfAttn-Align \
GloRet-SptSelfAttn-AttnPool GloRet-SptSelfAttn-CNN GloRet-SptSelfAttn-Align \
GloRet-Identity-AttnPool GloRet-Identity-CNN GloRet-Identity-Align GloRet-GNN-GraphPool GloRet-CLSTrans-Identity \
$L+dustbin $L+sinkhorn $L+sigmoid" ;;
  *) echo "ROUND is 1, 2, 3 or 4 ($ROUND)"; exit 2 ;;
esac

export ENCODER=gigapath POOLING=cls EPOCHS="${EPOCHS:-4}" N_TRAIN_WSI="${N_TRAIN_WSI:-20}" N_VAL_WSI="${N_VAL_WSI:-10}"
export HEADS TOP_P
export EXTRA="--tag round$ROUND --fov-per-visit 64 --steps-per-visit 4 --fovs-per-step 16 --val-fov-per-visit 32 --episodes 2 --warmup-epochs 2"
CMD=(sbatch -J "$JOB" --partition=8gpus --time="${TIME:-24:00:00}" --nodes=1 --gpus-per-node=4 --cpus-per-task=48 --mem=800G
     --exclude="$EXCLUDE")
[ -n "${AFTER:-}" ] && CMD+=(--dependency="afterok:$AFTER")      # AFTER=<job id>: start only when that job ended well (a test)
CMD+=(deploy/slurm/GloRet/GloRet.sh)
echo "round $ROUND: $(echo $HEADS | wc -w) heads, --top-p $TOP_P, $EPOCHS epochs of $N_TRAIN_WSI slides a dataset"
if [ "$DRY_RUN" = "1" ]; then
  echo "HEADS=\"$HEADS\""
  echo "${CMD[*]}"
else
  "${CMD[@]}"
fi
