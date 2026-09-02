#!/bin/bash
#SBATCH --job-name=ExtractPreTiles        # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=12:00:00                   # 6 slides x 6 rungs x 500 pre-tiles
#SBATCH --account=MST114560               # Account
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=8                 # openslide reads, not a model
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
#  spec.md 12 step 3c: cut the pre-tiles the training set is made of
# =============================================================================
#
# NO MODEL. The GPU line is there because this cluster wants one; every second
# here is openslide reading a MIRAX and cv2 writing a PNG.
#
# WHAT LANDS. A pre-tile is `tile * PRE_TILE_FACTOR` on a side -- 768 px for the
# 256 model -- centred on a position the richness contract admitted.
# The tile itself is never written: it is the centre crop, and it exists only
# inside the training loop. spec.md 6.6 has the reason (a warp of a bare tile is
# a third pure black, and pure black is a straight maximum-contrast edge with
# two right angles, which is what a corner detector fires on).
#
# THE COST IS 9x THE TILE, and that is this step's main number rather than a
# footnote: the 17,784 pre-tiles are 31.5 GB of 768 px RGB uncompressed.
#
# RUN 2026-08-27, exit 0: 36 cells, 17,784 pre-tiles, PNG at 45.1 per cent of
# raw -- 14.2 GB on disk, better than the twenty-something that was guessed.
# Glass compresses almost to nothing, and the coarse rungs hold more of it (ds
# 32 lands 191 MB per 500 tiles against ds 4's 402 MB). By the same ratio 512
# is about 48 GB and 1024 about 153 GB, which is an upper bound rather than a
# prediction: their rung mixes are not this one's.
#
# RESUMABLE. `index.csv` is written last and its presence is what marks a
# directory complete, so a walltime kill leaves directories that a re-run
# rebuilds -- not a short index that reads as a small dataset. Re-running this
# script is safe and skips what is already finished.
#
# BEFORE THIS RUNS, two things must be true, and neither is checked by the
# script because both are cheap to check by hand:
#
#   1. result/cache/masks/ holds six masks   (jobscripts/BuildMaskStore.sh)
#   2. tile_yield.csv says how many tiles each (ratio, tile, ds) cell can
#      actually supply, and N below has been set from it, and no floor is
#      reported unmet

# THE TISSUE GATE STOPPED BEING A SECOND MECHANISM, 2026-08-27. It scored the
# same quantity as the richness buckets -- background fraction -- and the two
# disagreeing produced the 475/500 corpus of 2026-08-26: 22.5 per cent of every
# rung was reserved for buckets the gate had already emptied, and the shortfall
# was read as a property of the slides.
#
# `--tissue-ratio` came back on 2026-09-01 pointing at the mechanism that
# actually runs: it is written INTO the caps (`caps_for_tissue_ratio`), so
# there is one place a tile can be refused rather than two that can disagree.
# The floors go with it -- a floor asking for a share of a bucket the caps just
# closed is the same shortfall by another route.
#
# WHAT REPLACED IT, as seven buckets on the background fraction with a floor and
# a cap each (utilities/TileSampler.py, RichnessConfig):
#
#   bucket     background   floor   cap        the unassigned 30 per cent is
#   bg00_15      < 15 %       5 %   15 %       split evenly over the three
#   bg15_30     15 - 30 %    15 %   25 %       buckets that ASKED, so the
#   bg30_50     30 - 50 %    50 %   60 %       targets are 15/25/60 and sum to
#   bg50_70     50 - 70 %      -    20 %       exactly 1. bg50_70 and bg70_85
#   bg70_85     70 - 85 %      -    20 %       therefore receive NOTHING unless
#   bg85_95     85 - 95 %      -     0         another bucket falls short --
#   bg95_100     > 95 %        -     0         their ceilings are the spillway.
#
# A zero cap is HARD, and it binds the inheritance set too: a carried centre
# whose footprint reaches into bg85_95 at this rung TRUNCATES its chain rather
# than being placed. Expect those breaks to cluster at the coarse rungs, where
# the footprint is 32x the fine one and the corpus is already thinnest.
#
# THE OPEN NUMBER IS WHETHER bg30_50 CAN SUPPLY 50 PER CENT. Nothing measured
# so far can say: the old corpus put ONE 15 per cent cap over what are now two
# buckets, so it recorded the cap and not the slide. probe_tile_yield now
# reports supply_<bucket> against floor_<bucket> and prints every cell that
# misses a floor -- run 3b before trusting this step's mix.

# =============================================================================
#  TWO CORPORA, ONE SCRIPT. `CORPUS` names which.
# =============================================================================
#
# The knobs below decide what gets cut, and the two answers are far apart -- one
# corpus trains Stage A and the other feeds Stage B's survival analysis. Every
# one of them is in `sampler_id`, so getting one wrong does not fail: it writes
# a THIRD corpus under a name nobody meant, in its own directory, indefinitely.
# Naming the two is what keeps that from being six things to remember.
#
#   stageA  the training corpus. No chains, the seven-bucket contract, a
#           disjoint lattice. These are the values of 2026-08-27, and they have
#           to stay these values: `_rng` is reset per rung so that this mode
#           reproduces that corpus bit for bit (utilities/TileSampler.py).
#
#   stageB  the chain corpus. Every centre carried to every rung, the gate at
#           50 per cent background, an overlapping lattice to put back the
#           candidates the gate removes. NOT a training corpus -- the gate and
#           `share=1.0` between them mean the bucket mix is ds 16's rather than
#           the contract's (see BUCKET_FRAME below).
#
# Any single knob can still be overridden on top, which is what a smoke run is:
#
#   CORPUS=stageB N=20 DS="1 2 4 8 16" WSI=<path> ROOT=<scratch> sbatch ...
#
CORPUS="${CORPUS:-stageB}"
case "$CORPUS" in
  stageA)
    _share=0     ; _source=      ; _frame=per_rung   ; _ratio=
    _step=0      ; _overlap=0    ; _ovshare=0
    _n=100       ; _root=/work/u26130998/result/cache/tiles
    ;;
  stageB)
    _share=1.0   ; _source=16    ; _frame=at_inherit ; _ratio=0.5
    _step=128    ; _overlap=0.5  ; _ovshare=1.0
    _n=200       ; _root=/work/u26130998/result/cache/tiles_chains
    ;;
  *)
    echo "unknown CORPUS: $CORPUS   (known: stageA stageB)" >&2
    exit 1
    ;;
esac

# 100 per (slide, rung), across TWELVE slides rather than six: five per stain
# in train and one per stain held out. The corpus is therefore 12 x 6 x 100
# rather than 6 x 6 x 500 -- 7,200 against 18,000 asked for, and a wider spread
# of slides for fewer tiles of each.
#
# That trade is the one worth making here. spec.md 6.5 already records that a
# held-out estimate from one slide per stain is noisy, and every criterion in
# spec.md 1 is a MARGIN OVER A DECOY rather than an absolute -- decoy and model
# see the same slide, so the slide's own character cancels. What does not
# cancel is having sampled two Ki67 batches out of ten.
#
# RAISED TO 200 FOR THE CHAIN CORPUS, 2026-09-01. Under `share=1.0` this number
# IS the chain count: `_choose_centres` asks for `share * n` centres and gets
# `min(that, what the source rung admits)`. ds 16 admits 187 a slide on
# average, so N=100 would leave 87 of them unused. The cost is linear and it is
# mostly not this job -- the extraction doubles, and so does MakeHaLabels,
# which was 72 cells x 1.1 min and becomes about 160.
N="${N:-$_n}"

# v1 is 256. 512 and 1024 are separate models, separate extractions and, at
# 106 GB and 340 GB, separate decisions (spec.md 6.5).
TILE=256

# 5x N, matching the 100/500 ratio of the runs spec.md 6.5 quotes.
MAX_TRIES=2500

# Empty = every slide, every rung in DsLadder's default. Set for a SMOKE RUN,
# which is worth doing before the full one because there is currently NO
# evidence that a chain gets built at all: `inherit` was never reachable from
# this CLI, and `_extract_slide` (one sampler over all rungs) is new. The
# number to read is `N chains` on the sampler line -- zero means the wiring is
# still not connected, and it costs twelve slides to find that out at full size.
#
#   N=20 DS="1 2 4 8 16" \
#     WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_BT/Type_N/BRACS_1598.svs \
#     ROOT=/work/u26130998/result/cache/tiles_smoke \
#     sbatch jobscripts/SuperPathPointJobs/ExtractPreTiles.sh
#
# WSI IS A PATH, NOT A STEM. Every other thing here is keyed by the stem --
# the mask store, the tile store, `--wsi-stem` in make_ha_labels -- so the stem
# is the expected mistake, and it used to surface four frames down as
# openslide's "Unsupported or missing image file", which reads as a corrupt
# slide. The CLI now says so by name and suggests the path.
#
# A SEPARATE ROOT FOR THE SMOKE RUN. `sampler_id` differs, so the directories
# would not collide -- but they would sit beside the real ones and `Datasets`
# refuses two stores for one (slide, rung). Keeping the smoke corpus somewhere
# else means deleting it is `rm -rf` of one directory.
DS="${DS:-}"
WSI="${WSI:-}"
# =============================================================================
#  CHAINS, added 2026-09-01. spec.md 3.2, plan.md P1.
# =============================================================================
#
# A chain is ONE level-0 centre with a tile at every rung -- the same physical
# tissue at every magnification -- and it is what Stage B's survival analysis
# reads. The corpus of 2026-08-27 has `inherit_id = -1` on all 6,388 rows, and
# that was not a setting that was wrong: the option was never wired to this
# CLI, and `extract_pretiles` ran one sampler PER RUNG, which cannot build a
# chain at all because each call chooses its own centres. Both are fixed.
#
# WHY source_rung IS THE COARSE END AND NOT THE DEFAULT FINE ONE.
# `_choose_centres` validates a centre at the SOURCE rung only; every other
# rung is checked as the chain is placed, and a refusal truncates it. A centre
# admissible at ds 16 (footprint 4096) fits at every finer rung by arithmetic
# -- the footprint only shrinks going down -- so the fit can never break the
# chain. Choosing at ds 1 instead maximises candidates and then loses them at
# the coarse rungs, where the corpus is already thinnest.
#
# WHAT source_rung STILL CANNOT GUARANTEE IS TISSUE. `caps[bg85_95] = 0` binds
# the inherited set (see above), and a centre whose 4096 window has tissue can
# have its central 256 window land in a gap. Those chains truncate at the fine
# end and are dropped whole. `n_inherit_refused` per rung is the only place
# that loss is visible, and it is now printed and in the CSV.
#
# 16 AND NOT 32, DECIDED 2026-09-01. ds 32 admits 583 positions over the twelve
# slides against ds 16's 2,242, and N binds before either: at N=200 the coarse
# source yields ~48 chains a slide and ds 16 yields the full 200. ds 32 is
# still IN the ladder -- a chain that also fits there gets a sixth member -- so
# the six-rung analysis runs on that subset and the five-rung one on all of it.
# `MppStack.chains` takes the rung list to be complete over, so both are reads
# of one corpus.
INHERIT_SHARE="${INHERIT_SHARE:-$_share}"
INHERIT_SOURCE_RUNG="${INHERIT_SOURCE_RUNG:-$_source}"

# WHERE THE CENTRES COME FROM, and the smoke run of 2026-09-01 is why this is
# set at all. `_choose_centres` draws UNIFORMLY from whatever the caps admit,
# and the settled contract admits up to 85 per cent background -- so on
# BRACS_1598 (24 per cent tissue) nine of fifteen chains were seeded from
# windows already more than half glass, and five of twenty centres had their
# FINEST tile land in a zero-capped bucket and truncate. 20 asked, 13 complete.
#
# 0.5 writes the gate into the caps instead: background <= 50 per cent, which
# is a bucket edge (a ratio that is not one is refused rather than rounded).
# The floors go with it -- a floor asking for a share of a bucket the caps just
# closed is the 475/500 shortfall of 2026-08-26 by another route.
#
# IT GATES EVERY TILE, NOT ONLY THE CENTRES. So this corpus is NOT the one to
# train Stage A on: that wants the seven-bucket contract's mix, and it is still
# on disk under `cache/tiles` with its own sampler_id.
# EMPTY MEANS THE SEVEN-BUCKET CONTRACT, which is what reproducing the
# 2026-08-27 corpus needs -- the flag is then not passed at all rather than
# passed as an empty string. That corpus is the Stage A one and it has to stay
# regenerable: `_rng` is reset per rung so that at `inherit.share = 0` the
# tiles come out bit-identical (utilities/TileSampler.py, sample()).
#
#   REPRODUCE THE 2026-08-27 CORPUS, one slide, into a scratch root:
#
#     N=100 INHERIT_SHARE=0 BUCKET_FRAME=per_rung TISSUE_RATIO= \
#       GRID_STEP=0 MAX_OVERLAP=0 OVERLAPPING_SHARE=0 \
#       WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_BT/Type_N/BRACS_1598.svs \
#       ROOT=/work/u26130998/result/cache/tiles_repro \
#       sbatch jobscripts/SuperPathPointJobs/ExtractPreTiles.sh
#
#   The directory names must come out `__d581d527` -- the same cfg_hash as the
#   old store, because the config is the same -- and `index.csv` must match it
#   row for row. The RUNGS AFTER ds 1 ARE THE TEST: ds 1 is the first consumer
#   of the stream and was never affected, so a check that stops there proves
#   nothing.
TISSUE_RATIO="${TISSUE_RATIO:-$_ratio}"

# AN OVERLAPPING LATTICE, because the gate above shrinks the candidate pool and
# this is what puts it back. grid_step 128 on a 256 tile halves the step in
# each axis, so the lattice is 4x denser: ds 16 offered 187 positions a slide
# disjoint, and offers roughly 750 here.
#
# The three knobs have to agree or `OverlapConfig.check` refuses them: step 128
# makes every adjacent pair overlap 50 per cent along an axis, so
# max_overlap_ratio must be at least 0.5 -- below it every adjacent position is
# illegal, the lattice degenerates to the disjoint one, and `sampler_id` still
# records 128. `overlapping_share` 1.0 leaves only the ratio binding.
#
# Inherited tiles were always exempt from the bound (they are the same tissue
# at every magnification, so at ds 16 they overlap each other by construction);
# what changes here is the CANDIDATE lattice they are chosen from.
GRID_STEP="${GRID_STEP:-$_step}"
MAX_OVERLAP="${MAX_OVERLAP:-$_overlap}"
OVERLAPPING_SHARE="${OVERLAPPING_SHARE:-$_ovshare}"

# 'at_inherit': the bucket is fixed at the source rung and carried, so a chain
# has ONE bucket. A survival analysis stratified by richness needs that -- under
# 'per_rung' a chain drifts between buckets as its footprint grows, and
# grouping by bucket at rung k groups a different set than at rung k+1. The
# question then produces a number rather than an error, which is worse.
#
# WHAT IT COSTS, and at share=1.0 it costs all of it: the per-rung floors act
# only on the NON-inherited remainder, and with the whole quota inherited there
# is no remainder. Every rung's bucket distribution is ds 16's, not the seven-
# bucket contract's. Accepted because this corpus is for Stage B; the 2026-08-27
# corpus is still on disk under its own sampler_id for anything that wants the
# contract's mix.
BUCKET_FRAME="${BUCKET_FRAME:-$_frame}"

# A SEPARATE ROOT, and not tidiness. `sampler_id` is in the directory name, so
# re-extracting adds directories beside the old ones rather than replacing
# them -- and `Datasets` used to read EVERY store matching (slide, ds) and call
# the union a corpus. It now refuses two, but a separate root means the
# situation never arises.
ROOT="${ROOT:-$_root}"

echo "======== ExtractPreTiles  corpus: $CORPUS ========"
echo "  tile $TILE   pre-tile $((TILE * 3))   n $N"
echo "  chains: share $INHERIT_SHARE   source ds $INHERIT_SOURCE_RUNG   bucket $BUCKET_FRAME"
echo "  gate  : tissue >= $TISSUE_RATIO   lattice step $GRID_STEP   max overlap $MAX_OVERLAP"
echo "  root  : $ROOT"
echo "  slides: ${WSI:-every mask in result/cache/masks/}   rungs: ${DS:-DsLadder default}"
echo ""

python training/SuperPathPoint/cli/extract_pretiles.py \
  --tile "$TILE" \
  --n "$N" \
  --root "$ROOT" \
  --inherit-share "$INHERIT_SHARE" \
  ${INHERIT_SOURCE_RUNG:+--inherit-source-rung "$INHERIT_SOURCE_RUNG"} \
  --bucket-frame "$BUCKET_FRAME" \
  ${TISSUE_RATIO:+--tissue-ratio "$TISSUE_RATIO"} \
  --grid-step "$GRID_STEP" \
  --max-overlap "$MAX_OVERLAP" \
  --overlapping-share "$OVERLAPPING_SHARE" \
  ${DS:+--ds $DS} \
  ${WSI:+--wsi $WSI} \
  --max-tries "$MAX_TRIES"

status=$?

echo ""
echo "======== done  (exit $status) ========"
echo "  pre-tiles -> $ROOT/<slide>__ds<d>__t${TILE}__<cfg8>/"
echo "  table     -> result/\${SLURM_JOB_NAME}/extract_pretiles.csv"
echo ""
echo "  Two numbers to read before anything else:"
echo "    n_got against n_requested per rung -- the coarse rungs are where the"
echo "      rejection sampler runs out, and the gap decides the rung balance"
echo "      switch (align-min or loss-weight, spec.md 6.5)."
echo "    the printed PNG-against-raw ratio -- it is what says whether 512 and"
echo "      1024 fit on disk in this shape."

exit $status
