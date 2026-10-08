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
#   1. result/cache/BuildMaskStore_mask/ holds the masks (BuildMaskStore.sh);
#      a missing one is made on the way, at a segmentation's cost
#   2. tile_yield.csv says how many tiles each (ratio, tile, ds) cell can
#      actually supply, and N below has been set from it, and no floor is
#      reported unmet

# NO `tissue_ratio`/`TISSUE_RATIO` -- richness is seven buckets on the
# background fraction with a floor and a cap each (utilities/TileSampler.py,
# RichnessConfig):
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
#  THE TRAINING CORPUS. This script does only this now (2026-09-06).
# =============================================================================
#
# F's and C's own extractions used to also go through a `CORPUS=` switch
# here (`stageB-fOwn`/`stageB-cOwn`), each with its own copy of the sampling
# knobs -- a second definition of numbers `cli/prepare_chain_stack.py` also
# needed in Python to compute `sampler_id()` without running anything, so
# the two could drift from each other silently. Reverted: F's and C's own
# corpora are now sampled directly in `prepare_chain_stack.py` (`TileSampler`
# + `PreTileStore`, no subprocess, no second copy of the knobs) -- this
# script goes back to being the one thing it was before that detour, the
# `stageA` training corpus, human-run.
#
# The defaults below ARE `common/Corpora.RECIPES['stageA']` -- the address a
# reader computes for "stageA" -- so a run with no override lands exactly where
# `--corpus stageA` looks. Any knob can still be overridden on top, which is
# what a smoke run is; the run then prints its own corpus key, which is what a
# reader passes as `--corpus <key>`:
#
#   N=20 DS="1 2 4 8 16" WSI=<path> CACHE_JOB=ExtractPreTilesSmoke sbatch ...
#
_share=0     ; _source=      ; _frame=per_rung
_step=0      ; _overlap=0    ; _ovshare=0
_n=100

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
#     CACHE_JOB=ExtractPreTilesSmoke \
#     sbatch jobscripts/SuperPathPointJobs/ExtractPreTiles.sh
#
# WSI IS A PATH, NOT A STEM. Every other thing here is keyed by the stem --
# the mask store, the tile store, `--wsi-stem` in make_ha_labels -- so the stem
# is the expected mistake, and it used to surface four frames down as
# openslide's "Unsupported or missing image file", which reads as a corrupt
# slide. The CLI now says so by name and suggests the path.
#
# A SEPARATE CACHE JOB FOR THE SMOKE RUN -- not because a reader would mix
# them (a corpus is read by its address, and N alone moves the address), but
# because deleting it is then `rm -rf result/cache/ExtractPreTilesSmoke_pretiles`
# instead of hunting one corpus out of the real root.
#
DS="${DS:-}"
WSI="${WSI:-}"
# DATASETS: AccessDatasets ids or <id>#<split>, N_WSI of each (empty: all) --
# the slides are named the way the routing heads name theirs. WSI adds paths.
DATASETS="${DATASETS:-bracs/test#val ki67_with_photo#val}"
N_WSI="${N_WSI:-}"
# =============================================================================
#  CHAINS, added 2026-09-01, REMOVED FROM THIS SCRIPT 2026-09-06.
# =============================================================================
#
# F's/C's own extractions (what this section's knobs were FOR) now sample
# directly in `cli/prepare_chain_stack.py` -- this script builds `stageA`
# only. Kept here as history for why `INHERIT_SHARE`/`INHERIT_SOURCE_RUNG`/
# `BUCKET_FRAME` exist as overridable knobs at all (a `stageA` smoke run can
# still legitimately want a different value on top), not as a description of
# what a normal run of this script does today.
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
# `ChainStack.chains` takes the rung list to be complete over, so both are reads
# of one corpus.
INHERIT_SHARE="${INHERIT_SHARE:-$_share}"
INHERIT_SOURCE_RUNG="${INHERIT_SOURCE_RUNG:-$_source}"

# WHERE THE CENTRES COME FROM, and the smoke run of 2026-09-01 is why
# `--inherit-source-rung` above matters at all. `_choose_centres` draws
# UNIFORMLY from whatever the richness caps admit -- so on BRACS_1598 (24 per
# cent tissue) nine of fifteen chains were seeded from windows already more
# than half glass, and five of twenty centres had their FINEST tile land in a
# zero-capped bucket and truncate. 20 asked, 13 complete.
#
#   REPRODUCE THE 2026-08-27 CORPUS, one slide, into a scratch root:
#
#     N=100 INHERIT_SHARE=0 BUCKET_FRAME=per_rung \
#       GRID_STEP=0 MAX_OVERLAP=0 OVERLAPPING_SHARE=0 \
#       WSI=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_BT/Type_N/BRACS_1598.svs \
#       CACHE_JOB=ExtractPreTilesRepro \
#       sbatch jobscripts/SuperPathPointJobs/ExtractPreTiles.sh
#
#   `index.csv` must match the 2026-08-27 corpus row for row. The RUNGS AFTER
#   ds 1 ARE THE TEST: ds 1 is the first consumer of the stream and was never
#   affected, so a check that stops there proves nothing. (The directory name
#   does not carry over: the mask recipe and the rung plan are in the address
#   now, and the old store had neither.)

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

# ONE TREE FOR EVERY CORPUS: result/cache/<CACHE_JOB>/. In it
# each corpus has its own address -- mask, region, sampler, rung plan, factor
# (utilities/Store.py, PreTileCorpus) -- so re-extracting at another setting
# adds a directory beside the old one, and a reader reads exactly one of them.
# prepare_chain_stack.py writes F's and C's own corpora into the same tree.
CACHE_JOB="${CACHE_JOB:-ExtractPreTiles}"
# Which mask the tiles are cut through, and which job made it.
SEG="${SEG:-uni2_pca}"
MASK_CACHE_JOB="${MASK_CACHE_JOB:-BuildMaskStore}"

echo "======== ExtractPreTiles  corpus: stageA ========"
echo "  tile $TILE   pre-tile $((TILE * 3))   n $N"
echo "  chains: share $INHERIT_SHARE   source ds $INHERIT_SOURCE_RUNG   bucket $BUCKET_FRAME"
echo "  lattice step $GRID_STEP   max overlap $MAX_OVERLAP"
echo "  cache : result/cache/${CACHE_JOB}/   mask: $SEG (${MASK_CACHE_JOB})"
echo "  slides: ${DATASETS} ${N_WSI:+(n $N_WSI each)} ${WSI}   rungs: ${DS:-DsLadder default}"
echo ""

python training/SuperPathPoint/cli/extract_pretiles.py \
  --tile "$TILE" \
  --n "$N" \
  --pretile-cache-job "$CACHE_JOB" \
  --seg "$SEG" \
  --mask-cache-job "$MASK_CACHE_JOB" \
  --inherit-share "$INHERIT_SHARE" \
  ${INHERIT_SOURCE_RUNG:+--inherit-source-rung "$INHERIT_SOURCE_RUNG"} \
  --bucket-frame "$BUCKET_FRAME" \
  --step "$GRID_STEP" \
  --max-overlap "$MAX_OVERLAP" \
  --overlapping-share "$OVERLAPPING_SHARE" \
  ${DS:+--rungs $DS} \
  ${WSI:+--wsi $WSI} \
  ${DATASETS:+--datasets $DATASETS} \
  ${N_WSI:+--n-wsi $N_WSI} \
  --max-tries "$MAX_TRIES"

status=$?

echo ""
echo "======== done  (exit $status) ========"
echo "  pre-tiles -> result/cache/${CACHE_JOB}/slide=<s>/.../draw=<sampler>/pretile=f3/ds=<d>/tiles/"
echo "             (the corpus key is printed at the top of the log)"
echo "  table     -> result/\${SLURM_JOB_NAME}/extract_pretiles.csv"
echo ""
echo "  Two numbers to read before anything else:"
echo "    n_got against n_requested per rung -- the coarse rungs are where the"
echo "      rejection sampler runs out, and the gap decides the rung balance"
echo "      switch (align-min or loss-weight, spec.md 6.5)."
echo "    the printed PNG-against-raw ratio -- it is what says whether 512 and"
echo "      1024 fit on disk in this shape."

exit $status
