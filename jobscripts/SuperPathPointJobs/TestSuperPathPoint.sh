#!/bin/bash
#SBATCH --job-name=TestSuperPathPoint     # -> log/%x, result/%x/
#SBATCH --partition=normal2               # Partition
#SBATCH --time=01:00:00                   # every stage here is seconds; the WSI
#SBATCH --account=MST114560               # reads are the only slow part
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gpus-per-node=1                 # GPUs per node (不要設0)
#SBATCH --cpus-per-task=2                 # CPU cores per task
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
#  SuperPathPoint -- the assertions in front of spec.md section 12 steps 1-4
# =============================================================================
#
# NO WEIGHTS EXCEPT `seg` AND `backbone-model`. Every other stage makes up its
# own arrays -- fake pyramids, fake slides, a fake detector, a fake ViT -- and
# finishes in seconds. The GPU line above is there because those two load a
# model and because this cluster wants one, not because the assertions need it.
#
# That is the point (spec.md 12): these are the cheapest unknowns, and the errors
# they catch are the ones that produce a plausible-looking wrong answer rather
# than a crash. Every stage but those two is seconds, and those two are minutes
# -- while the runs they stand in front of are hours and tens of GB.
#
#   homography   a matrix applied in the reverse direction gives a perfectly
#                reasonable warped image. A half-pixel error in the grid_sample
#                normalisation gives an image nobody can tell from the right one
#                and keypoints that are all offset. Neither raises.
#   ds_ladder    picking a level COARSER than the requested rung still returns
#                an image of the right size. What is in it is interpolation
#                texture, and a keypoint detector will learn to fire on it.
#   pre-tile     a centre crop off by one pixel shifts every training image
#                against every label. Training still converges; the model is
#                just slightly worse, and nothing says why.
#   from_mask    mask_ds is DERIVED from the mask's shape against the span it
#                covers. Pair a cropped mask with the full canvas instead and
#                every position is 0.09 percent off -- 55 level-0 px by the far
#                edge, which is invisible until a region boundary lands on the
#                wrong side of a tile.
#   backbone     a trunk's stride taken from model_spec.feat_hw is the grid it
#                produces at INFERENCE, after a resize and a centre crop the
#                training path does not do. 14 instead of 16, both plausible,
#                and the cell grid ends up a different size than the labels
#                were splatted onto.
#   decoder      the two permutes of depth-to-space in the wrong order
#                transpose every keypoint inside its own 8x8 cell. Right shape,
#                right count, plausible positions, every label 0-7 px off.
#   ha           projecting the warped probability back with H instead of
#                H_inv gives a smooth, plausible map offset by each homography,
#                and averaging over N draws partly cancels the offsets -- so it
#                reads as a blurry label rather than as a bug.
#   student      the loss splats labels into cells and the model decodes
#                predictions out of them. If the two disagree about which
#                channel is which pixel, every label is transposed inside its
#                own cell -- both halves self-consistent, loss falling.
#   stores       an identity field that stopped separating files means a
#                rebuild overwrites something a run has already read; a field
#                that started separating them means six slides of GPU time is
#                rebuilt because a mount point changed.
#
# So the checks that carry weight in every one of these files score against a
# DELIBERATELY WRONG alternative, not against a tolerance -- the discipline that
# found the R(-theta) bug in Camera.output_to_level0, where the first version
# passed at 0 and 180 degrees and lost to a point-reflected candidate 40/40
# times at 90 and 270. A margin over a decoy is robust; a threshold is a guess.
#
# -----------------------------------------------------------------------------
#  STAGES picks a subset; the default is all of them.
# -----------------------------------------------------------------------------
#
#   homography   test_homography.py            geometry, both warp paths
#   ladder       test_ds_ladder.py             level resolution, synthetic
#   mask         test_tissues_regions_mask.py  from_mask: the half of from_wsi
#                --wsi ''                      that Uni2PcaSegFunc and MaskStore
#                                              use, plus has_tissue at 0.5 --
#                                              query_sim's gate, not 3c's
#   store        test_mask_store.py            the three stores: identity,
#                test_pre_tile_store.py        round trips, and the refusals
#                test_keypoint_label_store.py  that keep a reader from getting
#                                              the WRONG artefact rather than
#                                              none
#   decoder      test_detector_decoder.py      depth-to-space placement, against
#                                              the transposed convention
#   ha           test_homographic_adaptation   H_inv vs H, and what the 3x
#                .py                           pre-tile bought
#   student      test_superpathpoint.py        files 11-19: the cell round
#                                              trip, the warp direction, the
#                                              descriptor alignment, and one
#                                              batch overfitted
#   backbone     test_encoder_backbone.py      the foundation-model trunk as a
#                                              Backbone: stride from the patch
#                                              and not from the inference grid,
#                                              and the two refusals. No weights
#   backbone-model  test_encoder_backbone.py   the same, on real weights, all
#                --with-model                  three encoders at the size their
#                                              own patch grid divides. Only the
#                                              real model can say what its patch
#                                              size and its input size are
#   ladder-wsi   test_ds_ladder.py --wsi       the plan for a real 4x and a
#                                              real 2x pyramid, printed
#   demo         demo_homography.py            the figure, synthetic + a WSI tile
#   seg          test_uni2_pca_seg.py          step 3a's segmenter. Loads UNI2,
#                                              and with backbone-model one of
#                                              the only two stages costing
#                                              minutes rather than seconds
#
#   sbatch jobscripts/TestSuperPathPoint.sh                     # all twelve
#   STAGES="homography ladder" sbatch jobscripts/TestSuperPathPoint.sh
#
# NOTHING HERE RUNS ON A LOGIN NODE, including the stages that need no data,
# no weights and no GPU. That is a standing rule (ClaudeRules section 3), not a
# property of these tests: a login node is shared, and "it only takes seconds"
# is exactly what every job that ends up hogging one was believed to be. The
# subset is selected with STAGES and still submitted:
#
#   STAGES="mask backbone" sbatch jobscripts/TestSuperPathPoint.sh
#
# Everything shares one log and one result directory, both keyed on
# SLURM_JOB_NAME, so --job-name moves the pair together:
#
#   STAGES=demo sbatch --job-name=TestSuperPathPoint_demo \
#       jobscripts/TestSuperPathPoint.sh

STAGES="${STAGES:-homography ladder mask sampler store decoder ha student pretrained pretrained-weights reeval survival survival-report backbone backbone-model ladder-wsi demo seg}"

# Every test this jobscript owns lives in one directory, named after the
# jobscript that runs them. The two that do NOT are deliberate: their subjects
# are shared modules, not SuperPathPoint's -- `TissuesRegionsMask` is used by
# LocaScopePipeline and PatchingLib, and `Uni2PcaSegFunc` is an encoder.
TESTS=utilities/test_modules/TestSuperPathPoint

BRACS=/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test
KI67=/work/u26130998/datasets/Ki67

# One of each pyramid shape, because that is the whole point of the ladder.
# Measured in spec.md 6.5: BRACS steps 4x and has four levels (ds 1, 4, 16, 32),
# Ki67 steps 2x and has ten. The rungs ds 2 and ds 8 exist natively on one and
# not on the other, and getting them by reading a COARSER level would be the
# silent failure -- ladder-wsi is where a human sees which level each rung
# actually resolved to.
WSI_4X="$BRACS/Group_AT/Type_ADH/BRACS_1228.svs"
WSI_2X="$KI67/S1104233,G7E,110208.mrxs"

# The demo's tile. ds 4 is native on both pyramids, so the picture is the
# homography and not a resampling artefact.
DEMO_DS=4
TILE=256

# How many homographies the statistical checks draw. 200 is enough for the
# allow_artifacts check to see the no-op appear at its ~1/26 rate; lower it and
# that check starts failing for want of samples rather than for a bug.
DRAWS=200

# How many homographies --calibrate draws. Each one costs a 768x768 nearest
# warp, so 2000 is a few seconds. It is the tail that matters here -- the
# assertion is on the MAXIMUM, not on a percentile -- so more draws is a
# strictly stronger check.
CALIBRATE=2000

# The seg stage reads at level 0 and has no magnification knob -- see
# Uni2PcaSegFunc.LEVEL. plane_ds was a config field, then a level argument, and
# both are gone; a --plane-ds here is what made the 2026-08-26 run exit 1.
# 200 rather than the config default of 1000, so the fit is minutes. It is a
# hashed field, so a run at 200 and a run at 1000 correctly get different ids.
SEG_FIT_TILES=200
# The tiling check reads a 4x4-tile square and splits it into quadrants. Must
# be even. Larger makes the decoy stronger -- a bigger plane has more for a
# per-quadrant refit to disagree about -- and costs one forward per extra tile.
SEG_PLANE_TILES=4

status=0
run () {   # run <label> <command...>
  echo ""
  echo "======== $1 ========"
  shift
  "$@" || status=1
}

echo "======== TestSuperPathPoint  stages: $STAGES ========"

for stage in $STAGES; do
  case "$stage" in

    homography)
      # Sections: identity / direction / torch / sampling. The torch section
      # skips itself with a printed line if torch is not importable, so this
      # stage is still meaningful in a bare environment.
      run "homography  (geometry, both warp paths, ${DRAWS} draws)" \
        python "$TESTS"/test_homography.py --draws "$DRAWS"

      # --calibrate lives here and not in `demo`, because it is an ASSERTION
      # and not a figure: it draws N homographies, asks each how much
      # pre-tile it needed, and fails if any one exceeds PRE_TILE_FACTOR.
      #
      # There is no calibration to do -- the worst case is derived (0.85 ->
      # 1.25 -> 1.50 -> 2.12 -> 2.49, and 3 is that plus slack for the one
      # approximation in it). This turns the derivation into a measured
      # statement for seconds, and it guards a silent failure: a draw that
      # runs off the pre-tile leaves a black wedge, and a black wedge is a
      # straight maximum-contrast edge with two right angles -- what a corner
      # detector fires on. The label would be confident and wrong.
      #
      # It prints TWO numbers that must agree: the analytic factor (the output
      # frame's corners pushed back through the matrix) and whether the warp
      # actually ran off the pre-tile (an all-ones array through the same
      # composed matrix). Different lines, different quantities, so agreement
      # is evidence rather than a tautology.
      run "homography  (--calibrate: is PRE_TILE_FACTOR 3 enough, ${CALIBRATE} draws)" \
        python training/SuperPathPoint/cli/demo_homography.py \
          --calibrate "$CALIBRATE" --tile-size "$TILE"
      ;;

    ladder)
      # Runs against the two pyramid shapes written out as plain lists, so it
      # needs no slide and cannot be broken by one being unreadable.
      run "ladder  (level resolution, synthetic pyramids)" \
        python "$TESTS"/test_ds_ladder.py
      ;;

    mask)
      # from_mask, which is everything from_wsi does AFTER a mask exists.
      # aiNNModel/Uni2PcaSegFunc.py reads the slide itself and hands the
      # finished mask here; utilities/MaskStore.py does the same coming back
      # out of the cache; probe_tile_yield and extract_pretiles are its two
      # readers. Every other validator in that file builds its fixture through
      # make_trm, which calls _search_tissue_regions directly and therefore
      # never exercises the three things from_mask decides -- what mask_ds is,
      # where the mask starts, and what the slide contributes.
      #
      # --wsi '' skips the real-slide tier. The synthetic tiers are the ones
      # that carry the assertions; the slide tier prints a figure.
      #
      # `has_tissue` at 0.5 stays under test even though the SAMPLER no
      # longer calls it: `TissuesRegionsMask` is shared, query_sim's Camera
      # still gates on it at 0.3, and a method that keeps a caller keeps its
      # test. What went away on 2026-08-27 is the sampler's OWN gate, which
      # scored the same quantity as the richness buckets -- see RichnessConfig.
      # Exactly half passes here, because has_tissue compares with >=.
      run "mask  (from_mask: derived ds, the span decoy, the 0.5 gate)" \
        python utilities/test_modules/test_tissues_regions_mask.py --wsi ''
      ;;

    sampler)
      # utilities/TileSampler.py, rewritten in place. Three axes -- richness,
      # overlap, inheritance -- set when sampling and queried afterwards,
      # because those are the same information read twice.
      #
      # The one it exists for is the lattice step. It converts OUTPUT pixels to
      # level-0 px, and the conversion is footprint/tile: that equals ds on an
      # 'F' rung and 1 on an 'R' one. Written as `grid_step * ds` it is right
      # for F and 32x too coarse for R at ds 32 -- the rung comes back with a
      # handful of candidates and reads as "there was no tissue at that
      # magnification".
      #
      # The overlap check is scored against the RANDOM ARM -- the sampler this
      # replaced, kept as a control -- and not against zero. That arm produced
      # 202,420 overlapping pairs over the real corpus, 69.2 per cent of tiles
      # touching another and every single tile at ds 32.
      run "sampler  (the three axes, the units, and the random-arm decoy)" \
        python utilities/test_modules/test_tile_sampler.py
      ;;

    store)
      # The three stores, in the order the pipeline fills them. All temp dirs,
      # no data, no model -- and the first of them is a regression test for a
      # bug that had already written six correct mask files before the probe
      # that reads them raised on `f'{meta.mask_ds:.0f}'`: lazy annotations make
      # `field.type` a STRING, so a decoder comparing it against `float` hands
      # every field back as str.
      run "store  (mask store: identity, components, the typed round trip)" \
        python "$TESTS"/test_mask_store.py
      run "store  (pre-tile store: centre crop, identity, codec)" \
        python "$TESTS"/test_pre_tile_store.py
      run "store  (label store: threshold/NMS/border/cap, padding, two rounds)" \
        python "$TESTS"/test_keypoint_label_store.py
      ;;

    decoder)
      # depth-to-space. The two permutes in the wrong order transpose every
      # keypoint inside its own 8x8 cell: right shape, right count, plausible
      # positions, every label 0-7 px off in a pattern that depends on where it
      # is. Scored against the transposed convention as an explicit decoy.
      run "decoder  (depth-to-space placement and the dustbin)" \
        python "$TESTS"/test_detector_decoder.py
      ;;

    ha)
      # Homographic Adaptation with a fake detector that answers "wherever this
      # view is brightest", so the aggregate has a known truth: a dot planted at
      # a known place has to come back to it.
      #
      # The failure it exists for is projecting with H instead of H_inv. That
      # gives a smooth, plausible map offset by each homography, and averaging
      # over N draws partly cancels the offsets -- so it reads as a slightly
      # blurry label rather than as a bug. Same shape of error as the
      # R(-theta) one in Camera.output_to_level0.
      #
      # It also pins what the 3x pre-tile bought, as a pair: the pre-tile mask
      # is full inside the eroded rim while the tile-sized mask is not, and the
      # gap between the two IS the measurement.
      run "ha  (direction, the pre-tile masks, counts, identity)" \
        python "$TESTS"/test_homographic_adaptation.py
      ;;

    student)
      # Files 11-19 of spec.md 14 -- Interfaces, Backbones, Decoders, Heads,
      # KeypointNet, Losses, Datasets, Trainer -- in one file, because what is
      # being tested is the CONTRACT BETWEEN them and eight files would each
      # hold half of a statement.
      #
      # The one it exists for is `space_to_depth` against `depth_to_space_prob`:
      # the loss splats labels into cells with one and the model decodes
      # predictions out of cells with the other, and if the two disagree about
      # which channel is which pixel, every label is transposed inside its own
      # cell against every prediction. Both halves stay self-consistent, the
      # loss falls, and nothing raises.
      #
      # The last section overfits one batch for 20 steps. Every other check here
      # is local -- shapes, directions, one function against another -- and none
      # of them would notice a loss that is computed correctly and connected to
      # nothing.
      run "student  (11-19: the cell round trip, the warp direction, one batch)" \
        python "$TESTS"/test_superpathpoint.py
      ;;

    pretrained)
      # spec.md 13: this one BLOCKS AN ARM. `--pretrained` initialises the
      # student from upstream SuperPoint v6, and its failure mode produces a
      # NUMBER rather than an error -- under strict=False the keys that did not
      # match stay random, the network trains, the loss falls, and the run
      # reads as "pretraining did not help" instead of "pretraining did not
      # happen". A trunk-only transfer is still far better than random, so even
      # a rename that misses two of the three groups looks like it worked.
      #
      # Every equality here is scored against a decoy, because the specific way
      # this test could pass vacuously is both sides being empty or unchanged:
      # the RGB check has the no-/3 arm, --with-weights has a shuffled trunk
      # layer, and one check simply asserts that loading moved a weight at all.
      run "pretrained  (the rename, repeat-and-divide, strict)" \
        python "$TESTS"/test_pretrained_init.py
      ;;

    pretrained-weights)
      # THE ONE THAT ACTUALLY GATES THE `_pre` ARMS. Everything in `pretrained`
      # runs on a fake state dict, which proves the rename and the arithmetic
      # and nothing about sp_v6 -- the two models could still be two models.
      #
      # This loads the real weights into a student and compares its `prob_map`
      # with the teacher's `dense_prob` elementwise. The two run through
      # completely different code (this repo's parts against the upstream
      # module), so agreement to 1e-4 is evidence and not a tautology.
      #
      # Scored against a decoy -- one trunk layer shuffled -- because the way
      # this could pass vacuously is both sides being degenerate. The fake
      # version already did exactly that once: `gap 0, decoy 1`, two exact
      # integers a float32 network cannot produce, because randn weights had
      # saturated the softmax to one-hot.
      #
      # Needs sp_v6 (5 MB) and one forward. Seconds, no GPU required.
      run "pretrained-weights  (sp_v6 into a student equals the teacher)" \
        python "$TESTS"/test_pretrained_init.py --with-weights --only fixture
      ;;

    reeval)
      # GATES ReEvalSuperPathPoint.sh, and it exists because the thing it
      # checks already went wrong once at full size. The 2026-08-31 training
      # run printed margins of 1.50 and 3.59 for two arms and they were not
      # comparable: `margin <= 1/decoy`, the decoy rises with point density,
      # and the two arms were scored at 420 and 159 points per view. Nothing
      # errored. The table looked like a result.
      #
      # The re-eval replaces the threshold with a fixed budget, so the three
      # claims worth blocking a rerun over are:
      #   budget   the cut returns EXACTLY N -- that IS "matched density", and
      #            a short column is a scale change with no error to say so
      #   uniform  1 - exp(-(2r+1)^2 N / tile^2), the arithmetic the whole
      #            diagnosis rests on, against two wrong box sizes rather than
      #            against a tolerance
      #   rebuild  the net is reconstructed from the checkpoint's SHAPES, and
      #            `_cell_of` reading the first detector conv instead of the
      #            last would raise nothing and rebuild the wrong grid
      #
      # numpy and CPU tensors. Seconds, no GPU, no checkpoint required.
      run "reeval  (the budget binds, the decoy formula, the rebuild)" \
        python "$TESTS"/test_reeval_density.py
      ;;

    survival)
      # spec.md 3.2, and it gates Stage B for the same reason `reeval` gates
      # the re-scoring: the two numbers Stage B exists to produce come out of
      # code that CANNOT FAIL LOUDLY.
      #
      #   `Patterns`      a classifier that calls [1,0,1] a contiguous band
      #                   raises nothing, and the band fraction is what
      #                   spec.md 3.3 reads to decide whether Stage C's head
      #                   can be two outputs. 0.97 says yes; 0.6 says the
      #                   simplification is a silent error.
      #   `Attribution`   a branch that reads the wrong column still returns a
      #                   label. Every branch here is scored against a decoy
      #                   that must NOT trigger it.
      #   `rung_scale`    level-0 px per output pixel is `ds` on the F axis and
      #                   1.0 on the R axis. Using `ds` for R scatters every
      #                   coarse point `ds` times too far from the centre, the
      #                   table fills, and the survival numbers become a
      #                   picture of the bug.
      #
      # numpy and a temp directory. Seconds, no GPU, no store, no slide --
      # which is the property the pure modules exist to have.
      run "survival  (the six patterns, the four causes, scale vs shrink)" \
        python "$TESTS"/test_survival.py
      ;;

    survival-report)
      # plan.md P1: the arithmetic that turns the table into the three numbers
      # Stage C's design turns on -- the band fraction, the late-born fraction
      # and the one-rung-only fraction.
      #
      # ALL THREE ARE RATIOS, and a ratio is the shape that fails silently: a
      # denominator counted over the wrong set, a null that conditions on
      # something the measurement does not, a merge that drops rows. Nothing
      # raises; the number just means something other than it says. So every
      # test is either hand-computable or a decoy that must NOT be produced --
      # `p**L` against the unconditioned `p**L`, radius 0 keeping every row,
      # the axes handed over the wrong way round.
      #
      # It also pins the fix that made the store re-cuttable: the anchor merge
      # radius is `nms_radius` level-0 px and NOT tau, so a tau sweep is a
      # re-read instead of another GPU run.
      #
      # numpy only. Seconds, no GPU, no store, no slide.
      run "survival-report  (the null, the merge, the sweep, the tau curve)" \
        python "$TESTS"/test_survival_report.py
      ;;

    backbone)
      # spec.md 5.3 step 8: the foundation-model trunk as a Backbone, with a
      # fake trunk instead of weights. A fake is not a weaker test here, it is
      # a stronger one -- everything this file decides is about NUMBERS, and a
      # fake can be told to lie about them on purpose where a real ViT cannot.
      #
      # The one it exists for: stride taken from model_spec.feat_hw instead of
      # from patch_size. feat_hw is crop_size // patch_size = 224 // 16 = 14 --
      # the grid the encoder produces at INFERENCE, after the resize and the
      # centre crop this backbone deliberately does not do. The real answer at
      # tile 256 is 16. Both are plausible, both are self-consistent, and the
      # wrong one leaves the cell grid a different size than the labels were
      # splatted onto.
      run "backbone  (trunk stride, the refusals, the frozen parameters)" \
        python "$TESTS"/test_encoder_backbone.py
      ;;

    backbone-model)
      # The one thing the fake cannot have: a position embedding that actually
      # has to interpolate. dynamic_img_size is a flag a fake can set; whether
      # timm then produces a 16x16 map from a 256 px input is a fact about the
      # real model.
      #
      # All three, each at a size its own patch grid divides. The table this
      # is checking, and every number in it is read off the model rather than
      # off its name:
      #
      #   gigapath   patch 16, input size FIXED at 224   -> tile 256
      #   uni2       patch 14, dynamic_img_size          -> tile 224
      #   conch_vit  patch 16, dynamic_img_size, 448     -> tile 256
      #
      # gigapath is the one to read first. Its arch is called
      # vit_giant_patch14_dinov2 and prov-gigapath's config.json then sets
      # model_args.patch_size to 16 -- the name says 14, the model is 16 -- and
      # its pretrained_cfg says fixed_input_size, so it asserts on anything but
      # 224 until set_input_size resamples its position embedding. Both facts
      # are only visible on the real weights, which is what this stage is for.
      #
      # uni2 at 224 is the patch-14 arm: 224 = 14 x 16, and 256 is not a
      # multiple of 14 at all. It checks the BACKBONE only. uni2 cannot be a
      # trunk for this project as things stand -- the decoder has to climb
      # patch/cell = 14/8 = 1.75, which is not a power of two and which no tile
      # size repairs because the tile cancels out of that ratio. The three
      # things that would are written out in EncoderBackbone.py and none is
      # chosen, because the patch-16 encoders need none of them.
      run "backbone-model  gigapath at tile $TILE  (fixed input size, patch 16)" \
        python "$TESTS"/test_encoder_backbone.py \
          --with-model --encoder gigapath --tile-size "$TILE"
      run "backbone-model  uni2 at tile 224  (patch 14, backbone only)" \
        python "$TESTS"/test_encoder_backbone.py \
          --with-model --encoder uni2 --tile-size 224
      run "backbone-model  conch_vit at tile $TILE" \
        python "$TESTS"/test_encoder_backbone.py \
          --with-model --encoder conch_vit --tile-size "$TILE"
      ;;

    ladder-wsi)
      # Assertions plus a printed plan. The print is the deliverable here: it
      # is where you read which level each rung landed on, and whether ds 2 and
      # ds 8 on the 4x pyramid are being reached by shrinking level 0 and level
      # 1 rather than by upsampling something coarser.
      run "ladder-wsi  4x pyramid  $(basename "$WSI_4X")" \
        python "$TESTS"/test_ds_ladder.py \
          --wsi "$WSI_4X" --tile-size "$TILE"
      run "ladder-wsi  2x pyramid  $(basename "$WSI_2X")" \
        python "$TESTS"/test_ds_ladder.py \
          --wsi "$WSI_2X" --tile-size "$TILE"
      ;;

    demo)
      # Two figures to ONE filename would overwrite, so the synthetic one is
      # redirected with --out. Both write under result/$SLURM_JOB_NAME/.
      #
      # Two panels are the ones to look at:
      #   "all off"           must be pixel-identical to "original". patch_ratio
      #                       reads like a crop and is not one.
      #   "point-warp check"  the green crosses must sit on the red dots. If they
      #                       are displaced, the SHAPE of the displacement says
      #                       which error it is -- a mirror-like scatter is the
      #                       reversed direction, a uniform drift is the
      #                       half-pixel normalisation.
      OUT_DIR="/work/u26130998/result/${SLURM_JOB_NAME:-TestSuperPathPoint}"
      mkdir -p "$OUT_DIR"
      run "demo  (synthetic pattern)" \
        python training/SuperPathPoint/cli/demo_homography.py \
          --tile-size "$TILE" \
          --out "$OUT_DIR/homography_operations__synthetic.png"
      run "demo  (WSI tile, ds $DEMO_DS)" \
        python training/SuperPathPoint/cli/demo_homography.py \
          --wsi "$WSI_4X" --ds "$DEMO_DS" --tile-size "$TILE" \
          --out "$OUT_DIR/homography_operations__bracs_ds${DEMO_DS}.png"
      ;;

    seg)
      # Step 3a: aiNNModel/Uni2PcaSegFunc.py. Three tiers in one file, and this
      # runs all three -- the config/helper checks need nothing, --with-model
      # loads UNI2, --wsi fits on a slide.
      #
      # The assertion that justifies the whole design is in the last tier:
      # AFTER fit(wsi), segmenting a plane in one call and in four quadrants
      # must agree pixel for pixel. That is the property from_wsi already names
      # as the line between methods it may tile and methods it may not, and it
      # holds only because the PCA is fitted in fit() rather than in __call__.
      # It is scored against the rejected design -- refit per quadrant -- so
      # what comes out is a margin, not a tolerance.
      #
      # Also PRINTS fit_report and asserts nothing about whether the mask is
      # right, because there is no tissue ground truth here and a threshold on
      # the foreground fraction would be a guess (ClaudeRules section 8). Read
      # the printed fraction against the measured tissue: BRACS 23-38 percent,
      # Ki67 3.5-9. Near 0.5 on a Ki67 slide means PC1 found something other
      # than tissue.
      #
      # FIT_TILES is 200, not the config default of 1000, so this is minutes.
      # The fit sample size changes the basis and is hashed, so a test run and
      # a production run correctly get different identity ids.
      run "seg  (config + helpers, no weights)" \
        python utilities/test_modules/test_uni2_pca_seg.py
      run "seg  (+model +slide, level 0, $(basename "$WSI_4X"))" \
        python utilities/test_modules/test_uni2_pca_seg.py \
          --wsi "$WSI_4X" \
          --fit-tiles "$SEG_FIT_TILES" \
          --plane-tiles "$SEG_PLANE_TILES"
      ;;

    *)
      echo "unknown stage: $stage   (known: homography ladder mask sampler store decoder ha student backbone backbone-model ladder-wsi demo seg)"
      status=1
      ;;
  esac
done

echo ""
echo "======== done  (exit $status) ========"
echo "  figures -> result/\${SLURM_JOB_NAME}/homography_operations__*.png"
echo ""
echo "  Two numbers in the output are worth more than the pass/fail:"
echo "    homography  'image warp matches point warp' prints how many times"
echo "                closer the correct direction is than the reversed decoy."
echo "    seg         'tiling is invariant' prints 0 differing pixels for the"
echo "                pre-fitted basis and a percentage for the per-quadrant"
echo "                refit. The second number is the measurement of how wrong"
echo "                a lazily-fitting __call__ would have been."
echo ""
echo "  And one that is neither: seg prints fit_report's foreground fraction,"
echo "  which nothing asserts. Read it against the measured tissue fractions"
echo "  -- BRACS 23-38 percent, Ki67 3.5-9 -- and if it sits near 0.5 the PCA"
echo "  found position or scanner banding rather than tissue."

exit $status
