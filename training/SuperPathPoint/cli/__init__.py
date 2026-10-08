"""Shared setup for SuperPathPoint CLI entry points.

Re-exports `_paths` rather than deriving the output root again. `_paths` lives
in `utilities/`, the library layer every package here already depends on, so
this import points DOWN rather than sideways at a sibling. What this package
owns is its default job names -- SuperPathPointDemo and the rest -- which are
arguments to `job_result_dir`, not a second copy of it.

An entry point sets the path before importing this package -- utilities/ by
hand, then `_paths.setup_import_paths('SuperPathPoint')`, which puts
`training/SuperPathPoint/` (this package's parent) there too.
"""

from _paths import LOG_DIR, OUTPUT_ROOT, RESULT_DIR, job_result_dir  # noqa: F401


# ── the pre-tile corpora every entry point here reads ──────────────────────
#
# ONE JOB FOR ALL OF THEM. `made_by` is the job that produced a cache, and
# every SuperPathPoint corpus -- stage A from extract_pretiles, F's and C's
# own from prepare_chain_stack, which calls the same writer -- is produced by
# the one extraction code path. So they share PRETILE_JOB's tree, the corpus
# address tells them apart in it (Store.PreTileCorpus), and a reader names
# another job only through `--pretile-cache-job`.
PRETILE_JOB = 'ExtractPreTiles'
#: The mask SuperPathPoint's corpora are drawn through, and who made it.
PRETILE_SEG = 'uni2_pca'
MASK_JOB = 'BuildMaskStore'


def add_pretile_args(ap, *, tile: bool = True) -> None:
    """`--pretile-cache-job`, `--mask-cache-job`, `--seg`, `--pre-tile-factor`
    (and `--tile` unless the caller owns it): everything a corpus address
    needs besides the recipe and the rungs."""
    from TileSampler import PRE_TILE_FACTOR                        # noqa: PLC0415
    from TissueMaskConfig import MASK_RECIPES                      # noqa: PLC0415
    ap.add_argument('--pretile-cache-job', default=PRETILE_JOB,
                    help='the job whose cache tree holds the pre-tiles')
    ap.add_argument('--mask-cache-job', default=MASK_JOB,
                    help='the job whose cache holds the masks: result/cache/<this>/')
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default=PRETILE_SEG,
                    help='mask recipe the pre-tiles were cut through')
    ap.add_argument('--pre-tile-factor', type=int, default=PRE_TILE_FACTOR,
                    help='pre-tile side / tile side. An identity field: a '
                         'different value is a different corpus')
    if tile:
        ap.add_argument('--tile', type=int, default=256)


def pretile_root(args) -> str:
    """The job whose tree holds the pre-tiles: what `PreTileCorpus` takes as
    its root."""
    return args.pretile_cache_job


# ── the two caches made from those corpora ───────────────────────────────────
#
# In the producer's tree like every other cache: the producer defaults to its
# own job name, a reader to the producer's, and either is named with the flag.
# The labels sit beside the rung they were made from (`KeypointLabelStore`),
# the chain-stack tiles under their slide (`ChainStack`).
#: Who makes the keypoint labels: make_ha_labels.py, MakeHaLabels.sh.
LABELS_JOB = 'MakeHaLabels'
#: The chain-stack tile cache has no single producer: whichever entry point
#: reads a C descendant from the slide writes it (survival_alpha_analysis and
#: demo_survival_analysis by default, prepare_chain_stack when asked), so each
#: defaults to its own job name and shares another's with --chainstack-cache-job.


def add_labels_args(ap, *, produces: bool = False) -> None:
    """`--labels-cache-job`: whose tree holds the keypoint labels. The
    producer defaults to its own job name, a reader to LABELS_JOB."""
    from Cache import job_name                                     # noqa: PLC0415
    ap.add_argument('--labels-cache-job',
                    default=job_name(LABELS_JOB) if produces else LABELS_JOB,
                    help='the job whose cache tree holds the keypoint labels')


def labels_root(args) -> str:
    """The job whose tree holds the keypoint labels: what
    `KeypointLabelStore` takes as its root."""
    return args.labels_cache_job


def add_chainstack_args(ap, job: str, *, on: bool) -> None:
    """`--chainstack-cache-job` (default `job`, this entry point's own name):
    whose chain-stack tiles (`slide=<s>/chainstack/` in its tree). `on` keeps the
    entry point's own default -- the analyses cache, prepare_chain_stack does
    not (RStack.from_own says why) -- and the flag flips it."""
    from Cache import job_name                                     # noqa: PLC0415
    ap.add_argument('--chainstack-cache-job', default=job_name(job),
                    help='the job whose chain-stack tile cache is read and '
                         'written')
    if on:
        ap.add_argument('--no-chainstack-cache', dest='chainstack_cache',
                        action='store_false', help='read every tile fresh')
    else:
        ap.add_argument('--chainstack-cache', dest='chainstack_cache',
                        action='store_true', help='cache the tiles read')


def chainstack_root(args):
    """The job whose tree holds the tile cache, or None when it is switched
    off."""
    if not args.chainstack_cache:
        return None
    return args.chainstack_cache_job


def corpus_from_args(args, name: str, rungs=None):
    """Recipe `name`'s corpus under the args' root, mask and factor."""
    from common.Corpora import corpus_of                                # noqa: PLC0415
    from TissueMaskConfig import MASK_RECIPES                      # noqa: PLC0415
    return corpus_of(name, pretile_root(args), MASK_RECIPES[args.seg],
                     tile=args.tile, rungs=rungs, factor=args.pre_tile_factor)


def add_corpus_arg(ap, default: str = 'stageA') -> None:
    """`--corpus` and `--corpus-rungs`: which corpus a reader reads.

    `--corpus` is a recipe name (common/Corpora.RECIPES) or a full corpus key
    as extract_pretiles prints it -- the name for the corpora this package
    defines, the key for anything cut by hand with other knobs. A reader
    never searches the root for something that looks right."""
    ap.add_argument('--corpus', default=default,
                    help='a recipe name (stageA, stageB-fOwn, stageB-cOwn) or '
                         'a corpus key <seg_id>/<region>_<sampler>_<plan>/f<k>')
    ap.add_argument('--corpus-rungs', type=float, nargs='+', default=None,
                    help='the rungs a recipe corpus was cut over -- part of '
                         "its address. Default: DsLadder's. Ignored for a key")


def corpus_arg(args):
    """`--corpus` resolved to a `Store.PreTileCorpus`."""
    from common.Corpora import RECIPES                             # noqa: PLC0415
    from Store import PreTileCorpus                                # noqa: PLC0415
    if args.corpus in RECIPES:
        return corpus_from_args(args, args.corpus, args.corpus_rungs)
    return PreTileCorpus.from_key(pretile_root(args), args.corpus)
