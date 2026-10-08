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

from typing import List, Optional

from _paths import LOG_DIR, OUTPUT_ROOT, RESULT_DIR, job_result_dir  # noqa: F401


# ── the corpora every entry point here reads ─────────────────────────────────
#
# ONE JOB FOR ALL THE DRAWS. `made_by` is the job that produced a cache, and
# whichever entry point first reads a corpus of a slide draws it -- so every
# one of them defaults to CORPUS_JOB's tree, the corpus address tells stage A
# and stage B apart in it (common/Corpora.Corpus), and a reader names another
# job only through `--draw-cache-job`.
CORPUS_JOB = 'SuperPathPointCorpus'
#: The mask SuperPathPoint's corpora are drawn through, and who made it.
CORPUS_SEG = 'uni2_pca'
MASK_JOB = 'BuildMaskStore'


def add_corpus_args(ap, *, tile: bool = True, corpus: Optional[str] = 'stageA'
                    ) -> None:
    """Everything a corpus needs besides its slides: `--corpus` (a recipe of
    common/Corpora.RECIPES; omitted when `corpus` is False), `--corpus-rungs`,
    `--draw-cache-job`, `--mask-cache-job`, `--seg`, `--pre-tile-factor`, and
    `--tile` unless the caller owns it."""
    from TileSampler import PRE_TILE_FACTOR                        # noqa: PLC0415
    from TissueMaskConfig import MASK_RECIPES                      # noqa: PLC0415
    if corpus is not False:
        from common.Corpora import RECIPES                         # noqa: PLC0415
        ap.add_argument('--corpus', default=corpus, choices=sorted(RECIPES),
                        help='the corpus recipe (common/Corpora.RECIPES)')
        ap.add_argument('--corpus-rungs', type=float, nargs='+', default=None,
                        help="the rungs the corpus is drawn over -- part of "
                             "its address. Default: DsLadder's")
    ap.add_argument('--draw-cache-job', default=CORPUS_JOB,
                    help='the job whose cache tree holds the draws')
    ap.add_argument('--mask-cache-job', default=MASK_JOB,
                    help='the job whose cache holds the masks: result/cache/<this>/')
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default=CORPUS_SEG,
                    help='mask recipe the corpus is drawn through')
    ap.add_argument('--pre-tile-factor', type=int, default=PRE_TILE_FACTOR,
                    help='pre-tile side / tile side. An identity field: a '
                         'different value is a different corpus')
    if tile:
        ap.add_argument('--tile', type=int, default=256)


def corpus_from_args(args, name: str, rungs=None):
    """Recipe `name`'s corpus under the args' jobs, mask and factor."""
    from common.Corpora import corpus_of                           # noqa: PLC0415
    from TissueMaskConfig import MASK_RECIPES                      # noqa: PLC0415
    return corpus_of(name, MASK_RECIPES[args.seg], tile=args.tile,
                     job=args.draw_cache_job, mask_job=args.mask_cache_job,
                     rungs=rungs, factor=args.pre_tile_factor)


def corpus_arg(args):
    """`--corpus` over `--corpus-rungs`, as a `common.Corpora.Corpus`."""
    return corpus_from_args(args, args.corpus, args.corpus_rungs)


def add_slides_args(ap) -> None:
    """`--datasets` / `--n-wsi` / `--wsi`: which slides, named the way the
    routing heads name theirs. Nothing searches a cache for slides."""
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='the slides of these datasets: AccessDatasets ids, '
                         '<id>#<split> for a recorded split')
    ap.add_argument('--n-wsi', type=int, default=None,
                    help='slides per dataset, a seeded pick; default every one')
    ap.add_argument('--split-cache-job', default=None,
                    help='whose recorded split a <id>#<split> dataset reads. '
                         'Default: MakeSplit')
    ap.add_argument('--wsi', nargs='+', default=None,
                    help='slide paths or AccessDatasets names, instead of or '
                         'beside --datasets')


def slides_from_args(args, seed: int = 0) -> List[str]:
    """The slide paths `add_slides_args` names, datasets first."""
    from AccessDatasets import list_names, locate, pick_wsi_names  # noqa: PLC0415
    from common.Corpora import Corpus                              # noqa: PLC0415
    paths = []
    for dataset in args.datasets or []:
        split_job = args.split_cache_job if '#' in dataset else None
        names = pick_wsi_names(list_names(dataset=dataset, split_job=split_job),
                               args.n_wsi, seed)
        paths += [str(locate(n, dataset=dataset, split_job=split_job).path)
                  for n in names]
    paths += [Corpus.path_of(w) for w in args.wsi or []]
    if not paths:
        raise SystemExit('name the slides: --datasets (an id or <id>#<split>) '
                         'or --wsi')
    return paths


# ── the two caches made from those corpora ───────────────────────────────────
#
# In the producer's tree like every other cache: the producer defaults to its
# own job name, a reader to the producer's, and either is named with the flag.
# The labels sit under the draw they were made on (`KeypointLabelStore`),
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
