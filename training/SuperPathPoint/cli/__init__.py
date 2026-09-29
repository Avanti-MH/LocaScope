"""Shared setup for SuperPathPoint CLI entry points.

Re-exports `_paths` rather than deriving the output root again, for the reason
spelled out at length in `query_sim/cli/__init__.py`: that package once carried
its own copy of the rule, the two drifted, and three of its four files ended up
using a different definition from the fourth. `_paths` lives in `utilities/`,
the library layer every package here already depends on, so this import points
DOWN rather than sideways at a sibling.

What this package owns is its default job names -- SuperPathPointDemo and the
rest -- which are arguments to `job_result_dir`, not a second copy of it.

Entry points must put `training/SuperPathPoint/` on sys.path themselves before
`from cli import job_result_dir`, since this package lives one level under it.
`setup_import_paths()` does that for the module tree; the two lines below do it
for `_paths` itself, which is the one import that cannot be bootstrapped by the
thing it bootstraps.

`setup_import_paths` here is NOT `_paths.setup_import_paths` re-exported
unchanged (2026-09-22): that function no longer puts any training
package's own directory on `sys.path` at all -- see its own docstring for
why (a bare `from Runtime import ...` used to resolve to whichever
training package `sys.path`'s own order favoured, not necessarily the
caller's own). This wrapper calls `_paths.add_training_package(
'SuperPathPoint')` right after, since every file that reaches
`setup_import_paths` THROUGH this module IS a SuperPathPoint entry point --
this package is the one place that fact is already known, so callers going
through here (`from cli import job_result_dir, setup_import_paths`) get it
automatically. A file that imports `_paths.setup_import_paths` directly
instead still has to call `add_training_package('SuperPathPoint')` itself.
"""

import os
import sys

_UTILITIES = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)

import _paths                                                       # noqa: E402
from _paths import LOG_DIR, OUTPUT_ROOT, RESULT_DIR, job_result_dir  # noqa: E402,F401


def setup_import_paths() -> None:
    _paths.setup_import_paths()
    _paths.add_training_package('SuperPathPoint')


# ── the pre-tile corpora every entry point here reads ──────────────────────
#
# ONE ROOT FOR ALL OF THEM. `made_by` is the job that produced a cache, and
# every SuperPathPoint corpus -- stage A from extract_pretiles, F's and C's
# own from prepare_chain_stack, which calls the same writer -- is produced by
# the one extraction code path. So they share `<PRETILE_JOB>_pretiles/`, the
# corpus key tells them apart below it, and a reader names another job's root
# only through `--pretile-cache-job`.
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
                    help='the job that made the pre-tiles: result/cache/'
                         '<this>_pretiles/')
    ap.add_argument('--mask-cache-job', default=MASK_JOB,
                    help='the job that made the masks: result/cache/<this>_mask/')
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default=PRETILE_SEG,
                    help='mask recipe the pre-tiles were cut through')
    ap.add_argument('--pre-tile-factor', type=int, default=PRE_TILE_FACTOR,
                    help='pre-tile side / tile side. An identity field: a '
                         'different value is a different corpus')
    if tile:
        ap.add_argument('--tile', type=int, default=256)


def pretile_root(args):
    from Cache import cache_root                                   # noqa: PLC0415
    return cache_root(args.pretile_cache_job, 'pretiles')


def mask_root(args):
    from Cache import cache_root                                   # noqa: PLC0415
    return cache_root(args.mask_cache_job, 'mask')


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
