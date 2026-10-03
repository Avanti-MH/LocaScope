"""Where this project writes, and how its packages find each other.

The ONE definition of OUTPUT_ROOT. It lived under utilities/test_modules/ and
was reachable only by scripts in that directory, so everything else either
inserted a test directory into sys.path to reach it -- nine CLI entry points
did -- or derived the rule again. query_sim/cli/__init__.py derived it again,
and said so: keep the two in step. They did not stay in step, and the proof was
inside that same package: three of its four files used the local copy while
diag_camera_skip.py (retired 2026-09-30) inserted utilities/test_modules to import this one.

It sits in utilities/ because that is the library layer every package already
depends on -- utilities/cli, query_sim/cli, test_modules and bench_modules all
point DOWN at it, and none of them points sideways at another. What each of
those legitimately owns is its default job name, which is already the argument
to job_result_dir.

It imports os and sys and nothing else on purpose. Files that are careful about
their startup cost -- analyze_locascope_metrics is one -- can import this
without pulling in torch or openslide.
"""

import os
import sys

UTILITIES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(UTILITIES_DIR, '..'))

#: Where runs write. One level ABOVE the repo, so that outputs are not inside
#: the working tree: an `rm -rf` of the checkout, a `git clean`, or a fresh
#: clone no longer takes 60 GB of results with it, and nothing under `result/`
#: can ever be staged by accident. Override with LOCASCOPE_OUTPUT_ROOT.
OUTPUT_ROOT = os.environ.get(
    'LOCASCOPE_OUTPUT_ROOT', os.path.abspath(os.path.join(PROJECT_ROOT, '..')))
RESULT_DIR = os.path.join(OUTPUT_ROOT, 'result')
LOG_DIR = os.path.join(OUTPUT_ROOT, 'log')
QUERY_SIM_DIR = os.path.join(PROJECT_ROOT, 'query_sim')
ESTIMATE_MPP_DIR = os.path.join(PROJECT_ROOT, '1_estimate_query_mpp')
RETRIEVAL_DIR = os.path.join(PROJECT_ROOT, '2_retrieval')
LOCALIZATION_DIR = os.path.join(PROJECT_ROOT, '3_localization')
AINM_DIR = os.path.join(PROJECT_ROOT, 'aiNNModel')

#: Generic encoder+head plumbing (`Heads.py`, `common/Head.py`/`Features.py`/
#: `Checkpoints.py`) shared by any task built on an encoder+head pair --
#: `training/MppRoutingHead/` today, `1_estimate_query_mpp/ClassifierEstMpp.py`
#: and the planned retrieval work (GraphNN/tree/reranking NN) after it. Two
#: entries, not one, for the same reason `AINM_DIR` and `SUPERPATHPOINT_DIR`
#: are separate from `PROJECT_ROOT`: what goes on sys.path is the directory
#: whose children are DIRECTLY importable modules, and `Head.py` lives one
#: level below `Heads.py`, not beside it.
AINM_MODELS_DIR = os.path.join(AINM_DIR, 'models')
AINM_MODELS_COMMON_DIR = os.path.join(AINM_MODELS_DIR, 'common')

#: The SuperPathPoint training package, NOT its parent `training/`. What goes on
#: sys.path is the directory whose children are importable subpackages -- so
#: `from common.Homography import ...` and `from cli import job_result_dir`
#: resolve, exactly as `query_sim/` on the path makes `from augment.geometry
#: import ...` resolve. `training/` itself holds no modules and adding it would
#: put the useless name `SuperPathPoint` in the import namespace instead.
SUPERPATHPOINT_DIR = os.path.join(PROJECT_ROOT, 'training', 'SuperPathPoint')

#: Same rule as SUPERPATHPOINT_DIR, same reason: the training package itself,
#: not `training/`, so `from Datasets import ...` and `from Runtime import
#: ...` resolve for `Runtime.py`/`cli/train.py`/`cli/evaluate.py`.
MPPROUTINGHEAD_DIR = os.path.join(PROJECT_ROOT, 'training', 'MppRoutingHead')

#: Same rule again -- `training/PrototypicalRoutingHead/spec.md`'s own
#: package, so its `Pooling.py`/`Runtime.py`/`Losses.py`/`cli/train.py`
#: resolve each other the same way MPPROUTINGHEAD_DIR's siblings do.
PROTOTYPICALROUTINGHEAD_DIR = os.path.join(
    PROJECT_ROOT, 'training', 'PrototypicalRoutingHead')

def setup_import_paths():
    """Make utilities/, query_sim/, 1_estimate_query_mpp/, 2_retrieval/,
    3_localization/, aiNNModel/ (+ its models/ and models/common/) and
    project root importable.

    Does NOT add any training package's own directory (2026-09-22 --
    before this, it added all three: SUPERPATHPOINT_DIR/MPPROUTINGHEAD_DIR/
    PROTOTYPICALROUTINGHEAD_DIR, unconditionally, every time ANY caller
    anywhere called this function). `SuperPathPoint`/`MppRoutingHead`/
    `PrototypicalRoutingHead` each have their OWN `Runtime.py` (the first
    two) or `Datasets.py`/`Losses.py`/etc, and having all three on
    `sys.path` at once makes a bare `from Runtime import ...` resolve to
    whichever one `sys.path`'s own order happens to put first -- NOT
    necessarily the one the calling file is actually a sibling of.

    `MppRoutingHead`/`PrototypicalRoutingHead` (2026-09-22, same day,
    later) no longer touch this mechanism at all: `training/`, `training/
    MppRoutingHead/` and `training/PrototypicalRoutingHead/` are now real
    Python packages (each with its own `__init__.py`), so their own
    cross-file imports are fully-qualified (`from training.MppRoutingHead.
    Runtime import ...`) and resolve unambiguously by import path, with no
    `sys.path` ordering involved -- see `training/__init__.py`'s own
    docstring. Only `SuperPathPoint` still calls `add_training_package`
    (naming itself alone, right after this) -- it has no top-level bare
    file that collides with anything in the other two, so a plain
    `sys.path` entry was never actually ambiguous for it.
    """
    for path in (UTILITIES_DIR, QUERY_SIM_DIR, ESTIMATE_MPP_DIR, RETRIEVAL_DIR,
                LOCALIZATION_DIR, AINM_DIR, AINM_MODELS_DIR, AINM_MODELS_COMMON_DIR,
                PROJECT_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)


#: `add_training_package`'s own name -> directory map. Keys are the SAME
#: names `training/<name>/` uses on disk, not a separate vocabulary.
TRAINING_PACKAGE_DIRS = {
    'SuperPathPoint': SUPERPATHPOINT_DIR,
    'MppRoutingHead': MPPROUTINGHEAD_DIR,
    'PrototypicalRoutingHead': PROTOTYPICALROUTINGHEAD_DIR,
}


def add_training_package(*names: str) -> None:
    """Put ONE OR MORE training packages' own directories on `sys.path`,
    by name (`TRAINING_PACKAGE_DIRS`' own keys) -- never all three by
    default the way `setup_import_paths` used to, see that function's own
    docstring for the bare-`from Runtime import ...` collision this
    avoids.

    ORDER MATTERS when a caller names more than one: each `insert(0, ...)`
    pushes the previous ones DOWN, so the LAST name given ends up FIRST in
    `sys.path` and wins any bare-import collision. Put the caller's OWN
    package last.

    Only `SuperPathPoint` calls this today, naming itself alone -- nothing
    to order with one name. `MppRoutingHead`/`PrototypicalRoutingHead`
    used to call this in combination (`add_training_package('MppRoutingHead',
    'PrototypicalRoutingHead')`, `MppRoutingHead` first/lower-priority,
    `PrototypicalRoutingHead` last/winning `Runtime`, the one name both
    directories had) until 2026-09-22, when both became real Python
    packages with fully-qualified cross-file imports instead -- see
    `training/__init__.py`'s own docstring. The ordering rule above still
    applies to any FUTURE caller that names more than one.
    """
    for name in names:
        path = TRAINING_PACKAGE_DIRS[name]
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def encoder_tag(encoder: str, head: str = '') -> str:
    """What names an encoder's outputs, in one place.

    The head is part of it because it is part of identity_id: conch_vit through
    its attentional pooler and through its trunk are 512-d and 768-d vectors in
    different spaces, and one directory holding both would read as one
    experiment. An empty head means the encoder's own single exit, so gigapath
    and uni2 get plain names.

    Here rather than in each caller because it is already a name computed in
    two places -- the output path and, for the pooling benches, a CSV column
    naming the arm -- and a tag computed twice is a tag that eventually differs
    in one of them. Takes strings, not an argparse Namespace, so that this file
    keeps importing os and sys and nothing else.
    """
    return f'{encoder}_{head}' if head else str(encoder)


def job_result_dir(default_name: str, *, encoder: str = '') -> str:
    """
    Return the per-job output directory: RESULT_DIR / (SLURM_JOB_NAME or default_name).
    Creates the directory if it doesn't exist.

    Usage:
        JOB_DIR = job_result_dir('TissueMaskTest')  # default when run locally
        out = args.out or os.path.join(JOB_DIR, 'tissue_mask__regions.png')

    Write it as `args.out or job_result_dir(...)` and then makedirs the result
    anyway: `or` short-circuits, so an explicit --out never reaches this
    function and nothing else would create that directory.

    `encoder` adds one more level, and the rule for when to pass it is: if
    running a different encoder would change what this writes, the directory
    says which one wrote it. It is a directory rather than a filename suffix
    because a run usually emits more than one file -- a CSV next to a
    figures/<category>/ tree -- and tagging only the CSV leaves a second
    encoder overwriting the first one's figures one at a time, which reads as a
    redrawn figure rather than as a collision. Pass encoder_tag(...) into it.
    """
    name = os.environ.get('SLURM_JOB_NAME') or default_name
    path = os.path.join(RESULT_DIR, name, encoder) if encoder \
        else os.path.join(RESULT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path
