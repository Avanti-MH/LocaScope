"""Where this project writes, and how its packages find each other.

The ONE definition of OUTPUT_ROOT. It sits in utilities/ because that is the
library layer every package already depends on -- utilities/cli, test_modules
and bench_modules all point DOWN at it, and none of them points sideways at
another. What each of those legitimately owns is its default job name, which
is the argument to job_result_dir.

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
#: clone does not take 60 GB of results with it, and nothing under `result/`
#: can ever be staged by accident. Override with LOCASCOPE_OUTPUT_ROOT.
OUTPUT_ROOT = os.environ.get(
    'LOCASCOPE_OUTPUT_ROOT', os.path.abspath(os.path.join(PROJECT_ROOT, '..')))
RESULT_DIR = os.path.join(OUTPUT_ROOT, 'result')
LOG_DIR = os.path.join(OUTPUT_ROOT, 'log')
#: Where the INPUTS are: the slides and photos (`AccessDatasets`), and the
#: slides moved out of them for scanner damage (`datasets_holed/`). Its own
#: root, because pointing outputs elsewhere must not move the data. Override
#: with LOCASCOPE_DATA_ROOT.
DATA_ROOT = os.environ.get(
    'LOCASCOPE_DATA_ROOT', os.path.abspath(os.path.join(PROJECT_ROOT, '..')))
DATASETS_DIR = os.path.join(DATA_ROOT, 'datasets')
HOLED_DATASETS_DIR = os.path.join(DATA_ROOT, 'datasets_holed')
#: Model weights: HF_HOME's default (jobscripts/_env.sh exports the same) and
#: the CONCH checkpoint directory under it.
MODEL_WEIGHTS_DIR = os.path.join(OUTPUT_ROOT, 'model_weights')
QUERY_SIM_DIR = os.path.join(PROJECT_ROOT, 'query_sim')
# stage1_estimation/, stage2_retrieval/ and stage3_localization/ are real
# packages: imported as `stage2_retrieval.X` off PROJECT_ROOT, so
# none of them is a sys.path entry and none has a constant here.
AINM_DIR = os.path.join(PROJECT_ROOT, 'aiNNModel')

#: Generic encoder+head plumbing (`Heads.py`, `common/Head.py`/`Features.py`/
#: `Checkpoints.py`) shared by any task built on an encoder+head pair --
#: `training/MppRoutingHead/` today, `stage1_estimation/ClassifierEstMpp.py`
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

def setup_import_paths(*training_packages: str) -> None:
    """Make utilities/, query_sim/, aiNNModel/ (+ its models/ and
    models/common/) and the project root importable -- the root is what makes
    the stage packages (`stage1_estimation`, `stage2_retrieval`,
    `stage3_localization`) and `training` importable by their full names --
    then the named training packages' own directories (`add_training_package`).

    THE ONE PLACE sys.path IS SET. An entry point puts utilities/ on the path
    by hand (this function lives in it) and calls this; no library module
    touches sys.path, so what a module can import is decided by the entry
    that runs it and by nothing else.

    No training package directory is added unless named: the packages each
    have their own `Runtime.py` / `Datasets.py` / ..., and with all of them
    on `sys.path` a bare `from Runtime import ...` resolves to whichever comes
    first, not the caller's sibling. `MppRoutingHead` and
    `PrototypicalRoutingHead` are real packages imported by full name
    (`training/__init__.py`); a SuperPathPoint entry names `'SuperPathPoint'`.
    """
    for path in (UTILITIES_DIR, QUERY_SIM_DIR,
                AINM_DIR, AINM_MODELS_DIR, AINM_MODELS_COMMON_DIR,
                PROJECT_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    add_training_package(*training_packages)


#: `add_training_package`'s own name -> directory map. Keys are the SAME
#: names `training/<name>/` uses on disk, not a separate vocabulary.
TRAINING_PACKAGE_DIRS = {
    'SuperPathPoint': SUPERPATHPOINT_DIR,
    'MppRoutingHead': MPPROUTINGHEAD_DIR,
    'PrototypicalRoutingHead': PROTOTYPICALROUTINGHEAD_DIR,
}


def add_training_package(*names: str) -> None:
    """Put ONE OR MORE training packages' own directories on `sys.path`,
    by name (`TRAINING_PACKAGE_DIRS`' own keys); see `setup_import_paths`
    for the bare-`from Runtime import ...` collision this avoids.

    ORDER MATTERS when a caller names more than one: each `insert(0, ...)`
    pushes the previous ones DOWN, so the LAST name given ends up FIRST in
    `sys.path` and wins any bare-import collision. Put the caller's OWN
    package last. Called by `setup_import_paths` with the names it was given.
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


def job_name(default: str) -> str:
    """SLURM_JOB_NAME, else `default`: the name a job's results AND the caches
    it made carry (`job_result_dir`, `Cache.cache_root`)."""
    return os.environ.get('SLURM_JOB_NAME') or default


def job_result_dir(default_name: str, *, encoder: str = '') -> str:
    """
    Return the per-job output directory: RESULT_DIR / (SLURM_JOB_NAME or default_name).
    Creates the directory if it doesn't exist.

    Usage:
        JOB_DIR = job_result_dir('TestTissueMask')  # default when run locally
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
    name = job_name(default_name)
    path = os.path.join(RESULT_DIR, name, encoder) if encoder \
        else os.path.join(RESULT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path
