"""Where this project reads and writes.

The ONE definition of OUTPUT_ROOT and DATA_ROOT. Every other package points DOWN
at it: none of them defines a root of its own, and what each legitimately owns is
its default job name, which is the argument to job_result_dir.

It imports os and nothing else on purpose. Files that are careful about their
startup cost can import this without pulling in torch or openslide.

How packages find each other is not decided here (nor anywhere with `sys.path`):
every module imports by its full name, `locascope.core.cache.Cache`, and an entry
point is run from the repo root as `python -m locascope.cli.x`.
"""

import os

#: The repo root: the directory that holds `locascope/`.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
#: Model weights: HF_HOME's default (`locascope/__init__.py` sets the same) and
#: the CONCH checkpoint directory under it.
MODEL_WEIGHTS_DIR = os.path.join(OUTPUT_ROOT, 'model_weights')


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
    keeps importing os and nothing else.
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
