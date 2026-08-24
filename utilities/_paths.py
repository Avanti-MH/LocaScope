"""Where this project writes, and how its packages find each other.

The ONE definition of OUTPUT_ROOT. It lived under utilities/test_modules/ and
was reachable only by scripts in that directory, so everything else either
inserted a test directory into sys.path to reach it -- nine CLI entry points
did -- or derived the rule again. query_sim/cli/__init__.py derived it again,
and said so: keep the two in step. They did not stay in step, and the proof was
inside that same package: three of its four files used the local copy while
diag_camera_skip.py inserted utilities/test_modules to import this one.

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

def setup_import_paths():
    """Make utilities/, query_sim/, 1_estimate_query_mpp/, 2_retrieval/, 3_localization/, aiNNModel/ and project root importable."""
    for path in (UTILITIES_DIR, QUERY_SIM_DIR, ESTIMATE_MPP_DIR, RETRIEVAL_DIR, LOCALIZATION_DIR, AINM_DIR, PROJECT_ROOT):
        if path not in sys.path:
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
