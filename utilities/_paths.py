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


def job_result_dir(default_name: str) -> str:
    """
    Return the per-job output directory: RESULT_DIR / (SLURM_JOB_NAME or default_name).
    Creates the directory if it doesn't exist.

    Usage:
        JOB_DIR = job_result_dir('TissueMaskTest')  # default when run locally
        out = args.out or os.path.join(JOB_DIR, 'tissue_mask__regions.png')

    Write it as `args.out or job_result_dir(...)` and then makedirs the result
    anyway: `or` short-circuits, so an explicit --out never reaches this
    function and nothing else would create that directory.
    """
    name = os.environ.get('SLURM_JOB_NAME') or default_name
    path = os.path.join(RESULT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path
