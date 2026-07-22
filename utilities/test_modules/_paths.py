"""Shared path helpers for utilities/test_modules scripts."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UTILITIES_DIR = os.path.abspath(os.path.join(HERE, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '../..'))
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result')
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

    Usage in test_modules scripts:
        JOB_DIR = job_result_dir('TissueMaskTest')  # default when run locally
        out = args.out or os.path.join(JOB_DIR, 'tissue_mask__regions.png')
    """
    name = os.environ.get('SLURM_JOB_NAME') or default_name
    path = os.path.join(RESULT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path
