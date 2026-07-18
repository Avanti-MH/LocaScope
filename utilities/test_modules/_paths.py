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
