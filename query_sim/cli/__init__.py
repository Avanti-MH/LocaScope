"""Shared helper for query_sim CLI entry points.

`job_result_dir(name)` mirrors `utilities/test_modules/_paths.job_result_dir`:
returns `<repo>/result/<SLURM_JOB_NAME or default>/`, creating it if needed.

demo.py / batch.py must add `query_sim/` to sys.path themselves BEFORE
`from cli import job_result_dir` (this package lives at `query_sim/cli/`).
"""

import os

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


def job_result_dir(default_name: str) -> str:
    name = os.environ.get('SLURM_JOB_NAME') or default_name
    path = os.path.join(REPO_ROOT, 'result', name)
    os.makedirs(path, exist_ok=True)
    return path
