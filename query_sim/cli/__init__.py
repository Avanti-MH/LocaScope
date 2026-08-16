"""Shared helper for query_sim CLI entry points.

`job_result_dir(name)` mirrors `utilities/test_modules/_paths.job_result_dir`:
returns `<OUTPUT_ROOT>/result/<SLURM_JOB_NAME or default>/`, creating it if
needed.

OUTPUT_ROOT is one level ABOVE the repo, not inside it. Results are the
expensive half of this project -- 60 GB of feature stores and figures -- and
keeping them out of the working tree means an `rm -rf` of the checkout, a
`git clean`, or a fresh clone cannot take them, and nothing under `result/`
can be staged by accident. Override with LOCASCOPE_OUTPUT_ROOT.

The rule is duplicated rather than imported because this package lives at
`query_sim/cli/` and `_paths` lives under `utilities/test_modules/`; a CLI
depending on a test helper would be the wrong direction. Keep the two in step.

demo.py / batch.py must add `query_sim/` to sys.path themselves BEFORE
`from cli import job_result_dir` (this package lives at `query_sim/cli/`).
"""

import os

HERE        = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(HERE, '..', '..'))
OUTPUT_ROOT = os.environ.get(
    'LOCASCOPE_OUTPUT_ROOT', os.path.abspath(os.path.join(REPO_ROOT, '..')))
RESULT_DIR  = os.path.join(OUTPUT_ROOT, 'result')
LOG_DIR     = os.path.join(OUTPUT_ROOT, 'log')


def job_result_dir(default_name: str) -> str:
    name = os.environ.get('SLURM_JOB_NAME') or default_name
    path = os.path.join(RESULT_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path
