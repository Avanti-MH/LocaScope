"""Shared helper for query_sim CLI entry points.

`job_result_dir(name)` returns `<OUTPUT_ROOT>/result/<SLURM_JOB_NAME or
default>/`, creating it if needed. OUTPUT_ROOT is one level ABOVE the repo, not
inside it: results are the expensive half of this project -- 60 GB of feature
stores and figures -- and keeping them out of the working tree means an
`rm -rf` of the checkout, a `git clean`, or a fresh clone cannot take them, and
nothing under `result/` can be staged by accident.

RE-EXPORTED, not redefined. This file used to carry its own copy of the rule,
on the argument that a CLI depending on a helper under utilities/test_modules
would be the wrong direction. The direction was the real problem; the copy was
not the fix. It ended with two definitions inside ONE package -- batch, demo
and multi_batch imported the local one while diag_camera_skip inserted
utilities/test_modules into sys.path to import the other -- which is what
`keep the two in step` means once nothing enforces it.

_paths lives in utilities/ now, the library layer this package already depends
on for TissuesRegionsMask and config, so the import points DOWN rather than
sideways at a sibling. What this package legitimately owns is its default job
names -- QuerySimBatch, QuerySimDemo, MultiBatch -- which are arguments to the
function, not a second copy of it.

demo.py / batch.py must add `query_sim/` to sys.path themselves BEFORE
`from cli import job_result_dir` (this package lives at `query_sim/cli/`).
"""

import os
import sys

_UTILITIES = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)

# Re-exported so that `from cli import job_result_dir` keeps working unchanged
# for the three callers that already spell it that way.
from _paths import (LOG_DIR, OUTPUT_ROOT, RESULT_DIR,  # noqa: E402,F401
                    job_result_dir)
