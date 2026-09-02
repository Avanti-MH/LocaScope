"""Shared setup for SuperPathPoint CLI entry points.

Re-exports `_paths` rather than deriving the output root again, for the reason
spelled out at length in `query_sim/cli/__init__.py`: that package once carried
its own copy of the rule, the two drifted, and three of its four files ended up
using a different definition from the fourth. `_paths` lives in `utilities/`,
the library layer every package here already depends on, so this import points
DOWN rather than sideways at a sibling.

What this package owns is its default job names -- SuperPathPointDemo and the
rest -- which are arguments to `job_result_dir`, not a second copy of it.

Entry points must put `training/SuperPathPoint/` on sys.path themselves before
`from cli import job_result_dir`, since this package lives one level under it.
`setup_import_paths()` does that for the module tree; the two lines below do it
for `_paths` itself, which is the one import that cannot be bootstrapped by the
thing it bootstraps.
"""

import os
import sys

_UTILITIES = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)

from _paths import (LOG_DIR, OUTPUT_ROOT, RESULT_DIR,  # noqa: E402,F401
                    job_result_dir, setup_import_paths)
