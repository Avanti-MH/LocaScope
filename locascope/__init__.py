"""LocaScope: locate a microscope photo in a whole-slide image.

This file runs before anything else in the package, whichever entry point is
used (`python -m locascope.cli.x` imports it first), and that is its one job:
HF_HOME must be in the environment BEFORE `huggingface_hub` is imported, because
that module reads it into a constant at its own import and nothing written
later is read by anyone. `setdefault`, not `=`: a value the caller exported
(`jobscripts` or a shell) wins. It imports `locascope.paths`, which imports os
and nothing else, so this costs no torch or openslide import.
"""

import os

from locascope.paths import MODEL_WEIGHTS_DIR

os.environ.setdefault('HF_HOME', MODEL_WEIGHTS_DIR)
