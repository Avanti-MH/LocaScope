"""Backward-compat shim: implementation moved to pipeline / cli.demo.

Existing callers rely on
    from simulate_microscope_photo import simulate_microscope_photo
with `query_sim/` on sys.path. This file keeps that entry point alive and
still supports `python query_sim/simulate_microscope_photo.py <wsi>` by
delegating to cli.demo.
"""

from pipeline import simulate_microscope_photo, simulate_with_gt  # noqa: F401


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cli'))
    from cli.demo import main
    main()
