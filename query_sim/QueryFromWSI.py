"""Backward-compat shim: import moved to source.wsi_query.

Existing callers (utilities/test_modules/*) rely on
    from QueryFromWSI import QueryFromWSI
with `query_sim/` on sys.path. This file keeps that entry point alive.
"""

from source.wsi_query import QueryFromWSI  # noqa: F401
