'''Makes this a real package (2026-09-22) so `Runtime.py`/`Datasets.py` are
reachable as `training.MppRoutingHead.Runtime`/`training.MppRoutingHead.
Datasets` -- fully-qualified, not a bare `from Runtime import ...` that
depended on `sys.path` order to disambiguate against `training.
PrototypicalRoutingHead.Runtime`. See `training/__init__.py`'s own
docstring for the collision this replaces.
'''
