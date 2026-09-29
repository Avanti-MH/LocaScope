'''Makes this a real package (2026-09-22) so `Runtime.py`/`Episodes.py`/
`Losses.py` are reachable as `training.PrototypicalRoutingHead.Runtime`/
`.Episodes`/`.Losses` -- fully-qualified, not a bare `from Runtime import
...` that depended on `sys.path` order to disambiguate against `training.
MppRoutingHead.Runtime`. See `training/__init__.py`'s own docstring for the
collision this replaces.
'''
