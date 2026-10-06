'''Makes this a real package so `Runtime.py`/`Episodes.py`/
`Losses.py` are reachable as `training.PrototypicalRoutingHead.Runtime`/
`.Episodes`/`.Losses` -- fully-qualified, so a bare `from Runtime import
...` cannot resolve to `training.MppRoutingHead.Runtime` by `sys.path` order.
See `training/__init__.py`.
'''
