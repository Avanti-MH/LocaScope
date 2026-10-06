'''Makes this a real package so `Runtime.py`/`Datasets.py` are
reachable as `training.MppRoutingHead.Runtime`/`training.MppRoutingHead.
Datasets` -- fully-qualified, so a bare `from Runtime import ...` cannot resolve to
`training.PrototypicalRoutingHead.Runtime` by `sys.path` order. See
`training/__init__.py`.
'''
