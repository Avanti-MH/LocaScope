'''Namespace package only -- `training/MppRoutingHead/` and `training/
PrototypicalRoutingHead/` are real Python packages under it (2026-09-22) so
their own cross-file imports can be fully-qualified (`from training.
MppRoutingHead.Runtime import ...`) instead of a bare `from Runtime import
...` that depended on `sys.path` ORDER to tell the two packages' same-named
files apart -- see `utilities/_paths.add_training_package`'s own docstring
for the collision this replaces.

`training/SuperPathPoint/` does NOT need this: it has no top-level bare
file that collides with anything in the other two packages, so it keeps
using `_paths.add_training_package('SuperPathPoint')` (`sys.path` entry,
bare imports) unchanged.
'''
