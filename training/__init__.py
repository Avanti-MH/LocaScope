'''Namespace package only -- `training/MppRoutingHead/` and `training/
PrototypicalRoutingHead/` are real Python packages under it so
their own cross-file imports can be fully-qualified (`from training.
MppRoutingHead.Runtime import ...`), so `sys.path` ORDER never has to tell
the two packages' same-named files apart -- see `utilities/_paths.setup_import_paths`.

`training/SuperPathPoint/` does NOT need this: it has no top-level bare
file that collides with anything in the other two packages, so it keeps
using `_paths.add_training_package('SuperPathPoint')` (`sys.path` entry,
bare imports).
'''
