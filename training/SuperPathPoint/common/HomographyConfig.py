"""The thirteen homography options as one hashable config.

    cfg = HomographyConfig()                     # upstream's export values
    sample = sample_homography(shape, rng=rng, **cfg.kwargs())

WHY THIS IS NOT IN common/Homography.py
-----------------------------------------
That module is deliberately torch-free at import: `warp_image_torch` imports
torch inside itself so that `cli/demo_homography.py` and the geometry half of
`test_homography` stay runnable without it. `ConfigIdentity` imports torch at
module level, so putting an `IdentifiedConfig` next to `HOMOGRAPHY_DEFAULTS`
would quietly take that property away -- and the way it would be noticed is a
demo that used to run on a login node failing to import.

WHY IT IS NOT THIRTEEN FIELDS IN EACH CONFIG
----------------------------------------------
Two configs need exactly these options: `HaConfig` (the N views of Homographic
Adaptation) and `PairDatasetConfig` (the pair a training step is built from).
Duplicating thirteen fields means thirteen chances for the two to drift, and
they must NOT drift for one specific reason: a student trained on pairs drawn
from a wider distribution than the one that produced its labels is being asked
to be invariant to transforms its teacher never voted on.

`ConfigIdentity.parts_against` recurses into a nested config and prefixes its
parts (`homography.patch_ratio=...`), so embedding it keeps every option in the
identity and keeps them distinguishable from the outer config's own fields.

The import-time check below is the other half: `sample_homography` takes
`**overrides`, so an option this class stopped naming would simply not be
passed -- the sampler would run on its own default while the config said
otherwise, with nothing raising.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ConfigIdentity import IdentifiedConfig

from common.Homography import HOMOGRAPHY_DEFAULTS


@dataclass(frozen=True)
class HomographyConfig(IdentifiedConfig):
    """Upstream's export config, field by field (spec.md 9).

    Not `@register`ed: it is never the top-level thing anyone asks for by name,
    and a registry entry that nothing looks up is a name to keep in step for no
    reader. It is always nested inside a config that IS registered.
    """
    perspective: bool = True
    scaling: bool = True
    rotation: bool = True
    translation: bool = True

    n_scales: int = 5
    n_angles: int = 25

    scaling_amplitude: float = 0.2
    perspective_amplitude_x: float = 0.2
    perspective_amplitude_y: float = 0.2

    #: The quad starts at 85 percent of the frame, which is where the pre-tile
    #: factor's `1 / patch_ratio` term comes from (spec.md 6.6).
    patch_ratio: float = 0.85

    #: pi in the export config; the function signature's own default is pi/2
    #: (`homographies.py:119`). The config wins, because the config is what ran.
    max_angle: float = float(np.pi)

    allow_artifacts: bool = True
    translation_overflow: float = 0.0

    def kwargs(self) -> Dict[str, object]:
        """Exactly what `sample_homography` takes, and nothing else."""
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)}


_FIELDS = {f.name for f in dataclasses.fields(HomographyConfig)}
if _FIELDS != set(HOMOGRAPHY_DEFAULTS):
    raise ImportError(
        f'HomographyConfig names {sorted(_FIELDS)} but common.Homography takes '
        f'{sorted(HOMOGRAPHY_DEFAULTS)}. One of the two gained or lost a field; '
        f'a mismatch here means the sampler silently runs on a default the '
        f'config does not know about')

#: The values above, as the dict a BASELINE wants. Derived rather than typed
#: again: a baseline that restated them could disagree with the defaults, and
#: then every config would hash as "differs from baseline" in a field nobody
#: changed. ConfigIdentity's zero point still cannot MOVE -- these defaults are
#: upstream's and are not ours to edit -- but it is spelled once.
HOMOGRAPHY_BASELINE = HomographyConfig()
