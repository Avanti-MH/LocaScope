"""Pieces shared by all three SuperPathPoint stages.

Flat re-exports, matching `query_sim/augment/__init__.py`: with
`training/SuperPathPoint/` on sys.path (`_paths.setup_import_paths(
'SuperPathPoint')`), callers spell `from common.Homography import
sample_homography`, or take the short names from here.
"""

from DsLadder import DEFAULT_RUNGS, DsLadder, LEVEL_REL_TOL, RungPlan
from common import KeypointLabelStore
from common.KeypointLabelStore import (LabelBatch, LabelMeta, LabelMismatch,
                                       batch_from_lists, cap_for,
                                       nms_max_pool, points_from_prob)
from TileSampler import (PRE_TILE_FACTOR, centre_crop, centre_margin,
                         pre_tile_px)
from common.Homography import (HOMOGRAPHY_DEFAULTS, HomographySample,
                               erode_valid, erosion_anchor, identity,
                               inside, invert, points_input_to_output,
                               points_output_to_input, quad_polygon,
                               sample_homography, valid_mask, warp_image,
                               warp_image_torch)
from common.HomographyConfig import HOMOGRAPHY_BASELINE, HomographyConfig
from common.Interfaces import (Backbone, DescriptorHead, DetectorDecoder,
                               ShapeMismatch, check_shapes)

__all__ = [
    # Homography
    'HOMOGRAPHY_DEFAULTS', 'HomographySample', 'sample_homography',
    'identity', 'invert',
    'points_input_to_output', 'points_output_to_input', 'inside',
    'warp_image', 'warp_image_torch', 'valid_mask', 'erode_valid',
    'erosion_anchor', 'quad_polygon',
    # HomographyConfig -- the thirteen sampler options, shared by HaConfig and
    # PairDatasetConfig so the two cannot drift apart.
    'HomographyConfig', 'HOMOGRAPHY_BASELINE',
    # Interfaces
    'Backbone', 'DetectorDecoder', 'DescriptorHead', 'check_shapes',
    'ShapeMismatch',
    # DsLadder
    'DEFAULT_RUNGS', 'DsLadder', 'RungPlan', 'LEVEL_REL_TOL',
    # The pre-tile geometry (utilities/TileSampler.py). A corpus is
    # common/Corpora.Corpus, imported by name.
    'PRE_TILE_FACTOR', 'pre_tile_px', 'centre_margin', 'centre_crop',
    # KeypointLabelStore: the module, because `save`,
    # `load` and `find_one` are generic names.
    'KeypointLabelStore', 'LabelBatch', 'LabelMeta', 'LabelMismatch',
    'points_from_prob', 'nms_max_pool', 'batch_from_lists', 'cap_for',
]
