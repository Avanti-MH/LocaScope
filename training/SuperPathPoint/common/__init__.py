"""Pieces shared by all three SuperPathPoint stages.

Flat re-exports, matching `query_sim/augment/__init__.py`: callers put
`training/SuperPathPoint/` on sys.path (via `_paths.setup_import_paths()`) and
then spell `from common.Homography import sample_homography`, or take the short
names from here.
"""

from DsLadder import (DEFAULT_RUNGS, DsLadder, LEVEL_REL_TOL, RungPlan,
                             finer_level_for_downsample)
from common import KeypointLabelStore
import PreTileStore
from common.KeypointLabelStore import (LabelBatch, LabelMeta, LabelMismatch,
                                       batch_from_lists, cap_for,
                                       nms_max_pool, points_from_prob)
from PreTileStore import (PRE_TILE_FACTOR, PreTileMeta, PreTileMismatch,
                                 PreTileRecord, centre_crop, centre_margin,
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
    'finer_level_for_downsample',
    # PreTileStore. The MODULE is exported too, not just these names: its
    # read/write half is `PreTileStore.create` / `.save_tile` / `.load_index`,
    # and flattening those would put `create` and `find_one` in a namespace
    # shared with two other stores that have functions of the same name.
    'PreTileStore', 'PRE_TILE_FACTOR', 'PreTileMeta', 'PreTileRecord',
    'PreTileMismatch', 'pre_tile_px', 'centre_margin', 'centre_crop',
    # KeypointLabelStore. The module for the same reason as above -- `save`,
    # `load` and `find_one` are names three stores here share.
    'KeypointLabelStore', 'LabelBatch', 'LabelMeta', 'LabelMismatch',
    'points_from_prob', 'nms_max_pool', 'batch_from_lists', 'cap_for',
]
