"""Stage A: Homographic Adaptation and the joint detector+descriptor training.

Named after upstream's stages 2 and 3, which is what this package reproduces
(spec.md 3.1). Stage 1 -- MagicPoint on synthetic shapes -- is skipped in favour
of the released weights, and `Teacher.py` is the only thing that loads them.

Flat re-exports, matching `common/__init__.py`: with
`training/SuperPathPoint/` on sys.path (`_paths.setup_import_paths()`), callers
spell `from SuperPoint.Teacher import TeacherConfig` or take the short names
from here.

The import order below is the dependency order, and it is not alphabetical:
`Decoders` owns the depth-to-space that `Teacher` imports, and `Backbones` owns
the `VggBlock` that both heads are built from.
"""

from SuperPoint.Backbones import (VggBackbone, VggBackboneConfig, VggBlock,
                                  to_model_channels)
from SuperPoint.Decoders import (DepthToSpaceDecoder,
                                 DepthToSpaceDecoderConfig, UpsampleDecoder,
                                 UpsampleDecoderConfig, depth_to_space_prob)
from SuperPoint.Heads import (DescriptorHead, DescriptorHeadConfig,
                              sample_descriptors)
from SuperPoint.Teacher import SuperPointTeacher, TeacherConfig
from SuperPoint.HomographicAdaptation import (HaConfig, HaResult,
                                              HomographicAdaptation)
from SuperPoint.KeypointNet import (KeypointNet, KeypointNetConfig,
                                    KeypointOutput, Keypoints,
                                    extract_keypoints)
from SuperPoint.Losses import (SuperPointLoss, SuperPointLossConfig,
                               cell_labels, descriptor_loss, detector_loss,
                               space_to_depth)
from SuperPoint.Datasets import (BALANCE_MODES, HomographyPairDataset,
                                 PairDatasetConfig, splat)
from SuperPoint.Trainer import Trainer, TrainerConfig

__all__ = [
    # Backbones
    'VggBackboneConfig', 'VggBackbone', 'VggBlock', 'to_model_channels',
    # Decoders
    'DepthToSpaceDecoderConfig', 'DepthToSpaceDecoder',
    'UpsampleDecoderConfig', 'UpsampleDecoder', 'depth_to_space_prob',
    # Heads
    'DescriptorHeadConfig', 'DescriptorHead', 'sample_descriptors',
    # Teacher
    'TeacherConfig', 'SuperPointTeacher',
    # Homographic Adaptation
    'HaConfig', 'HomographicAdaptation', 'HaResult',
    # KeypointNet
    'KeypointNetConfig', 'KeypointNet', 'KeypointOutput', 'Keypoints',
    'extract_keypoints',
    # Losses
    'SuperPointLossConfig', 'SuperPointLoss', 'detector_loss',
    'descriptor_loss', 'cell_labels', 'space_to_depth',
    # Datasets
    'PairDatasetConfig', 'HomographyPairDataset', 'BALANCE_MODES', 'splat',
    # Trainer
    'TrainerConfig', 'Trainer',
]
