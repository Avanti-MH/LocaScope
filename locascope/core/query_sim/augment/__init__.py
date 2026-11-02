"""Atomic augmentation primitives — each fn is img -> img (numpy uint8 RGB)."""

from locascope.core.query_sim.augment.color import apply_color, apply_color_temp, apply_brightness_contrast, apply_jpeg
from locascope.core.query_sim.augment.field import apply_vignette, apply_stage_shift
from locascope.core.query_sim.augment.lens import apply_distortion, apply_defocus, apply_chromatic
from locascope.core.query_sim.augment.geometry import apply_rotation, apply_scale
from locascope.core.query_sim.augment.noise import apply_noise

__all__ = [
    'apply_color', 'apply_color_temp', 'apply_brightness_contrast', 'apply_jpeg',
    'apply_vignette', 'apply_stage_shift',
    'apply_distortion', 'apply_defocus', 'apply_chromatic',
    'apply_rotation', 'apply_scale',
    'apply_noise',
]
