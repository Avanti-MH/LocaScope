"""Atomic augmentation primitives — each fn is img -> img (numpy uint8 RGB)."""

from augment.color    import apply_color, apply_color_temp, apply_brightness_contrast, apply_jpeg
from augment.field    import apply_vignette, apply_stage_shift
from augment.lens     import apply_distortion, apply_defocus, apply_chromatic
from augment.geometry import apply_rotation, apply_scale
from augment.noise    import apply_noise

__all__ = [
    'apply_color', 'apply_color_temp', 'apply_brightness_contrast', 'apply_jpeg',
    'apply_vignette', 'apply_stage_shift',
    'apply_distortion', 'apply_defocus', 'apply_chromatic',
    'apply_rotation', 'apply_scale',
    'apply_noise',
]
