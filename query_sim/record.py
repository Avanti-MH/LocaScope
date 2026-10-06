"""Per-FOV ground-truth record (one row per synthetic FOV in gt.csv).

The sensor (fov_width x fov_height) and the nominal mpp are the camera's, so
one row is enough to reproduce the exact crop from the same WSI + coordinates.
"""

from dataclasses import dataclass


@dataclass
class FOVRecord:
    filename: str
    wsi_path: str          # full path to the source WSI file
    level:    int          # source camera's WSI pyramid level (0 for single-cam runs)

    # the camera: sensor and objective
    nominal_mpp:   float   # the camera's mpp, ds * base_mpp
    effective_mpp: float   # nominal_mpp / scale for THIS shot (post-jitter GT)
    fov_width:     int     # sensor width, output px
    fov_height:    int     # sensor height, output px

    # Position (level-0 top-left of the pre-augment crop)
    gt_x: int
    gt_y: int

    # Geometry
    rot_deg:      int      # 0 / 90 / 180 / 270 chosen from rotation_choices
    angle_jitter: float    # small extra rotation (degrees) on top of rot_deg
    scale:        float

    # Photometric (all recorded so runs are reproducible from the row alone)
    vignette_strength: float
    color_temp:        float
    brightness:        float
    contrast:          float
    distortion_k1:     float
    defocus_radius:    int
    chromatic_shift:   int
    stage_shift_dx:    int
    stage_shift_dy:    int
    noise_sigma:       float
    jpeg_quality:      int

    @classmethod
    def from_capture(cls, filename: str, wsi_path: str, camera, x: int, y: int,
                     params: dict, level: int = 0) -> 'FOVRecord':
        """The row of one photo: `camera` (a `Render`) took it at the FoV
        rectangle's level-0 top-left (x, y), and `params` is what
        `capture_with_gt` returned with it. The sensor and the nominal mpp are
        the camera's own (`camera.sensor`, `camera.mpp`)."""
        p = params
        return cls(
            filename      = filename,
            wsi_path      = wsi_path,
            level         = level,
            nominal_mpp   = camera.mpp,
            effective_mpp = float(p['effective_mpp']),
            fov_width     = camera.output_w,
            fov_height    = camera.output_h,
            gt_x          = int(x),
            gt_y          = int(y),
            rot_deg           = int(p['rot_deg']),
            angle_jitter      = round(float(p['angle_jitter']), 3),
            scale             = round(float(p['scale']), 4),
            vignette_strength = round(float(p['vignette_strength']), 3),
            color_temp        = round(float(p['color_temp']), 3),
            brightness        = round(float(p['brightness']), 3),
            contrast          = round(float(p['contrast']), 3),
            distortion_k1     = round(float(p['distortion_k1']), 4),
            defocus_radius    = int(p['defocus_radius']),
            chromatic_shift   = int(p['chromatic_shift']),
            stage_shift_dx    = int(p['stage_shift_dx']),
            stage_shift_dy    = int(p['stage_shift_dy']),
            noise_sigma       = float(p['noise_sigma']),
            jpeg_quality      = int(p['jpeg_quality']),
        )
