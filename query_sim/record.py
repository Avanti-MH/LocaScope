"""Per-FOV ground-truth record (one row per synthetic FOV in gt.csv).

Shape fields follow QueryFromWSI's spec (wh_ratio + MPixels + query_mpp), so
one row is enough to reproduce the exact crop from the same WSI + coordinates.
"""

from dataclasses import dataclass


@dataclass
class FOVRecord:
    filename: str
    wsi_path: str          # full path to the source WSI file
    level:    int          # source Camera's WSI pyramid level (0 for single-cam runs)

    # QueryFromWSI spec (the microscope-photo shape + scale)
    wh_ratio:      str     # e.g. '4:3'
    MPixels:       float   # total pixel budget
    query_mpp:     float   # cfg.query_mpp (nominal, Camera-level target)
    nominal_mpp:   float   # alias of query_mpp for clarity
    effective_mpp: float   # query_mpp / scale for THIS shot (post-jitter GT)
    fov_width:     int     # QFW-derived output pixel width
    fov_height:    int     # QFW-derived output pixel height

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
