#!/usr/bin/env python3
"""show_camera_effects -- every effect of the camera's domain gap on its own, then all of them
together, on a slide crop and on a checkerboard.

    python utilities/cli/demo/show_camera_effects.py --wsi BRACS_1228
    python utilities/cli/demo/show_camera_effects.py            # the checkerboard only
    sbatch jobscripts/ShowCameraEffects.sh

The effects are `query_sim/pipeline._apply_params`'s own, run through that function (the same
order, the same lens margin, the same sensor crop) with a parameter set that is neutral except for
the one effect shown, so nothing here re-implements an effect. Each is shown at the most outward
value `DomainGapConfig` can draw, or at its fixed value when it is a constant of the config
(defocus, chromatic shift, noise, JPEG):

    rotation 90 / jitter +3 deg      scale 1.15 / 0.90         stage shift 3 px
    brightness +0.08                 contrast +0.08            colour temperature +0.12
    lens distortion k1 +0.04 / -0.04 defocus radius 2          chromatic shift 2 px
    vignette 0.45                    noise sigma 3             JPEG quality 85

Then two "all together" tiles: every effect at those values at once ("all, extreme"), and one
ordinary random draw of the config (`_sample_params`, --seed) as training would make it.

"none" is the same chain with every effect neutral (JPEG at 100), the reference for the
difference images: its own tile is what a camera with the gap switched off hands over.

WRITES, per input, under result/<job>/:
    camera_effects_<input>.png        the tiles, each labelled with its effect and value
    camera_effects_<input>_diff.png   the same tiles as |tile - none| x --gain (the effects that are
                                      hard to see: vignette, noise, JPEG, a 0.08 brightness), with
                                      the mean difference in 0..255 in each label
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))                          # utilities/
from _paths import job_result_dir, setup_import_paths                # noqa: E402

setup_import_paths()

import numpy as np                                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

from config import DomainGapConfig                                   # noqa: E402
from pipeline import _apply_params, _sample_params                   # noqa: E402

Params = Dict[str, float]

#: The sensor the effects are cut to, and the oversized read they are applied to (the slack the
#: rotation, the zoom-out and the lens margin draw on).
OUTPUT_WH: Tuple[int, int] = (512, 384)
INPUT_SIDE: int = 896


def neutral_params() -> Params:
    """Every effect off. JPEG at 100 is the nearest the chain gets to none."""
    return dict(rot_deg=0, angle_jitter=0.0, scale=1.0, effective_mpp=None,
                brightness=0.0, contrast=0.0, color_temp=0.0, vignette_strength=0.0,
                distortion_k1=0.0, distortion_k2=0.0, defocus_radius=0, chromatic_shift=0,
                stage_shift_dx=0, stage_shift_dy=0, noise_sigma=0.0, noise_seed=0,
                jpeg_quality=100, saturation=1.0)


def effects(cfg: DomainGapConfig) -> List[Tuple[str, Params]]:
    """`(label, the parameters that differ from neutral)` for each effect alone, at the most
    outward value the config draws."""
    return [
        ('rotation 90', dict(rot_deg=90)),
        (f'rotation jitter +{cfg.angle_jitter_deg:g}', dict(angle_jitter=cfg.angle_jitter_deg)),
        (f'scale {cfg.scale_range[1]:g} (zoom in)', dict(scale=cfg.scale_range[1])),
        (f'scale {cfg.scale_range[0]:g} (zoom out)', dict(scale=cfg.scale_range[0])),
        (f'stage shift {cfg.stage_shift_max} px',
         dict(stage_shift_dx=cfg.stage_shift_max, stage_shift_dy=cfg.stage_shift_max)),
        (f'brightness {cfg.brightness_range[1]:+g}', dict(brightness=cfg.brightness_range[1])),
        (f'contrast {cfg.contrast_range[1]:+g}', dict(contrast=cfg.contrast_range[1])),
        (f'colour temp {cfg.color_temp_range[1]:+g}', dict(color_temp=cfg.color_temp_range[1])),
        (f'distortion k1 {cfg.distortion_k1_range[1]:+g}',
         dict(distortion_k1=cfg.distortion_k1_range[1], distortion_k2=cfg.distortion_k2)),
        (f'distortion k1 {cfg.distortion_k1_range[0]:+g}',
         dict(distortion_k1=cfg.distortion_k1_range[0], distortion_k2=cfg.distortion_k2)),
        (f'defocus radius {cfg.defocus_radius}', dict(defocus_radius=cfg.defocus_radius)),
        (f'chromatic shift {cfg.chromatic_shift} px', dict(chromatic_shift=cfg.chromatic_shift)),
        (f'vignette {cfg.vignette_range[1]:g}', dict(vignette_strength=cfg.vignette_range[1])),
        (f'noise sigma {cfg.noise_sigma:g}', dict(noise_sigma=cfg.noise_sigma, noise_seed=1)),
        (f'JPEG q{cfg.jpeg_quality}', dict(jpeg_quality=cfg.jpeg_quality)),
    ]


def extreme_params(cfg: DomainGapConfig) -> Params:
    """All the effects above at once. Rotation takes its 90 and its jitter, scale the zoom-in,
    distortion the positive k1."""
    p: Params = neutral_params()
    for label, change in effects(cfg):
        if label.startswith(('scale', 'distortion k1 -')) and not label.endswith('(zoom in)'):
            continue                                  # one of each pair is enough
        p.update(change)
    return p


def checkerboard(side: int = INPUT_SIDE, square: int = 56) -> np.ndarray:
    """Grey and white squares (not black and white, so brightness and contrast do not clip), a
    few coloured ones for the colour temperature, and a thin red line through the middle."""
    yy, xx = np.mgrid[0:side, 0:side]
    board: np.ndarray = (((yy // square) + (xx // square)) % 2).astype(np.uint8)
    img: np.ndarray = np.where(board[..., None] == 1, 215, 70).astype(np.uint8).repeat(3, axis=2)
    for (row, col), colour in {(2, 3): (200, 50, 50), (5, 8): (50, 160, 60),
                               (9, 4): (50, 70, 200), (12, 11): (220, 180, 40)}.items():
        img[row * square:(row + 1) * square, col * square:(col + 1) * square] = colour
    img[side // 2 - 1:side // 2 + 1, :] = (230, 40, 40)
    return img


def slide_crop(wsi_name: str, level: int, seed: int) -> np.ndarray:
    """The most textured of a few random INPUT_SIDE squares of the slide at `level`, skipping
    ones that are mostly glass."""
    from AccessDatasets import locate                                # noqa: PLC0415
    from SafeSlide import SafeSlide                                  # noqa: PLC0415
    wsi = SafeSlide(str(locate(wsi_name).path))
    ds: float = float(wsi.level_downsamples[level])
    width, height = wsi.dimensions
    rng: np.random.Generator = np.random.default_rng(seed)
    best: Optional[np.ndarray] = None
    best_std: float = -1.0
    for _ in range(60):
        x: int = int(rng.integers(0, max(1, width - int(INPUT_SIDE * ds))))
        y: int = int(rng.integers(0, max(1, height - int(INPUT_SIDE * ds))))
        crop: np.ndarray = np.asarray(wsi.read_region_rgb((x, y), level, (INPUT_SIDE, INPUT_SIDE)))
        if float(crop.mean()) > 215.0:                               # glass
            continue
        std: float = float(crop.std())
        if std > best_std:
            best, best_std = crop, std
    wsi.close()
    if best is None:
        raise RuntimeError(f'{wsi_name}: no tissue crop found at level {level}')
    return best


def label_tile(tile: np.ndarray, text: str) -> Image.Image:
    """The tile under a one-line caption."""
    band: int = 22
    canvas: Image.Image = Image.new('RGB', (tile.shape[1], tile.shape[0] + band), (252, 252, 251))
    canvas.paste(Image.fromarray(tile), (0, band))
    ImageDraw.Draw(canvas).text((6, 5), text, fill=(11, 11, 11))
    return canvas


def sheet(tiles: List[Image.Image], columns: int, scale: float) -> Image.Image:
    """The labelled tiles in a grid, `columns` wide."""
    w: int = int(tiles[0].width * scale)
    h: int = int(tiles[0].height * scale)
    rows: int = (len(tiles) + columns - 1) // columns
    out: Image.Image = Image.new('RGB', (columns * w + (columns + 1) * 6, rows * h + (rows + 1) * 6),
                                 (225, 224, 217))
    for i, tile in enumerate(tiles):
        out.paste(tile.resize((w, h), Image.LANCZOS), (6 + (i % columns) * (w + 6), 6 + (i // columns) * (h + 6)))
    return out


def run_input(name: str, source: np.ndarray, cfg: DomainGapConfig, out_dir: Path, *,
              seed: int, gain: float, columns: int, scale: float) -> None:
    def shot(change: Params) -> np.ndarray:
        p: Params = neutral_params()
        p.update(change)
        return _apply_params(source, cfg, p, output_wh=OUTPUT_WH)

    cases: List[Tuple[str, np.ndarray]] = [('none', shot({}))]
    for label, change in effects(cfg):
        cases.append((label, shot(change)))
    cases.append(('ALL, extreme values', _apply_params(source, cfg, extreme_params(cfg), output_wh=OUTPUT_WH)))
    drawn: Params = _sample_params(cfg, random.Random(seed))
    cases.append((f'ALL, random draw (seed {seed})', _apply_params(source, cfg, drawn, output_wh=OUTPUT_WH)))

    base: np.ndarray = cases[0][1].astype(np.float32)
    images: List[Image.Image] = []
    diffs: List[Image.Image] = []
    for label, tile in cases:
        delta: np.ndarray = np.abs(tile.astype(np.float32) - base)
        images.append(label_tile(tile, label))
        diffs.append(label_tile(np.clip(delta * gain, 0, 255).astype(np.uint8),
                                f'{label}   mean|d| {float(delta.mean()):.2f}'))
        print(f'  {name:12s} {label:34s} mean|d| {float(delta.mean()):7.2f}', flush=True)
    path: Path = out_dir / f'camera_effects_{name}.png'
    sheet(images, columns, scale).save(path)
    sheet(diffs, columns, scale).save(out_dir / f'camera_effects_{name}_diff.png')
    print(f'  -> {path}  (+ _diff)', flush=True)


def main() -> int:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument('--wsi', default=None, help='a slide name (AccessDatasets.locate); omitted: the checkerboard only')
    ap.add_argument('--level', type=int, default=1, help='pyramid level the slide crop is read at')
    ap.add_argument('--seed', type=int, default=0, help='the crop position and the random draw')
    ap.add_argument('--gain', type=float, default=8.0, help='the difference images are |d| x gain')
    ap.add_argument('--columns', type=int, default=3)
    ap.add_argument('--scale', type=float, default=1.0, help='tile size on the sheet')
    ap.add_argument('--out', default=None, help='default result/<SLURM_JOB_NAME or ShowCameraEffects>/')
    args: argparse.Namespace = ap.parse_args()

    cfg: DomainGapConfig = DomainGapConfig()
    out_dir: Path = Path(args.out or job_result_dir('ShowCameraEffects'))
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs: List[Tuple[str, np.ndarray]] = [('checkerboard', checkerboard())]
    if args.wsi:
        inputs.insert(0, (args.wsi.replace(',', '_'), slide_crop(args.wsi, args.level, args.seed)))
    for name, source in inputs:
        Image.fromarray(source).save(out_dir / f'camera_effects_{name}_input.png')
        run_input(name, source, cfg, out_dir, seed=args.seed, gain=args.gain,
                  columns=args.columns, scale=args.scale)
    return 0


if __name__ == '__main__':
    sys.exit(main())
