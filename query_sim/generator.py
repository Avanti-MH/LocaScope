"""Layer 2: batch synthesise N FOVs from a WSI, with per-FOV ground truth.

    Render     HOW a FoV looks (`camera.Render`, instance `camera`): the read
               its spec asks its SlideReader for, then the domain gap
               (`capture_with_gt`). Takes a position; knows no mask.
    FovSupply  WHERE the FoVs are: positions a `TileSampler` draws for the
               camera's spec, rendered through that camera, batch after batch.
    generate   file naming + gt.csv writing.

WHERE THE FOVs ARE is a `SamplerConfig` (richness buckets, overlap); the
tissue-ratio / region-protrusion parameters this used to take are gone.
`RICHNESS_PRESETS` names the two mixes the CLIs offer.

`FovSupply` was `Camera.set_sampler` / `FoV_bank` / `__iter__` until
2026-10-03. It moved because placing is not a property of the microscope: the
training packages build Cameras with no mask and never draw, and a Camera that
carried a sampler, a draw cache and a mask was two objects in one.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import os
import random
import sys
from dataclasses import asdict
from typing import Iterator, List, Optional

from PIL import Image

# ── utilities/ on sys.path so TissueMask is importable ───────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILITIES = os.path.abspath(os.path.join(_HERE, '..', 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)

from TissueMask import TissueMask   # noqa: E402
from TileSampler import (RichnessConfig, SamplerConfig,  # noqa: E402
                         TileSampler, camera_plan)
from ConfigArgs import add_config_args, config_from_args   # noqa: E402

from config  import DomainGapConfig          # noqa: E402
from record  import FOVRecord                # noqa: E402
from camera  import CameraShot, Render         # noqa: E402
from SlideReader import SlideReader          # noqa: E402


#: The mixes the CLIs can name.
#:   default  `RichnessConfig()`: mostly tissue-dense FoVs, a share of edges,
#:            nothing above 85 per cent background (the production contract).
#:   open     no floors, any FoV up to 85 per cent background, first come over
#:            the shuffle -- what "any place with tissue" means.
RICHNESS_PRESETS = {
    'default': RichnessConfig(),
    'open': RichnessConfig(floors=(0.0,) * 7,
                           caps=(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)),
}


def sampler_cfg_for(n: int, seed: int = 0, richness: str = 'default') -> SamplerConfig:
    """`n` FoV positions under a named richness mix. Disjoint, no inheritance."""
    if richness not in RICHNESS_PRESETS:
        raise ValueError(f'richness must be one of {sorted(RICHNESS_PRESETS)}, '
                         f'got {richness!r}')
    return SamplerConfig(n_per_rung=n, seed=seed,
                         richness=RICHNESS_PRESETS[richness])


def add_domain_gap_args(ap, base: DomainGapConfig, skip=()) -> None:
    """One `--camera-*` flag per field of the camera's `DomainGapConfig`, e.g.
    `--camera-noise-sigma`, `--camera-rotation-choices`, `--camera-scale-range`.
    A flag not given leaves the field at what `base` holds (see `ConfigArgs`).
    `skip` names fields a caller sets itself (a bench sets `query_mpp` from each
    level), so no flag exists that would be overwritten."""
    add_config_args(ap, base, 'camera', skip=skip)


def domain_gap_from_args(args, base: DomainGapConfig, skip=()) -> DomainGapConfig:
    """`base` with every `--camera-*` flag that was given applied."""
    return config_from_args(args, base, 'camera', skip=skip)


def _slide_tag(wsi_path: str, max_len: int = 12) -> str:
    return os.path.splitext(os.path.basename(wsi_path))[0][:max_len]


def base_mask(wsi, masks) -> TissueMask:
    """The level-INDEPENDENT mask: the recipe's segmentation and region prep.

    `masks` is the caller's `TissueMaskConfig.MaskMaker` -- recipe, device and
    optional cache. Taken as an object and never imported, so query_sim stays
    importable without the segmenters behind it.

    Nothing here looks at a camera, so the result is valid for every pyramid
    level of the slide, and a caller that sweeps levels builds it once.
    """
    return masks.mask(wsi)[0]


def fov_plan_of(cam: Render):
    """The `RungPlan` that places `cam`'s FoVs: the footprint, the FoV
    rectangle and the reserve are the camera's (`ReadSpec.place`), at the
    FoV's downsample and the level `cam` reads from, so every position the
    sampler offers is one `capture_with_gt` can read."""
    # ds as `rect_w_l0 / output_w`, not `cam.ds`: the footprint is then the
    # FoV rectangle's own integer width, which the lattice steps by.
    return camera_plan(cam.wsi.level_downsamples, cam.spec,
                       ds=cam.rect_w_l0 / float(cam.output_w),
                       level=cam.level)


class FovSupply:
    """FoV positions drawn by a `TileSampler`, rendered by a `Render`.

        supply = FovSupply(cam, mask, SamplerConfig(n_per_rung=25))
        supply.sampler            # the first draw (its reports say what it saw)
        bank = supply.bank()      # one CameraShot per position of that draw
        for shot in supply: ...   # shots forever (see __iter__)

    `cfg` is WHERE (richness, overlap, count, seed); None is `SamplerConfig()`.
    How big a FoV is and what is read around it are the camera's. `plan` is
    optional: without one it is `fov_plan_of(cam)`; one passed in is checked
    against the camera and refused if it describes a different FoV.

    Draw 0 uses `cfg.seed`, draw k `cfg.seed + k`, so the sequence is fixed by
    the config and the same for every camera. Each draw happens on first use
    and is kept.
    """

    #: Draws in a row that may bring nothing new before the supply is called
    #: exhausted. One is not enough: a draw of 8 out of 26 can, by chance, be
    #: eight already seen.
    EXHAUSTED_AFTER = 3

    def __init__(self, cam: Render, mask: TissueMask,
                 cfg: Optional[SamplerConfig] = None, plan=None):
        if cfg is None:
            cfg = SamplerConfig()
        elif not isinstance(cfg, SamplerConfig):
            raise TypeError(f'FovSupply takes a SamplerConfig, got '
                            f'{type(cfg).__name__}')
        if plan is not None:
            ds = cam.rect_w_l0 / float(cam.output_w)
            tile = cam.spec.long_side
            if plan.tile_size != tile or abs(plan.rung_ds - ds) > 0.02 * ds:
                raise ValueError(
                    f'plan is tile {plan.tile_size} at ds {plan.rung_ds:g}, but '
                    f'this camera is a {tile} px footprint at ds {ds:g}')
        self.cam, self.mask, self.cfg = cam, mask, cfg
        self.plan = plan if plan is not None else fov_plan_of(cam)
        self._draws = {}          # draw index -> (TileSampler, [SampleMeta])

    @property
    def sampler(self) -> TileSampler:
        """The sampler of the FIRST draw (the one `bank` renders)."""
        return self.draw(0)[0]

    def draw(self, k: int):
        """(TileSampler, [SampleMeta]) of the `k`-th batch of positions. Each
        batch honours the quotas and the overlap bound on its own; two batches
        are drawn independently, so FoVs from different batches can overlap or
        coincide."""
        if k not in self._draws:
            cfg = self.cfg
            if k:
                cfg = dataclasses.replace(cfg, seed=cfg.seed + k)
            sampler = TileSampler(self.cam.wsi, self.mask, cfg).sample([self.plan])
            metas = [s.meta for s in sampler]
            if k == 0 and not metas:
                raise RuntimeError(
                    f'No FoV position for a {self.cam.rect_w_l0}x'
                    f'{self.cam.rect_h_l0} level-0 rect. What the sampler saw:\n'
                    f'{sampler.reports[self.plan.rung_ds].line()}')
            self._draws[k] = (sampler, metas)
        return self._draws[k]

    def _rng(self, meta, pass_index: int) -> random.Random:
        """Derived from the position and the pass, so a shot does not depend on
        what was drawn before it: the same FoV in the same pass is the same
        picture whichever draw found it, and the next pass is a different one."""
        key = f'{self.cfg.seed}|{meta.x}|{meta.y}|{meta.ds:g}|{pass_index}'
        return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))

    def shot(self, meta, draw: int, pass_index: int) -> Optional[CameraShot]:
        """One rendered FoV, or None if its read runs off the slide."""
        x0, y0, _, _ = meta.fov_rect
        image, params = self.cam.capture_with_gt(
            x0, y0, rng=self._rng(meta, pass_index))
        if image is None:
            return None
        return CameraShot(
            image=image, gt_x=x0, gt_y=y0, params=params,
            bucket=meta.bucket, fov_background=meta.score,
            origin=meta.origin, overlap_max=meta.overlap_max,
            draw_index=draw, pass_index=pass_index)

    def bank(self) -> List[CameraShot]:
        """One rendered shot per position of the FIRST draw -- as many as
        `n_per_rung` allows the slide to give. A position whose read runs off
        the slide is skipped, so the list can be shorter than the draw."""
        _, metas = self.draw(0)
        shots = (self.shot(m, 0, 0) for m in metas)
        return [s for s in shots if s is not None]

    def __iter__(self) -> Iterator[CameraShot]:
        """Shots forever.

        A batch of positions is drawn, rendered, and the next batch is drawn
        with the next seed, and so on -- so the FoVs keep being new places on
        the slide. A position already shown is not shown again. When several
        draws in a row bring nothing new the slide is out of positions, and
        from then on the ones it gave are repeated, each pass with a fresh
        domain gap. A position whose read runs off the slide is skipped, and if
        every one does this raises rather than spinning.
        """
        seen = set()
        shown = []                     # (meta, draw), first-show order
        k = 0
        barren = 0
        while barren < self.EXHAUSTED_AFTER:
            _, metas = self.draw(k)
            fresh = [m for m in metas if (m.x, m.y) not in seen]
            barren = 0 if fresh else barren + 1
            for meta in fresh:
                seen.add((meta.x, meta.y))
                shot = self.shot(meta, k, 0)
                if shot is None:
                    continue
                shown.append((meta, k))
                yield shot
            k += 1
        if not shown:
            raise RuntimeError(
                'Every FoV position reads off the slide; nothing to yield')
        # The slide has given everything it has. Repeat it, each pass a new
        # domain gap: the gap is unbounded, so from here the supply is too.
        pass_index = 1
        while True:
            for meta, d in shown:
                shot = self.shot(meta, d, pass_index)
                if shot is not None:
                    yield shot
            pass_index += 1


def _record_from_shot(
    shot:     CameraShot,
    filename: str,
    wsi_path: str,
    cfg:      DomainGapConfig,
    fov_w:    int,
    fov_h:    int,
    level:    int = 0,
) -> FOVRecord:
    p = shot.params
    return FOVRecord(
        filename      = filename,
        wsi_path      = wsi_path,
        level         = level,
        wh_ratio      = cfg.wh_ratio,
        MPixels       = cfg.MPixels,
        query_mpp     = cfg.query_mpp,
        nominal_mpp   = cfg.query_mpp,
        effective_mpp = float(p['effective_mpp']),
        fov_width     = fov_w,
        fov_height    = fov_h,
        gt_x          = shot.gt_x,
        gt_y          = shot.gt_y,
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


def generate(
    wsi_path:      str,
    out_dir:       str,
    n:             int,
    cfg:           Optional[DomainGapConfig] = None,
    seed:          int    = 0,
    sampler_cfg:   Optional[SamplerConfig] = None,
    *,
    masks,
) -> List[FOVRecord]:
    """Generate `n` synthetic FOVs into `out_dir/images/` + `out_dir/gt.csv`.

    `masks` is a `TissueMaskConfig.MaskMaker`; see `base_mask`. `sampler_cfg`
    decides where the FoVs are; the default asks for `n` positions under the
    default richness mix, and iterating draws further batches (next seed) if
    more shots are needed. Only when the slide has no new positions left do
    shots repeat one, each with a fresh domain gap."""
    cfg = cfg or DomainGapConfig()

    cam = Render(SlideReader(wsi_path), cfg=cfg, seed=seed)
    print(f'FOV spec  : {cfg.wh_ratio}  {cfg.MPixels}MP  @ mpp={cfg.query_mpp}', flush=True)
    print(f'            output {cam.output_w}x{cam.output_h} px, '
          f'level-0 rect {cam.rect_w_l0}x{cam.rect_h_l0}', flush=True)
    print(f'            bounding square side (level-0) = {cam.bounding_square_side_l0}  '
          f'(rotation-safe read window)', flush=True)

    print(f'Building tissue mask ({masks.cfg.seg_id()}) ...', flush=True)
    mask = base_mask(cam.wsi, masks)
    print(f'            tissue_frac={mask.tissue_fraction()*100:.1f}%, '
          f'regions={len(mask.tissue_regions)}', flush=True)
    supply = FovSupply(cam, mask, sampler_cfg or sampler_cfg_for(n, seed))
    # Drawn here, not on the first shot, so a slide with no room says so before
    # any image is written -- and says what it saw: per bucket, asked vs taken.
    print(next(iter(supply.sampler.reports.values())).line(), flush=True)

    slide_tag = _slide_tag(wsi_path)
    img_dir   = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    gt_path   = os.path.join(out_dir, 'gt.csv')

    records: List[FOVRecord] = []
    for shot in supply:
        if len(records) >= n:
            break
        idx = len(records)
        fname = f'{slide_tag}_syn{idx:05d}.png'
        Image.fromarray(shot.image).save(os.path.join(img_dir, fname))
        records.append(_record_from_shot(
            shot, fname, wsi_path, cfg, cam.output_w, cam.output_h,
        ))
        print(f'  [saved] {len(records)}/{n}  {fname}', flush=True)

    with open(gt_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    print(f'\n{len(records)} synthetic FOVs -> {img_dir}')
    print(f'GT -> {gt_path}')
    return records
