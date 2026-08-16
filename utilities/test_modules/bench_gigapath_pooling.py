#!/usr/bin/env python3
"""Tile-level retrieval bench: does a different pooling find the tile GigaPath's
CLS token misses?

Retrieval's failures split three ways on the synthetic corpus -- rank-1 correct
43.2%, truth ranked but not first 24.5%, truth never proposed 32.3%. The last
bucket is a statement about the descriptor, and the descriptor throws away 196 of
its 197 tokens (timm pools with global_pool='token'). This bench asks whether
keeping some of them helps, without touching retrieval or the pipeline.

Two phases, because they cost very different things:

    --phase dump    reads the WSIs, encodes, writes token stores. Needs a GPU
                    (one is enough) and a few hours for the full corpus.
    --phase eval    reads only the stores. No GPU, no WSI, no model -- runs on a
                    login node in seconds, so every pooling idea after the first
                    is free.

The shape of the question
-------------------------
For each (slide, level):

    reference set   tiles at the PRODUCTION grid positions -- main grid and the
                    half-tile-shifted overlap grid, the same two the retriever
                    scores. Coordinates come from PatchGrid.from_size, which is
                    pure geometry, so nothing large is read: a mask_all region at
                    L0 is 18.7 Gpx and reading it whole once cost 3h43m.

    query set       whole FoVs photographed by query_sim's Camera, then cut into
                    5x4 tiles. Going through a FoV rather than augmenting each
                    tile is not fussiness: vignette, field mask and lens
                    distortion are all defined relative to the image they are
                    given (field.py:19-23, lens.py:12-13), so applied to a 256px
                    tile they paint a radial gradient centred on every tile --
                    which is exactly what ring pooling measures, and would decide
                    the comparison on an artefact.

    the answer      computed, not searched: Camera.output_tile_origins inverts the
                    capture to a level-0 coordinate, and the nearest tile in each
                    of the two grids is an answer. Both are recorded separately,
                    since which one wins is the measurable half of "is the
                    half-tile overlap grid earning its keep".

delta -- how far a query tile sits from the grid position it is matched to -- is
recorded, not swept. A FoV lands wherever it lands, and its 20 tiles come with a
natural spread. The union of the two grids is a checkerboard, so delta reaches
128px (half a tile) rather than 64: the deep holes are at points like (128, 0),
equidistant from (0,0) and (128,128).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('utilities', 'aiNNModel', 'query_sim', 'utilities/test_modules'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import FeatureStore as FS                                           # noqa: E402
from PatchingLib import PatchGrid                                   # noqa: E402
import ReferenceSampler as RS                                       # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TissuesRegionsMask import TissuesRegionsMask                   # noqa: E402
from TissueSegFunc import HestSegConfig                             # noqa: E402
from GigaPathFunc import GigaPathEncoderConfig, pool_tokens         # noqa: E402
from camera import Camera                                           # noqa: E402
from config import DomainGapConfig                                  # noqa: E402

TILE = 256
POOLINGS = ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2')

#: Covering radius of the main and overlap grids together, in level-n px. Main
#: sits at (256i, 256j) and overlap at (256i+128, 256j+128), so the union is a
#: checkerboard whose deep holes, like (128, 0), are 128 from every occupied
#: point. A query further than this from both is in the uncovered margin at a
#: region edge -- see dump_one. inspect_feature_store.py gates on the same
#: number, and must, or the gate and the dump would disagree about what is legal.
DELTA_UNION = 128.0


# ── grid coordinates, without reading any pixels ──────────────────────────────

def grid_coords(mask, level: int, ds: float, tile_size: int = TILE):
    """Every production grid position at this level, as arrays.

    Returns (xy, region, rowcol, kind) where xy is [n, 2] level-0 top-left.
    PatchGrid.from_size takes sizes and offsets only -- no image -- so this costs
    nothing even where the region is the whole slide. PatchInfo.kind already
    separates the main grid from the half-tile-shifted one, so the two grids need
    no arithmetic here.
    """
    xs, ys, regs, rows, cols, kinds = [], [], [], [], [], []
    for ri, region in enumerate(mask.tissue_regions):
        w_n, h_n = int(region.w / ds), int(region.h / ds)
        if w_n < tile_size or h_n < tile_size:
            continue
        grid = PatchGrid.from_size(
            w_n, h_n, tile_size, overlap=True,
            x_offset=int(region.x / ds), y_offset=int(region.y / ds),
            ds=ds, level=level,
        )
        for info in grid.iter_infos():
            xs.append(int(round(info.x * ds)))
            ys.append(int(round(info.y * ds)))
            regs.append(ri)
            rows.append(info.row)
            cols.append(info.col)
            kinds.append(0 if info.kind == 'main' else 1)
    if not xs:
        return None
    return (np.array([xs, ys], dtype=np.int64).T,
            np.array(regs, dtype=np.int64),
            np.array([rows, cols], dtype=np.int64).T,
            np.array(kinds, dtype=np.int64))


def nearest_in(centres: np.ndarray, pts: np.ndarray, chunk: int = 4096):
    """Index of the nearest row of `centres` for each row of `pts`, plus the
    offset. Brute force in chunks -- the arrays are ~1e5 by ~1e3, which numpy
    does in a second, and an exact answer beats a spatial index that would need
    its own test."""
    idx = np.empty(len(pts), dtype=np.int64)
    off = np.empty((len(pts), 2), dtype=np.float64)
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        d = ((centres[None, :, 0] - p[:, None, 0]) ** 2
             + (centres[None, :, 1] - p[:, None, 1]) ** 2)
        k = d.argmin(axis=1)
        idx[s:s + chunk] = k
        off[s:s + chunk] = p - centres[k]
    return idx, off


# ── one (slide, level) ────────────────────────────────────────────────────────

def dump_one(wsi_path: str, level: int, out_root: Path, *,
             mask, encoder, device, spec,
             k: int, k_floor: int, n_query: int, seed: int,
             mask_id: str, encoder_id: str, batch_size: int,
             sampler_cfg: RS.SamplerConfig,
             min_std: float = 8.0, rots=(0, 90)) -> dict:
    slide = SafeSlide(wsi_path)
    stem = Path(wsi_path).stem
    base_mpp = slide.base_mpp  # SafeSlide.base_mpp: mean of mpp-x/y, one definition
    ds = float(slide.level_downsamples[level])
    level_mpp = base_mpp * ds

    g = grid_coords(mask, level, ds)
    if g is None:
        print(f'  L{level}: no region can host a {TILE}px tile -- skipped', flush=True)
        slide.close()
        return {}
    grid_xy, grid_region, grid_rowcol, grid_kind = g
    n_grid_positions = len(grid_xy)
    gcen = grid_xy + (TILE * ds) / 2.0                 # grid tile centres, level-0

    # queries: FoVs -> tiles -> level-0 centres -> the two grid answers
    cfg = DomainGapConfig(
        wh_ratio='45:32', MPixels=1.47456, query_mpp=level_mpp,
        angle_jitter_deg=0.0, scale_range=(1.0, 1.0), query_mpp_jitter=0.0,
        stage_shift_max=0,
    )
    cam = Camera(slide, cfg=cfg, mask=mask, seed=seed)
    per_fov = (cam.qfw.output_h // TILE) * (cam.qfw.output_w // TILE)
    n_fov = max(1, math.ceil(n_query / (per_fov * len(rots))))

    rng = np.random.default_rng(seed)
    query_imgs, query_centres, query_rots, query_fov_ids, query_rowcol = [], [], [], [], []
    n_fov_made = 0
    tries = 0
    while n_fov_made < n_fov and tries < n_fov * 20:
        tries += 1
        ri = int(rng.integers(0, len(mask.tissue_regions)))
        reg = mask.tissue_regions[ri]
        if reg.w < cam.qfw.rect_w_l0 * 2 or reg.h < cam.qfw.rect_h_l0 * 2:
            continue
        x = int(rng.integers(reg.x, reg.x + reg.w - cam.qfw.rect_w_l0))
        y = int(rng.integers(reg.y, reg.y + reg.h - cam.qfw.rect_h_l0))
        shot = [cam.capture_with_gt(x, y, rotation=r) for r in rots]
        if any(im is None for im, _ in shot):
            continue
        # A region's bbox contains plenty of blank glass, and a FoV that lands on
        # it yields 20 featureless tiles -- questions with no answer that every
        # pooling gets wrong equally, diluting the comparison. Judge the shot,
        # not the mask, for the same reason the coordinate test does.
        if float(np.asarray(shot[0][0], dtype=np.float32).std()) < min_std:
            continue
        for (img, _), r in zip(shot, rots):
            for rr, cc, u, v, cx, cy in cam.output_tile_origins(
                    x, y, TILE, rot_deg=r, scale=1.0):
                query_imgs.append(np.ascontiguousarray(img[v:v + TILE, u:u + TILE]))
                query_centres.append((cx, cy))
                query_rots.append(r)
                query_fov_ids.append(n_fov_made)
                query_rowcol.append((rr, cc))
        n_fov_made += 1

    if not query_imgs:
        print(f'  L{level}: no FoV could be placed -- skipped', flush=True)
        slide.close()
        return {}

    query_centres = np.asarray(query_centres, dtype=np.float64)
    main_idx = np.where(grid_kind == 0)[0]
    ovlp_idx = np.where(grid_kind == 1)[0]
    am_rel, ans_main_offset = nearest_in(gcen[main_idx], query_centres)
    ao_rel, ans_ovlp_offset = nearest_in(gcen[ovlp_idx], query_centres) if len(ovlp_idx) else (am_rel, ans_main_offset)
    ans_main_g = main_idx[am_rel]
    ans_ovlp_g = ovlp_idx[ao_rel] if len(ovlp_idx) else ans_main_g

    # Drop the tiles no grid position covers.
    #
    # from_size lays whole tiles only, so a region's right and bottom keep a
    # margin of up to 255 px with no main grid point in it, and the overlap grid
    # is inset another 128. A FoV near the region edge puts tiles in that margin,
    # where the nearest grid position sits 300+ px away -- past a 256 px tile, so
    # the "answer" shares no pixels with the query. Every pooling gets those
    # wrong, equally, which is dilution rather than evidence -- the same reason
    # min_std rejects blank glass above.
    #
    # 128 is the covering radius of the two grids together: main at (256i, 256j)
    # and overlap at (256i+128, 256j+128) form a checkerboard whose deep holes
    # are 128 from every occupied point. Inside the grids' extent this bound
    # always holds, so what it removes is exactly the uncovered margin -- 0.52%
    # of queries measured over 23 combinations, losing no FoV entirely.
    near = np.minimum(np.hypot(ans_main_offset[:, 0], ans_main_offset[:, 1]),
                      np.hypot(ans_ovlp_offset[:, 0], ans_ovlp_offset[:, 1]))
    keep = near <= DELTA_UNION * ds
    if not keep.all():
        n_drop = int((~keep).sum())
        # Split by rotation, because two different things put a tile in the
        # margin and they call for different fixes. The FoV rect is sampled
        # inside the region bbox, but only unrotated: at rot 90 the footprint is
        # the rect turned about its centre, 208 level-0 px taller here, and that
        # overhang is not in the sampling constraint. If the drops sit at one
        # rotation, tighten the sampling; if they are even, it is purely the
        # margin PatchGrid leaves and no sampling change would help.
        rot_a = np.asarray(query_rots)[~keep]
        by_rot = '  '.join(f'rot{r}={int((rot_a == r).sum())}' for r in rots)
        query_imgs = [im for im, k in zip(query_imgs, keep) if k]
        query_centres, query_rots = query_centres[keep], [r for r, k in zip(query_rots, keep) if k]
        query_fov_ids = [f for f, k in zip(query_fov_ids, keep) if k]
        query_rowcol = [rc for rc, k in zip(query_rowcol, keep) if k]
        ans_main_offset, ans_ovlp_offset = ans_main_offset[keep], ans_ovlp_offset[keep]
        ans_main_g, ans_ovlp_g = ans_main_g[keep], ans_ovlp_g[keep]
        print(f'  L{level}: dropped {n_drop} of {len(keep)} query tiles that fell '
              f'in the uncovered margin at a region edge  ({by_rot})', flush=True)

    # reference: a QUOTA-CONTROLLED draw, plus every answer.
    #
    # The draw used to be uniform over the grid, which sounds neutral and is
    # not: 46% of level-0 grid positions on BRACS_1228 are pure background
    # (result/RefStore, white p50 = 0.72, p75 = 1.00). Those are distractors
    # that can never outrank an answer, so a nominal pool of 3000 was an
    # effective pool of roughly half that -- and the share differs per level, so
    # the per-level numbers were comparing descriptor difficulty and pool
    # composition at the same time. ReferenceSampler holds the composition
    # fixed, which is what makes one pooling comparable with another.
    #
    # The answers stay mandatory whatever the quota chose: an answer missing
    # from the pool is an unanswerable question that scores as a miss.
    white = mask.white_fractions(grid_xy, level, TILE)
    level_cfg = dataclasses.replace(sampler_cfg,
                                n_target=min(n_grid_positions,
                                             max(int(round(k / (ds * ds))),
                                                 k_floor)),
                                seed=seed)
    level_geom = RS.LevelGeoms(level=level, ds=ds, footprint_l0=int(TILE * ds),
                         xy=grid_xy, region=grid_region.astype(np.int32),
                         grid_rc=grid_rowcol.astype(np.int32),
                         kind=grid_kind.astype(np.int8), white=white,
                         bucket=RS.assign_buckets(white, level_cfg))
    plan = RS.plan_level(level_geom, level_cfg)
    sampler = RS.ReferenceSampler({level: level_geom}, level_cfg, mask=mask)
    sample = sampler.plan(level, plan)

    ans_all = np.unique(np.concatenate([ans_main_g, ans_ovlp_g]))
    already_chosen = set(zip(sample.x.tolist(), sample.y.tolist()))
    answers_to_add = np.array([i for i in ans_all
                    if (int(grid_xy[i, 0]), int(grid_xy[i, 1])) not in already_chosen],
                   dtype=np.int64)

    ref_xy = np.concatenate([np.stack([sample.x, sample.y], 1), grid_xy[answers_to_add]])
    ref_region = np.concatenate([sample.region, grid_region[answers_to_add].astype(np.int32)])
    ref_rowcol = np.concatenate([sample.grid_rc, grid_rowcol[answers_to_add].astype(np.int32)])
    ref_kind = np.concatenate([sample.kind, grid_kind[answers_to_add].astype(np.int8)])
    ref_white = np.concatenate([sample.white_frac, white[answers_to_add]])
    ref_bucket = np.concatenate([sample.bucket, level_geom.bucket[answers_to_add]])
    ref_origin = np.concatenate([sample.origin, np.zeros(len(answers_to_add), np.int8)])
    ref_parent_x = np.concatenate([sample.parent_x,
                                   np.full(len(answers_to_add), -1, np.int64)])
    ref_parent_y = np.concatenate([sample.parent_y,
                                   np.full(len(answers_to_add), -1, np.int64)])

    # Keyed by COORDINATE, not by grid index: a displaced tile has no grid
    # index, and an answer is always a grid position, so the coordinate is the
    # one key both sides share.
    where = {(int(x), int(y)): i for i, (x, y) in enumerate(ref_xy)}

    print(f'  L{level}  mpp={level_mpp:.4f}  grid={n_grid_positions:,} '
          f'(main {len(main_idx):,} / ovlp {len(ovlp_idx):,})  '
          f'ref={len(ref_xy):,} (quota {plan.got} + answers {len(answers_to_add)})  '
          f'queries={len(query_imgs):,} from {n_fov_made} FoV', flush=True)
    bucket_counts = '  '.join(f'{b}={int((ref_bucket == i).sum())}'
                    for i, b in enumerate(RS.BUCKETS))
    print(f'      sampler {level_cfg.sampler_id()}   {bucket_counts}   '
          f'displaced={int((ref_origin == 1).sum())}', flush=True)

    # read the reference tiles, discarding the ones the scanner never
    # photographed.
    #
    # SafeSlide fills a hole and any unscanned canvas with a flat colour, and a
    # flat tile encodes to the same vector at every level -- up to 13% of one
    # slide's deep bank turned out to be bit-identical twins that way. As a
    # distractor such a tile is worse than useless: it is a pool slot that can
    # never outrank anything.
    #
    # An ANSWER is kept whatever its validity. Dropping it would make that query
    # unanswerable, which is the one thing this store's construction forbids --
    # so its valid_frac is recorded instead and the eval can decide. A distractor
    # is replaced from its OWN bucket, because holes come in contiguous patches
    # and topping up from anywhere would move the composition the quota holds.
    answer_coords = {(int(grid_xy[i, 0]), int(grid_xy[i, 1])) for i in ans_all}
    pending = [dict(x=int(x), y=int(y), row=i) for i, (x, y) in enumerate(ref_xy)]

    t0 = time.time()
    ref_imgs, kept_rows, valid_fracs = [], [], []
    n_rejected = n_replaced = n_bad_answers = 0
    position = 0
    while position < len(pending):
        item = pending[position]
        position += 1
        image, valid = slide.read_region_valid((item['x'], item['y']), level,
                                               (TILE, TILE))
        valid_fraction = float(valid.mean())
        is_answer = (item['x'], item['y']) in answer_coords
        if valid_fraction < level_cfg.min_valid and not is_answer:
            n_rejected += 1
            more = sampler.replace(level, int(ref_bucket[item['row']]), n=1)
            if more:
                new_x, new_y, grid_index = more[0]
                pending.append(dict(x=new_x, y=new_y, row=None,
                                    grid_index=grid_index))
                n_replaced += 1
            continue
        if valid_fraction < level_cfg.min_valid:
            n_bad_answers += 1
        ref_imgs.append(image)
        kept_rows.append(item)
        valid_fracs.append(valid_fraction)
    t_read = time.time() - t0

    def _column(source_array, grid_array, dtype):
        return np.array(
            [source_array[item['row']] if item['row'] is not None
             else grid_array[item['grid_index']] for item in kept_rows],
            dtype=dtype)

    ref_xy = np.array([[item['x'], item['y']] for item in kept_rows],
                      dtype=np.int64)
    ref_region = _column(ref_region, grid_region, np.int32)
    ref_rowcol = np.array(
        [ref_rowcol[item['row']] if item['row'] is not None
         else grid_rowcol[item['grid_index']] for item in kept_rows],
        dtype=np.int32).reshape(len(kept_rows), 2)
    ref_kind = _column(ref_kind, grid_kind, np.int8)
    ref_white = _column(ref_white, white, np.float32)
    ref_bucket = _column(ref_bucket, level_geom.bucket, np.int8)
    ref_origin = np.array([ref_origin[item['row']] if item['row'] is not None
                           else 0 for item in kept_rows], dtype=np.int8)
    ref_parent_x = np.array([ref_parent_x[item['row']] if item['row'] is not None
                             else -1 for item in kept_rows], dtype=np.int64)
    ref_parent_y = np.array([ref_parent_y[item['row']] if item['row'] is not None
                             else -1 for item in kept_rows], dtype=np.int64)
    ref_valid = np.array(valid_fracs, dtype=np.float32)
    where = {(int(x), int(y)): i for i, (x, y) in enumerate(ref_xy)}

    if n_rejected or n_bad_answers:
        print(f'      holes: {n_rejected} distractors below valid '
              f'{level_cfg.min_valid:.2f} ({n_replaced} replaced), '
              f'{n_bad_answers} answers kept anyway', flush=True)
    ref_tokens = encoder.tokens(ref_imgs)
    query_tokens = encoder.tokens(query_imgs)
    print(f'      read {t_read:.0f}s   encode {time.time() - t0 - t_read:.0f}s',
          flush=True)

    common = dict(wsi_stem=stem, wsi_path=str(wsi_path), level=level, ds=ds,
                  mpp=level_mpp, base_mpp=base_mpp, tile_size=TILE, overlap=True,
                  dim=spec['dim'], token_grid=tuple(spec['token_grid']),
                  num_prefix=spec['num_prefix'], encoder_id=encoder_id,
                  mask_id=mask_id, coverage='sample', sample_seed=seed,
                  sampler_id=level_cfg.sampler_id())

    written = {}
    for tag, tok in (('ref', ref_tokens), ('query', query_tokens)):
        feats, slots, layout = pool_tokens(tok, 'tokens', spec)
        n = feats.shape[0]
        if tag == 'ref':
            xy, reg, rc = ref_xy, ref_region, ref_rowcol
            extra = {
                'kind': torch.from_numpy(ref_kind.astype(np.int16)),
                'white_frac': torch.from_numpy(ref_white),
                'bucket': torch.from_numpy(ref_bucket),
                'origin': torch.from_numpy(ref_origin),
                'parent_x': torch.from_numpy(ref_parent_x),
                'parent_y': torch.from_numpy(ref_parent_y),
                'valid_frac': torch.from_numpy(ref_valid),
            }
            pooling = 'tokens'
        else:
            xy = np.stack([query_centres[:, 0] - TILE * ds / 2,
                           query_centres[:, 1] - TILE * ds / 2], 1).astype(np.int64)
            reg = np.full(n, -1, dtype=np.int64)
            rc = np.asarray(query_rowcol, dtype=np.int64)
            extra = {
                'ans_main': torch.tensor(
                    [where[(int(grid_xy[v, 0]), int(grid_xy[v, 1]))]
                     for v in ans_main_g], dtype=torch.int32),
                'ans_ovlp': torch.tensor(
                    [where[(int(grid_xy[v, 0]), int(grid_xy[v, 1]))]
                     for v in ans_ovlp_g], dtype=torch.int32),
                # level-n px, so it is comparable across levels
                'delta_main': torch.from_numpy((ans_main_offset / ds).astype(np.int16)),
                'delta_ovlp': torch.from_numpy((ans_ovlp_offset / ds).astype(np.int16)),
                'fov_id': torch.tensor(query_fov_ids, dtype=torch.int32),
                'rot': torch.tensor(query_rots, dtype=torch.int32),
            }
            pooling = 'query_tokens'

        meta = FS.StoreMeta(pooling=pooling, slots=slots, slot_layout=layout,
                            n_available=(n_grid_positions if tag == 'ref' else n),
                            n_tiles=n, **common)
        p = FS.save(out_root, meta=meta,
                    features=feats.to(torch.float16),
                    x=torch.from_numpy(xy[:, 0].astype(np.int32)),
                    y=torch.from_numpy(xy[:, 1].astype(np.int32)),
                    region=torch.from_numpy(reg.astype(np.int16)),
                    grid_rc=torch.from_numpy(rc.astype(np.int32)),
                    extra=extra)
        written[tag] = p
        print(f'      {tag:5s} -> {p.name}  '
              f'{p.stat().st_size / 1e9:.2f} GB', flush=True)

    slide.close()
    return written


# ── eval ──────────────────────────────────────────────────────────────────────
#
# Reads stores only -- no GPU, no WSI, no model. The cross-level answers are not
# in the dump (ans_main / ans_ovlp index the SAME level) but they do not need to
# be: every store carries each tile's level-0 x/y, so "which tile of ref(L-1)
# covers this query" is a nearest-neighbour question answerable from
# coordinates. That is what those four tensors are for.

def _centres(t, ds: float) -> np.ndarray:
    return np.stack([t['x'].numpy().astype(np.float64) + TILE * ds / 2.0,
                     t['y'].numpy().astype(np.float64) + TILE * ds / 2.0], 1)


def combine_slots(qf: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """[Nq, n, D] x [Nr, n, D] -> [Nq, Nr]: the mean of the per-slot cosines.

    Every slot is already unit norm, so this averages similarities rather than
    computing the similarity of an average -- which is the whole reason slots are
    stored stacked instead of concatenated. How slots SHOULD be weighted is an
    open question, so it lives in this one replaceable function rather than being
    baked into the vectors at dump time.
    """
    return torch.einsum('qnd,rnd->qr', qf, rf) / qf.shape[1]


# ── whitening ─────────────────────────────────────────────────────────────────
#
# Cosine similarity is dominated by whichever directions carry the most variance,
# and on one slide those directions encode what the slide IS -- stain, tissue
# type, scanner. Every tile shares them, so they add a large near-constant to
# every similarity and compress the range the ranking is decided in. The
# location-specific signal lives in low-variance directions underneath.
#
# Whitening rescales each principal direction to equal variance; dropping the top
# k removes the shared ones outright. Both are closed form -- no training, no
# labels.
#
# Fitted on the REFERENCE pool, never on queries. That is not a convenience: the
# reference pool is built from the WSI during build(), before any query exists,
# so a per-slide transform fitted this way is something a deployment can actually
# do. Fitting on queries would be test-time leakage and would not transfer.
#
# Reported alongside the identity so the comparison is against the current
# production path, not against another variant.

#: p = the power the eigenvalues are divided by (1.0 is full whitening, 0.5 the
#: usual partial compromise -- full whitening amplifies the smallest directions,
#: which is where noise lives). dropN removes the N leading directions and
#: rescales nothing. 'centre' isolates how much of any gain is mean removal
#: alone, which costs one subtraction and is worth knowing separately.
WHITENS = ('none', 'centre', 'drop1', 'drop4', 'p0.5', 'p1.0')


def _fit_whiten(ref: torch.Tensor):
    """Mean and eigenbasis of one slot of the reference pool, descending.

    Covariance plus eigh, not SVD of the data matrix: D is 1536 and Nr a few
    thousand, so the DxD route is the small one, and it is reused by every
    variant -- the decomposition is fitted once and applied six times.
    """
    mu = ref.mean(0, keepdim=True)
    rc = (ref - mu).double()
    cov = (rc.T @ rc) / max(1, rc.shape[0] - 1)
    lam, vec = torch.linalg.eigh(cov)              # ascending
    return mu, lam.flip(0).clamp_min(1e-10).float(), vec.flip(1).float()


def _apply_whiten(x: torch.Tensor, mu, lam, vec, spec: str) -> torch.Tensor:
    if spec == 'none':
        return x
    z = (x - mu) @ vec
    if spec.startswith('drop'):
        z = z[:, int(spec[4:]):]
    elif spec.startswith('p'):
        z = z * lam.pow(-float(spec[1:]) / 2)
    # 'centre' is the rotation alone; an orthogonal rotation does not change
    # cosine, so what it measures is exactly the mean removal.
    return torch.nn.functional.normalize(z, dim=-1)


def _rank_stats(sim: torch.Tensor, ans: np.ndarray):
    """(rank of the answer, fraction of the pool that beat it)."""
    n_pool = sim.shape[1]
    a = sim[np.arange(len(ans)), ans]
    better = (sim > a[:, None]).sum(dim=1).numpy()
    return better + 1, better / max(1, n_pool - 1)


def _spec_of(m) -> dict:
    return {'dim': m.dim, 'token_grid': m.token_grid, 'num_prefix': m.num_prefix}


def _table(rows, head, title, note=''):
    out = [f'\n{title}']
    if note:
        out.append(f'  {note}')
    out.append('  ' + head)
    out.extend('  ' + r for r in rows)
    return out


def eval_one(query_path: Path, query_meta, refs: dict, poolings, rec: dict = None,
             whitens=WHITENS) -> list:
    """One (slide, level): three tables, each answering a different question.

    Fills `rec` with the same numbers so eval_all can ask the question no single
    combination can answer -- whether an ordering holds everywhere.
    """
    query_tensors, _ = FS.load(query_path)

    # Same rule as dump_one, applied again here so a store written before that
    # rule existed still scores correctly. A query further than DELTA_UNION from
    # both grids sits in the margin PatchGrid leaves at a region edge: no
    # reference tile shares pixels with it, so every pooling misses it equally
    # and it only dilutes the comparison. Measured at 0.52% of queries.
    _near = np.minimum(
        np.hypot(*query_tensors['delta_main'].numpy().astype(np.float64).T),
        np.hypot(*query_tensors['delta_ovlp'].numpy().astype(np.float64).T))
    _keep = torch.from_numpy(_near <= DELTA_UNION)
    if not bool(_keep.all()):
        query_tensors = {k: v[_keep] for k, v in query_tensors.items()}

    query_centres = _centres(query_tensors, query_meta.ds)
    n_queries = query_tensors['features'].shape[0]

    loaded, ans = {}, {}
    for level_delta, (ref_path, ref_meta) in refs.items():
        ref_tensors, _ = FS.load(ref_path)
        loaded[level_delta] = (ref_tensors, ref_meta)
        ans[level_delta], _ = nearest_in(_centres(ref_tensors, ref_meta.ds), query_centres)

    sampler_name = getattr(query_meta, 'sampler_id', '') or 'uniform (pre-quota)'
    if rec is not None:
        rec['sampler'] = sampler_name
    lines = [f'\n{"=" * 74}',
             f'sampler {sampler_name}',
             f'{query_meta.wsi_stem}  L{query_meta.level}   queries={n_queries}   '
             + '   '.join(f'ref L{query_meta.level + level_delta}={loaded[level_delta][0]["features"].shape[0]:,}'
                          for level_delta in sorted(loaded))]

    # δ is a per-FoV property (the FoV's tiles and the grid share a 256 lattice,
    # so one shot yields one offset), and it is what the third table splits on.
    dmain = np.hypot(*query_tensors['delta_main'].numpy().astype(np.float64).T)
    dovlp = np.hypot(*query_tensors['delta_ovlp'].numpy().astype(np.float64).T)
    delta = np.minimum(dmain, dovlp)
    rc = query_tensors['grid_rc'].numpy()
    n_rows, n_cols = rc[:, 0].max() + 1, rc[:, 1].max() + 1
    edge = ((rc[:, 0] == 0) | (rc[:, 0] == n_rows - 1)
            | (rc[:, 1] == 0) | (rc[:, 1] == n_cols - 1))
    # Half the shots are rendered at 90 deg. Kept apart rather than averaged:
    # GigaPath is not rotation invariant, some poolings are (rings are concentric,
    # cls_avg/cls_std are global) and one is not (grid2x2's quadrants permute), so
    # a single number would score "better descriptor" and "happens to be rotation
    # invariant" as the same thing.
    rot = query_tensors['rot'].numpy()

    pooled_q, pooled_r = {}, {}
    for mode in poolings:
        pooled_q[mode] = pool_tokens(query_tensors['features'].float(), mode, _spec_of(query_meta))[0]
        for level_delta, (ref_tensors, ref_meta) in loaded.items():
            pooled_r[(mode, level_delta)] = pool_tokens(ref_tensors['features'].float(), mode,
                                               _spec_of(ref_meta))[0]

    # One decomposition per (pooling, slot), reused by every whitening variant.
    # Only the same-level pool: whitening across scales would mix two questions.
    fitted = {}
    if whitens:
        for mode in poolings:
            rf = pooled_r[(mode, 0)]
            for s in range(rf.shape[1]):
                fitted[(mode, s)] = _fit_whiten(rf[:, s, :])

    # ── 1. each level searched on its own ────────────────────────────────────
    head = f'{"pooling":10s}' + ''.join(
        f'{f"L{query_meta.level + level_delta:+d}" if level_delta else "L (same)":>12s}{"":>10s}'
        for level_delta in sorted(loaded))
    head = f'{"pooling":10s}' + ''.join(
        f'{("L%+d" % level_delta if level_delta else "L"):>9s}{"r@1":>7s}{"pct50":>8s}'
        for level_delta in sorted(loaded))
    rows = []
    for mode in poolings:
        cells = []
        for level_delta in sorted(loaded):
            ref_meta = loaded[level_delta][1]
            rank, pct = _rank_stats(combine_slots(pooled_q[mode],
                                                  pooled_r[(mode, level_delta)]), ans[level_delta])
            cells.append(f'{f"x{ref_meta.ds / query_meta.ds:.2f}":>9s}'
                         f'{np.mean(rank == 1) * 100:6.1f}%'
                         f'{np.median(pct) * 100:7.2f}%')
            if rec is not None and level_delta == 0:
                rec['p1'][mode] = (float(np.mean(rank == 1) * 100),
                                   float(np.median(pct) * 100))
                rec['pool'] = int(pooled_r[(mode, 0)].shape[0])
            if rec is not None and level_delta != 0:
                # Adjacent-level ds ratio names the pyramid without opening the
                # slide: 4.0 on an SVS, 2.0 on a MIRAX.
                rec['step'] = round(max(ref_meta.ds / query_meta.ds, query_meta.ds / ref_meta.ds), 1)
        rows.append(f'{mode:10s}' + ''.join(cells))
    lines += _table(rows, head, 'phase 1 -- each level searched on its own',
                    'r@1 = answer ranked first;  pct50 = median fraction of the '
                    'pool that beat it (comparable across pool sizes)')

    # ── 2. all levels in one pool: does it confuse the scale? ────────────────
    if len(loaded) > 1:
        order = sorted(loaded)
        offs, base = {}, 0
        for level_delta in order:
            offs[level_delta] = base
            base += loaded[level_delta][0]['features'].shape[0]
        rows = []
        for mode in poolings:
            sim = torch.cat([combine_slots(pooled_q[mode], pooled_r[(mode, level_delta)])
                             for level_delta in order], dim=1)
            top = sim.argmax(dim=1).numpy()
            # "at L+-1" is the co-located tile at another scale; anything else is
            # a different PLACE, which is a different failure and gets its own
            # column. Counted rather than left as 100 - sum: the three are
            # mutually exclusive so the subtraction is valid, but a residual
            # carries floating point noise and hides an empty category behind an
            # arithmetic identity.
            hit = {level_delta: np.mean(top == ans[level_delta] + offs[level_delta]) * 100 for level_delta in order}
            elsewhere = np.ones(len(top), dtype=bool)
            for level_delta in order:
                elsewhere &= top != ans[level_delta] + offs[level_delta]
            none = elsewhere.mean() * 100
            cells = ''.join(f'{hit[level_delta]:15.1f}%' for level_delta in order)
            rows.append(f'{mode:10s}{cells}{none:11.1f}%')
            if rec is not None:
                rec['p2'][mode] = (float(hit.get(0, 0.0)),
                                   float(sum(v for k, v in hit.items() if k != 0)),
                                   float(none))
        head = (f'{"pooling":10s}'
                + ''.join(f'{("right spot L%+d" % level_delta if level_delta else "right spot L"):>16s}'
                          for level_delta in order)
                + f'{"wrong spot":>12s}')
        n_ans = len(order)
        lines += _table(rows, head,
                        f'phase 2 -- all levels in one pool (n={base:,})',
                        f'The columns are not "which level did the winner come '
                        f'from" -- every candidate comes from some level. They '
                        f'are "was the winner THE tile covering the query\'s own '
                        f'position, at that scale". Only {n_ans} of the {base:,} '
                        f'candidates qualify; the other {base - n_ans:,} are the '
                        f'right slide in the wrong place and land in the last '
                        f'column, which will dominate. The question is how the '
                        f'rest divides: right spot at L is success, right spot at '
                        f'L+-1 is the scale confusion stage 1 suffers from')

    # ── 3. what makes it harder ──────────────────────────────────────────────
    bins = [(0, 32), (32, 64), (64, 96), (96, 128)]
    rows = []
    for mode in poolings:
        rank, _ = _rank_stats(combine_slots(pooled_q[mode], pooled_r[(mode, 0)]),
                              ans[0])
        cells = ''.join(
            f'{np.mean(rank[(delta >= lo) & (delta < hi)] == 1) * 100:8.1f}%'
            if ((delta >= lo) & (delta < hi)).any() else f'{"-":>9s}'
            for lo, hi in bins)
        rows.append(f'{mode:10s}{cells}'
                    f'{np.mean(rank[~edge] == 1) * 100:9.1f}%'
                    f'{np.mean(rank[edge] == 1) * 100:8.1f}%'
                    f'{np.mean(rank[rot == 0] == 1) * 100:8.1f}%'
                    f'{np.mean(rank[rot != 0] == 1) * 100:8.1f}%')
        if rec is not None:
            rec['delta'][mode] = [
                float(np.mean(rank[(delta >= lo) & (delta < hi)] == 1) * 100)
                if ((delta >= lo) & (delta < hi)).any() else float('nan')
                for lo, hi in bins]
            rec['rot'][mode] = (float(np.mean(rank[rot == 0] == 1) * 100),
                                float(np.mean(rank[rot != 0] == 1) * 100))
            rec['n_fov'] = int(len(np.unique(query_tensors['fov_id'].numpy())))
    head = (f'{"pooling":10s}'
            + ''.join(f'{f"{lo}-{hi}":>9s}' for lo, hi in bins)
            + f'{"interior":>10s}{"edge":>8s}{"rot0":>8s}{"rot90":>8s}')
    counts = ', '.join(f'{lo}-{hi}: {int(((delta >= lo) & (delta < hi)).sum())}'
                       for lo, hi in bins)
    lines += _table(rows, head, 'phase 1 at L, r@1 split by what makes it harder',
                    f'delta px (nearest of both grids) [{counts}];  '
                    f'interior/edge = position in the FoV '
                    f'({int((~edge).sum())}/{int(edge.sum())});  '
                    f'rot0/rot90 = the query turned 90 deg against a reference '
                    f'that was not, with NO rotation search here -- production '
                    f'wraps the encoder in one, so rot90 is a lower bound and a '
                    f'pooling can lead it just by being rotation invariant '
                    f'({int((rot == 0).sum())}/{int((rot != 0).sum())})')

    # ── 4. does whitening make the features more distinctive ────────────────
    rows = []
    for mode in poolings:
        qf, rf = pooled_q[mode], pooled_r[(mode, 0)]
        cells = ''
        for spec in whitens:
            qs, rs = [], []
            for s in range(rf.shape[1]):
                mu, lam, vec = fitted[(mode, s)]
                rs.append(_apply_whiten(rf[:, s, :], mu, lam, vec, spec))
                qs.append(_apply_whiten(qf[:, s, :], mu, lam, vec, spec))
            rank, pct = _rank_stats(
                combine_slots(torch.stack(qs, 1), torch.stack(rs, 1)), ans[0])
            cells += f'{np.mean(rank == 1) * 100:9.1f}%'
            if rec is not None:
                rec['white'].setdefault(mode, {})[spec] = float(
                    np.mean(rank == 1) * 100)
        rows.append(f'{mode:10s}{cells}')
    lines += _table(rows,
                    f'{"pooling":10s}' + ''.join(f'{w:>10s}' for w in whitens),
                    'phase 1 at L, r@1 after whitening the pool',
                    'fitted on the reference pool of THIS slide and level, never '
                    'on queries -- the pool exists at build() time, so this is '
                    'deployable as-is. "none" is the production path')
    return lines


def _summary(recs: list, poolings, whitens=WHITENS) -> list:
    """The one table 25 per-combination tables cannot give you.

    Averaging r@1 across combinations would be wrong: pools run from a few
    hundred at the top of a 4x pyramid to a few thousand at L0, and r@1 is not
    comparable across that. So the headline columns count PLACES, not points --
    how often a pooling ranked first, and how often it beat the CLS baseline.
    Those survive the pool-size difference, and they are the question worth
    asking: a pooling that leads on one slide and not the next has told you
    nothing. That is exactly how classify_region died (M4.2 in log/TODO.log).
    """
    if not recs:
        return []
    # Which sampling rules are in here. eval_all keys stores by cfg_hash, so a
    # root holding a quota dump and an older uniform one is scored twice and
    # BOTH land in recs -- and every number below is a median across them. A
    # median over two different distractor compositions is not a comparison of
    # poolings, it is a blend, so the count is stated and more than one is
    # called out rather than left to be noticed.
    _sids = sorted({r.get('sampler') or 'uniform (pre-quota)' for r in recs})
    lines = [f'\n{"=" * 74}',
             f'sampler {", ".join(_sids)}',
             f'SUMMARY over {len(recs)} (slide, level) combinations '
             f'-- {sum(r["n_fov"] for r in recs)} FoV in total']
    if len(_sids) > 1:
        lines.append(
            f'!! {len(_sids)} sampling rules are being averaged together. Every'
            f' median below mixes')
        lines.append(
            '   distractor pools of different composition. Split the roots, or'
            ' read the')
        lines.append(
            '   per-combination tables above instead.')

    def med(vals):
        v = [x for x in vals if x == x]
        return float(np.median(v)) if v else float('nan')

    # ── who wins, and how often ──────────────────────────────────────────────
    best = {m: 0 for m in poolings}
    beat = {m: 0 for m in poolings}
    for r in recs:
        top = max(poolings, key=lambda m: r['p1'].get(m, (-1,))[0])
        best[top] += 1
        base = r['p1'].get('cls', (0.0,))[0]
        for m in poolings:
            if m != 'cls' and r['p1'].get(m, (0.0,))[0] > base:
                beat[m] += 1
    rows = []
    for m in poolings:
        r1 = med([r['p1'][m][0] for r in recs if m in r['p1']])
        pc = med([r['p1'][m][1] for r in recs if m in r['p1']])
        vs = '-' if m == 'cls' else f'{beat[m]}/{len(recs)}'
        rows.append(f'{m:10s}{best[m]:>4d}/{len(recs):<4d}{vs:>10s}'
                    f'{r1:>13.1f}%{pc:>14.2f}%')
    lines += _table(rows,
                    f'{"pooling":10s}{"best in":>8s}{"beat cls":>10s}'
                    f'{"median r@1":>14s}{"median pct50":>15s}',
                    'phase 1 at L -- consistency',
                    'counts of combinations, not averaged points: r@1 is not '
                    'comparable across pools of 600 and 3,000')

    # ── does it hold on both pyramids ────────────────────────────────────────
    groups = {}
    for r in recs:
        groups.setdefault(r.get('step'), []).append(r)
    if len(groups) > 1:
        rows = []
        for m in poolings:
            cells = ''
            for step in sorted(groups, key=lambda s: (s is None, s)):
                g = groups[step]
                cells += (f'{sum(1 for r in g if max(poolings, key=lambda k: r["p1"].get(k, (-1,))[0]) == m):>10d}'
                          f'{med([r["p1"][m][0] for r in g if m in r["p1"]]):>11.1f}%')
            rows.append(f'{m:10s}{cells}')
        head = f'{"pooling":10s}' + ''.join(
            f'{f"{s}x best":>10s}{f"(n={len(groups[s])}) r@1":>12s}'
            for s in sorted(groups, key=lambda s: (s is None, s)))
        lines += _table(rows, head, 'by pyramid step',
                        'a 4x slide and a 2x slide are not the same experiment; '
                        'an ordering that only holds on one is not an ordering')

    # ── scale confusion ─────────────────────────────────────────────────────
    if any(r['p2'] for r in recs):
        rows = []
        for m in poolings:
            got = [r['p2'][m] for r in recs if m in r['p2']]
            if not got:
                continue
            atl = med([g[0] for g in got])
            pm1 = med([g[1] for g in got])
            found = atl + pm1
            share = (atl / found * 100) if found > 0 else float('nan')
            rows.append(f'{m:10s}{atl:>15.1f}%{pm1:>17.1f}%'
                        f'{med([g[2] for g in got]):>15.1f}%{share:>15.1f}%')
        lines += _table(rows,
                        f'{"pooling":10s}{"right spot L":>16s}'
                        f'{"right spot L+-1":>18s}{"wrong spot":>16s}'
                        f'{"L / found":>16s}',
                        'phase 2 -- scale confusion, median across combinations',
                        'the last column is the one to read: OF the times it '
                        'found the place at all, how often it also got the scale '
                        'right. "wrong spot" measures something else (not found)')

    # ── how fast each decays with misalignment ──────────────────────────────
    rows = []
    for m in poolings:
        cells = ''.join(f'{med([r["delta"][m][i] for r in recs if m in r["delta"]]):>10.1f}%'
                        for i in range(4))
        rows.append(f'{m:10s}{cells}')
    lines += _table(rows,
                    f'{"pooling":10s}' + ''.join(f'{b:>11s}' for b in
                                                 ('0-32', '32-64', '64-96', '96-128')),
                    'phase 1 at L by delta -- median r@1 across combinations',
                    'delta is how far the query sits from the nearest position '
                    'retrieval scores. A pooling that only leads in the leftmost '
                    'column is betting on an alignment the pipeline does not '
                    'guarantee')

    # ── how much of the lead is just rotation invariance ────────────────────
    if any(r['rot'] for r in recs):
        rows = []
        base0 = med([r['rot']['cls'][0] for r in recs if 'cls' in r['rot']])
        base9 = med([r['rot']['cls'][1] for r in recs if 'cls' in r['rot']])
        for m in poolings:
            got = [r['rot'][m] for r in recs if m in r['rot']]
            if not got:
                continue
            a, b = med([g[0] for g in got]), med([g[1] for g in got])
            lead0 = a - base0 if m != 'cls' else float('nan')
            lead9 = b - base9 if m != 'cls' else float('nan')
            rows.append(f'{m:10s}{a:>11.1f}%{b:>12.1f}%'
                        + (f'{"-":>15s}{"-":>15s}' if m == 'cls' else
                           f'{lead0:>+14.1f}%{lead9:>+14.1f}%'))
        lines += _table(rows,
                        f'{"pooling":10s}{"rot0 r@1":>11s}{"rot90 r@1":>13s}'
                        f'{"lead at rot0":>15s}{"lead at rot90":>15s}',
                        'phase 1 at L by rotation -- median r@1 across combinations',
                        'the two lead columns are the point. A pooling whose lead '
                        'is only at rot90 is not a better descriptor, it is a '
                        'rotation-invariant one, and production already gets that '
                        'from the rotation search around the encoder. Read the '
                        'rot0 lead as the descriptor claim')

    # ── does a closed-form transform recover anything ───────────────────────
    if whitens and any(r['white'] for r in recs):
        rows = []
        for m in poolings:
            got = [r['white'][m] for r in recs if m in r['white']]
            if not got:
                continue
            rows.append(f'{m:10s}' + ''.join(
                f'{med([g[w] for g in got if w in g]):>10.1f}%' for w in whitens))
        lines += _table(rows,
                        f'{"pooling":10s}' + ''.join(f'{w:>11s}' for w in whitens),
                        'phase 1 at L after whitening -- median r@1 across '
                        'combinations',
                        'no training and no labels: fitted on the reference pool '
                        'the build already computes. If a column here matches what '
                        'a learned head would buy, the head is not worth its cost')
    return lines


def eval_all(root: Path, wsi_filter=None, poolings=POOLINGS, out_txt=None,
             whitens=WHITENS) -> int:
    stores = {}
    for p in sorted(Path(root).glob('*.safetensors')):
        try:
            m = FS.load_meta(p)
        except Exception:                                   # noqa: BLE001
            continue
        if wsi_filter and wsi_filter not in m.wsi_stem:
            continue
        stores[(m.wsi_stem, m.level, m.pooling, m.cfg_hash())] = (p, m)

    todo = sorted(k for k in stores if 'query' in k[2])
    if not todo:
        sys.exit(f'no query stores under {root}')

    lines, recs = [], []
    for key in todo:
        stem, lv, _, cfg = key
        query_path, query_meta = stores[key]
        refs = {level_delta: stores[(stem, lv + level_delta, 'tokens', cfg)]
                for level_delta in (-1, 0, 1) if (stem, lv + level_delta, 'tokens', cfg) in stores}
        if 0 not in refs:
            lines.append(f'{stem} L{lv}: no same-level reference -- skipped')
            continue
        rec = {'stem': stem, 'level': lv, 'p1': {}, 'p2': {}, 'delta': {},
               'rot': {}, 'white': {}, 'step': None, 'pool': 0, 'n_fov': 0,
               'sampler': ''}
        lines += eval_one(query_path, query_meta, refs, poolings, rec=rec, whitens=whitens)
        recs.append(rec)

    lines += _summary(recs, poolings, whitens)
    text = '\n'.join(lines)
    print(text)
    if out_txt:
        Path(out_txt).write_text(text + '\n')
        print(f'\nwrote {out_txt}')
    return 0


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['dump', 'eval'], default='dump')
    ap.add_argument('--report', default=None,
                    help='eval only: also write the tables to this path')
    ap.add_argument('--poolings', nargs='+', default=list(POOLINGS),
                    help=f'eval only, default {" ".join(POOLINGS)}')
    ap.add_argument('--whitens', nargs='*', default=list(WHITENS),
                    help=f'eval only: whitening variants to compare, fitted on '
                         f'the reference pool. Pass none at all to skip the '
                         f'table -- it costs one eigh per (pooling, slot). '
                         f'Default {" ".join(WHITENS)}')
    ap.add_argument('--gt-csv', default='result/MultiBatch1440/gt.csv',
                    help='only read for its wsi_path/level pairs')
    ap.add_argument('--out', default='result/cache/features')
    ap.add_argument('--wsi', default=None, help='substring filter, for a small run')
    ap.add_argument('--levels', type=int, nargs='+', default=None)
    ap.add_argument('-k', type=int, default=5000, help='reference tiles at L0')
    ap.add_argument('--k-floor', type=int, default=500)
    ap.add_argument('--queries', type=int, default=400,
                    help='query tiles per (slide, level) -- the same unit as -k '
                         'and as every table eval prints. It used to be per '
                         'slide and divided by the level count, which silently '
                         'gave a third of what was asked for on a 3-level slide.')
    ap.add_argument('--mask-ds', type=float, default=4.0)
    ap.add_argument('--seg-chunk-px', type=float, default=4e6)
    ap.add_argument('--quota-tile', type=int, default=TILE,
                    help='tile size the background quota is defined on')
    ap.add_argument('--quota-jitter-cap', type=float, default=0.20,
                    help='most of a bucket that may be filled by displacing an '
                         'existing tile')
    ap.add_argument('--quota-floor-lt15', type=float, default=0.85,
                    help='least of the pool that must be tissue-dense. The '
                         'distractors this bench slides over used to be 46%% '
                         'pure background at level 0, which is a pool half the '
                         'size it looked')
    ap.add_argument('--min-region-ratio', type=float, default=0.01,
                    help='filter_regions threshold, matching '
                         'LocaScopePipeline. It gates on BBOX area, so a value '
                         'like 0.10 lets one large legitimate region set a bar '
                         'the rest cannot clear -- see log/TODO.log.')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--min-std', type=float, default=8.0,
                    help='reject a FoV whose pixels vary less than this; blank '
                         'glass makes questions no pooling can answer')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.phase == 'eval':
        # No GPU, no WSI, no model -- everything needed is in the stores.
        return eval_all(Path(args.out), wsi_filter=args.wsi,
                        poolings=tuple(args.poolings), out_txt=args.report,
                        whitens=tuple(args.whitens))

    import csv
    rows = list(csv.DictReader(open(args.gt_csv, newline='')))
    combos = {}
    for r in rows:
        if args.wsi and args.wsi not in r['wsi_path']:
            continue
        lv = int(r['level'])
        if args.levels and lv not in args.levels:
            continue
        combos.setdefault(r['wsi_path'], set()).add(lv)
    if not combos:
        sys.exit('no (slide, level) pairs matched')

    out_root = Path(args.out)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  out={out_root}')

    # fp32 and single card on purpose -- this writes stores that existing ones
    # have to stay comparable with. encoder_id is derived rather than typed, so
    # a changed checkpoint or precision cannot keep the old name.
    encoder = GigaPathEncoderConfig(batch_size=args.batch_size).with_model(dtype='fp32').build(device)
    spec = encoder.spec
    encoder_id = encoder.identity_id()
    print(f'spec={spec}\n')

    hest_method = HestSegConfig().build(device)
    # The rule is part of the identity, not a footnote: 'best' and 'nearest' pick
    # different levels and so give different region boundaries. Without it in
    # mask_id the two would share a cfg_hash and could be silently mixed.
    mask_id = f'hest@ds{args.mask_ds:g}/nearest'

    # n_target and seed are set per level inside dump_one -- the bench keeps its
    # own k / ds**2 sizing rule -- so this carries only the composition.
    sampler_cfg = RS.SamplerConfig(tile=args.quota_tile,
                                   jitter_cap=args.quota_jitter_cap,
                                   floor_lt15=args.quota_floor_lt15,
                                   inherit_frac=0.0, seed=args.seed)

    for wsi_path, levels in sorted(combos.items()):
        print(f'== {Path(wsi_path).stem}   levels {sorted(levels)}', flush=True)
        slide = SafeSlide(wsi_path)
        t0 = time.time()
        # level_rule='nearest': asking for ds=4 with openslide's rule lands on
        # level 0 whenever the pyramid reports 4.00003, which segmented
        # BRACS_1228 over 6.58 Gpx in 646 s instead of 411 Mpx in about 40.
        # Recorded in mask_id below, because the two rules give different region
        # boundaries and their stores must not be mixed.
        mask = TissuesRegionsMask.from_wsi(
            slide, ds=args.mask_ds, method=hest_method,
            seg_chunk_px=int(args.seg_chunk_px), stitch_overlap=128,
            level_rule='nearest')
        n_raw = len(mask.tissue_regions)
        # The same two stages LocaScopePipeline.build() runs, in the same order
        # and with the same default (0.01, not the 0.10 that once let one large
        # legitimate region set a threshold that deleted every other). They are
        # level-independent, which is why they belong here and the per-level
        # "can this region host a tile" test lives in grid_coords.
        #
        # Done BEFORE the Camera is constructed: the Camera keeps a reference to
        # this mask, so filtering afterwards would change what it samples from
        # under it.
        mask.filter_regions(min_ratio=args.min_region_ratio)
        mask.merge_overlapping()
        print(f'  mask: tissue={mask.tissue_fraction() * 100:.1f}%  '
              f'regions {n_raw} -> {len(mask.tissue_regions)} after '
              f'filter_regions({args.min_region_ratio}) + merge  '
              f'({time.time() - t0:.0f}s)', flush=True)
        if not mask.tissue_regions:
            print('  no region survived the filters -- skipped', flush=True)
            slide.close()
            continue
        n_per_level = args.queries
        for lv in sorted(levels):
            try:
                dump_one(wsi_path, lv, out_root, mask=mask,
                         encoder=encoder, device=device, spec=spec,
                         k=args.k, k_floor=args.k_floor, n_query=n_per_level,
                         seed=args.seed, mask_id=mask_id, encoder_id=encoder_id,
                         batch_size=args.batch_size, min_std=args.min_std,
                         sampler_cfg=sampler_cfg)
            except Exception as e:                          # noqa: BLE001
                import traceback
                print(f'  L{lv} FAILED: {type(e).__name__}: {e}', flush=True)
                traceback.print_exc()
        slide.close()

    print('\ndone. inspect with:')
    print(f'  python utilities/cli/inspect_feature_store.py {out_root}/*.safetensors')
    return 0


if __name__ == '__main__':
    sys.exit(main())
