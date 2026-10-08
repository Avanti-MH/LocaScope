#!/usr/bin/env python3
"""Tile-level retrieval bench: does a different pooling find the tile GigaPath's
CLS token misses?

Retrieval's failures split three ways on the synthetic corpus -- rank-1 correct
43.2%, truth ranked but not first 24.5%, truth never proposed 32.3%. The last
bucket is a statement about the descriptor, and the descriptor throws away 196 of
its 197 tokens (timm pools with global_pool='token'). This bench asks whether
keeping some of them helps, without touching retrieval or the pipeline.

    --phase dump    reads the slides, encodes, writes the token stores. Needs a
                    GPU; a few hours for the full corpus.
    --phase eval    reads only the stores. No GPU, no slide pixels, no model --
                    every pooling idea after the first is free.
    --phase all     both.

The shape of the question
-------------------------
For each (slide, level), every store a cache entry in --cache-job's tree:

    reference   a TileSampler draw on this level's rung under the reference-bank
                recipe (max(k / ds^2, k_floor) tiles), its tokens beside it:
                .../plan=ladder-<ds>-cam256x256/draw=<ref sampler>/features/
                    features_ds<d>-tokens-<encoder>.safetensors

    queries     one 256 x 256 photo per position of a random draw
                (`query_fov`: FOV_RECIPES['bench']'s gap on a one-tile sensor),
                through FovSupply's render, each at the rotation the gap drew:
                .../draw=<query sampler>/render=<gap>/features/
                    features_<id>.safetensors   the query tiles
                    answers_<id>.safetensors    the grid tiles they answer to
                (<id> = ds<d>-query_tokens-<encoder>)

    the answer  computed, not searched: Render.output_tile_origins inverts the
                capture to a level-0 coordinate, and the nearest tile in each
                of the two production grids (main, and the half-tile-shifted
                overlap grid) is an answer. `ans_main` / `ans_ovlp` index the
                `answers` file of the same variant.

The same-level pool is the answers and the reference draw; a level either side
is its own draw, scored by the nearest tile to each query's centre.

delta -- how far a query tile sits from the grid position it is matched to -- is
recorded, not swept. A query position is drawn uniformly, so delta comes with a
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # utilities/
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402

import Cache                                                        # noqa: E402
from AccessDatasets import list_names, locate                        # noqa: E402
from ConfigIdentity import record                                   # noqa: E402
from Store import FeatureStore as FS, encoder_names as enc_names, feature_id  # noqa: E402
from dump_function import RetrievalReport as RR                     # noqa: E402
from PatchingLib import PatchGrid                                   # noqa: E402
from ReadGeometry import ReadSpec                                   # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from TileSampler import (SAMPLER_RECIPES, PlanSpec, SamplerConfig,  # noqa: E402
                         TileSampler)
from TissueMaskConfig import MaskMaker, add_mask_args, mask_cfg_from_args  # noqa: E402
from TileEncoderFunc import (encoder_config, encoder_names,          # noqa: E402
                             pool_slots, pooling_kinds)
from camera import Render                                           # noqa: E402
from FovSupply import FOV_RECIPES, FovRecipe, FovSupply              # noqa: E402
from SlideReader import SlideReader                                 # noqa: E402

TILE = 256
POOLINGS = ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2')

#: Covering radius of the main and overlap grids together, in level-n px. Main
#: sits at (256i, 256j) and overlap at (256i+128, 256j+128), so the union is a
#: checkerboard whose deep holes, like (128, 0), are 128 from every occupied
#: point. A query further than this from both is in the uncovered margin at a
#: region edge -- see dump_one.
DELTA_UNION = 128.0

#: 0 grid, 1 jitter, 2 inherit -- TileSampler's `origin`, as a column.
ORIGIN_CODE = {'grid': 0, 'jitter': 1, 'inherit': 2}


# ── grid coordinates, without reading any pixels ──────────────────────────────

def grid_coords(mask, level: int, ds: float, tile_size: int = TILE):
    """Every production grid position at this level, as arrays.

    Returns (xy, region, rowcol, kind) where xy is [n, 2] level-0 top-left.
    PatchGrid.from_size takes sizes and offsets only -- no image -- so this costs
    nothing even where the region is the whole slide. PatchInfo.kind already
    separates the main grid from the half-tile-shifted one.
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


# ── where each store is ──────────────────────────────────────────────────────

def reference_config(k: int, k_floor: int, seed: int, ds: float) -> SamplerConfig:
    """The draw one level's reference pool is cut by: the reference-bank
    recipe, `max(k / ds**2, k_floor)` tiles -- k at level 0, fewer where a
    coarse level has fewer positions. A level that holds fewer comes back
    short in the sampler's report."""
    return dataclasses.replace(
        SAMPLER_RECIPES['reference-bank'],
        n_per_rung=max(int(round(k / (ds * ds))), k_floor), seed=seed)


def query_fov(n_positions: int, seed: int) -> FovRecipe:
    """The query photos: `FOV_RECIPES['bench']` -- its gap, its draw -- on a
    one-tile sensor, `n_positions` positions drawn uniformly
    (`candidates='random'`). Not the lattice, which IS the main grid and would
    put every query exactly on its answer."""
    fov = FOV_RECIPES['bench']
    return dataclasses.replace(
        fov, sensor=(TILE, TILE),
        sampler=dataclasses.replace(fov.sampler, n_per_rung=n_positions,
                                    seed=seed, candidates='random'))


class Level:
    """One (slide, level): the reference draw's and the queries' addresses,
    computed from the flags alone -- no mask, no pixels, no model."""

    def __init__(self, path: str, level: int, ds: float, args, masks, encoder_cfg):
        self.path, self.level, self.ds = path, int(level), float(ds)
        self.stem = Cache.wsi_stem_of(path)
        job = args.cache_job
        self.ref_cfg = reference_config(args.k, args.k_floor, args.seed, ds)
        self.ref_plan = PlanSpec('ladder', (self.ds,), camera=ReadSpec(TILE, TILE))
        self.ref_entry = TileSampler.draw_address(
            job, self.stem, masks.cfg, self.ref_plan
        ).at(draw=self.ref_cfg.identity_id()).entry('features')
        self.ref_id = feature_id(encoder_cfg, 'tokens', self.ds)
        self.q = query_fov(args.queries, args.seed)
        self.cam = Render(SlideReader(path), self.q.sensor, self.q.gap, ds=self.ds,
                          seed=args.seed)
        self.q_plan = PlanSpec('ladder', (self.ds,), camera=self.cam.spec)
        self.q_entry = TileSampler.draw_address(
            job, self.stem, masks.cfg, self.q_plan
        ).at(draw=self.q.sampler.identity_id(),
             render=self.cam.cfg.identity_id()).entry('features')
        self.q_id = feature_id(encoder_cfg, 'query_tokens', self.ds)


# ── one (slide, level) ────────────────────────────────────────────────────────

def _tensors(rows_xy, extra, region=None, grid_rc=None) -> dict:
    out = dict(x=torch.from_numpy(rows_xy[:, 0].astype(np.int32)),
               y=torch.from_numpy(rows_xy[:, 1].astype(np.int32)), extra=extra)
    if region is not None:
        out.update(region=torch.from_numpy(region.astype(np.int16)),
                   grid_rc=torch.from_numpy(grid_rc.astype(np.int32)))
    return out


def dump_one(lv: Level, *, mask, masks, encoder, args) -> None:
    slide = SafeSlide(lv.path)
    try:
        _dump_one(lv, slide, mask=mask, masks=masks, encoder=encoder, args=args)
    finally:
        slide.close()


def _dump_one(lv: Level, slide, *, mask, masks, encoder, args) -> None:
    level, ds = lv.level, lv.ds
    base_mpp = slide.base_mpp
    spec = encoder.model_spec
    reader = SlideReader(slide)
    rec_ref = record(encoder, also=(FS, SlideReader), draw=lv.ref_cfg.identity_id())
    rec_q = record(encoder, also=(FS, SlideReader, FovSupply),
                   render=lv.cam.cfg.identity_id())
    common = dict(wsi_stem=lv.stem, wsi_path=str(lv.path), level=level, ds=ds,
                  mpp=base_mpp * ds, base_mpp=base_mpp, tile_size=TILE,
                  overlap=True, dim=spec['dim'], feat_hw=tuple(spec['feat_hw']),
                  num_prefix=spec['num_prefix'], encoder_id=enc_names(encoder)[1],
                  seg_id=masks.cfg.seg_id(), region_id=masks.cfg.region_id(),
                  coverage='sample')
    slots, layout = pool_slots('tokens', spec)

    # ── the reference draw ──────────────────────────────────────────────────
    if lv.ref_entry.status(lv.ref_id, rec_ref)[0] == 'hit':
        print(f'  L{level}: reference {FS.path(lv.ref_entry, lv.ref_id)} -- there',
              flush=True)
    else:
        draw = TileSampler.cached(lv.path, lv.ref_cfg, lv.ref_plan, args.cache_job,
                                  masks=masks)
        metas = [s.meta for s in draw]
        names = list(lv.ref_cfg.richness.names)
        t0 = time.time()
        imgs = reader.read_samples(metas, ReadSpec(TILE, TILE))
        feats = pooling_kinds(encoder.tokens(imgs), 'tokens', spec)
        col = lambda f, dt: torch.from_numpy(np.array([f(m) for m in metas], dtype=dt))  # noqa: E731
        meta = FS.Meta(pooling='tokens', slots=slots, slot_layout=layout,
                       n_available=len(metas), n_tiles=len(metas),
                       sampler_id=lv.ref_cfg.identity_id(), plan=lv.ref_plan.key(),
                       sample_seed=args.seed, buckets=tuple(names), **common)
        xy = np.array([[m.x, m.y] for m in metas], dtype=np.int64).reshape(-1, 2)
        FS.save(lv.ref_entry, lv.ref_id, rec_ref, meta=meta,
                features=feats.to(torch.float16), **_tensors(xy, {
                    'white_frac': col(lambda m: m.score, np.float32),
                    'bucket': col(lambda m: names.index(m.bucket), np.int8),
                    'origin': col(lambda m: ORIGIN_CODE[m.origin], np.int8),
                    'parent_x': col(lambda m: m.parent_x, np.int64),
                    'parent_y': col(lambda m: m.parent_y, np.int64)}))
        print(f'  L{level}: reference {len(metas):,} tiles '
              f'(asked {lv.ref_cfg.n_per_rung:,})  {time.time() - t0:.0f}s  -> '
              f'{FS.path(lv.ref_entry, lv.ref_id)}', flush=True)

    # ── the queries and their answers ───────────────────────────────────────
    if lv.q_entry.status(lv.q_id, rec_q)[0] == 'hit':
        print(f'  L{level}: queries {FS.path(lv.q_entry, lv.q_id)} -- there',
              flush=True)
        return
    g = grid_coords(mask, level, ds)
    if g is None:
        print(f'  L{level}: no region can host a {TILE}px tile -- skipped', flush=True)
        return
    grid_xy, grid_region, grid_rowcol, grid_kind = g
    gcen = grid_xy + (TILE * ds) / 2.0                 # grid tile centres, level-0
    try:
        supply = FovSupply.cached(lv.cam, lv.q_plan, lv.q.sampler, masks=masks,
                                  draw_job=args.cache_job, render_job=args.cache_job)
    except RuntimeError as exc:
        print(f'  L{level}: no query position -- skipped '
              f'({str(exc).splitlines()[0]})', flush=True)
        return
    imgs, centres, rots, fov_ids, rowcol = [], [], [], [], []
    for index, meta, img, params in supply.shots(workers=args.workers):
        x, y = meta.fov_rect[0], meta.fov_rect[1]
        angle = float(params['rot_deg']) + float(params['angle_jitter'])
        for rr, cc, u, v, cx, cy in supply.camera_for(meta.ds).output_tile_origins(
                x, y, TILE, rot_deg=angle, scale=float(params['scale'])):
            imgs.append(np.ascontiguousarray(img[v:v + TILE, u:u + TILE]))
            centres.append((cx, cy))
            rots.append(int(params['rot_deg']))
            fov_ids.append(index)
            rowcol.append((rr, cc))
    if not imgs:
        print(f'  L{level}: no query tile -- skipped', flush=True)
        return
    centres = np.asarray(centres, dtype=np.float64)
    main_idx = np.where(grid_kind == 0)[0]
    ovlp_idx = np.where(grid_kind == 1)[0]
    am, off_main = nearest_in(gcen[main_idx], centres)
    if len(ovlp_idx):
        ao, off_ovlp = nearest_in(gcen[ovlp_idx], centres)
        ans_ovlp_g = ovlp_idx[ao]
    else:
        ans_ovlp_g, off_ovlp = main_idx[am], off_main
    ans_main_g = main_idx[am]

    # Drop the tiles no grid position covers: from_size lays whole tiles only,
    # so a region's right and bottom keep a margin with no grid point, and a
    # query there shares no pixels with its "answer". Every pooling gets those
    # wrong equally, which is dilution rather than evidence. 128 is the
    # covering radius of the two grids together; inside their extent it always
    # holds, so what it removes is exactly the uncovered margin.
    near = np.minimum(np.hypot(*off_main.T), np.hypot(*off_ovlp.T))
    keep = near <= DELTA_UNION * ds
    if not keep.all():
        print(f'  L{level}: dropped {int((~keep).sum())} of {len(keep)} query '
              f'tiles in the uncovered margin at a region edge', flush=True)
    sel = np.where(keep)[0]
    imgs = [imgs[i] for i in sel]
    centres, off_main, off_ovlp = centres[sel], off_main[sel], off_ovlp[sel]
    ans_main_g, ans_ovlp_g = ans_main_g[sel], ans_ovlp_g[sel]
    rots = [rots[i] for i in sel]
    fov_ids = [fov_ids[i] for i in sel]
    rowcol = [rowcol[i] for i in sel]

    # the answers: every grid tile some query answers to, once
    ans_g = np.unique(np.concatenate([ans_main_g, ans_ovlp_g]))
    pos = {int(g_): i for i, g_ in enumerate(ans_g)}
    t0 = time.time()
    ans_imgs = []
    for i in ans_g:
        image = reader.read(int(grid_xy[i, 0]), int(grid_xy[i, 1]),
                            ReadSpec(TILE, TILE), ds, level=level)
        if image is None:
            raise RuntimeError(f'answer tile {grid_xy[i].tolist()} runs off the slide')
        ans_imgs.append(image)
    ans_feats = pooling_kinds(encoder.tokens(ans_imgs), 'tokens', spec)
    q_feats = pooling_kinds(encoder.tokens(imgs), 'tokens', spec)
    q_xy = np.stack([centres[:, 0] - TILE * ds / 2,
                     centres[:, 1] - TILE * ds / 2], 1).astype(np.int64)
    qs = dict(sampler_id=lv.q.sampler.identity_id(), plan=lv.q_plan.key(),
              sample_seed=args.seed)
    n_q, n_a = len(imgs), len(ans_g)
    FS.write(lv.q_entry, lv.q_id, rec_q, {
        'features': dict(
            meta=FS.Meta(pooling='query_tokens', slots=slots, slot_layout=layout,
                         n_available=n_q, n_tiles=n_q, **qs, **common),
            features=q_feats.to(torch.float16),
            **_tensors(q_xy, {
                'ans_main': torch.tensor([pos[int(v)] for v in ans_main_g],
                                         dtype=torch.int32),
                'ans_ovlp': torch.tensor([pos[int(v)] for v in ans_ovlp_g],
                                         dtype=torch.int32),
                # level-n px, so it is comparable across levels
                'delta_main': torch.from_numpy((off_main / ds).astype(np.int16)),
                'delta_ovlp': torch.from_numpy((off_ovlp / ds).astype(np.int16)),
                'fov_id': torch.tensor(fov_ids, dtype=torch.int32),
                'rot': torch.tensor(rots, dtype=torch.int32)},
                region=np.full(n_q, -1), grid_rc=np.asarray(rowcol).reshape(-1, 2))),
        'answers': dict(
            meta=FS.Meta(pooling='tokens', slots=slots, slot_layout=layout,
                         n_available=len(grid_xy), n_tiles=n_a, **qs, **common),
            features=ans_feats.to(torch.float16),
            **_tensors(grid_xy[ans_g], {
                'kind': torch.from_numpy(grid_kind[ans_g].astype(np.int16))},
                region=grid_region[ans_g], grid_rc=grid_rowcol[ans_g]))})
    print(f'  L{level}: {n_q:,} query tiles from {len(set(fov_ids))} FoV, '
          f'{n_a:,} answer tiles  {time.time() - t0:.0f}s  -> '
          f'{FS.path(lv.q_entry, lv.q_id)}', flush=True)


# ── eval ──────────────────────────────────────────────────────────────────────
#
# Reads stores only -- no GPU, no model. The cross-level answers are not in the
# dump (ans_main / ans_ovlp index the SAME level's answers) but they do not need
# to be: every store carries each tile's level-0 x/y, so "which tile of ref(L-1)
# covers this query" is a nearest-neighbour question answerable from
# coordinates.

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


def slidewin_rows(query_meta, ref_tensors, pooled_q, pooled_r, poolings,
                  query_tensors, dmain, dovlp) -> list:
    """The same rows bench_window_retrieval stores, for the same tables.

    That bench asks its question of a whole FoV window through stage 2; this one
    asks it of a single tile against a store. Different systems, one metric
    vocabulary -- W/L/T against the baseline, top@f%, truth@k, gap@k -- so the
    numbers here can be read beside the ones in log/WindowRetrievalBench rather than
    only against each other.

    Only the same-level pool (level_delta == 0). truth and fine are defined by
    which of the two INTERLEAVED GRIDS the query sat closer to, and the other
    two levels have their own grids at their own scale; ranking against them is
    the scale question, which phase 1 and phase 2 already answer above.

    ans_main / ans_ovlp are the stored indices rather than a recomputed nearest,
    because the main/overlap distinction is what gap@k is made of and only the
    dump knows which grid each index came from.
    """
    n = pooled_q[poolings[0]].shape[0]
    pool = int(ref_tensors['features'].shape[0])
    a_main = query_tensors['ans_main'].numpy().astype(np.int64)
    a_ovlp = query_tensors['ans_ovlp'].numpy().astype(np.int64)
    fov = query_tensors['fov_id'].numpy().astype(np.int64)
    # fov_id is the query POSITION and repeats across its rotations, and
    # RetrievalReport pairs arms on its key. Two shots of one position would
    # collide, so the pairing key is the row itself -- each query tile is its
    # own retrieval question here, and the position is kept as `shot_id`.
    qid = np.arange(n, dtype=np.int64)

    out = []
    for mode in poolings:
        sim = combine_slots(pooled_q[mode], pooled_r[(mode, 0)])
        rank_main, _ = _rank_stats(sim, a_main)
        rank_ovlp, _ = _rank_stats(sim, a_ovlp)
        for i in range(n):
            out.append({
                'slide': query_meta.wsi_stem, 'level': int(query_meta.level),
                'fov_id': int(qid[i]), 'shot_id': int(fov[i]), 'pool': pool,
                'arm': mode,
                'rank_main': int(rank_main[i]), 'rank_overlap': int(rank_ovlp[i]),
                'd_main': float(dmain[i]), 'd_overlap': float(dovlp[i]),
            })
    return out


def _table(rows, head, title, note=''):
    out = [f'\n{title}']
    if note:
        out.append(f'  {note}')
    out.append('  ' + head)
    out.extend('  ' + r for r in rows)
    return out


def eval_one(query_tensors: dict, query_meta, refs: dict, poolings,
             rec: dict = None, whitens=WHITENS) -> list:
    """One (slide, level): three tables, each answering a different question.
    `refs` is `{level_delta: (tensors, meta)}`, the pools already read.

    Fills `rec` with the same numbers so eval_all can ask the question no single
    combination can answer -- whether an ordering holds everywhere.
    """
    # The rule dump_one applies, held here too. A query further than DELTA_UNION from
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

    loaded, ans = dict(refs), {}
    for level_delta, (ref_tensors, ref_meta) in loaded.items():
        ans[level_delta], _ = nearest_in(_centres(ref_tensors, ref_meta.ds), query_centres)

    sampler_name = f'{query_meta.sampler_id}_{query_meta.plan}'
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
        pooled_q[mode] = pooling_kinds(query_tensors['features'].float(), mode,
                                     query_meta)
        for level_delta, (ref_tensors, ref_meta) in loaded.items():
            pooled_r[(mode, level_delta)] = pooling_kinds(
                ref_tensors['features'].float(), mode, ref_meta)

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
            + f'{"interior":>10s}{"edge":>8s}{"rot0":>8s}{"rot!=0":>8s}')
    counts = ', '.join(f'{lo}-{hi}: {int(((delta >= lo) & (delta < hi)).sum())}'
                       for lo, hi in bins)
    lines += _table(rows, head, 'phase 1 at L, r@1 split by what makes it harder',
                    f'delta px (nearest of both grids) [{counts}];  '
                    f'interior/edge = position in the FoV '
                    f'({int((~edge).sum())}/{int(edge.sum())});  '
                    f'rot0/rot!=0 = the query at 0, or turned 90/180/270 deg, against a reference '
                    f'that was not, with NO rotation search here -- production '
                    f'wraps the encoder in one, so rot!=0 is a lower bound and a '
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

    if rec is not None:
        rec['rows'] = slidewin_rows(query_meta, loaded[0][0], pooled_q, pooled_r,
                                    poolings, query_tensors, dmain, dovlp)
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
    # Which sampling rules are in here. eval_all scores every query store under
    # its root, so a root holding two draws is scored twice and BOTH land in
    # recs -- and every number below is a median across them. A
    # median over two different distractor compositions is not a comparison of
    # poolings, it is a blend, so the count is stated and more than one is
    # called out rather than left to be noticed.
    _sids = sorted({r['sampler'] for r in recs})
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
                        f'{"pooling":10s}{"rot0 r@1":>11s}{"rot!=0 r@1":>13s}'
                        f'{"lead at rot0":>15s}{"lead at rot!=0":>15s}',
                        'phase 1 at L by rotation -- median r@1 across combinations',
                        'the two lead columns are the point. A pooling whose lead '
                        'is only at rot!=0 is not a better descriptor, it is a '
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



def _pool(lv: Level):
    """`(tensors, meta)` of the same-level pool: the answers, then every
    reference tile that is not one of them. None when either is missing."""
    qpath = FS.path(lv.q_entry, lv.q_id)
    apath = FS.path(lv.q_entry, lv.q_id, 'answers')
    rpath = FS.path(lv.ref_entry, lv.ref_id)
    if not (qpath.is_file() and apath.is_file() and rpath.is_file()):
        return None
    a, _ = FS.load(apath)
    r, rmeta = FS.load(rpath)
    seen = {(int(x), int(y)) for x, y in zip(a['x'], a['y'])}
    extra = torch.tensor([(int(x), int(y)) not in seen
                          for x, y in zip(r['x'], r['y'])], dtype=torch.bool)
    pool = {k: torch.cat([a[k], r[k][extra]]) for k in ('features', 'x', 'y')}
    return pool, rmeta


def _ref(lv: Level):
    rpath = FS.path(lv.ref_entry, lv.ref_id)
    return FS.load(rpath) if rpath.is_file() else None


def eval_all(levels: list, poolings=POOLINGS, out_txt=None, whitens=WHITENS) -> int:
    """Every (slide, level) the flags address, against its pools: the same
    level's (answers + its reference draw) and one level either side (their
    draws)."""
    lines, recs = [], []
    by_slide = {}
    for lv in levels:
        by_slide.setdefault(lv.stem, {})[lv.level] = lv
    for stem, at in sorted(by_slide.items()):
        for level, lv in sorted(at.items()):
            qpath = FS.path(lv.q_entry, lv.q_id)
            same = _pool(lv)
            if same is None:
                lines.append(f'{stem} L{level}: no query or reference store '
                             f'({qpath}) -- skipped')
                continue
            query_tensors, query_meta = FS.load(qpath)
            refs = {0: same}
            for d in (-1, 1):
                if level + d in at:
                    got = _ref(at[level + d])
                    if got is not None:
                        refs[d] = got
            rec = {'stem': stem, 'level': level, 'p1': {}, 'p2': {},
                   'delta': {}, 'rot': {}, 'white': {}, 'step': None, 'pool': 0,
                   'n_fov': 0, 'sampler': '', 'rows': []}
            lines += eval_one(query_tensors, query_meta, refs, poolings, rec=rec,
                              whitens=whitens)
            recs.append(rec)
    if not recs:
        sys.exit('no (slide, level) has both its stores')

    lines += _summary(recs, poolings, whitens)

    # The same tables bench_window_retrieval prints, from the same code in
    # utilities/dump_function/RetrievalReport.py. Last because they are the
    # widest reading: everything above is one (slide, level) at a time.
    rows = [r for rec in recs for r in rec['rows']]
    if rows:
        lines.append(f'\n\n{"#" * 90}')
        lines.append('# paired tables -- same metrics as log/WindowRetrievalBench')
        lines.append('# baseline is `cls`, which is what production keeps.')
        lines.append(f'{"#" * 90}')
        RR.report(RR.attach_baseline(rows, 'cls'), list(poolings), 'cls',
                  per_slide=True, emit=lines.append)
    text = '\n'.join(lines)
    print(text)
    if out_txt:
        Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(out_txt).write_text(text + '\n')
        print(f'\nwrote {out_txt}')
    return 0


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--phase', choices=['dump', 'eval', 'all'], default='all')
    ap.add_argument('--report', default=None,
                    help='eval: also write the tables here. Default '
                         'result/<job>/<encoder>/reference_report.txt')
    ap.add_argument('--poolings', nargs='+', default=list(POOLINGS),
                    help=f'eval only, default {" ".join(POOLINGS)}')
    ap.add_argument('--whitens', nargs='*', default=list(WHITENS),
                    help=f'eval only: whitening variants to compare, fitted on '
                         f'the reference pool. Pass none at all to skip the '
                         f'table -- it costs one eigh per (pooling, slot). '
                         f'Default {" ".join(WHITENS)}')
    # The slides: the first --n-wsi of each dataset's recorded --split, the
    # ones every other bench scores. Every native level of each.
    ap.add_argument('--datasets', nargs='+', default=['bracs/test', 'ki67_with_photo'])
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--n-wsi', type=int, default=5, help='slides per dataset')
    ap.add_argument('--cache-job', default=None,
                    help="whose cache the stores are written to (dump) and read "
                         "from (eval). Default: this job")
    ap.add_argument('--mask-cache-job', default='MppRoutingHead',
                    help="masks are read from this job's cache")
    ap.add_argument('--wsi', default=None, help='substring filter, for a small run')
    ap.add_argument('--levels', type=int, nargs='+', default=None)
    ap.add_argument('-k', type=int, default=5000, help='reference tiles at L0')
    ap.add_argument('--k-floor', type=int, default=500)
    ap.add_argument('--queries', type=int, default=400,
                    help='query positions per (slide, level), one tile each')
    add_mask_args(ap)
    ap.add_argument(
        '--encoder', default='gigapath', choices=encoder_names(),
        help='which tile encoder. Its config names every store '
             '(Store.feature_id), so two encoders cannot overwrite each other.')
    ap.add_argument(
        '--head', default='',
        help="which exit of the model, empty for its own default. CONCH needs "
             "--head trunk here: this bench calls encoder.tokens(), and CONCH's "
             "default attentional pooler hands back ONE vector with no token axis.")
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    args.cache_job = args.cache_job or Cache.job_name('TileRetrievalBench')
    from CpuBudget import CpuBudget                               # noqa: PLC0415
    args.workers = CpuBudget.for_job(processes=1).apply().workers

    # fp32 and single card on purpose: the stores have to stay comparable
    # across runs. The config alone names them, so eval builds no model.
    over = {'head': args.head} if args.head else {}
    cfg = encoder_config(args.encoder, batch_size=args.batch_size, **over)\
        .with_model(dtype='fp32')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masks = MaskMaker(mask_cfg_from_args(args), args.mask_cache_job, device)

    levels = []
    for dataset in args.datasets:
        dataset_split = f'{dataset}#{args.split}'
        names = list_names(dataset=dataset_split)
        if args.n_wsi > len(names):
            sys.exit(f'--n-wsi {args.n_wsi} but {dataset_split} holds {len(names)}')
        for name in names[:args.n_wsi]:
            path = str(locate(name, dataset=dataset_split).path)
            if args.wsi and args.wsi not in path:
                continue
            with SafeSlide(path) as slide:
                lds = list(slide.level_downsamples)
            levels += [Level(path, lv, ds, args, masks, cfg)
                       for lv, ds in enumerate(lds)
                       if not args.levels or lv in args.levels]
    if not levels:
        sys.exit('no (slide, level) pairs matched')
    print(f'cache {args.cache_job}   encoder {enc_names(cfg)[0]} '
          f'{enc_names(cfg)[1]}   mask {masks.cfg.seg_id()}/{masks.cfg.region_id()}',
          flush=True)

    if args.phase in ('dump', 'all'):
        encoder = cfg.build(device)
        print(f'device={device}  spec={encoder.model_spec}\n', flush=True)
        stems = {}
        for lv in levels:
            stems.setdefault(lv.path, []).append(lv)
        for path, todo in stems.items():
            print(f'== {Path(path).stem}   levels {[lv.level for lv in todo]}',
                  flush=True)
            with SafeSlide(path) as slide:
                mask, _ = masks.mask(slide)
            if not mask.tissue_regions:
                print('  no region survived the filters -- skipped', flush=True)
                continue
            for lv in todo:
                try:
                    dump_one(lv, mask=mask, masks=masks, encoder=encoder, args=args)
                except Exception as e:                      # noqa: BLE001
                    import traceback
                    print(f'  L{lv.level} FAILED: {type(e).__name__}: {e}', flush=True)
                    traceback.print_exc()
        masks.close()
        del encoder

    if args.phase in ('eval', 'all'):
        report = args.report or str(Path(_paths.job_result_dir(
            'TileRetrievalBench', encoder=enc_names(cfg)[0])) / 'reference_report.txt')
        return eval_all(levels, poolings=tuple(args.poolings), out_txt=report,
                        whitens=tuple(args.whitens))
    return 0


if __name__ == '__main__':
    sys.exit(main())
