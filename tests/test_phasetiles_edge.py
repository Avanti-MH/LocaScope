#!/usr/bin/env python3
"""Unit test for the region boxes' `edge_px` (SlideLevels.geometry) and what rides on it (phasetiles).

    python tests/test_phasetiles_edge.py

Geometry: a fake slide (two levels) and a fake mask, no slide opened. The slide's canvas is 20000 x 20000 px at
level 0; level 1 is downsampled 4x. The spectra of `zncc_maps` and the scores built on them are held bitwise to the unsplit
computation, on random maps (the GPU when there is one, else the CPU).
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch

from locascope.core.stage2_retrieval.Context import SlideLevels, frame_mismatch
from locascope.core.stage2_retrieval.RetrieverPhaseTiles import (PHASETILES_RECIPES, CosineKernels, PhaseCandidate, QuerySpectra,
                                                                 RetrieverPhaseTiles, RetrieverPhaseTilesConfig, _box_sum, centre_cells,
                                                                 check_edge, coae_latent_config, cosine_block, cosine_of, lattice_cells,
                                                                 region_spectrum, stitch, tent, zncc_maps, zncc_of)
from locascope.core.tiles.PatchingLib import PatchGrid, region_grids
from locascope.core.tiles.TissueMask import TissueRegion

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []

TILE: int = 256


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n        {type(e).__name__}: {e}')


def rejects(fn: Callable[[], Any], needle: str = '') -> None:
    try:
        fn()
    except Exception as e:                                   # noqa: BLE001
        if needle and needle not in str(e):
            raise AssertionError(f'raised, but the message never mentions {needle!r}: {e}') from None
        return
    raise AssertionError('should have raised, returned normally')


# ── fixtures ──────────────────────────────────────────────────────────────────

class FakeWsi:
    level_downsamples: Tuple[float, ...] = (1.0, 4.0)
    level_dimensions: Tuple[Tuple[int, int], ...] = ((20000, 20000), (5000, 5000))


class FakeView:
    def __init__(self, tissue_regions: List[TissueRegion]) -> None:
        self.tissue_regions: List[TissueRegion] = tissue_regions


class FakeMask:
    def __init__(self, boxes: List[Tuple[int, int, int, int]]) -> None:
        self.boxes: List[Tuple[int, int, int, int]] = boxes

    def patchable(self, side: float) -> FakeView:
        return FakeView([TissueRegion(x=x, y=y, w=w, h=h, index=i) for i, (x, y, w, h) in enumerate(self.boxes)])


def levels(boxes: List[Tuple[int, int, int, int]], edge_px: int) -> SlideLevels:
    return SlideLevels(FakeWsi(), FakeMask(boxes), TILE, True, edge_px)


def box_of(r: TissueRegion) -> Tuple[int, int, int, int]:
    return (r.x, r.y, r.w, r.h)


def grid_key(g: PatchGrid) -> Tuple[Any, ...]:
    return (g.origin_l0, g.width, g.height, g.grid_rows, g.grid_cols, g.overlap_rows, g.overlap_cols)


# ── tests ─────────────────────────────────────────────────────────────────────

def t_edge_zero_is_the_box() -> None:
    boxes: List[Tuple[int, int, int, int]] = [(5000, 6000, 3000, 2000), (100, 50, 3000, 2000), (17000, 18500, 2900, 1400)]
    for level, ds in ((0, 1.0), (1, 4.0)):
        regions, grids = levels(boxes, 0).geometry(level)
        raw: List[TissueRegion] = FakeMask(boxes).patchable(TILE * ds).tissue_regions
        want: List[PatchGrid] = region_grids(raw, ds=ds, level=level, tile_size=TILE, overlap=True)
        assert len(regions) == len(raw)
        for r, q in zip(regions, raw):
            assert (r.x, r.y, r.w, r.h, r.index) == (q.x, q.y, q.w, q.h, q.index), (level, box_of(r), box_of(q))
        for g, h in zip(grids, want):
            assert grid_key(g) == grid_key(h), (level, grid_key(g), grid_key(h))


def t_edge_grows_every_side() -> None:
    regions, _ = levels([(5000, 6000, 3000, 2000)], 1024).geometry(0)
    assert box_of(regions[0]) == (3976, 4976, 5048, 4048), box_of(regions[0])
    assert (5000 - 3976) % 256 == 0


def t_edge_stops_at_the_canvas() -> None:
    regions, _ = levels([(100, 50, 3000, 2000), (17000, 18500, 2900, 1400)], 1024).geometry(0)
    assert box_of(regions[0]) == (0, 0, 3000 + 100 + 1024, 2000 + 50 + 1024), box_of(regions[0])
    x, y, w, h = box_of(regions[1])
    assert (x, y) == (17000 - 1024, 18500 - 1024), (x, y)
    assert x + w == 20000 and y + h == 20000, (x + w, y + h)
    assert w == 2900 + 1024 + 100 and h == 1400 + 1024 + 100, (w, h)


def t_edge_is_level_px() -> None:
    regions, _ = levels([(8000, 8000, 3000, 2000)], 1024).geometry(1)
    assert box_of(regions[0]) == (8000 - 4096, 8000 - 4096, 3000 + 2 * 4096, 2000 + 2 * 4096), box_of(regions[0])


def t_small_region_holds_a_window() -> None:
    query: np.ndarray = np.zeros((1024, 1440, 3), dtype=np.uint8)
    for edge, want in ((0, ()), (1024, (0, 90, 180, 270))):
        lv: SlideLevels = levels([(8000, 8000, 900, 900)], edge)
        got: Tuple[int, ...] = tuple(lv.fitting_rotations(0, lv.query_dims(query)))
        assert got == want, (edge, got, want)


def t_frame_mismatch() -> None:
    boxes: List[Tuple[int, int, int, int]] = [(5000, 6000, 3000, 2000), (100, 50, 3000, 2000)]
    a: List[PatchGrid] = levels(boxes, 0).geometry(0)[1]
    b: List[PatchGrid] = levels(boxes, 0).geometry(0)[1]
    c: List[PatchGrid] = levels(boxes, 1024).geometry(0)[1]
    d: List[PatchGrid] = levels(boxes[:1], 0).geometry(0)[1]
    assert frame_mismatch(a, b) is None
    assert isinstance(frame_mismatch(a, c), str)
    assert isinstance(frame_mismatch(a, d), str)


def t_config_edge() -> None:
    rejects(lambda: RetrieverPhaseTilesConfig(edge_px=100), 'edge_px')
    rejects(lambda: RetrieverPhaseTilesConfig(edge_px=-256), 'edge_px')
    old: str = dataclasses.replace(PHASETILES_RECIPES['localphase'], edge_px=0).identity_id()
    assert old == RetrieverPhaseTilesConfig().identity_id(), 'edge_px = 0 must be the old id'
    assert PHASETILES_RECIPES['localphase'].identity_id() != old, 'edge_px = 1024 must be a new id'


def t_query_too_long() -> None:
    check_edge(1024, 96, 1024, 1440)                                  # 720 + 96 = 816 <= 1024
    check_edge(1024, 96, 1856, 1000)                                  # 928 + 96 = 1024: the longest that is supported
    rejects(lambda: check_edge(1024, 96, 1000, 1858), 'edge_px')      # 929 + 96 = 1025
    rejects(lambda: check_edge(1024, 96, 2000, 500), 'edge_px')
    check_edge(0, 96, 5000, 5000)                                     # 0 = the boxes as they are: not checked


def t_lattice_cell_is_the_query_centre() -> None:
    assert centre_cells(1440, 8) == 90 and centre_cells(1024, 8) == 64
    assert lattice_cells([0, 256, 512], 0, 1440, 8).tolist() == [90, 122, 154]        # 720 px, not the tile grid's 640
    assert lattice_cells([0, 256, 512], 0, 1024, 8).tolist() == [64, 96, 128]
    assert lattice_cells([0, 256], 128, 1440, 8).tolist() == [106, 138]               # the offset lattice: half a tile in
    # run reads where window_scores writes: a window whose top-left is a lattice start has shift 0
    for extent in (1024, 1440, 1100):
        assert (lattice_cells([0, 256], 0, extent, 8) - centre_cells(extent, 8)).tolist() == [0, 32]


def t_shift_puts_the_origin_on_the_top_left() -> None:
    frame = levels([(5000, 6000, 6144, 5120)], 0).frame(0)
    grid: PatchGrid = frame.grids[0]
    half: int = TILE // 2 // 8
    for h, w in ((1024, 1440), (1440, 1024), (1100, 1300)):                            # the turned query's size
        rows_q, cols_q = h // TILE, w // TILE
        for lattice in ('main', 'offset'):
            n_r, n_c = grid.lattice_dims(lattice)
            n_r, n_c = n_r - rows_q + 1, n_c - cols_q + 1
            lo: int = grid._half(lattice)
            yi: np.ndarray = lattice_cells(grid._main_row_starts[:n_r], lo, h, 8)
            xi: np.ndarray = lattice_cells(grid._main_col_starts[:n_c], lo, w, 8)
            for tx, ty in ((1000, 808), (1128, 936), (1256, 1064), (2048, 1024), (2176, 1152), (2296, 1272)):   # top-lefts, multiples of 8
                py, px = ty // 8 + centre_cells(h, 8), tx // 8 + centre_cells(w, 8)    # the cell window_scores keeps this window at
                row, col = int(np.abs(yi - py).argmin()), int(np.abs(xi - px).argmin())
                assert abs(int(yi[row]) - py) <= half and abs(int(xi[col]) - px) <= half, (h, w, lattice, tx, ty)
                c = PhaseCandidate(0, lattice, row, col, 0, 1.0, shift=((px - int(xi[col])) * 8, (py - int(yi[row])) * 8))
                assert frame.origin_l0(c) == (5000 + tx, 6000 + ty), (h, w, lattice, tx, ty, frame.origin_l0(c))


def t_lattice_covers_every_top_left() -> None:
    for extent in (1024, 1440, 1100):
        q: int = extent // TILE
        starts: List[int] = list(range(0, 5120, TILE))                                 # a 5120 px side: 20 tiles
        cells: np.ndarray = lattice_cells(starts[:len(starts) - q + 1], 0, extent, 8)  # the main lattice alone
        for t in range(0, starts[len(starts) - q] + 1, 8):                             # every top-left up to the last lattice window's
            assert int(np.abs(cells - (t // 8 + centre_cells(extent, 8))).min()) <= TILE // 2 // 8, (extent, t)


def zncc_reference(region: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """`zncc_maps` as it was before the spectra were split (the backup's body), the reference the split is held to."""
    c_, big_h, big_w = region.shape
    _, _, h, w = queries.shape
    q: torch.Tensor = (queries - queries.mean(dim=(2, 3), keepdim=True)).double()
    qn: torch.Tensor = q.pow(2).sum(dim=(1, 2, 3)).sqrt().clamp(min=1e-12)
    r64: torch.Tensor = region.double()
    spec: torch.Tensor = torch.fft.rfft2(r64)
    corr: torch.Tensor = torch.fft.irfft2((spec[None] * torch.fft.rfft2(q, s=(big_h, big_w)).conj()).sum(1), s=(big_h, big_w))
    corr = corr[:, :big_h - h + 1, :big_w - w + 1]
    s1: torch.Tensor = _box_sum(r64, h, w)
    s2: torch.Tensor = _box_sum(r64 * r64, h, w)
    var: torch.Tensor = (s2 - s1.pow(2) / (h * w)).sum(0, keepdim=True)
    valid: torch.Tensor = var > 1e-12
    out: torch.Tensor = torch.where(valid, corr / (var.clamp(min=1e-12).sqrt() * qn[:, None, None]), torch.zeros_like(corr))
    return out.float()


DEVICE: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def t_zncc_split_is_bitwise() -> None:
    torch.manual_seed(0)
    for C, H, W, n in ((6, 96, 120, 3), (6, 128, 128, 12), (3, 70, 64, 2)):
        region: torch.Tensor = torch.randn(C, H, W, device=DEVICE).half().float()
        region[:, :40, :40] = 0                                                        # a block of zeros: the valid = False path
        tiles: torch.Tensor = torch.randn(n, C, 32, 32, device=DEVICE)
        ref: torch.Tensor = zncc_reference(region, tiles)
        assert torch.equal(zncc_maps(region, tiles), ref), (C, H, W, n)
        qs: QuerySpectra = QuerySpectra(tiles)
        rs = region_spectrum(region, 32, 32)
        assert torch.equal(zncc_of(rs, qs), ref) and torch.equal(zncc_of(rs, qs), ref), (C, H, W, n)   # the second time: the kept spectra


def t_scores_cached_is_bitwise() -> None:
    torch.manual_seed(0)
    cfg: RetrieverPhaseTilesConfig = dataclasses.replace(PHASETILES_RECIPES['localphase'], block_px=2048)    # small blocks: ~40, edge shapes too
    r: RetrieverPhaseTiles = RetrieverPhaseTiles(cfg, DEVICE)
    photo: np.ndarray = (torch.rand(1024, 1440, 3) * 255).byte().numpy()
    region: torch.Tensor = torch.randn(6, 520, 700).half()                             # cpu float16, as `prepare` makes it
    region[:, 100:200, 300:420] = 0
    for rot in (0, 90):
        tiles, tl, hw = r.query_tiles(photo, rot)
        ref: torch.Tensor = r.window_scores(region, tiles, tl, hw)
        qs: QuerySpectra = QuerySpectra(tiles)
        a: torch.Tensor = r.scores(region, qs, tl, hw, (0, 0))
        hits0: int = r.spectra_hits
        b: torch.Tensor = r.scores(region, qs, tl, hw, (0, 0))
        assert torch.equal(a, ref) and torch.equal(b, ref) and r.spectra_hits > hits0, rot
    r.spectra_budget = 0
    r.spectra = {}
    r.spectra_bytes = 0
    c: torch.Tensor = r.scores(region, QuerySpectra(tiles), tl, hw, (0, 0))
    assert torch.equal(c, ref) and len(r.spectra) == 0 and r.spectra_skipped > 0


def t_spec_cache_gb_is_not_identity() -> None:
    rec: RetrieverPhaseTilesConfig = PHASETILES_RECIPES['localphase']
    assert dataclasses.replace(rec, spec_cache_gb=0.0).identity_id() == rec.identity_id()
    rejects(lambda: RetrieverPhaseTilesConfig(spec_cache_gb=-1.0), 'spec_cache_gb')


def t_localphase_keeps_its_identity_parts() -> None:
    """The parts of `phasetiles:localphase` are what its stage id cf65f987a76f50ec was made of: a field at its baseline writes none
    (`source` is None), the class and `edge_px` are all there is. Naming a map (`MAPS`) and a source must not move them."""
    assert PHASETILES_RECIPES['localphase'].identity_parts() == ['retriever="RetrieverPhaseTilesConfig"', 'edge_px=1024'], \
        PHASETILES_RECIPES['localphase'].identity_parts()
    assert PHASETILES_RECIPES['congruency'].identity_parts() == ['retriever="RetrieverPhaseTilesConfig"', 'edge_px=1024', 'rep="congruency"']


def t_stitch_gives_back_the_map_the_tiles_were_cut_from() -> str:
    torch.manual_seed(0)
    side, step, dim, ny, nx = 7, 4, 8, 5, 6
    shape: Tuple[int, int] = (step * (ny - 1) + side, step * (nx - 1) + side)
    truth: torch.Tensor = torch.randn(shape[0], shape[1], dim)
    keep: List[Tuple[int, int]] = [(y, x) for y in range(ny) for x in range(nx) if (y, x) not in ((0, 0), (2, 3))]
    lat: torch.Tensor = torch.stack([truth[step * y:step * y + side, step * x:step * x + side] for y, x in keep])
    gy: torch.Tensor = torch.tensor([y for y, _ in keep], dtype=torch.int64)
    gx: torch.Tensor = torch.tensor([x for _, x in keep], dtype=torch.int64)
    out: torch.Tensor = stitch(lat, gy, gx, tent(side, step), step, shape).permute(1, 2, 0)
    covered: torch.Tensor = torch.zeros(shape, dtype=torch.bool)
    for y, x in keep:
        covered[step * y:step * y + side, step * x:step * x + side] = True
    assert tuple(out.shape) == (shape[0], shape[1], dim) and bool((~covered).any())
    assert float((out[covered] - truth[covered]).abs().max()) < 1e-5, 'a covered cell is the map it was cut from'
    assert float(out[~covered].abs().max()) == 0.0, 'a cell no tile covers is 0'
    assert torch.allclose(tent(7, 4), torch.tensor([.25, .5, .75, 1., .75, .5, .25]))
    rejects(lambda: tent(9, 4), 'tent')
    return f'{len(keep)} tiles, {int((~covered).sum())} cells uncovered'


def cosine_by_hand(region: torch.Tensor, kernel: torch.Tensor, y: int, x: int) -> float:
    w: torch.Tensor = region[:, y:y + kernel.shape[1], x:x + kernel.shape[2]].double().flatten()
    q: torch.Tensor = kernel.double().flatten()
    den: float = float(w.norm() * q.norm())
    return 0.0 if den < 1e-6 else float(w @ q) / den


def t_cosine_of_is_the_plain_cosine() -> str:
    torch.manual_seed(0)
    chans, big_h, big_w, t = 6, 40, 50, 7
    region: torch.Tensor = torch.randn(chans, big_h, big_w, device=DEVICE)
    region[:, 10:25, 20:35] = 0                                                       # a block of zeros: the valid = False path
    where: List[Tuple[int, int]] = [(3, 4), (30, 40), (0, 0), (33, 43), (26, 5)]
    tiles: torch.Tensor = torch.stack([region[:, y:y + t, x:x + t] for y, x in where])
    out: torch.Tensor = cosine_of(cosine_block(region, t), CosineKernels(tiles))
    assert tuple(out.shape) == (len(where), big_h - t + 1, big_w - t + 1), tuple(out.shape)
    for j, (y, x) in enumerate(where):
        assert float(out[j, y, x]) >= 0.9999, (j, float(out[j, y, x]))
    worst: float = 0.0
    for j in range(len(where)):
        for y in range(0, big_h - t + 1, 3):
            for x in range(0, big_w - t + 1, 3):
                worst = max(worst, abs(float(out[j, y, x]) - cosine_by_hand(region, tiles[j], y, x)))
    assert worst < 1e-5, worst
    assert float(out[:, 12, 22].abs().max()) == 0.0, 'a window of zeros scores 0'
    assert float(out.max()) <= 1 + 1e-4, float(out.max())
    return f'worst difference to the cosine by hand {worst:.2e}'


class LatentStub:
    """The `MapProvider` of `coae-latent` without a network: its query, block and match are the real cosine's three."""
    margin_px: int = 256
    step_px: int = 128
    tile_cells: int = 7
    dtype: torch.dtype = torch.float32
    model: Optional[Any] = None

    def query(self, tiles: torch.Tensor) -> CosineKernels:
        return CosineKernels(tiles)

    def block(self, sub: torch.Tensor) -> Any:
        return cosine_block(sub, self.tile_cells)

    def match(self, block: Any, query: CosineKernels) -> torch.Tensor:
        return cosine_of(block, query)


def t_a_planted_window_is_found_by_the_latent_map_search() -> str:
    from locascope.core.backbones.GigaPath import GigaPathEncoderConfig                # noqa: PLC0415
    torch.manual_seed(0)
    cfg: RetrieverPhaseTilesConfig = coae_latent_config(GigaPathEncoderConfig(), '/nonexistent.pt', feature='latent', overlap=True,
                                                        raw_topk=1000, min_sep_tiles=1.0)
    r: RetrieverPhaseTiles = RetrieverPhaseTiles(cfg, DEVICE)
    r._provider = LatentStub()                                                         # type: ignore[assignment]
    region: torch.Tensor = torch.randn(16, 120, 150)
    ys: List[int] = [0, 384, 768]                                                      # the query's tile grid, 1440 x 1024 px: 3 x 4 tiles
    xs: List[int] = [16, 400, 784, 1168]
    tl: np.ndarray = np.array([(y, x) for y in ys for x in xs])
    oy: np.ndarray = np.round(tl[:, 0] / cfg.cell).astype(int)
    ox: np.ndarray = np.round(tl[:, 1] / cfg.cell).astype(int)
    move_y: List[int] = [-3, -2, -1, 0, 1, 2, 3, -3, 2, -1, 3, 0]                        # every tile its own move, rows and columns different,
    move_x: List[int] = [3, -3, 0, 2, -2, 1, -1, 3, -3, 0, 2, -2]                        # both extremes (+-3 = pool_px / cell) taken
    at: Tuple[int, int] = (30 + centre_cells(1024, cfg.cell), 40 + centre_cells(1440, cfg.cell))

    def search(my: List[int]) -> torch.Tensor:
        tiles: torch.Tensor = torch.stack([region[:, 30 + oy[j] + my[j]:30 + oy[j] + my[j] + 7, 40 + ox[j] + move_x[j]:40 + ox[j] + move_x[j] + 7]
                                           for j in range(len(tl))])
        return r.window_scores(region, tiles.to(DEVICE), tl, (1024, 1440)).cpu()

    scores: torch.Tensor = search(move_y)
    best: int = int(scores.flatten().argmax())
    assert float(scores.max()) >= 0.9999, float(scores.max())
    assert divmod(best, scores.shape[1]) == at, (divmod(best, scores.shape[1]), at)
    far: List[int] = list(move_y)
    far[0] = 4                                                                          # one tile moved more than the search reaches
    assert float(search(far)[at]) < 0.9999
    return f'planted at {at}: {float(scores[at]):.6f}'


def main() -> int:
    print('phasetiles edge_px')
    check('edge_px = 0 is the box',            t_edge_zero_is_the_box)
    check('every side grows',                  t_edge_grows_every_side)
    check('growing stops at the canvas',       t_edge_stops_at_the_canvas)
    check('edge_px is level px',               t_edge_is_level_px)
    check('a small region holds a window',     t_small_region_holds_a_window)
    check('frame mismatch',                    t_frame_mismatch)
    check('config: edge_px and identity',      t_config_edge)
    check('a query too long raises',           t_query_too_long)
    check('lattice cell is the query centre',  t_lattice_cell_is_the_query_centre)
    check('shift puts origin on the top-left', t_shift_puts_the_origin_on_the_top_left)
    check('lattice covers every top-left',     t_lattice_covers_every_top_left)
    check('zncc split is bitwise',             t_zncc_split_is_bitwise)
    check('cached scores are bitwise',         t_scores_cached_is_bitwise)
    check('spec_cache_gb is not identity',     t_spec_cache_gb_is_not_identity)
    check('localphase keeps its id parts',     t_localphase_keeps_its_identity_parts)
    check('stitch gives the map back',         t_stitch_gives_back_the_map_the_tiles_were_cut_from)
    check('cosine_of is the plain cosine',     t_cosine_of_is_the_plain_cosine)
    check('a planted window is found',         t_a_planted_window_is_found_by_the_latent_map_search)

    bad: List[str] = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
