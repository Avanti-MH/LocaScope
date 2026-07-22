#!/usr/bin/env python3
"""
Comprehensive visual + assertion test for QueryPatchContainer and TissuePatchContainer.

Synthetic (always run):
  QPC:  main / overlap pixel correctness, corner-pixel relationship,
        overlap=False, factory methods (from_path/from_pil/from_array),
        multichannel input (RGBA, grayscale), error cases
  TPC:  case1/2/3 pixel correctness, case3 overlap == case2 overlap,
        ds != 1.0, region y offset, at_level in PatchInfo,
        overlap=False, factory methods, corner-pixel, error cases

Real data (default paths pre-filled; skipped if file missing):
  --query   Ki67 BMP           → QPC
  --roi     BRACS RoI PNG      → QPC (roi-as-query) + TPC
  --wsi     Ki67 mrxs          → TPC via read_region crop + from_openslide

Usage:
  python test_modules/test_tissue_patch_container.py
  python test_modules/test_tissue_patch_container.py --size 64 --rsize 128
"""

import argparse
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage

from _paths import job_result_dir, setup_import_paths
setup_import_paths()

from PatchingLib import QueryPatchContainer, TissuePatchContainer
from TissuesRegionsMask import TissueRegion


# ── Synthetic image ───────────────────────────────────────────────────────────

def make_gradient_image(width: int, height: int) -> np.ndarray:
    """Each pixel encodes (x, y) in R/G channels → unique values everywhere."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = (np.arange(width,  dtype=np.float32) * 255 / max(width  - 1, 1)
                    ).astype(np.uint8)[np.newaxis, :]
    img[:, :, 1] = (np.arange(height, dtype=np.float32) * 255 / max(height - 1, 1)
                    ).astype(np.uint8)[:, np.newaxis]
    img[:, :, 2] = 128
    return img


# ── Grid geometry helpers ─────────────────────────────────────────────────────

def main_origins(w: int, h: int, size: int):
    rows = [i for i in range(0, h, size) if i + size <= h]
    cols = [j for j in range(0, w, size) if j + size <= w]
    return [(i, j) for i in rows for j in cols]


def overlap_origins(w: int, h: int, size: int):
    rows = [i for i in range(0, h, size) if i + size <= h]
    cols = [j for j in range(0, w, size) if j + size <= w]
    half = size // 2
    return [
        (rows[ri] + half, cols[ci] + half)
        for ri in range(len(rows) - 1)
        for ci in range(len(cols) - 1)
    ]


# ── Universal helpers ─────────────────────────────────────────────────────────

def validate_patch_shapes(container, size: int, label: str = ''):
    bad = [(i, p.shape) for i, p in enumerate(container)
           if p.shape != (size, size, 3)]
    assert not bad, (
        f'{label}shape mismatch at indices {[i for i,_ in bad]}: '
        f'{[s for _,s in bad]}, expected ({size},{size},3)'
    )
    print(f'[PASS] {label}shapes: all {len(list(container))} patches are ({size},{size},3)')


def validate_iterators(container, label: str = ''):
    flat = list(container)
    assert flat == [container[i] for i in range(len(container))]

    grid = container.grid
    main_by_iter = list(container.iter_main())
    main_by_idx  = [container[grid.flat_index_for_main(info.row, info.col)]
                    for info in grid.main_patch_infos]
    assert main_by_iter == main_by_idx

    if grid.has_overlap:
        ovl_by_iter = list(container.iter_overlap())
        ovl_by_idx  = [container[grid.flat_index_for_overlap(info.row, info.col)]
                       for info in grid.overlap_patch_infos]
        assert ovl_by_iter == ovl_by_idx
        assert len(flat) == len(main_by_iter) + len(ovl_by_iter)

    assert [p for b in container.iter_batches(batch_size=3) for p in b] == flat
    print(f'[PASS] {label}iterators: __iter__ / iter_main / iter_overlap / iter_batches OK')


# ── QueryPatchContainer ───────────────────────────────────────────────────────

def validate_qpc_main(qc: QueryPatchContainer, size: int, label: str = 'QPC '):
    grid = qc.grid
    origins = main_origins(qc.width, qc.height, size)
    main_patches = list(qc.iter_main())
    assert len(main_patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        r, c = divmod(idx, grid.grid_cols)
        expected = qc.img[y:y + size, x:x + size]
        flat_i = grid.flat_index_for_main(r, c) if grid.has_overlap else idx
        grid_i = (2*r, 2*c) if grid.has_overlap else (r, c)
        for lbl, patch in (
            (f'iter_main[{idx}]', main_patches[idx]),
            (f'[{flat_i}]', qc[flat_i]),
            (f'[{grid_i}]', qc[grid_i]),
        ):
            assert np.array_equal(patch, expected), f'{label}{lbl} mismatch at ({r},{c})'
    print(f'[PASS] {label}main: {len(origins)} patches, 3 access methods verified')


def validate_qpc_overlap(qc: QueryPatchContainer, size: int, label: str = 'QPC '):
    grid = qc.grid
    half = size // 2
    origins = overlap_origins(qc.width, qc.height, size)
    ovl_patches = list(qc.iter_overlap())
    assert len(ovl_patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        r, c = divmod(idx, grid.overlap_cols)
        expected = qc.img[y:y + size, x:x + size]
        flat_i = grid.flat_index_for_overlap(r, c)
        for lbl, patch in (
            (f'iter_overlap[{idx}]', ovl_patches[idx]),
            (f'[{flat_i}]', qc[flat_i]),
            (f'[{2*r+1},{2*c+1}]', qc[2*r+1, 2*c+1]),
        ):
            assert np.array_equal(patch, expected), f'{label}{lbl} mismatch at ({r},{c})'
        # Corner-pixel 4-neighbour relationship
        p = qc[2*r+1, 2*c+1]
        assert np.array_equal(p[:half, :half],  qc[2*r,   2*c  ][half:, half:])
        assert np.array_equal(p[:half, half:],  qc[2*r,   2*c+2][half:, :half])
        assert np.array_equal(p[half:, :half],  qc[2*r+2, 2*c  ][:half, half:])
        assert np.array_equal(p[half:, half:],  qc[2*r+2, 2*c+2][:half, :half])
    print(f'[PASS] {label}overlap: {len(origins)} patches, pixel + corner-pixel OK')


def validate_qpc_no_overlap(img: np.ndarray, size: int):
    qc = QueryPatchContainer(img.copy())
    qc.extract_all(size, overlap=False)
    assert not qc.grid.has_overlap
    assert list(qc.iter_overlap()) == []
    assert len(qc) == len(list(qc.iter_main()))
    assert list(qc) == list(qc.iter_main())
    # Without overlap, any in-range (r, c) is valid — no parity restriction
    qc[0, 1]
    print(f'[PASS] QPC overlap=False: {len(qc)} main patches, mixed-parity tuple OK')


def validate_qpc_factory_methods(img: np.ndarray, size: int):
    ref = QueryPatchContainer(img.copy())
    ref.extract_all(size, overlap=True)
    ref_patches = list(ref)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmppath = f.name
    try:
        PILImage.fromarray(img).save(tmppath)
        cases = [
            ('from_path',  QueryPatchContainer.from_path(tmppath)),
            ('from_pil',   QueryPatchContainer.from_pil(PILImage.open(tmppath).convert('RGB'))),
            ('from_array', QueryPatchContainer.from_array(img.copy())),
        ]
        for label, qc in cases:
            qc.extract_all(size, overlap=True)
            for i, (p, r) in enumerate(zip(qc, ref_patches)):
                assert np.array_equal(p, r), f'QPC {label}: patch[{i}] differs'
            print(f'[PASS] QPC {label}: identical to direct constructor')
    finally:
        os.unlink(tmppath)


def validate_qpc_multichannel(size: int):
    H, W = 256, 256
    # RGBA → drop alpha, keep RGB
    rgba = np.random.randint(0, 200, (H, W, 4), dtype=np.uint8)
    qc_rgba = QueryPatchContainer(rgba.copy())
    assert qc_rgba.img.shape == (H, W, 3), f'RGBA shape: {qc_rgba.img.shape}'
    assert np.array_equal(qc_rgba.img, rgba[:, :, :3])
    print('[PASS] QPC RGBA→RGB: alpha dropped, RGB preserved')

    # Grayscale (2D) → stack 3 identical channels
    gray = np.random.randint(0, 200, (H, W), dtype=np.uint8)
    qc_gray = QueryPatchContainer(gray.copy())
    assert qc_gray.img.shape == (H, W, 3), f'gray shape: {qc_gray.img.shape}'
    assert np.all(qc_gray.img[:, :, 0] == gray)
    assert np.all(qc_gray.img[:, :, 1] == gray)
    assert np.all(qc_gray.img[:, :, 2] == gray)
    print('[PASS] QPC grayscale→RGB: all 3 channels equal original')


def validate_qpc_errors(img: np.ndarray, size: int):
    fresh = QueryPatchContainer(img.copy())
    try:
        _ = fresh[0]
        raise AssertionError('expected RuntimeError before extract_all')
    except RuntimeError:
        pass

    qc = QueryPatchContainer(img.copy())
    qc.extract_all(size, overlap=True)

    try:
        _ = qc[len(qc)]
        raise AssertionError('expected IndexError for OOB flat')
    except IndexError:
        pass

    try:
        _ = qc[0, 1]
        raise AssertionError('expected IndexError for mixed parity (0,1) with overlap')
    except IndexError:
        pass

    print('[PASS] QPC errors: RuntimeError / OOB IndexError / mixed-parity IndexError')


# ── TissuePatchContainer ─────────────────────────────────────────────────────

def validate_tpc_case1(tc: TissuePatchContainer, img: np.ndarray, size: int):
    origins = main_origins(tc.width, tc.height, size)
    patches = list(tc.iter_main())
    assert len(patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        assert np.array_equal(patches[idx], img[y:y+size, x:x+size]), (
            f'case1 mismatch at ({y},{x})')
    print(f'[PASS] TPC case1 (full, no region): {len(origins)} patches')


def validate_tpc_case2(tc: TissuePatchContainer, img: np.ndarray,
                        region: TissueRegion, size: int, ds: float):
    rx, ry = int(region.x / ds), int(region.y / ds)
    rw, rh = int(region.w / ds), int(region.h / ds)
    origins = main_origins(rw, rh, size)
    patches = list(tc.iter_main())
    assert len(patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        expected = img[ry+y:ry+y+size, rx+x:rx+x+size]
        assert np.array_equal(patches[idx], expected), (
            f'case2 mismatch at global ({ry+y},{rx+x})')
    print(f'[PASS] TPC case2 (full + region): {len(origins)} patches, offset verified')
    return patches


def validate_tpc_case3(tc: TissuePatchContainer, size: int, ref: list):
    patches = list(tc.iter_main())
    assert len(patches) == len(ref)
    for i, (p3, p2) in enumerate(zip(patches, ref)):
        assert np.array_equal(p3, p2), f'case3 main[{i}] differs from case2'
    print(f'[PASS] TPC case3 (is_crop + region): {len(patches)} main patches match case2')
    return patches


def validate_tpc_case3_overlap(tc2: TissuePatchContainer, tc3: TissuePatchContainer):
    """Overlap patches from is_crop must be pixel-identical to full-image + region."""
    ovl2 = list(tc2.iter_overlap())
    ovl3 = list(tc3.iter_overlap())
    assert len(ovl2) == len(ovl3), (
        f'overlap count: case2={len(ovl2)}, case3={len(ovl3)}')
    for i, (p2, p3) in enumerate(zip(ovl2, ovl3)):
        assert np.array_equal(p2, p3), f'case3 overlap[{i}] differs from case2'
    print(f'[PASS] TPC case3 overlap: {len(ovl2)} overlap patches match case2')


def validate_tpc_ds_not_1(size: int):
    """
    img_ds=4.0 (level-2 equivalent): verify both x and y region offsets are
    correctly divided by ds, and that at_level / ds are forwarded to PatchInfo.

    Synthetic image: 512×512 at level-N (ds=4), representing a 2048×2048 level-0 WSI.
    Region (level-0): x=256, y=384, w=1024, h=768
    → level-N:        x=64,  y=96,  w=256,  h=192
    """
    W, H = 512, 512
    img   = make_gradient_image(W, H)
    ds    = 4.0
    level = 2
    # Level-0 region coords
    region = TissueRegion(x=256, y=384, w=1024, h=768)

    tc = TissuePatchContainer(img.copy(), region=region, img_ds=ds,
                              is_crop=False, at_level=level)
    tc.extract_all(size, overlap=False)

    rx_n = int(region.x / ds)   # 64
    ry_n = int(region.y / ds)   # 96
    rw_n = int(region.w / ds)   # 256
    rh_n = int(region.h / ds)   # 192

    origins = main_origins(rw_n, rh_n, size)
    patches = list(tc.iter_main())
    assert len(patches) == len(origins), (
        f'ds=4 patch count {len(patches)} != {len(origins)}')
    for idx, (y, x) in enumerate(origins):
        expected = img[ry_n+y:ry_n+y+size, rx_n+x:rx_n+x+size]
        assert np.array_equal(patches[idx], expected), (
            f'ds=4 mismatch at level-N ({ry_n+y},{rx_n+x})')

    # PatchInfo metadata must reflect the constructor arguments
    for info in tc.grid.main_patch_infos:
        assert info.ds    == ds,    f'PatchInfo.ds={info.ds}'
        assert info.level == level, f'PatchInfo.level={info.level}'
        # x/y in PatchInfo are level-N global coords (include grid offset)
        assert info.x >= rx_n, f'PatchInfo.x={info.x} < rx_n={rx_n}'
        assert info.y >= ry_n, f'PatchInfo.y={info.y} < ry_n={ry_n}'

    print(f'[PASS] TPC ds=4.0 level={level}: {len(origins)} patches, '
          f'x/y offset ({rx_n},{ry_n}), ds/level in PatchInfo verified')


def validate_tpc_region_y_offset(img: np.ndarray, size: int):
    """Region with non-zero y: both x and y offsets must be applied."""
    H, W = img.shape[:2]
    ds = 1.0
    region = TissueRegion(x=W // 2, y=H // 2, w=W // 2, h=H // 2)
    tc = TissuePatchContainer(img.copy(), region=region, img_ds=ds, is_crop=False)
    tc.extract_all(size, overlap=False)

    rx, ry = W // 2, H // 2
    rw, rh = W // 2, H // 2
    origins = main_origins(rw, rh, size)
    patches = list(tc.iter_main())
    assert len(patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        expected = img[ry+y:ry+y+size, rx+x:rx+x+size]
        assert np.array_equal(patches[idx], expected), (
            f'y-offset mismatch at global ({ry+y},{rx+x})')

    print(f'[PASS] TPC region y_offset={H//2}: {len(origins)} patches verified')


def validate_tpc_patchinfo_meta(img: np.ndarray, size: int):
    """at_level must be forwarded to every PatchInfo in the grid."""
    ds, lv = 4.0, 2
    tc = TissuePatchContainer(img.copy(), img_ds=ds, at_level=lv)
    tc.extract_all(size, overlap=True)
    for info in tc.grid.iter_infos():
        assert info.ds    == ds,  f'PatchInfo.ds={info.ds}, expected {ds}'
        assert info.level == lv,  f'PatchInfo.level={info.level}, expected {lv}'
    print(f'[PASS] TPC PatchInfo meta: ds/level forwarded to all {len(tc.grid)} patches')


def validate_tpc_no_overlap(img: np.ndarray, region: TissueRegion, ds: float, size: int):
    tc = TissuePatchContainer(img.copy(), region=region, img_ds=ds, is_crop=False)
    tc.extract_all(size, overlap=False)
    assert not tc.grid.has_overlap
    assert list(tc.iter_overlap()) == []
    assert len(tc) == len(list(tc.iter_main()))
    print(f'[PASS] TPC overlap=False: {len(tc)} main patches, no overlap')


def validate_tpc_factory_methods(img: np.ndarray, size: int):
    ref = TissuePatchContainer(img.copy())
    ref.extract_all(size, overlap=True)
    ref_patches = list(ref)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmppath = f.name
    try:
        PILImage.fromarray(img).save(tmppath)
        cases = [
            ('from_path',  TissuePatchContainer.from_path(tmppath)),
            ('from_pil',   TissuePatchContainer.from_pil(PILImage.open(tmppath).convert('RGB'))),
            ('from_array', TissuePatchContainer.from_array(img.copy())),
        ]
        for label, tc in cases:
            tc.extract_all(size, overlap=True)
            for i, (p, r) in enumerate(zip(tc, ref_patches)):
                assert np.array_equal(p, r), f'TPC {label}: patch[{i}] differs'
            print(f'[PASS] TPC {label}: identical to direct constructor')
    finally:
        os.unlink(tmppath)


def validate_tpc_overlap_corner(tc: TissuePatchContainer, size: int, label: str = 'TPC '):
    grid = tc.grid
    if not grid.has_overlap:
        print(f'[SKIP] {label}overlap corner: grid too small for overlap')
        return
    half = size // 2
    for info in grid.overlap_patch_infos:
        r, c = info.row, info.col
        p = tc[2*r+1, 2*c+1]
        assert np.array_equal(p[:half, :half],  tc[2*r,   2*c  ][half:, half:])
        assert np.array_equal(p[:half, half:],  tc[2*r,   2*c+2][half:, :half])
        assert np.array_equal(p[half:, :half],  tc[2*r+2, 2*c  ][:half, half:])
        assert np.array_equal(p[half:, half:],  tc[2*r+2, 2*c+2][:half, :half])
    print(f'[PASS] {label}overlap corner-pixel: {len(grid.overlap_patch_infos)} patches verified')


def validate_tpc_errors(img: np.ndarray, size: int):
    try:
        TissuePatchContainer(img.copy(), is_crop=True, region=None)
        raise AssertionError('expected ValueError for is_crop without region')
    except ValueError:
        pass

    fresh = TissuePatchContainer(img.copy())
    try:
        _ = fresh[0]
        raise AssertionError('expected RuntimeError before extract_all')
    except RuntimeError:
        pass

    tc = TissuePatchContainer(img.copy())
    tc.extract_all(size, overlap=True)
    try:
        _ = tc[len(tc)]
        raise AssertionError('expected IndexError for OOB flat')
    except IndexError:
        pass

    print('[PASS] TPC errors: ValueError / RuntimeError / OOB IndexError')


# ── crop() ────────────────────────────────────────────────────────────────────

def validate_container_crop(container, label: str = ''):
    """
    Verify container.crop() correctness across grid / pixel units, size / bottom_right,
    padding, clamping, and error cases.

    Assertions:
      - grid + bottom_right: img_origin & new-grid offset correct, patches identical
      - grid + size + pad:   pad clamps within grid bounds
      - pixel + size:        top-left floor / bottom-right ceil (round-up rule)
      - pixel + pad:         pad_unit='pixel' ceils to full grid tiles
      - PatchInfo.x/y stays in level-N global coords after crop
      - 6 invalid argument combos raise ValueError
    """
    grid = container.grid
    ts   = grid.tile_size
    R, C = grid.grid_rows, grid.grid_cols
    if R < 2 or C < 2:
        print(f'[SKIP] {label}crop: grid too small ({R}x{C})')
        return

    # ── (1) grid unit: (r0,c0) + bottom_right ─────────────────────────────────
    r0, c0, r1, c1 = 0, 0, min(2, R), min(2, C)
    sub = container.crop((r0, c0), bottom_right=(r1, c1), unit='grid')

    assert sub.width  == (c1 - c0) * ts
    assert sub.height == (r1 - r0) * ts
    assert sub.img_origin_x == grid.x_offset + c0 * ts
    assert sub.img_origin_y == grid.y_offset + r0 * ts
    assert sub.grid.x_offset == sub.img_origin_x   # invariant: grid starts at img[0,0]
    assert sub.grid.y_offset == sub.img_origin_y

    for r in range(r1 - r0):
        for c in range(c1 - c0):
            new_idx = (2*r, 2*c)         if sub.grid.has_overlap else (r, c)
            old_idx = (2*(r0+r), 2*(c0+c)) if grid.has_overlap    else (r0+r, c0+c)
            assert np.array_equal(sub[new_idx], container[old_idx]), \
                f'{label}crop main patch ({r},{c}) mismatch'

    # PatchInfo.x/y is still level-N global
    for info in sub.grid.main_patch_infos:
        orig_r = (info.y - grid.y_offset) // ts
        orig_c = (info.x - grid.x_offset) // ts
        assert r0 <= orig_r < r1 and c0 <= orig_c < c1, \
            f'{label}crop PatchInfo.x/y not global: got ({info.y},{info.x})'
    print(f'[PASS] {label}crop grid+corners: {sub.grid.grid_rows}x{sub.grid.grid_cols}, '
          f'origin=({sub.img_origin_x},{sub.img_origin_y}), patches match, x/y stays global')

    # ── (2) grid unit: size + pad ─────────────────────────────────────────────
    sub2 = container.crop((0, 0), size=(1, 1), unit='grid', pad=1)
    # size=(1,1) → [0,1); pad=1 → [-1,2); clamp → [0, min(R,2))
    assert sub2.grid.grid_rows == min(2, R)
    assert sub2.grid.grid_cols == min(2, C)
    print(f'[PASS] {label}crop grid+size+pad: {sub2.grid.grid_rows}x{sub2.grid.grid_cols} '
          'after clamp')

    # ── (3) pixel unit: non-aligned size → round up ───────────────────────────
    x0, y0 = grid.x_offset + 1, grid.y_offset + 1
    w, h   = ts + 3, ts + 3
    sub3 = container.crop((x0, y0), size=(w, h), unit='pixel')
    # top-left floor: (1//ts, 1//ts) = (0, 0)
    # bottom-right ceil: ceil((ts+4)/ts) = 2
    assert sub3.grid.grid_rows == min(2, R)
    assert sub3.grid.grid_cols == min(2, C)
    print(f'[PASS] {label}crop pixel+size+roundup: {sub3.grid.grid_rows}x{sub3.grid.grid_cols} '
          '(bottom-right ceiled to tile boundary)')

    # ── (4) pixel unit: pad (should ceil to whole grid tiles) ─────────────────
    sub4 = container.crop((grid.x_offset, grid.y_offset), size=(ts, ts),
                          unit='pixel', pad=ts + 1, pad_unit='pixel')
    # size=(ts,ts) → grid [0,1); pad_g = ceil((ts+1)/ts) = 2 → [-2, 3); clamp → [0, min(R,3))
    assert sub4.grid.grid_rows == min(3, R)
    assert sub4.grid.grid_cols == min(3, C)
    print(f'[PASS] {label}crop pixel-pad (ceil to grid): '
          f'{sub4.grid.grid_rows}x{sub4.grid.grid_cols}')

    # ── (5) invalid arg combos → ValueError ───────────────────────────────────
    bad_cases = [
        dict(top_left=(0, 0)),                                          # no br, no size
        dict(top_left=(0, 0), bottom_right=(1, 1), size=(1, 1)),        # both
        dict(top_left=(0, 0), size=(1, 1), unit='what'),                # bad unit
        dict(top_left=(0, 0), size=(1, 1), pad=1, pad_unit='what'),     # bad pad_unit
        dict(top_left=(0, 0), bottom_right=(0, 0)),                     # empty (r0==r1)
        dict(top_left=(R, C), size=(1, 1)),                             # out-of-bounds → empty
    ]
    for kwargs in bad_cases:
        try:
            container.crop(**kwargs)
            raise AssertionError(f'{label}crop expected ValueError for {kwargs}')
        except ValueError:
            pass
    print(f'[PASS] {label}crop errors: {len(bad_cases)} invalid arg combos raise ValueError')

    return sub  # for downstream figure demo


def validate_crop_before_extract():
    """crop() before extract_all() must raise RuntimeError (same contract as __getitem__)."""
    img  = make_gradient_image(128, 128)
    fresh_qc = QueryPatchContainer(img.copy())
    fresh_tc = TissuePatchContainer(img.copy())
    for c, name in [(fresh_qc, 'QPC'), (fresh_tc, 'TPC')]:
        try:
            c.crop((0, 0), size=(1, 1), unit='grid')
            raise AssertionError(f'{name} crop before extract must raise RuntimeError')
        except RuntimeError:
            pass
    print('[PASS] crop before extract: QPC / TPC both raise RuntimeError')


def validate_tpc_crop_extra_fields(tc: TissuePatchContainer, label: str = 'TPC '):
    """_copy_extra_after_crop must shallow-copy tissue_region / img_ds / is_crop / at_level."""
    sub = tc.crop((0, 0), size=(2, 2), unit='grid')
    assert sub.tissue_region is tc.tissue_region, f'{label}crop lost tissue_region'
    assert sub.img_ds        == tc.img_ds,        f'{label}crop img_ds mismatch'
    assert sub.is_crop       == tc.is_crop,       f'{label}crop is_crop mismatch'
    assert sub.at_level      == tc.at_level,      f'{label}crop at_level mismatch'
    print(f'[PASS] {label}crop extra fields shallow-copied '
          '(tissue_region / img_ds / is_crop / at_level)')


# ── Real data tests ───────────────────────────────────────────────────────────

def test_real_query(path: str, size: int) -> QueryPatchContainer:
    qc = QueryPatchContainer(path)
    qc.extract_all(size, overlap=True)
    validate_patch_shapes(qc, size, f'real-query({os.path.basename(path)}) ')
    validate_iterators(qc, f'real-query ')
    validate_qpc_main(qc, size, label=f'real-query ')
    if qc.grid.has_overlap:
        validate_qpc_overlap(qc, size, label=f'real-query ')
    # from_path vs from_array must yield identical patches
    qc2 = QueryPatchContainer.from_array(qc.img.copy())
    qc2.extract_all(size, overlap=True)
    for i, (p1, p2) in enumerate(zip(qc, qc2)):
        assert np.array_equal(p1, p2), f'real query from_array patch[{i}] differs'
    print(f'[PASS] Real query {os.path.basename(path)}: {qc.width}x{qc.height}, '
          f'{len(qc)} patches (size={size})')
    return qc


def test_real_roi_as_query(path: str, size: int) -> QueryPatchContainer:
    """RoI PNG used as a plain query image (no region info)."""
    qc = QueryPatchContainer(path)
    qc.extract_all(size, overlap=True)
    validate_patch_shapes(qc, size, 'roi-as-query ')
    validate_iterators(qc, 'roi-as-query ')
    validate_qpc_main(qc, size, label='roi-as-query ')
    if qc.grid.has_overlap:
        validate_qpc_overlap(qc, size, label='roi-as-query ')
    print(f'[PASS] RoI as query {os.path.basename(path)}: {qc.width}x{qc.height}, '
          f'{len(qc)} patches (size={size})')
    return qc


def test_real_roi(path: str, size: int) -> TissuePatchContainer:
    tc = TissuePatchContainer(path)
    tc.extract_all(size, overlap=True)
    validate_patch_shapes(tc, size, 'real-roi ')
    validate_iterators(tc, 'real-roi ')
    validate_tpc_overlap_corner(tc, size, 'real-roi ')
    print(f'[PASS] Real RoI {os.path.basename(path)}: {tc.width}x{tc.height}, '
          f'{len(tc)} patches (size={size})')
    return tc


def test_real_wsi(path: str, level: int, size: int) -> TissuePatchContainer:
    """Load the full WSI level image."""
    import openslide
    wsi = openslide.OpenSlide(path)
    W_l, H_l = wsi.level_dimensions[level]
    ds = wsi.level_downsamples[level]
    arr = np.array(wsi.read_region((0, 0), level, (W_l, H_l)).convert('RGB'))
    wsi.close()

    tc = TissuePatchContainer(arr, img_ds=ds)
    tc.extract_all(size, overlap=True)
    validate_patch_shapes(tc, size, 'real-wsi-full ')
    validate_iterators(tc, 'real-wsi-full ')
    validate_tpc_overlap_corner(tc, size, 'real-wsi-full ')
    print(f'[PASS] Real WSI {os.path.basename(path)} level={level} (ds={ds:.0f}) '
          f'{W_l}x{H_l}: {len(tc)} patches (size={size})')
    return tc


def test_real_wsi_from_openslide(path: str, level: int, size: int) -> TissuePatchContainer:
    """Test from_openslide factory method."""
    import openslide
    wsi = openslide.OpenSlide(path)
    level = min(level, wsi.level_count - 1)
    W_l, H_l = wsi.level_dimensions[level]
    tc = TissuePatchContainer.from_openslide(wsi, at_level=level)
    wsi.close()
    tc.extract_all(size, overlap=True)
    validate_patch_shapes(tc, size, 'from_openslide ')
    validate_iterators(tc, 'from_openslide ')
    validate_tpc_overlap_corner(tc, size, 'from_openslide ')
    print(f'[PASS] from_openslide level={level} ({W_l}x{H_l}): {len(tc)} patches')
    return tc


# ── Reconstruction ────────────────────────────────────────────────────────────

def reconstruct_image(container, main_only: bool = True):
    """
    Stitch patches back using PatchInfo.x/y as the destination coordinates.

    For QPC   : info.x/y are image-local coords (img_origin = 0).
    For TPC   : info.x/y are level-N global; subtract img_origin to get local.

    Returns
    -------
    canvas   : (H, W, 3) uint8 — reconstructed image (uncovered pixels = black)
    coverage : (H, W) bool    — True where at least one patch was written
    """
    canvas   = np.zeros((container.height, container.width, 3), dtype=np.uint8)
    coverage = np.zeros((container.height, container.width), dtype=bool)
    ox = getattr(container, 'img_origin_x', 0)
    oy = getattr(container, 'img_origin_y', 0)
    grid = container.grid

    if main_only:
        pairs = [(grid.flat_index_for_main(info.row, info.col), info)
                 for info in grid.main_patch_infos]
    else:
        pairs = [(i, grid.patch_info_at(i)) for i in range(len(grid))]

    for flat_i, info in pairs:
        patch = container[flat_i]
        lx = info.x - ox
        ly = info.y - oy
        s  = info.size_px
        canvas[ly:ly + s, lx:lx + s] = patch
        coverage[ly:ly + s, lx:lx + s] = True
    return canvas, coverage


def draw_reconstruction_row(axes_row, container, source_img: np.ndarray,
                             size: int, title: str = ''):
    """
    Fill one row of 4 axes with the reconstruction comparison:
      col 0 : source image + main-grid overlay
      col 1 : reconstructed from main patches
      col 2 : reconstructed from main + overlap patches (overlap overwrites)
      col 3 : per-pixel max abs-diff between source and main-reconstruction,
               masked to covered area; uncovered pixels shown as grey
    """
    ax_src, ax_main, ax_all, ax_diff = axes_row

    grid = container.grid
    ox = getattr(container, 'img_origin_x', 0)
    oy = getattr(container, 'img_origin_y', 0)

    # col 0: source + grid overlay
    ax_src.imshow(source_img)
    for info in grid.main_patch_infos:
        lx, ly = info.x - ox, info.y - oy
        ax_src.add_patch(mpatches.Rectangle(
            (lx, ly), size, size,
            fill=False, edgecolor='lime', linewidth=1.0,
        ))
    for info in grid.overlap_patch_infos:
        lx, ly = info.x - ox, info.y - oy
        ax_src.add_patch(mpatches.Rectangle(
            (lx, ly), size, size,
            fill=False, edgecolor='red', linewidth=1.0, linestyle='--',
        ))
    ax_src.set_title(f'{title}\noriginal + grid\n'
                     f'{grid.grid_rows}×{grid.grid_cols} main, '
                     f'{grid.overlap_rows}×{grid.overlap_cols} overlap')
    ax_src.legend(handles=[
        mpatches.Patch(edgecolor='lime', facecolor='none', label='main'),
        mpatches.Patch(edgecolor='red',  facecolor='none', label='overlap'),
    ], loc='upper right', fontsize=6)

    # col 1: reconstruct from main only
    recon_main, cov_main = reconstruct_image(container, main_only=True)
    ax_main.imshow(recon_main)
    pct = cov_main.mean() * 100
    ax_main.set_title(f'Reconstructed (main only)\ncoverage {pct:.1f}%')

    # col 2: reconstruct from main + overlap
    recon_all, cov_all = reconstruct_image(container, main_only=False)
    ax_all.imshow(recon_all)
    pct_all = cov_all.mean() * 100
    ax_all.set_title(f'Reconstructed (main + overlap)\ncoverage {pct_all:.1f}%')

    # col 3: diff (source vs main recon, covered pixels only)
    src_crop = source_img[:container.height, :container.width]
    diff = np.abs(src_crop.astype(np.int16) - recon_main.astype(np.int16)).max(axis=-1)
    # Show diff only where covered; grey elsewhere
    diff_vis = np.full((*diff.shape, 3), 180, dtype=np.uint8)
    diff_vis[cov_main] = np.stack([diff[cov_main]] * 3, axis=-1).clip(0, 255)
    ax_diff.imshow(diff_vis, vmin=0, vmax=20)
    max_d = int(diff[cov_main].max()) if cov_main.any() else 0
    ax_diff.set_title(f'|source − recon| (main)\nmax diff = {max_d} (expect 0)')
    ax_diff.text(source_img.shape[1] // 2, source_img.shape[0] // 2,
                 f'max={max_d}',
                 ha='center', va='center', fontsize=14,
                 color='lime' if max_d == 0 else 'red')

    for ax in axes_row:
        ax.axis('off')


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_rects(ax, origins, size, color, lw=1.2, linestyle='-'):
    for y, x in origins:
        ax.add_patch(mpatches.Rectangle(
            (x, y), size, size,
            fill=False, edgecolor=color, linewidth=lw, linestyle=linestyle,
        ))


def draw_region_bbox(ax, region: TissueRegion, ds: float, color='yellow'):
    rx, ry = int(region.x / ds), int(region.y / ds)
    rw, rh = int(region.w / ds), int(region.h / ds)
    ax.add_patch(mpatches.Rectangle(
        (rx, ry), rw, rh, fill=False, edgecolor=color, linewidth=2,
    ))


def show_patch_grid(ax, patches, n_cols: int = 4, title: str = ''):
    n = len(patches)
    if n == 0:
        ax.set_title(title + '\n(no patches)')
        ax.axis('off')
        return
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    s = patches[0].shape[0]
    canvas = np.ones((n_rows * s, n_cols * s, 3), dtype=np.uint8) * 220
    for idx, p in enumerate(patches[:n_cols * n_rows]):
        r, c = divmod(idx, n_cols)
        canvas[r*s:(r+1)*s, c*s:(c+1)*s] = p
    ax.imshow(canvas)
    ax.set_title(title)
    ax.axis('off')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size',  type=int, default=128,
                    help='tile size for synthetic tests')
    ap.add_argument('--rsize', type=int, default=256,
                    help='tile size for real-data tests')
    ap.add_argument('--query',
                    default='/work/u26130998/datasets/Ki67/S1103037_ki67/2.bmp',
                    help='real query image (BMP/PNG)')
    ap.add_argument('--roi',
                    default='/work/u26130998/datasets/histoimage.na.icar.cnr.it/'
                            'BRACS_RoI/latest_version/test/0_N/BRACS_264_N_5.png',
                    help='real RoI image, tested as both QPC and TPC')
    ap.add_argument('--wsi',
                    default='/work/u26130998/datasets/Ki67/S1103037,G7E,110122.mrxs',
                    help='real WSI (.mrxs/.svs)')
    ap.add_argument('--level', type=int, default=2,
                    help='WSI crop level (default 8, ~3MB for mrxs)')
    ap.add_argument('--openslide-level', type=int, default=9,
                    help='WSI level for from_openslide test (default 9, ~1MB for mrxs)')
    ap.add_argument('--out', default=None, help='output figure path')
    args = ap.parse_args()

    size  = args.size
    W, H  = 512, 512
    img   = make_gradient_image(W, H)

    # ── QPC synthetic ─────────────────────────────────────────────────────────
    qc = QueryPatchContainer(img.copy())
    qc.extract_all(size, overlap=True)
    validate_qpc_main(qc, size)
    validate_qpc_overlap(qc, size)
    validate_iterators(qc, 'QPC ')
    validate_patch_shapes(qc, size, 'QPC ')
    validate_qpc_no_overlap(img, size)
    validate_qpc_factory_methods(img, size)
    validate_qpc_multichannel(size)
    validate_qpc_errors(img, size)

    # ── TPC synthetic ─────────────────────────────────────────────────────────
    region = TissueRegion(x=W // 2, y=0, w=W // 2, h=H, index=0)
    ds = 1.0

    tc1 = TissuePatchContainer(img.copy(), img_ds=ds, is_crop=False)
    tc1.extract_all(size, overlap=True)
    validate_tpc_case1(tc1, img, size)
    validate_iterators(tc1, 'TPC-case1 ')
    validate_patch_shapes(tc1, size, 'TPC-case1 ')
    validate_tpc_overlap_corner(tc1, size, 'TPC-case1 ')

    tc2 = TissuePatchContainer(img.copy(), region=region, img_ds=ds, is_crop=False)
    tc2.extract_all(size, overlap=True)
    case2_patches = validate_tpc_case2(tc2, img, region, size, ds)
    validate_iterators(tc2, 'TPC-case2 ')
    validate_patch_shapes(tc2, size, 'TPC-case2 ')
    validate_tpc_overlap_corner(tc2, size, 'TPC-case2 ')

    rx = int(region.x / ds)
    crop_img = img[:, rx:].copy()
    tc3 = TissuePatchContainer(crop_img, region=region, img_ds=ds, is_crop=True)
    tc3.extract_all(size, overlap=True)
    validate_tpc_case3(tc3, size, case2_patches)
    validate_tpc_case3_overlap(tc2, tc3)
    validate_patch_shapes(tc3, size, 'TPC-case3 ')

    validate_tpc_ds_not_1(size)
    validate_tpc_region_y_offset(img, size)
    validate_tpc_patchinfo_meta(img, size)
    validate_tpc_no_overlap(img, region, ds, size)
    validate_tpc_factory_methods(img, size)
    validate_tpc_errors(img, size)

    # ── crop() ────────────────────────────────────────────────────────────────
    qc_sub  = validate_container_crop(qc,  label='QPC ')
    tc1_sub = validate_container_crop(tc1, label='TPC-case1 ')
    tc2_sub = validate_container_crop(tc2, label='TPC-case2 ')
    tc3_sub = validate_container_crop(tc3, label='TPC-case3 ')
    validate_crop_before_extract()
    validate_tpc_crop_extra_fields(tc2, 'TPC-case2 ')
    validate_tpc_crop_extra_fields(tc3, 'TPC-case3 ')

    # ── Real data ─────────────────────────────────────────────────────────────
    rsize = args.rsize
    real_qc = real_roi_qc = real_roi_tc = real_wsi_tc = None

    if args.query and os.path.exists(args.query):
        real_qc = test_real_query(args.query, rsize)
    elif args.query:
        print(f'[SKIP] query not found: {args.query}')

    if args.roi and os.path.exists(args.roi):
        real_roi_qc = test_real_roi_as_query(args.roi, rsize)
        real_roi_tc = test_real_roi(args.roi, rsize)
    elif args.roi:
        print(f'[SKIP] roi not found: {args.roi}')

    if args.wsi and os.path.exists(args.wsi):
        real_wsi_tc = test_real_wsi(args.wsi, args.level, rsize)
        test_real_wsi_from_openslide(args.wsi, args.openslide_level, rsize)
    elif args.wsi:
        print(f'[SKIP] wsi not found: {args.wsi}')

    # ── Figure ────────────────────────────────────────────────────────────────
    has_real = any(x is not None for x in [real_qc, real_roi_qc, real_roi_tc, real_wsi_tc])
    nrows = 3 if has_real else 2
    fig, axes = plt.subplots(nrows, 4, figsize=(24, 6 * nrows))

    # Row 0: QPC synthetic
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'QPC original\n{W}x{H}')

    axes[0, 1].imshow(img)
    draw_rects(axes[0, 1], main_origins(W, H, size), size, 'lime')
    axes[0, 1].set_title(f'QPC main grid\n{qc.grid.grid_rows}x{qc.grid.grid_cols} '
                         f'= {len(list(qc.iter_main()))} patches')

    axes[0, 2].imshow(img)
    draw_rects(axes[0, 2], main_origins(W, H, size), size, 'lime')
    draw_rects(axes[0, 2], overlap_origins(W, H, size), size, 'red', lw=1.5, linestyle='--')
    axes[0, 2].set_title(f'QPC +overlap\n+{len(list(qc.iter_overlap()))} corner patches')
    axes[0, 2].legend(handles=[
        mpatches.Patch(edgecolor='lime', facecolor='none', label='main'),
        mpatches.Patch(edgecolor='red',  facecolor='none', label='overlap'),
    ], loc='upper right', fontsize=7)

    show_patch_grid(axes[0, 3], list(qc.iter_main())[:8], n_cols=4,
                    title=f'QPC first 8 main patches (size={size})')

    # Row 1: TPC synthetic
    axes[1, 0].imshow(img)
    draw_rects(axes[1, 0], main_origins(W, H, size), size, 'lime')
    axes[1, 0].set_title(f'TPC case1: full, no region\n{len(list(tc1.iter_main()))} patches')

    axes[1, 1].imshow(img)
    draw_region_bbox(axes[1, 1], region, ds)
    rw_n, rh_n = int(region.w / ds), int(region.h / ds)
    glob_orig = [(y, rx + x) for y, x in main_origins(rw_n, rh_n, size)]
    draw_rects(axes[1, 1], glob_orig, size, 'cyan')
    axes[1, 1].set_title(f'TPC case2: full + region\n{len(case2_patches)} patches')
    axes[1, 1].legend(handles=[
        mpatches.Patch(edgecolor='yellow', facecolor='none', label='region bbox'),
        mpatches.Patch(edgecolor='cyan',   facecolor='none', label='region grid'),
    ], loc='upper left', fontsize=7)

    axes[1, 2].imshow(crop_img)
    draw_rects(axes[1, 2], main_origins(rw_n, rh_n, size), size, 'cyan')
    axes[1, 2].set_title(f'TPC case3: is_crop + region\n(same pixels as case2)')

    diffs = [np.abs(p2.astype(int) - p3.astype(int)).max()
             for p2, p3 in zip(case2_patches, list(tc3.iter_main()))]
    max_diff = max(diffs) if diffs else 0
    axes[1, 3].imshow(np.zeros((size, size, 3), dtype=np.uint8))
    axes[1, 3].text(size // 2, size // 2,
                    f'case2 vs case3\nmax diff={max_diff}\n(expect 0)',
                    ha='center', va='center', fontsize=13,
                    color='lime' if max_diff == 0 else 'red')
    axes[1, 3].set_title('Pixel diff panel')

    # Row 2: Real data
    if has_real:
        real_items = [
            (real_qc,     args.query, 'Real query (QPC)'),
            (real_roi_qc, args.roi,   'RoI as query (QPC)'),
            (real_roi_tc, args.roi,   'Real RoI (TPC)'),
            (real_wsi_tc, args.wsi,   'Real WSI crop (TPC)'),
        ]
        for col, (container, path, label) in enumerate(real_items):
            ax = axes[2, col]
            if container is None:
                ax.axis('off')
                continue
            patches = list(container.iter_main())
            show_patch_grid(ax, patches[:8], n_cols=4,
                            title=f'{label}\n{os.path.basename(path or "")}\n'
                                  f'{container.width}x{container.height} '
                                  f'→ {len(patches)} main / {len(list(container.iter_overlap()))} ovl')

    for row in axes:
        for ax in row:
            ax.axis('off')
    fig.tight_layout()

    out = args.out or os.path.join(job_result_dir('TissuePatchContainerTest'),
                                    'patch_container__grid.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # ── Reconstruction comparison figure ──────────────────────────────────────
    # Each row: [source+grid | recon-main | recon-all | diff]
    recon_cases = [
        (qc,   img,       'QPC synthetic'),
        (tc1,  img,       'TPC case1 (full, no region)'),
        (tc2,  img,       'TPC case2 (full + region)'),
        (tc3,  crop_img,  'TPC case3 (is_crop + region)'),
    ]
    # crop() demos — sub.img is exactly the pixel slice, so recon vs sub.img
    # gives diff=0 iff crop preserves patch coordinates correctly
    for sub, title in [
        (qc_sub,  'QPC crop (2x2 grid)'),
        (tc2_sub, 'TPC-case2 crop (2x2 grid)'),
    ]:
        if sub is not None:
            recon_cases.append((sub, sub.img, title))
    real_recon = [
        (real_qc,     None,  'Real query (QPC)'),
        (real_roi_qc, None,  'RoI as query (QPC)'),
        (real_roi_tc, None,  'Real RoI (TPC)'),
        (real_wsi_tc, None,  'Real WSI crop (TPC)'),
    ]
    for container, _, lbl in real_recon:
        if container is not None:
            recon_cases.append((container, container.img, lbl))

    fig2, axes2 = plt.subplots(len(recon_cases), 4,
                               figsize=(24, 6 * len(recon_cases)))
    if len(recon_cases) == 1:
        axes2 = axes2[np.newaxis, :]

    for row_axes, (container, src, title) in zip(axes2, recon_cases):
        tile = container.grid.tile_size
        draw_reconstruction_row(row_axes, container, src, tile, title)

    fig2.suptitle('Patch reconstruction comparison\n'
                  '(col 1: source + grid  |  col 2: stitched main  |  '
                  'col 3: stitched main+overlap  |  col 4: diff)', fontsize=11)
    fig2.tight_layout()

    out2 = os.path.join(os.path.dirname(out), 'patch_container__reconstruction.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved {out2}')

    print('\nAll checks passed.')


if __name__ == '__main__':
    main()
