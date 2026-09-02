#!/usr/bin/env python3
"""Tests for utilities/PreTileStore.py.

    python utilities/test_modules/test_pre_tile_store.py

No model, no GPU, no WSI, seconds. Everything runs against arrays this file
makes up and a temporary directory it deletes.

WHY THIS EXISTS BEFORE THE EXTRACTION DOES
-------------------------------------------
Step 3c reads six slides, writes ~32 GB and takes hours, and CLAUDE.md's rule is
to put the seconds-long check in front of it. Ask what a wrong result would look
like here:

    a centre crop off by one       every image is shifted one pixel against
                                   every label. Training converges. The model
                                   is slightly worse and nothing says why.
    a pre-tile of the wrong size   `centre_crop` still returns a square, of the
                                   wrong place, at the wrong scale
    two extractions colliding      the second silently replaces the first, and
                                   the identity that differed (seed, mask,
                                   sampler_id) is invisible in the name
    a lossy codec                  ringing at an 8x8 block boundary IS a corner.
                                   A keypoint detector would learn the codec

None of those raise on their own. All four are cheap to pin, and three of them
are pinned against a DECOY rather than a tolerance -- an off-by-one crop, a
neighbouring identity, a re-encode -- because a margin over a decoy is robust
and a threshold is a guess (CLAUDE.md, "a cheap assertion in front of an
expensive run").

Sections:
  1. geometry  -- the centre crop, against an off-by-one decoy
  2. identity  -- what makes two extractions two directories
  3. roundtrip -- write, read, and the losslessness the format depends on
  4. guards    -- what has to raise rather than quietly do something
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib
import tempfile

# `utilities/` holds the one definition of the output roots (`_paths.py`), and
# `setup_import_paths` -- which puts every other package on the path -- is
# inside it, so it has to be reachable before anything else is imported.
#
# BOTH parents are inserted, and that is deliberate rather than sloppy: this
# file runs from `utilities/test_modules/` and from
# `utilities/test_modules/TestSuperPathPoint/`, one level deeper, and inserting
# both means the move needs no edit here. The one that is not `utilities/` is
# either the repo root or `test_modules/`; neither holds a `_paths.py`, and
# `setup_import_paths` puts the repo root on the path anyway.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402

import PreTileStore                                  # noqa: E402
from PreTileStore import (PRE_TILE_FACTOR, PreTileMeta,    # noqa: E402
                                 PreTileMismatch, PreTileRecord,
                                 centre_crop, centre_margin, pre_tile_px)

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


TILE = 64          # small enough to be instant, even enough to have a centre
FACTOR = PRE_TILE_FACTOR


def _meta(**over) -> PreTileMeta:
    """A meta with plausible values. Not built through `of()` -- that needs a
    slide handle and a RungPlan, and none of these checks are about those."""
    base = dict(wsi_stem='SLIDE_A', ds=4.0, tile=TILE, sampler_id='aaaa1111',
                seed=0, segmenter_id='deadbeef', pre_tile_factor=FACTOR,
                level=1, level_ds=4.0, shrink=1.0,
                read_size=pre_tile_px(TILE, FACTOR))
    base.update(over)
    return PreTileMeta(**base)


def _planted_pre(marker: int = 200) -> np.ndarray:
    """A pre-tile that is uniform except for a `TILE x TILE` block in the exact
    centre. Any crop that is not the centre crop sees some of the surround."""
    pre = pre_tile_px(TILE, FACTOR)
    image = np.zeros((pre, pre, 3), np.uint8)
    off = centre_margin(TILE, FACTOR)
    image[off:off + TILE, off:off + TILE] = marker
    return image


# ══════════════════════════════════════════════════════════════════════════════
#  1. geometry
# ══════════════════════════════════════════════════════════════════════════════

def t_centre_crop_finds_the_planted_block():
    """The crop returns exactly the block, and the block only."""
    crop = centre_crop(_planted_pre(), TILE)
    assert crop.shape == (TILE, TILE, 3), crop.shape
    assert (crop == 200).all(), 'the crop is not the planted block'
    return f'{pre_tile_px(TILE, FACTOR)} -> {TILE}'


def t_off_by_one_crop_is_detectably_wrong():
    """The DECOY. A crop one pixel off must not pass the same check.

    This is the check that carries the section. A centre crop that is right and
    a centre crop that is one pixel off both return a `TILE x TILE` square of
    plausible tissue, so a shape assertion would pass on either. Only content
    separates them, and only a planted marker makes content separable.
    """
    image = _planted_pre()
    off = centre_margin(TILE, FACTOR)
    for shift in (-1, +1):
        wrong = image[off + shift:off + shift + TILE,
                      off + shift:off + shift + TILE]
        assert not (wrong == 200).all(), (
            f'a crop shifted by {shift} px looks identical to the centre crop, '
            f'so this test cannot see an off-by-one at all')
    return 'both +-1 shifts are visible'


def t_margin_is_symmetric_and_whole():
    """`pre - tile` is even for every tile size we plan to use.

    Not a property of the code -- a property of the numbers. 256/512/1024 at
    factor 3 all leave an even margin; the check exists so that changing either
    number is what fails, rather than the crop silently sitting half a pixel
    off centre on one side.
    """
    for tile in (256, 512, 1024, TILE):
        pre = pre_tile_px(tile, FACTOR)
        assert pre == tile * FACTOR, (tile, pre)
        assert (pre - tile) % 2 == 0, (tile, pre)
        assert centre_margin(tile, FACTOR) * 2 + tile == pre, tile
    return f'tiles 256/512/1024 at factor {FACTOR}'


def t_level0_geometry_is_self_consistent():
    """The record's derived positions agree with the meta's own footprints.

    `x`/`y` are the TILE's top-left; the pre-tile's is `margin_l0` up and left;
    the centre is half a tile footprint in. Three quantities written by three
    different expressions, so agreement is evidence rather than restatement.
    """
    meta = _meta(ds=4.0)
    record = PreTileRecord(index=0, x=10_000, y=20_000)

    assert meta.tile_footprint_l0 == TILE * 4.0, meta.tile_footprint_l0
    assert meta.margin_l0 == meta.tile_footprint_l0 * (FACTOR - 1) / 2

    px, py = record.pre_origin_l0(meta)
    cx, cy = record.centre_l0(meta)
    # The pre-tile's centre and the tile's centre are the same point. That is
    # the invariant the whole store is arranged around; if it ever stops being
    # true, every crop is of the wrong place.
    pre_foot = meta.tile_footprint_l0 * FACTOR
    assert abs((px + pre_foot / 2) - cx) < 1e-6, (px, cx)
    assert abs((py + pre_foot / 2) - cy) < 1e-6, (py, cy)
    return f'tile foot {meta.tile_footprint_l0:g} L0, margin {meta.margin_l0:g}'


# ══════════════════════════════════════════════════════════════════════════════
#  2. identity
# ══════════════════════════════════════════════════════════════════════════════

def t_every_identity_field_changes_the_directory():
    """Change one identity field at a time; the hash must move each time.

    Written as a loop over `_IDENTITY_FIELDS` rather than seven hand-written
    cases, so a field added to that tuple is covered the moment it is added --
    which is the failure mode `ConfigIdentity` exists to prevent.
    """
    base = _meta()
    moved = {'wsi_stem': 'SLIDE_B', 'tile': 128, 'pre_tile_factor': 2,
             'ds': 8.0, 'sampler_id': 'bbbb2222', 'seed': 1,
             'segmenter_id': 'cafebabe'}
    assert set(moved) == set(PreTileStore._IDENTITY_FIELDS), (
        f'this test covers {sorted(moved)} but the module identifies on '
        f'{sorted(PreTileStore._IDENTITY_FIELDS)}')
    for field, value in moved.items():
        other = _meta(**{field: value})
        assert other.cfg_hash() != base.cfg_hash(), (
            f'{field} changed and the hash did not, so two datasets that '
            f'differ in {field} would share a directory')
    return f'{len(moved)} fields'


def t_provenance_does_not_change_the_directory():
    """A slide that moved between mounts must not invalidate a cache.

    The other half of the identity rule, and the one that is easy to get wrong
    in the safe-looking direction: hashing everything makes every re-run a new
    directory, and 32 GB of pre-tiles is not something to rebuild because a
    path changed.
    """
    base = _meta()
    for field, value in (('wsi_path', '/other/mount/SLIDE_A.svs'),
                         ('level', 3), ('read_size', 999),
                         ('n_tiles', 500), ('n_clipped', 7),
                         ('created_at', '2020-01-01T00:00:00')):
        assert _meta(**{field: value}).cfg_hash() == base.cfg_hash(), (
            f'{field} is provenance or a derived fact, but it moved the hash')
    return 'wsi_path, level, read_size, counts, created_at'


def t_dirname_says_what_it_is_and_hashes_what_it_cannot():
    meta = _meta()
    name = meta.dirname()
    assert name.startswith('SLIDE_A__ds4__t64__'), name
    assert name.endswith(meta.cfg_hash()), name
    # Two extractions differing ONLY in a field the readable part cannot show.
    assert _meta(seed=1).dirname() != name, 'seed vanished from the name'
    assert _meta(sampler_id='cccc3333').dirname() != name
    return name


# ══════════════════════════════════════════════════════════════════════════════
#  3. roundtrip
# ══════════════════════════════════════════════════════════════════════════════

def t_write_then_read_returns_the_same_pixels():
    """Byte-exact, not close. The format's whole justification.

    A lossy codec would pass any tolerance-based version of this check and still
    put ringing at every 8x8 block boundary -- which is a corner, and the thing
    the detector is being trained to find. Exact equality is what excludes it.
    """
    image = _rand_pre()
    with tempfile.TemporaryDirectory() as root:
        meta = _meta()
        folder = PreTileStore.create(root, meta)
        record = PreTileRecord(index=3, x=1024, y=2048, clip_px=0)
        PreTileStore.save_tile(folder, record, image, meta)
        PreTileStore.write_index(folder, [record], meta)

        back = PreTileStore.read_tile(folder, record)
        assert back.shape == image.shape, (back.shape, image.shape)
        # Also the channel order: `save_tile` writes BGR and `read_tile` reads
        # it back, two reversals that have to cancel. This catches it only
        # because the image is NOISE -- on a grey or a synthetic-tissue pre-tile
        # an R/B swap can survive an all-equal check.
        assert (back == image).all(), (
            'the pixels changed between write and read: a lossy codec, or the '
            'channel order reversed once instead of twice')
    return f'{image.shape[0]} px, exact'


def t_index_and_meta_survive_the_round_trip():
    with tempfile.TemporaryDirectory() as root:
        meta = _meta()
        folder = PreTileStore.create(root, meta)
        records = [PreTileRecord(index=i, x=100 * i, y=200 * i,
                                 clip_px=(37 if i == 2 else 0))
                   for i in range(4)]
        for record in records:
            PreTileStore.save_tile(folder, record, _rand_pre(), meta)
        PreTileStore.write_index(folder, records, meta)

        back = PreTileStore.load_index(folder)
        assert back == records, 'the index did not come back as it went in'

        got = PreTileStore.load_meta(folder)
        assert got.cfg_hash() == meta.cfg_hash(), 'identity did not survive'
        # write_index fills these in, so they must NOT match what create wrote.
        assert got.n_tiles == 4, got.n_tiles
        assert got.n_clipped == 1, got.n_clipped
        assert got.created_at, 'create() did not stamp a time'
    return '4 records, counts filled by write_index'


def t_the_three_axes_survive_the_index_round_trip():
    """bucket / score / overlap_max / inherit_id / origin / parent.

    Recorded and not recomputed: every one is a property of the RUN that placed
    the tile -- `bucket` on the scorer and the edges, `overlap_max` on what
    else that rung took, `inherit_id` on a set fixed before any rung was
    filled. None is recoverable from (x, y) afterwards, so a column lost here
    is a question that can no longer be asked of the corpus.
    """
    with tempfile.TemporaryDirectory() as root:
        meta = _meta()
        folder = PreTileStore.create(root, meta)
        want = [
            PreTileRecord(index=0, x=100, y=200, bucket='gt80', score=0.83,
                          overlap_max=0.25, inherit_id=7, origin='jitter',
                          parent_x=36, parent_y=200),
            PreTileRecord(index=1, x=300, y=400, bucket='lt15', score=0.02,
                          inherit_id=-1, origin='grid'),
        ]
        PreTileStore.write_index(folder, want, meta)
        got = PreTileStore.load_index(folder)
        assert got == want, f'{got}\n{want}'
    return 'six axis columns, exact'


def t_an_index_without_the_axis_columns_still_loads():
    """A store cut before 2026-08-27 has `index,x,y,clip_px,file` and nothing
    else. It must read back with the DEFAULTS rather than a KeyError -- but a
    missing `x` is a corrupt index and must still raise, so the two are read
    differently on purpose (`.get` for the axes, `[...]` for the four)."""
    with tempfile.TemporaryDirectory() as root:
        meta = _meta()
        folder = PreTileStore.create(root, meta)
        (pathlib.Path(folder) / 'index.csv').write_text(
            'index,x,y,clip_px,file\n0,100,200,0,000000.png\n')
        got = PreTileStore.load_index(folder)
        assert len(got) == 1 and got[0].x == 100
        assert (got[0].bucket, got[0].inherit_id, got[0].origin) == \
            ('mid', -1, 'grid')

        (pathlib.Path(folder) / 'index.csv').write_text(
            'index,y,clip_px,file\n0,200,0,000000.png\n')
        try:
            PreTileStore.load_index(folder)
        except KeyError:
            pass
        else:
            raise AssertionError('an index with no x column loaded')
    return 'old format defaults, corrupt one raises'


def t_find_ignores_the_unfinished_and_refuses_the_ambiguous():
    """An interrupted directory is not a small dataset.

    This is what makes the extraction resumable after a walltime kill: the index
    is written last, so a killed job leaves a directory `find` does not see.
    """
    with tempfile.TemporaryDirectory() as root:
        done, half = _meta(seed=0), _meta(seed=1)
        done_dir = PreTileStore.create(root, done)
        PreTileStore.write_index(done_dir, [], done)
        PreTileStore.create(root, half)              # no index: interrupted

        hits = PreTileStore.find(root)
        assert [p.name for p in hits] == [done_dir.name], [p.name for p in hits]
        assert PreTileStore.find_one(root, seed=0) == done_dir

        # Two finished stores of the same slide and rung, differing only in a
        # field the caller did not mention -> refuse, do not pick one.
        other = _meta(seed=2)
        PreTileStore.write_index(PreTileStore.create(root, other), [], other)
        try:
            PreTileStore.find_one(root, wsi_stem='SLIDE_A')
        except PreTileMismatch as e:
            assert '2 stores' in str(e), str(e)
        else:
            raise AssertionError('find_one picked one of two candidates')
    return 'unfinished skipped, ambiguous refused'


def _rand_pre() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (pre_tile_px(TILE, FACTOR),
                                 pre_tile_px(TILE, FACTOR), 3), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  4. guards
# ══════════════════════════════════════════════════════════════════════════════

def _raises(fn, what):
    try:
        fn()
    except Exception as e:                                        # noqa: BLE001
        return type(e).__name__
    raise AssertionError(f'{what} did not raise')


def t_an_odd_margin_raises():
    """An odd tile at an even factor cannot have a centre. Better to refuse than
    to be off by half a pixel on one side."""
    return _raises(lambda: pre_tile_px(65, 2), 'tile 65 at factor 2')


def t_a_pre_tile_of_the_wrong_size_raises():
    """The one error that survives everything downstream if it is not caught
    here: `centre_crop` would still return a square, of the wrong place."""
    with tempfile.TemporaryDirectory() as root:
        meta = _meta()
        folder = PreTileStore.create(root, meta)
        wrong = np.zeros((pre_tile_px(TILE, FACTOR) - 2,
                          pre_tile_px(TILE, FACTOR) - 2, 3), np.uint8)
        return _raises(
            lambda: PreTileStore.save_tile(
                folder, PreTileRecord(0, 0, 0), wrong, meta),
            'a pre-tile 2 px short')


def t_reading_an_unfinished_store_raises():
    with tempfile.TemporaryDirectory() as root:
        folder = PreTileStore.create(root, _meta())
        return _raises(lambda: PreTileStore.load_index(folder),
                       'load_index on a store with no index')


def t_overwriting_a_finished_store_needs_saying_so():
    """A finished directory may already be a dataset a run has read. Adding to
    it would change its size out from under that run."""
    with tempfile.TemporaryDirectory() as root:
        meta = _meta()
        folder = PreTileStore.create(root, meta)
        PreTileStore.write_index(folder, [], meta)
        name = _raises(lambda: PreTileStore.create(root, meta),
                       'create over a finished store')
        again = PreTileStore.create(root, meta, overwrite=True)
        assert again == folder, (again, folder)
        return name


def t_a_non_square_pre_tile_raises():
    return _raises(lambda: centre_crop(np.zeros((64, 96, 3), np.uint8), 32),
                   'a non-square pre-tile')


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = {
    'geometry':  ['t_centre_crop_finds_the_planted_block',
                  't_off_by_one_crop_is_detectably_wrong',
                  't_margin_is_symmetric_and_whole',
                  't_level0_geometry_is_self_consistent'],
    'identity':  ['t_every_identity_field_changes_the_directory',
                  't_provenance_does_not_change_the_directory',
                  't_dirname_says_what_it_is_and_hashes_what_it_cannot'],
    'roundtrip': ['t_write_then_read_returns_the_same_pixels',
                  't_index_and_meta_survive_the_round_trip',
                  't_the_three_axes_survive_the_index_round_trip',
                  't_an_index_without_the_axis_columns_still_loads',
                  't_find_ignores_the_unfinished_and_refuses_the_ambiguous'],
    'guards':    ['t_an_odd_margin_raises',
                  't_a_pre_tile_of_the_wrong_size_raises',
                  't_reading_an_unfinished_store_raises',
                  't_overwriting_a_finished_store_needs_saying_so',
                  't_a_non_square_pre_tile_raises'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS),
                    help='run only these sections (default: all)')
    args = ap.parse_args()

    for section in (args.only or sorted(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            check(name[2:].replace('_', ' '), globals()[name])

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
