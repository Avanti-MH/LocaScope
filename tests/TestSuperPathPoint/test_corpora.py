#!/usr/bin/env python3
"""Tests for training/SuperPathPoint/common/Corpora.py -- a corpus is a draw.

    python utilities/test_modules/TestSuperPathPoint/test_corpora.py
    python utilities/test_modules/TestSuperPathPoint/test_corpora.py --wsi <slide>

Run through `jobscripts/SuperPathPointJobs/TestSuperPathPoint.sh` (the `store`
stage without a slide, `ladder-wsi` with one per pyramid shape).

    address   the key is what the draw is -- recipe, mask, rungs, tile, factor --
              and not where it is; the address is the draw's own
              (`TileSampler.draw_address`) grown by `draw=<sampler_id>`
    read      `--wsi` only: the centre crop of `Corpus.read` IS the tile at the
              draw's position, scored against the same read shifted by one tile
              -- a pre-tile read around the wrong corner would still be a
              square of tissue, and only a decoy can tell
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                           # noqa: E402

setup_import_paths('SuperPathPoint')

import numpy as np                                               # noqa: E402

from ReadGeometry import ReadSpec                                # noqa: E402
from TileSampler import SampleMeta, TileSampler, centre_crop     # noqa: E402
from common.Corpora import Corpus, Tile, corpus_of               # noqa: E402

_RESULTS = []
TILE = 256
_MASK = SimpleNamespace(seg_id=lambda: 'seg0', region_id=lambda: 'reg0')


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _corpus(**over) -> Corpus:
    base = dict(tile=TILE, job='JobA', mask_job='MaskA', rungs=(1.0, 4.0))
    base.update(over)
    return corpus_of(base.pop('name', 'stageA'), _MASK, **base)


# ── address ──────────────────────────────────────────────────────────────────

def t_the_key_is_what_the_draw_is_and_not_where():
    base = _corpus()
    for what, other in (('recipe', _corpus(name='stageB-cOwn')),
                        ('rungs', _corpus(rungs=(1.0, 2.0))),
                        ('tile', _corpus(tile=128)),
                        ('factor', _corpus(factor=2))):
        assert other.key != base.key, f'a different {what} kept the key'
    assert _corpus(job='JobB', mask_job='MaskB').key == base.key, (
        'the jobs are where, not what, and moved the key')
    return f'{base.key}'


def t_the_address_is_the_draws_own():
    corpus = _corpus()
    want = TileSampler.draw_address('JobA', 'SLIDE_A', _MASK, corpus.plan
                                    ).at(draw=corpus.sampler_id)
    assert corpus.address('SLIDE_A') == want, (corpus.address('SLIDE_A'), want)
    assert corpus.address('/x/y/SLIDE_A.svs') == want, 'a path is not its stem'
    assert corpus.address('SLIDE_A', 'Labels').made_by == 'Labels'
    assert corpus.address('SLIDE_A', 'Labels') == want.on('Labels')
    return f'.../draw={corpus.sampler_id}'


# ── read ─────────────────────────────────────────────────────────────────────

def _textured_tile(corpus, path, ds):
    """A `Tile` at a textured position of rung `ds`: the read is checked
    against a shifted decoy, and on glass every crop matches every other."""
    from SafeSlide import SafeSlide                               # noqa: PLC0415
    from SlideReader import SlideReader                           # noqa: PLC0415
    wsi = SafeSlide(path)
    reader = SlideReader(wsi, resize='area')
    w, h = wsi.dimensions
    rng = np.random.default_rng(0)
    best, best_std = None, -1.0
    margin = int(TILE * ds * corpus.factor)
    for _ in range(60):
        x = int(rng.integers(margin, w - 2 * margin))
        y = int(rng.integers(margin, h - 2 * margin))
        img = reader.read(x, y, ReadSpec(TILE, TILE), ds)
        if img is not None and float(img.std()) > best_std:
            best, best_std = (x, y), float(img.std())
    assert best is not None and best_std > 8.0, f'no textured tile ({best_std:.1f})'
    meta = SampleMeta(slide='S', ds=float(ds), level=0, x=best[0], y=best[1],
                      tile_size=TILE, read_size=TILE,
                      footprint_l0=int(round(TILE * ds)))
    return Tile(path, 0, meta), reader


def t_the_read_is_centred_on_the_tile(path):
    corpus = _corpus()
    lines = []
    for ds in corpus.rungs:
        tile, reader = _textured_tile(corpus, path, ds)
        m = tile.meta
        pre = corpus.read(tile)
        side = TILE * corpus.factor
        assert pre.shape == (side, side, 3), pre.shape
        got = centre_crop(pre, TILE).astype(np.float32)
        here = reader.read(m.x, m.y, ReadSpec(TILE, TILE), ds).astype(np.float32)
        step = int(round(TILE * ds))
        decoys = [reader.read(m.x + step, m.y, ReadSpec(TILE, TILE), ds),
                  reader.read(m.x, m.y + step, ReadSpec(TILE, TILE), ds)]
        mad = float(np.abs(got - here).mean())
        worst = min(float(np.abs(got - d.astype(np.float32)).mean())
                    for d in decoys if d is not None)
        assert mad < 0.5 * worst and worst - mad > 1.0, (
            f'ds {ds:g}: MAD {mad:.2f} to the tile, {worst:.2f} to a tile one '
            f'step over -- the pre-tile is not centred on its tile')
        lines.append(f'ds {ds:g} {mad:.2f} vs {worst:.2f}')
    return '; '.join(lines)


_SECTIONS = {
    'address': ['t_the_key_is_what_the_draw_is_and_not_where',
                't_the_address_is_the_draws_own'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wsi', default=None,
                    help='a slide for the read section; without it only the '
                         'address section runs')
    args = ap.parse_args()

    for section, names in _SECTIONS.items():
        print(f'\n[{section}]')
        for name in names:
            check(name[2:].replace('_', ' '), globals()[name])
    if args.wsi:
        print(f'\n[read]  {os.path.basename(args.wsi)}')
        check('the read is centred on the tile',
              lambda: t_the_read_is_centred_on_the_tile(args.wsi))

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
