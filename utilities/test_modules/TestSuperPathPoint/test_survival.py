#!/usr/bin/env python3
"""Tests for Stage B. spec.md 3.2.

    python utilities/test_modules/TestSuperPathPoint/test_survival.py

WHAT THIS BLOCKS
==================
Stage B produces two numbers that Stage C's design turns on: what fraction of
survival vectors are contiguous bands (spec.md 3.3 -- 0.97 means the head can
be two outputs, 0.6 means that simplification is a silent error), and how many
late births are the neighbourhood rather than blur. Both come out of code that
CANNOT FAIL LOUDLY: a classifier that calls `[1,0,1]` a band raises nothing, an
attribution branch that reads the wrong column returns a label, and a coordinate
scale used on the wrong axis fills the table with plausible numbers.

Sections:
  patterns     the six 樣態, every one by hand, plus the boundaries between them
  alive        `alive` is derived from score and dist, and both cuts bite
  attribution  the four causes, each against a decoy that must NOT get it
  scale        `rung_scale` vs `rung_shrink` -- two quantities, equal on 'F'
  table        the store round-trips, and refuses a rung count that disagrees
  degrade      the 'R' degradation is the one in TileSampler, not a copy
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                        # noqa: E402

setup_import_paths()

import numpy as np                                           # noqa: E402

from TileSampler import degrade_resolution                   # noqa: E402
from PointsAnalysisByMpp import Attribution, MppStack        # noqa: E402
from PointsAnalysisByMpp import Patterns, SurvivalTable      # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def _v(text):
    """`'110000'` -> a boolean vector. Six rungs, finest first."""
    return np.array([c == '1' for c in text], bool)


# ── 1. patterns ──────────────────────────────────────────────────────────────

_CASES = {
    '111111': '一直存活',
    '111000': '細部存活',
    '100000': '只在一階',
    '000001': '只在一階',      # width 1 at the COARSEST end, not 晚生型
    '001111': '晚生型',
    '011110': '中間帶',
    '101000': '不連續',
    '100001': '不連續',
    '000000': Patterns.EMPTY,
}


def t_the_six_patterns_by_hand():
    wrong = {}
    for text, want in _CASES.items():
        got, lo, hi = Patterns.classify(_v(text))
        if got != want:
            wrong[text] = (got, want)
    if wrong:
        raise AssertionError(f'misclassified: {wrong}')
    return f'{len(_CASES)} vectors'


def t_width_one_at_the_coarsest_end_is_not_late_born():
    """The boundary the ordering of the tests exists for.

    `000001` satisfies 晚生型's condition (`j_hi` is the coarsest) AND
    只在一階's (`j_lo == j_hi`). It has to be the second: 只在一階 is the
    stronger claim -- alive at this rung and NO other -- and the class whose
    members are alive at several rungs must not absorb it, because that class
    is the one Stage 1 would read as "this point tells me which rung I am on".
    """
    got, _, _ = Patterns.classify(_v('000001'))
    if got != '只在一階':
        raise AssertionError(
            f'`000001` came back as {got!r}. 晚生型 would then contain points '
            f'alive at exactly one rung, and the scale-signature class would '
            f'be diluted by them')
    if Patterns.classify(_v('001111'))[0] != '晚生型':
        raise AssertionError('the ordering broke 晚生型 in the other direction')


def t_a_flicker_is_not_a_band_and_still_reports_its_span():
    lo, hi, is_band = Patterns.band_of(_v('101000'))
    if is_band:
        raise AssertionError('`101000` reported as a contiguous band')
    if (lo, hi) != (0, 2):
        raise AssertionError(
            f'span came back {(lo, hi)}, expected (0, 2). A flickering point '
            f'still has a range, and reporting it is what tells "flickers '
            f'across a wide span" from "flickers between two neighbours"')


def t_band_fraction_ignores_empty_rows_rather_than_counting_them():
    """An all-dead row is a corrupt table, not a non-band.

    Counting it as a non-band pushes the fraction down, and the fraction is
    what spec.md 3.3 reads. A broken table would then look like a finding
    about scale structure.
    """
    rows = np.array([_v('111111'), _v('111000'), _v('000000')])
    got = Patterns.band_fraction(rows)
    if abs(got - 1.0) > 1e-9:
        raise AssertionError(
            f'band fraction {got} over two bands and one empty row; the empty '
            f'row was counted')
    mixed = np.array([_v('111111'), _v('101000')])
    if abs(Patterns.band_fraction(mixed) - 0.5) > 1e-9:
        raise AssertionError('a real non-band was not counted')
    return '1.00 with an empty row present, 0.50 with a real flicker'


def t_the_band_fraction_over_multi_rung_points_is_the_one_that_decides():
    """A width-one band is a band by arithmetic and decides nothing.

    spec.md 3.3 reads the band fraction to choose whether Stage C's head can be
    two outputs `(j_lo, j_hi)`. A point alive at exactly one rung has no
    interval to get wrong, so counting it as evidence FOR the simplification
    counts a non-answer as a yes -- and on the 2026-09-01 corpus 69 per cent of
    points were single-rung, which put the unfiltered fraction at 0.944 and the
    one that decides at 0.82.
    """
    rows = np.array([_v('100000'), _v('010000'), _v('001000'),  # three width-1
                     _v('111000'),                              # a real band
                     _v('101000')])                             # a real flicker
    every = Patterns.band_fraction(rows)
    multi = Patterns.band_fraction(rows, multi_only=True)
    if abs(every - 0.8) > 1e-9:
        raise AssertionError(
            f'{every}: four of five vectors are bands, the three singletons '
            f'trivially so')
    if abs(multi - 0.5) > 1e-9:
        raise AssertionError(
            f'{multi}: among the two multi-rung vectors exactly one is a band. '
            f'The singletons must not be in this denominator')
    return 'all: 0.80, >1 rung: 0.50 -- the singletons carried the first'


def t_every_pattern_has_an_ascii_label():
    """matplotlib has no CJK glyphs, and the pattern names ARE the axis labels.

    All three of Stage B's figures rendered their labels as empty boxes on
    2026-09-01. A pattern added without an entry here would do it again, and
    the failure is a warning on stderr rather than an error.
    """
    missing = [p for p in Patterns.PATTERNS if p not in Patterns.ASCII_NAMES]
    if missing:
        raise AssertionError(f'{missing} have no ASCII label for plotting')
    if len(set(Patterns.ASCII_NAMES.values())) != len(Patterns.PATTERNS):
        raise AssertionError(
            f'two patterns share an ASCII label: '
            f'{sorted(Patterns.ASCII_NAMES.values())}')
    return ', '.join(Patterns.ASCII_NAMES[p] for p in Patterns.PATTERNS)


def t_summarise_keeps_every_key_even_at_zero():
    out = Patterns.summarise(np.array([_v('111111')]))
    missing = [k for k in Patterns.PATTERNS if k not in out]
    if missing:
        raise AssertionError(
            f'{missing} absent from the summary. A bar chart built from a dict '
            f'that dropped its empty classes silently renumbers its own axis')


# ── 2. alive is derived ──────────────────────────────────────────────────────

def t_alive_needs_both_the_score_and_the_distance():
    """Two cuts, and each one alone must be able to kill a rung.

    If only the score bit were applied, tau would be decorative and the coarse
    rungs would report matches that are hundreds of level-0 px away. If only
    the distance bit were, the permissive store cut would become the analysis
    threshold and nothing could be swept.
    """
    tau = np.array([2.0, 4.0, 8.0], float)
    score = np.array([0.9, 0.9, 0.001])
    dist = np.array([0.0, 100.0, 0.0])
    got = Patterns.alive_from(score, dist, score_threshold=0.01, tau=tau)
    if list(got) != [True, False, False]:
        raise AssertionError(
            f'{list(got)} -- rung 1 must die on distance and rung 2 on score')
    unmatched = Patterns.alive_from(np.array([0.9]), np.array([-1.0]),
                                    score_threshold=0.01, tau=np.array([9e9]))
    if unmatched[0]:
        raise AssertionError(
            '`dist = -1` means no partner was found at that rung, which is '
            'dead. A tau of infinity must not resurrect it')
    return 'score cut, tau cut, and the -1 sentinel all bite'


# ── 3. attribution ───────────────────────────────────────────────────────────

def _row(f, r, score=None, sup=None):
    """One row of both axes. `sup` is the RIVAL: the strongest response inside
    the NMS radius of this location's peak. Defaults to 0, i.e. nothing beat
    it -- `NONE` would mean "not measured", which is a different claim."""
    length = len(f)
    return dict(
        f_alive=_v(f), r_alive=_v(r),
        f_score=np.array(score if score is not None else [0.0] * length),
        suppressed_score=np.array(sup if sup is not None else [0.0] * length))


def t_born_at_the_finest_rung_is_not_a_birth_to_explain():
    got = Attribution.attribute(**_row('111000', '111000'))
    if got != Attribution.NOT_LATE:
        raise AssertionError(f'{got!r}; there is no rung below the finest')


def t_blur_is_when_the_r_axis_is_born_at_the_same_rung():
    got = Attribution.attribute(**_row('001111', '001111'))
    if got != Attribution.BLUR:
        raise AssertionError(f'{got!r}, expected 模糊新生')
    # THE DECOY: same F vector, but R was born at the finest rung, so the birth
    # is NOT explained by detail loss. Without this the blur branch would fire
    # on anything the R axis happened to detect.
    decoy = Attribution.attribute(**_row('001111', '111111'))
    if decoy == Attribution.BLUR:
        raise AssertionError(
            'a point alive on R from the finest rung was still called blur; '
            'the branch is not reading `r_born == born`')
    return 'blur fires on a matched R birth and not on an early one'


def t_a_score_rise_is_the_neighbourhood():
    got = Attribution.attribute(**_row(
        '001111', '000000', score=[0.0, 0.001, 0.30, 0.3, 0.3, 0.3]))
    if got != Attribution.NEIGHBOURHOOD_SCORE:
        raise AssertionError(f'{got!r}, expected 鄰域新生（分數）')
    # THE DECOY: the same birth with a score that did not move. It must fall
    # through to another branch, or the score test is not being applied.
    flat = Attribution.attribute(**_row(
        '001111', '000000', score=[0.0, 0.014, 0.015, 0.3, 0.3, 0.3]))
    if flat == Attribution.NEIGHBOURHOOD_SCORE:
        raise AssertionError(
            'a rise of 0.001 was accepted as a score rise; the threshold is '
            'not being applied')
    return 'a 0.30 rise fires, a 0.001 rise does not'


def t_a_released_suppressor_is_the_other_neighbourhood_branch():
    """Outranked below, not outranked at the birth rung.

    `rival > score` is the whole of it: with `max_keypoints` off, the only
    competition left is a stronger response inside the NMS radius, and this is
    the column that records one.
    """
    flat = [0.0, 0.02, 0.02, 0.02, 0.02, 0.02]
    got = Attribution.attribute(**_row(
        '001111', '000000', score=flat,
        sup=[0.0, 0.9, 0.0, 0.0, 0.0, 0.0]))
    if got != Attribution.NEIGHBOURHOOD_RELEASE:
        raise AssertionError(f'{got!r}, expected 鄰域新生（壓制解除）')
    # DECOY ONE: still outranked at the birth rung. Nothing was released.
    still = Attribution.attribute(**_row(
        '001111', '000000', score=flat, sup=[0.0, 0.9, 0.9, 0.0, 0.0, 0.0]))
    if still == Attribution.NEIGHBOURHOOD_RELEASE:
        raise AssertionError(
            'a point still outranked at its birth rung was called released')
    # DECOY TWO: the rung below was never probed (`NONE`). Absence of a
    # measurement must not read as "it was outranked there".
    unprobed = Attribution.attribute(**_row(
        '001111', '000000', score=flat,
        sup=[0.0, Attribution.NONE, 0.0, 0.0, 0.0, 0.0]))
    if unprobed == Attribution.NEIGHBOURHOOD_RELEASE:
        raise AssertionError(
            'a NONE sentinel at the rung below was read as a suppressor; -1 '
            'means "not measured", not "something beat it"')
    return 'release needs an outranking below AND none at the birth rung'


def t_nothing_that_fits_is_undecided_and_not_silently_bucketed():
    got = Attribution.attribute(**_row(
        '001111', '000000', score=[0.0, 0.02, 0.02, 0.02, 0.02, 0.02]))
    if got != Attribution.UNDECIDED:
        raise AssertionError(
            f'{got!r} -- a late birth with no score rise and no release has no '
            f'explanation, and forcing it into one would put a number under a '
            f'label nothing measured')


def t_the_summary_carries_the_denominator():
    out = Attribution.summarise([Attribution.BLUR, Attribution.NOT_LATE])
    if Attribution.NOT_LATE not in out:
        raise AssertionError(
            'NOT_LATE absent. An attribution split over late-born points reads '
            'as a statement about all points unless the rest is beside it')


# ── 4. scale vs shrink ───────────────────────────────────────────────────────

def t_scale_and_shrink_agree_on_F_and_disagree_on_R():
    """The two quantities that are both `ds` on one axis and are not the same.

    `rung_scale` is level-0 px per output PIXEL -- the mapping. `rung_shrink`
    is how far a position can be off -- the tolerance. On 'F' both are `ds`; on
    'R' the mapping is 1.0 (the frame never moves) while the tolerance is still
    `ds` (the image was degraded). Using `ds` as the 'R' mapping scatters every
    coarse point `ds` times too far and the table still fills.
    """
    for ds in (1.0, 4.0, 32.0):
        if MppStack.rung_scale(ds, 'F') != ds:
            raise AssertionError(f"F scale at ds {ds} is not ds")
        if MppStack.rung_shrink(ds, 'F') != ds:
            raise AssertionError(f"F shrink at ds {ds} is not ds")
        if MppStack.rung_scale(ds, 'R') != 1.0:
            raise AssertionError(
                f"R scale at ds {ds} came back "
                f"{MppStack.rung_scale(ds, 'R')}, not 1.0")
        if MppStack.rung_shrink(ds, 'R') != ds:
            raise AssertionError(f"R shrink at ds {ds} is not ds")
    return "F: scale == shrink == ds.  R: scale 1.0, shrink ds"


# ── 5. the store ─────────────────────────────────────────────────────────────

def _batch(n=5, length=6):
    rng = np.random.default_rng(0)
    return SurvivalTable.SurvivalBatch(
        x0=rng.uniform(0, 1000, n).astype(np.float32),
        y0=rng.uniform(0, 1000, n).astype(np.float32),
        chain=np.arange(n, dtype=np.int32),
        score=rng.uniform(0, 1, (n, length)).astype(np.float32),
        dist=rng.uniform(0, 10, (n, length)).astype(np.float32),
        suppressed_by_score=np.full((n, length), -1, np.float32),
        suppressed_by_dist=np.full((n, length), -1, np.float32),
        decoy_score=rng.uniform(0, 1, (n, length)).astype(np.float32),
        decoy_dist=rng.uniform(0, 10, (n, length)).astype(np.float32))


def t_the_store_round_trips_and_keeps_the_rung_order():
    meta = SurvivalTable.SurvivalMeta(
        wsi_stem='S1103627,G7E,110127', stack_kind='F', tile=256,
        rungs=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0), detector_id='abcd1234')
    with tempfile.TemporaryDirectory() as root:
        path = SurvivalTable.save(root, _batch(), meta)
        back, back_meta = SurvivalTable.load(path)
        if back_meta.rungs != meta.rungs:
            raise AssertionError(
                f'{back_meta.rungs} != {meta.rungs}; the [N, L] columns are '
                f'indexed by this tuple')
        if back_meta.wsi_stem != meta.wsi_stem:
            raise AssertionError(
                f'{back_meta.wsi_stem!r} -- the stem has commas in it and they '
                f'have to survive the metadata round trip')
        if not np.allclose(back.score, _batch().score):
            raise AssertionError('score did not round trip')
    return 'commas in the stem survive, rung order preserved'


def t_a_column_count_that_disagrees_with_the_rungs_is_refused():
    """The failure that would relabel every rung and say nothing.

    Six columns against five named rungs means every column after the mismatch
    describes a different magnification than its label. Nothing downstream
    checks it -- `Patterns` takes a vector, not a rung list.
    """
    meta = SurvivalTable.SurvivalMeta(
        wsi_stem='A', stack_kind='F', tile=256, rungs=(1.0, 2.0, 4.0))
    with tempfile.TemporaryDirectory() as root:
        try:
            SurvivalTable.save(root, _batch(length=6), meta)
        except SurvivalTable.SurvivalMismatch:
            return 'refused'
    raise AssertionError('a 6-column batch was written against 3 named rungs')


def t_an_unnamed_axis_is_refused():
    meta = SurvivalTable.SurvivalMeta(
        wsi_stem='A', stack_kind='', tile=256, rungs=tuple([1.0] * 6))
    with tempfile.TemporaryDirectory() as root:
        try:
            SurvivalTable.save(root, _batch(), meta)
        except SurvivalTable.SurvivalMismatch:
            return 'refused'
    raise AssertionError(
        "a table with no stack_kind was written; a survival number that does "
        "not say which axis it is about is meaningless")


def t_tau_grows_with_the_rung_and_never_collapses():
    """The failure spec.md 3.2 calls a wrong answer that looks like a discovery.

    A fixed level-0 tau makes the coarse rungs unable to match BY DEFINITION --
    one of their pixels is `ds` level-0 px, so a real point cannot be located
    better than that -- and the output reads as "keypoints all die at coarse
    resolution".
    """
    meta = SurvivalTable.SurvivalMeta(
        wsi_stem='A', stack_kind='F', tile=256,
        rungs=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0), tau_alpha=1.5)
    tau = meta.tau()
    if not np.all(np.diff(tau) > 0):
        raise AssertionError(f'tau {tau} does not grow with the rung')
    if abs(tau[-1] - 48.0) > 1e-9:
        raise AssertionError(f'tau at ds 32 is {tau[-1]}, expected 1.5 * 32')
    return f'tau {tau[0]:.1f} .. {tau[-1]:.1f} level-0 px'


# ── 6. the degradation is shared ─────────────────────────────────────────────

def t_the_r_degradation_loses_detail_and_keeps_the_frame():
    """'R' at ds d: same size out, `tile/d` real samples in it.

    Checked against a decoy that would pass a shape assertion: an image that
    was resized and resized back with the SAME filter both ways keeps more
    high-frequency content than the INTER_AREA/INTER_LINEAR pair, so a
    'degradation' that lost nothing would show up here as a variance that did
    not drop.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    out = degrade_resolution(img, 8.0, 256)
    if out.shape != img.shape:
        raise AssertionError(f'{out.shape} != {img.shape}; the frame moved')
    before = float(np.var(np.diff(img[..., 0].astype(np.float32), axis=1)))
    after = float(np.var(np.diff(out[..., 0].astype(np.float32), axis=1)))
    if after >= before * 0.25:
        raise AssertionError(
            f'horizontal detail variance {before:.0f} -> {after:.0f}; a ds 8 '
            f'degradation should remove most of it')
    same = degrade_resolution(img, 1.0, 256)
    if not np.array_equal(same, img):
        raise AssertionError(
            'ds 1 changed the image. ds 1 is the identity on both axes, and '
            'that is the assertion the whole F/R comparison rests on')
    return f'detail variance {before:.0f} -> {after:.0f}, ds 1 is exact'


_SECTIONS = {
    'patterns':    ['t_the_six_patterns_by_hand',
                    't_width_one_at_the_coarsest_end_is_not_late_born',
                    't_a_flicker_is_not_a_band_and_still_reports_its_span',
                    't_band_fraction_ignores_empty_rows_rather_than_counting_them',
                    't_summarise_keeps_every_key_even_at_zero',
                    't_the_band_fraction_over_multi_rung_points_is_the_one_that_decides',
                    't_every_pattern_has_an_ascii_label'],
    'alive':       ['t_alive_needs_both_the_score_and_the_distance'],
    'attribution': ['t_born_at_the_finest_rung_is_not_a_birth_to_explain',
                    't_blur_is_when_the_r_axis_is_born_at_the_same_rung',
                    't_a_score_rise_is_the_neighbourhood',
                    't_a_released_suppressor_is_the_other_neighbourhood_branch',
                    't_nothing_that_fits_is_undecided_and_not_silently_bucketed',
                    't_the_summary_carries_the_denominator'],
    'scale':       ['t_scale_and_shrink_agree_on_F_and_disagree_on_R'],
    'table':       ['t_the_store_round_trips_and_keeps_the_rung_order',
                    't_a_column_count_that_disagrees_with_the_rungs_is_refused',
                    't_an_unnamed_axis_is_refused',
                    't_tau_grows_with_the_rung_and_never_collapses'],
    'degrade':     ['t_the_r_degradation_loses_detail_and_keeps_the_frame'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    args = ap.parse_args()

    for section in (args.only or list(_SECTIONS)):
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
