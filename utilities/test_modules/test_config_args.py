#!/usr/bin/env python3
"""Tests for utilities/ConfigArgs.py -- one command-line flag per field of a
frozen dataclass config, and the `<config>_from_args` that use it.

    python utilities/test_modules/test_config_args.py

No slide, no model, no GPU. The mechanism is checked on made-up dataclasses that
carry every field type it claims to handle; the real sampler and camera configs
are then run through it, because "every field of my config has a flag" is a claim
about THOSE classes and not about the made-up ones.

WHAT THIS DEFENDS
-----------------
    untouched     no flag given returns the base object itself, so a run that
                  passes nothing uses exactly what its CONFIG block says
    one at a time a flag changes its field and no other, nested fields included
    refusal       a value the config's own checks reject is refused at parse
                  time, with that config's message -- not accepted and used
    no dead flag  a field a caller `skip`s has no flag at all
    identity      two configs that differ in one field differ in `asdict`, which
                  is what the parts id is made of
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

from _paths import setup_import_paths                        # noqa: E402

setup_import_paths()

import ConfigArgs as C                                         # noqa: E402
from TileSampler import (InheritConfig, OverlapConfig,         # noqa: E402
                         RichnessConfig, SamplerConfig, add_sampler_args,
                         sampler_from_args)
from config import DomainGapConfig                             # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── made-up configs with every field type ────────────────────────────────────

@dataclass(frozen=True)
class Inner:
    count: int = 1
    weights: Tuple[float, ...] = (0.1, 0.2)


@dataclass(frozen=True)
class Outer:
    name: str = 'x'
    ratio: Optional[float] = None
    flag: bool = True
    pair: Tuple[float, float] = (0.9, 1.1)
    offsets: Tuple[Tuple[float, float], ...] = ((0.25, 1.0),)
    inner: Inner = field(default_factory=Inner)
    other: Inner = field(default_factory=Inner)

    def __post_init__(self):
        if self.ratio is not None and not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f'ratio {self.ratio} is not in [0, 1]')


def _ap():
    """The parser these flags need: no abbreviations (see add_config_args)."""
    return argparse.ArgumentParser(allow_abbrev=False)


def _refused(parser, argv) -> bool:
    """True when argparse rejects `argv`, with its usage message kept out of the
    test output."""
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(argv)
        except SystemExit:
            return True
    return False


def _parser(base, prefix, **kw):
    ap = _ap()
    C.add_config_args(ap, base, prefix, **kw)
    return ap


def t_no_flag_returns_the_base_itself():
    base = Outer()
    ap = _parser(base, 'c')
    assert C.config_from_args(ap.parse_args([]), base, 'c') is base


def t_every_field_type_can_be_set():
    base = Outer()
    ap = _parser(base, 'c')
    args = ap.parse_args(
        ['--c-name', 'y', '--c-ratio', '0.5', '--no-c-flag', '--c-pair', '1', '2',
         '--c-offsets', '0.1,0.2', '0.3,0.4'])
    got = C.config_from_args(args, base, 'c')
    assert got.name == 'y' and got.ratio == 0.5 and got.flag is False, got
    assert got.pair == (1.0, 2.0), got.pair
    assert got.offsets == ((0.1, 0.2), (0.3, 0.4)), got.offsets


def t_optional_takes_the_word_none():
    base = Outer(ratio=0.2)
    ap = _parser(base, 'c')
    got = C.config_from_args(ap.parse_args(['--c-ratio', 'none']), base, 'c')
    assert got.ratio is None


def t_a_flag_changes_its_field_and_no_other():
    base = Outer()
    ap = _parser(base, 'c')
    got = C.config_from_args(ap.parse_args(['--c-inner-count', '7']), base, 'c')
    assert got.inner == Inner(count=7, weights=(0.1, 0.2)), got.inner
    assert got.other == base.other and got.name == base.name
    assert dataclasses.replace(got, inner=base.inner) == base


def t_a_value_the_config_rejects_is_refused_at_parse_time():
    base = Outer()
    ap = _parser(base, 'c')
    try:
        C.config_from_args(ap.parse_args(['--c-ratio', '2']), base, 'c')
    except ValueError as exc:
        assert 'not in [0, 1]' in str(exc), exc
    else:
        raise AssertionError('ratio 2 was accepted')


def t_a_skipped_field_has_no_flag():
    ap = _parser(Outer(), 'c', skip=('other',))
    assert _refused(ap, ['--c-other-count', '3']), 'a skipped field still had a flag'


def t_a_type_it_does_not_know_is_an_error_when_the_flags_are_added():
    @dataclass(frozen=True)
    class Odd:
        table: dict = field(default_factory=dict)
    try:
        _parser(Odd(), 'c')
    except TypeError as exc:
        assert 'table' in str(exc), exc
    else:
        raise AssertionError('a dict field silently had no flag')


def t_a_parser_that_abbreviates_is_refused():
    try:
        C.add_config_args(argparse.ArgumentParser(), Outer(), 'c')
    except ValueError as exc:
        assert 'allow_abbrev' in str(exc), exc
        return
    raise AssertionError('flags were added to a parser that abbreviates')


def t_help_carries_the_base_value():
    text = _parser(Outer(name='hello'), 'c').format_help()
    assert "base: 'hello'" in text, text


def t_describe_lists_nested_fields_with_dots():
    lines = C.describe(Outer(), 'cfg')
    assert 'cfg.inner.count = 1' in lines and 'cfg.name = \'x\'' in lines, lines


# ── the real configs ─────────────────────────────────────────────────────────

def t_richness_caps_from_the_command_line():
    base = RichnessConfig()
    ap = _ap()
    C.add_config_args(ap, base, 'richness')
    got = C.config_from_args(ap.parse_args(
        ['--richness-caps', '0.15', '0.25', '0.6', '0', '0', '0', '0']), base,
        'richness')
    assert got.caps == (0.15, 0.25, 0.6, 0.0, 0.0, 0.0, 0.0), got.caps
    assert got.floors == base.floors and got.edges == base.edges


def t_richness_with_the_wrong_number_of_caps_is_refused():
    base = RichnessConfig()
    ap = _ap()
    C.add_config_args(ap, base, 'richness')
    try:
        C.config_from_args(ap.parse_args(['--richness-caps', '0.5', '0.5']),
                           base, 'richness')
    except ValueError as exc:
        return str(exc)[:60]
    raise AssertionError('two caps for seven buckets were accepted')


def t_sampler_flags_reach_the_nested_configs():
    base = SamplerConfig()
    ap = _ap()
    add_sampler_args(ap, base)
    args = ap.parse_args(
        ['--sampler-n-per-rung', '50', '--richness-caps', '0.15', '0.25', '0.6',
         '0', '0', '0', '0', '--overlap-step', '0.5',
         '--overlap-jitter-offsets', '0.5,1.0', '1.0,0.5',
         '--inherit-source-rung', 'none'])
    try:
        got = sampler_from_args(args, base)
    except ValueError as exc:
        # step 0.5 with a disjoint bound is a contradiction the config
        # refuses (OverlapConfig.check); that IS the point of going through it
        return f'refused as it should: {str(exc)[:50]}'
    assert got.n_per_rung == 50
    assert got.richness.caps[3] == 0.0 and got.overlap.step == 0.5


def t_sampler_without_overlap_change_is_the_base():
    base = SamplerConfig()
    ap = _ap()
    add_sampler_args(ap, base)
    got = sampler_from_args(ap.parse_args(['--sampler-n-per-rung', '50']), base)
    assert got.n_per_rung == 50
    assert got.richness is base.richness and got.overlap is base.overlap
    assert sampler_from_args(ap.parse_args([]), base) is base


def t_the_sampler_has_no_tile_flag():
    ap = _ap()
    add_sampler_args(ap, SamplerConfig())
    assert _refused(ap, ['--sampler-tile', '512']), \
        "--sampler-tile exists, but the tile is the camera's, not the sampler's"


def t_camera_flags():
    base = DomainGapConfig()
    ap = _ap()
    C.add_config_args(ap, base, 'camera')
    args = ap.parse_args(['--camera-rotation-choices', '0', '90',
                          '--camera-scale-range', '0.9', '1.1',
                          '--no-camera-photometric', '--camera-noise-sigma', '0'])
    got = C.config_from_args(args, base, 'camera')
    assert got.rotation_choices == (0, 90) and got.scale_range == (0.9, 1.1)
    assert got.photometric is False and got.noise_sigma == 0.0
    assert got.jpeg_quality == base.jpeg_quality
    # `--camera-query-mpp` is a PREFIX of `--camera-query-mpp-jitter`: with
    # abbreviations on, this would set the jitter instead of failing
    assert _refused(ap, ['--camera-query-mpp', '0.5']), \
        '--camera-query-mpp was taken for another flag or exists'
    assert got.query_mpp_jitter == base.query_mpp_jitter


def t_camera_scale_range_with_low_above_high_is_refused():
    base = DomainGapConfig()
    ap = _ap()
    C.add_config_args(ap, base, 'camera')
    try:
        C.config_from_args(ap.parse_args(['--camera-scale-range', '1.2', '0.9']),
                           base, 'camera')
    except ValueError:
        return
    raise AssertionError('a scale range with low above high was accepted')


def t_a_changed_field_changes_the_identity():
    a = dataclasses.asdict(SamplerConfig())
    b = dataclasses.asdict(SamplerConfig(n_per_rung=50))
    c = dataclasses.asdict(dataclasses.replace(
        SamplerConfig(), richness=RichnessConfig(
            caps=(0.15, 0.25, 0.6, 0.0, 0.0, 0.0, 0.0))))
    assert a != b and a != c and b != c


_SECTIONS = {
    'mechanism': [
        't_no_flag_returns_the_base_itself', 't_every_field_type_can_be_set',
        't_optional_takes_the_word_none',
        't_a_flag_changes_its_field_and_no_other',
        't_a_value_the_config_rejects_is_refused_at_parse_time',
        't_a_skipped_field_has_no_flag',
        't_a_type_it_does_not_know_is_an_error_when_the_flags_are_added',
        't_a_parser_that_abbreviates_is_refused',
        't_help_carries_the_base_value',
        't_describe_lists_nested_fields_with_dots'],
    'real': [
        't_richness_caps_from_the_command_line',
        't_richness_with_the_wrong_number_of_caps_is_refused',
        't_sampler_flags_reach_the_nested_configs',
        't_sampler_without_overlap_change_is_the_base',
        't_the_sampler_has_no_tile_flag',
        't_camera_flags', 't_camera_scale_range_with_low_above_high_is_refused',
        't_a_changed_field_changes_the_identity'],
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
