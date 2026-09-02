#!/usr/bin/env python3
"""Unit test for utilities/ConfigIdentity.

    python utilities/test_modules/test_config_identity.py

No GPU, no model download, about a second. Everything here is string and hash
bookkeeping, which is the point: this module decides whether two runs are
allowed to share a cached result, and it has to be checkable without paying for
the thing being cached.

What it is for
--------------
GigaPathFunc and HESTSegFunc grew the same machinery independently -- a frozen
baseline, `_enc`, `_parts_against`, a lazy sha256 of a state dict, an id and a
json. The two copies of `_enc` were byte-identical, which is the shape of a bug
waiting: fix float formatting in one and the encoder and the segmenter start
hashing by different rules, silently, in the one place this project has decided
correctness depends on.

So the scheme lives once. What each config MEANS stays with the thing it
configures.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('aiNNModel', 'utilities'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import torch                                                # noqa: E402

import ConfigIdentity as CI                                 # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}')
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n        {type(e).__name__}: {e}')


def rejects(fn, needle=''):
    try:
        fn()
    except Exception as e:                                   # noqa: BLE001
        if needle and needle not in str(e):
            raise AssertionError(
                f'raised, but the message never mentions {needle!r}: {e}') from None
        return
    raise AssertionError('should have raised, returned normally')


# ── fixtures ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Inner(CI.IdentifiedConfig):
    a: int = 1
    b: float = 0.5


@dataclass(frozen=True)
class Outer(CI.IdentifiedConfig):
    inner: Inner = dataclasses.field(default_factory=Inner)
    name: str = 'x'
    size: int = 8
    scratch: int = 0

    NOT_IDENTITY = ('scratch',)


#: The frozen zero point for Outer. Inner's zero point rides inside it -- a
#: nested config is compared against the value the OUTER baseline holds, so a
#: nested config never carries a baseline of its own.
BASE = {'inner': Inner(a=1, b=0.5), 'name': 'x', 'size': 8}


# ── enc ───────────────────────────────────────────────────────────────────────

def t_enc_is_stable():
    """Two spellings of one value must not become two hashes."""
    assert CI.enc(0.24255) == CI.enc(0.242550)
    assert CI.enc(4.00003) != CI.enc(4.0), \
        'the 0.04% between a requested ds and a level own must survive'
    assert CI.enc((1, 2, 3)) == '1,2,3'
    assert CI.enc((0.1, 0.2)) == CI.enc([0.1, 0.2])
    assert CI.enc(True) == CI.enc(True) and CI.enc(True) != CI.enc(False)
    # Fixed precision and not repr(): repr of a float is version- and
    # platform-dependent in principle, and this feeds a hash that has to come
    # out the same everywhere.
    assert '.12g' not in CI.enc(1 / 3) and len(CI.enc(1 / 3)) <= 14, CI.enc(1 / 3)


# ── parts_against ─────────────────────────────────────────────────────────────

def t_baseline_values_are_omitted():
    assert CI.parts_against(Outer(), BASE) == [], \
        'a config equal to its baseline should contribute nothing'


def t_differences_appear_sorted():
    parts = CI.parts_against(Outer(name='y', size=9), BASE)
    assert parts == ['name=y', 'size=9'], parts


def t_not_identity_is_skipped():
    # Read off the config by default, the way the nested branch always did.
    assert CI.parts_against(Outer(scratch=99), BASE) == []
    # An explicit skip still wins, or the default would be unoverridable.
    assert CI.parts_against(Outer(scratch=99), BASE, skip=()) == ['scratch=99']


def t_field_absent_from_baseline_always_counts():
    """A field added after the baseline was frozen has nothing to equal.

    Dropping it would let a new knob change the output without changing the
    name -- the exact failure the baseline scheme exists to prevent, arriving
    through the door marked 'backwards compatible'.
    """
    thin = {'name': 'x'}          # baseline written before size existed
    parts = CI.parts_against(Outer(), thin)
    assert 'size=8' in parts, parts
    assert 'inner.a=1' in parts, parts


# ── nested configs ────────────────────────────────────────────────────────────

def t_nested_config_is_prefixed():
    parts = CI.parts_against(Outer(inner=Inner(a=2)), BASE)
    assert parts == ['inner.a=2'], parts


def t_nested_baseline_comes_from_the_outer_one():
    """Inner carries no baseline; the outer one supplies its zero point.

    So the same Inner can be baseline under one owner and not under another,
    which is what lets TransformConfig be shared while its numbers stay with
    the model that validated them.
    """
    other = {'inner': Inner(a=2, b=0.5), 'name': 'x', 'size': 8}
    assert CI.parts_against(Outer(inner=Inner(a=2)), other) == []
    assert CI.parts_against(Outer(inner=Inner(a=1)), other) == ['inner.a=1']


def t_nested_depth_two():
    @dataclass(frozen=True)
    class Deep(CI.IdentifiedConfig):
        outer: Outer = dataclasses.field(default_factory=Outer)

    base = {'outer': Outer()}
    assert CI.parts_against(Deep(), base) == []
    parts = CI.parts_against(Deep(outer=Outer(name='z')), base)
    assert parts == ['outer.name=z'], parts


# ── ids ───────────────────────────────────────────────────────────────────────

def t_short_id_tracks_the_parts():
    a = CI.short_id(['x=1'])
    assert a == CI.short_id(['x=1'])
    assert a != CI.short_id(['x=2'])
    assert a != CI.short_id(['x=1', 'y=1'])
    assert len(a) == 8, a
    # Order is the caller's: parts_against sorts, and short_id must not
    # re-sort, or two genuinely different orderings would collide.
    assert CI.short_id(['a=1', 'b=2']) != CI.short_id(['b=2', 'a=1'])


def t_identified_config_mixin():
    cfg = Outer(name='y', scratch=5)
    assert cfg.identity_parts(BASE) == ['name=y']


# ── weights_id ────────────────────────────────────────────────────────────────

def t_weights_id_is_content():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    a = CI.weights_id(m)
    assert len(a) == 16, a
    assert a == CI.weights_id(m), 'not deterministic on the same module'

    same = torch.nn.Linear(4, 3)
    same.load_state_dict(m.state_dict())
    assert CI.weights_id(same) == a, 'same parameters gave a different id'

    with torch.no_grad():
        m.weight[0, 0] += 0.5
    assert CI.weights_id(m) != a, 'a changed parameter did not move the id'


def t_weights_id_ignores_device_and_layout():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    a = CI.weights_id(m)
    if torch.cuda.is_available():
        assert CI.weights_id(m.cuda()) == a, \
            'moving to the GPU changed the id -- .cpu() is missing'
    # dtype is genuinely different numbers and MUST move it
    assert CI.weights_id(m.half()) != a


def t_weights_id_of_nothing():
    assert CI.weights_id(None) == '', \
        "no model means no weights to record; '' is the honest answer"


# ── registry ──────────────────────────────────────────────────────────────────

def t_registry_round_trip():
    @CI.register('t_demo')
    @dataclass(frozen=True)
    class Demo(CI.IdentifiedConfig):
        k: int = 3

    assert CI.config_from('t_demo') == Demo()
    assert CI.config_from('t_demo', k=9) == Demo(k=9)
    assert CI.config_from_json(CI.config_json(Demo(k=9))) == Demo(k=9)


#: MODULE LEVEL ON PURPOSE. This file has `from __future__ import annotations`
#: at the top, exactly as every module that defines a real config does, so
#: `_Outer.inner` is annotated with the STRING '_Inner'. A class defined inside
#: a test function would not reproduce that: `typing.get_type_hints` cannot see
#: a function's locals, which sends the resolution down its fallback path and
#: tests the branch that was already working.
@CI.register('t_inner')
@dataclass(frozen=True)
class _Inner(CI.IdentifiedConfig):
    a: int = 1


@CI.register('t_outer')
@dataclass(frozen=True)
class _Outer(CI.IdentifiedConfig):
    inner: _Inner = dataclasses.field(default_factory=_Inner)
    maybe: Optional[_Inner] = None
    k: int = 3


def t_json_round_trip_rebuilds_a_nested_config():
    """The one that was missing, and it cost a job.

    `config_from_json` guarded its nested branch with `isinstance(f.type, type)`,
    which is False for every config in this repo because PEP 563 makes `f.type`
    a string. So a nested config came back as a plain DICT, the outer dataclass
    accepted it without complaint, and the failure surfaced later and elsewhere:
    `'dict' object has no attribute 'kwargs'`, inside a DataLoader worker, four
    stack frames from anything to do with configs.

    The old `t_registry_round_trip` passed throughout -- its Demo has one int
    field, so it never had a nested config to lose.
    """
    text = CI.config_json(_Outer(inner=_Inner(a=7), k=9))
    back = CI.config_from_json(text)

    if not isinstance(back.inner, _Inner):
        raise AssertionError(
            f'inner came back as {type(back.inner).__name__}, not _Inner. A '
            f'dict here is accepted by the constructor and fails somewhere '
            f'else entirely')
    if back != _Outer(inner=_Inner(a=7), k=9):
        raise AssertionError(f'round trip changed the value: {back}')


def t_an_optional_nested_config_is_rebuilt_too():
    """`Optional[X]` is not a class, so a bare `issubclass` check misses it.

    Not hypothetical: `KeypointNetConfig.descriptor` is
    `Optional[DescriptorHeadConfig]`, because a detector with no descriptor
    head is MagicPoint. Both states have to survive the trip -- None because it
    is a real configuration, and the present one because that is the branch a
    class-only check drops.
    """
    for value in (None, _Inner(a=5)):
        back = CI.config_from_json(CI.config_json(_Outer(maybe=value)))
        if back.maybe != value or (value is not None
                                   and not isinstance(back.maybe, _Inner)):
            raise AssertionError(
                f'Optional nested config came back as {back.maybe!r}, '
                f'expected {value!r}')


def t_a_dict_that_is_not_a_config_is_refused_loudly():
    """The decoy for the two above: the failure must be an ERROR, not a value.

    What made this expensive was not the wrong branch, it was that the wrong
    branch RETURNED something. A field annotated as a plain dict cannot be a
    serialised config -- `_as_plain` only ever writes one for a nested config --
    so the honest answer is to refuse rather than to hand back the dict and let
    it fail four frames away.
    """
    @CI.register('t_notaconfig')
    @dataclass(frozen=True)
    class _NotAConfig(CI.IdentifiedConfig):
        k: int = 3

    text = json.dumps({'name': 't_notaconfig', 'fields': {'k': {'a': 1}}})
    try:
        CI.config_from_json(text)
    except TypeError:
        return
    raise AssertionError(
        'a dict in a non-config field was accepted. That is the exact shape of '
        'the bug this section exists for')


def t_registry_names_what_it_has():
    """An unknown name must list the known ones.

    A registry fills by import side effect, so the usual failure is not a typo
    -- it is a module nobody imported, and 'unknown: uni' cannot tell those two
    apart while 'known: gigapath, hest' can.
    """
    rejects(lambda: CI.config_from('no_such_thing_here'), 'no_such_thing_here')
    try:
        CI.config_from('no_such_thing_here')
    except KeyError as e:
        assert 't_demo' in str(e), f'the error did not list what IS registered: {e}'


def t_registry_refuses_a_second_claim():
    @CI.register('t_dup')
    @dataclass(frozen=True)
    class A(CI.IdentifiedConfig):
        pass

    def again():
        @CI.register('t_dup')
        @dataclass(frozen=True)
        class B(CI.IdentifiedConfig):
            pass

    rejects(again, 't_dup')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    argparse.ArgumentParser().parse_args()

    print('enc')
    check('stable across spellings',          t_enc_is_stable)

    print('parts_against')
    check('baseline values are omitted',      t_baseline_values_are_omitted)
    check('differences appear sorted',        t_differences_appear_sorted)
    check('NOT_IDENTITY is skipped',          t_not_identity_is_skipped)
    check('a field the baseline lacks counts', t_field_absent_from_baseline_always_counts)

    print('nested configs')
    check('nested is prefixed',               t_nested_config_is_prefixed)
    check('nested zero point is the outer one', t_nested_baseline_comes_from_the_outer_one)
    check('nesting survives two levels',      t_nested_depth_two)

    print('ids')
    check('short_id tracks the parts',        t_short_id_tracks_the_parts)
    check('IdentifiedConfig mixin',           t_identified_config_mixin)

    print('weights_id')
    check('hashes content, not names',        t_weights_id_is_content)
    check('ignores device, not dtype',        t_weights_id_ignores_device_and_layout)
    check('empty when there is no model',     t_weights_id_of_nothing)

    print('registry')
    check('name -> config -> json -> config', t_registry_round_trip)
    check('json rebuilds a NESTED config',    t_json_round_trip_rebuilds_a_nested_config)
    check('Optional nested too',              t_an_optional_nested_config_is_rebuilt_too)
    check('a non-config dict is refused',     t_a_dict_that_is_not_a_config_is_refused_loudly)
    check('an unknown name lists the known',  t_registry_names_what_it_has)
    check('one name, one claimant',           t_registry_refuses_a_second_claim)

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
