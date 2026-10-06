#!/usr/bin/env python3
"""Unit test for utilities/ConfigIdentity, and for every config that uses it.

    python utilities/test_modules/test_config_identity.py

No GPU, no model download. The first sections are the machinery on fixtures.
The `every config` section imports every module that defines an
IdentifiedConfig and holds each class to ConfigIdentity's rules 1 and 2 -- a
class added later is covered without anyone writing a test for it. `lint`
holds the repo to rule 5.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / 'utilities'))
import _paths                                               # noqa: E402

_paths.setup_import_paths()

import torch                                                # noqa: E402

import ConfigIdentity as CI                                 # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
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

    BASELINE = {'a': 1, 'b': 0.5}


@dataclass(frozen=True)
class OtherInner(CI.IdentifiedConfig):
    a: int = 1
    b: float = 0.5

    BASELINE = {'a': 1, 'b': 0.5}


@dataclass(frozen=True)
class Outer(CI.IdentifiedConfig):
    inner: Inner = dataclasses.field(default_factory=Inner)
    name: str = 'x'
    size: int = 8
    scratch: int = 0

    BASELINE = {'inner': 'Inner', 'name': 'x', 'size': 8}
    NOT_IDENTITY = ('scratch',)


# ── enc ───────────────────────────────────────────────────────────────────────

def t_enc_keeps_one_value_one_string():
    """Two spellings of one value are one part."""
    assert CI.enc(0.24255) == CI.enc(0.242550)
    assert CI.enc(4) == CI.enc(4.0), 'ds 4 and ds 4.0 read the same pixels'
    assert CI.enc((0.1, 0.2)) == CI.enc([0.1, 0.2])
    assert CI.enc(4.00003) != CI.enc(4.0), \
        'the 0.04% between a requested ds and a level own must survive'
    assert CI.enc(1 / 3) == CI.enc(float(f'{1 / 3:.12g}'))


def t_enc_is_injective():
    """The decoys: values that differ must never write the same part, or a
    record comparison passes over different output."""
    pairs = [(None, ''), (None, 0), ('', 0), (True, 1), (False, 0),
             (('a,b',), ('a', 'b')), (((1, 2), (3,)), ((1,), (2, 3))),
             ('1', 1), ('a|b', 'a'), ('x=1', 'x')]
    same = [(a, b) for a, b in pairs if CI.enc(a) == CI.enc(b)]
    assert not same, f'collide: {same}'


# ── parts_of ──────────────────────────────────────────────────────────────────

def t_baseline_values_are_omitted():
    assert Outer().identity_parts() == [], Outer().identity_parts()


def t_differences_appear_sorted():
    parts = Outer(name='y', size=9).identity_parts()
    assert parts == ['name="y"', 'size=9'], parts


def t_not_identity_is_skipped():
    assert Outer(scratch=99).identity_parts() == []
    assert Outer(scratch=99).provenance() == {'scratch': 99}


def t_field_absent_from_baseline_always_counts():
    """A field added without a baseline entry has nothing to equal, so it is in:
    a renamed id, never a kept id over changed output."""
    @dataclass(frozen=True)
    class Thin(CI.IdentifiedConfig):
        name: str = 'x'
        size: int = 8

        BASELINE = {'name': 'x'}

    assert Thin().identity_parts() == ['size=8'], Thin().identity_parts()


def t_nested_is_against_its_own_baseline():
    assert Outer(inner=Inner(a=2)).identity_parts() == ['inner.a=2']


def t_nested_class_is_part_of_the_identity():
    """Two nested configs at their own baselines have no parts each, so the
    class name is what keeps them apart."""
    a, b = Outer(), Outer(inner=OtherInner())
    assert a.identity_id() != b.identity_id(), (a.identity_parts(), b.identity_parts())
    assert b.identity_parts() == ['inner="OtherInner"'], b.identity_parts()


def t_a_baseline_holding_a_config_is_refused():
    """Rule 2: a config instance in a baseline moves with the defaults."""
    @dataclass(frozen=True)
    class Bad(CI.IdentifiedConfig):
        inner: Inner = dataclasses.field(default_factory=Inner)

        BASELINE = {'inner': Inner()}

    rejects(lambda: Bad().identity_parts(), 'literal')


def t_exclude_drops_by_name():
    assert CI.parts_of(Outer(inner=Inner(a=2), name='y'),
                       exclude=('inner',)) == ['name="y"']


# ── ids, versions, records ────────────────────────────────────────────────────

def t_short_id_tracks_the_parts():
    a = CI.short_id(['x=1'])
    assert a == CI.short_id(['x=1'])
    assert a != CI.short_id(['x=2'])
    assert a != CI.short_id(['x=1', 'y=1'])
    assert len(a) == CI.ID_HEX == 16, a
    assert CI.short_id(['a=1', 'b=2']) != CI.short_id(['b=2', 'a=1'])
    assert CI.short_id(['a=1|b=2']) != CI.short_id(['a=1', 'b=2']), \
        'the separator must not be forgeable'


def t_versions_follow_the_owner():
    """A bump in a base class reaches every subclass, under the base's name;
    a nested config's version is collected too; zero is not recorded."""
    @dataclass(frozen=True)
    class Base(CI.IdentifiedConfig):
        k: int = 0
        BASELINE = {'k': 0}
        VERSION = 2

    @dataclass(frozen=True)
    class Child(Base):
        BASELINE = {'k': 0}

    @dataclass(frozen=True)
    class Holder(CI.IdentifiedConfig):
        child: Child = dataclasses.field(default_factory=Child)
        BASELINE = {'child': 'Child'}

    assert CI.versions_of(Child()) == {Base.__qualname__: 2}
    assert CI.versions_of(Holder()) == {Base.__qualname__: 2}
    assert CI.versions_of(Outer()) == {}
    assert Child().identity_id() == Child().identity_id()


def t_record_names_what_differs():
    want = CI.record(Outer(), seg_id='hest-1')
    assert CI.record_diff(dict(want), want) == []
    assert CI.record_diff(None, want) == ['no identity record']
    other = CI.record(Outer(name='y'), seg_id='hest-1')
    diff = CI.record_diff(other, want)
    assert any(d.startswith('id:') for d in diff), diff
    assert 'part only in stored: name="y"' in diff, diff
    moved = dict(want, env=dict(want['env'], numpy='0.0.1'))
    assert any(d.startswith('env.numpy') for d in CI.record_diff(moved, want))
    up = CI.record(Outer(), seg_id='hest-2')
    assert CI.record_diff(up, want) == \
        ["upstream.seg_id: stored 'hest-2', now 'hest-1'"], CI.record_diff(up, want)


def t_environment_names_the_libraries():
    env = CI.environment()
    assert 'python' in env and 'numpy' in env and 'torch' in env, env
    return ', '.join(f'{k} {v}' for k, v in env.items())


# ── every config in the repo ──────────────────────────────────────────────────

#: Configs with a field that has no default, and the values to build one with.
def _factories():
    from GigaPathFunc import GigaPathEncoderConfig
    from HestSegFunc import HestSegConfig
    return {
        'TissueMaskConfig': lambda c: c(seg=HestSegConfig()),
        'KnnEstMppConfig': lambda c: c(encoder='gigapath'),
        'ClassifierEstMppConfig': lambda c: c(
            encoder='gigapath', classifier='linear', reduction='cls',
            tile_size=256, weights='/nonexistent.pt'),
        'PrototypeEstMppConfig': lambda c: c(
            encoder='gigapath', pooling='cls', support_context='none',
            query_context='none', collapse='mean', routing_head='none',
            tile_size=256, weights='/nonexistent.pt'),
        'SlidingWinSimRotConfig': lambda c: c(encoder=GigaPathEncoderConfig()),
    }


_MODULES = ('TileEncoderFunc', 'GigaPathFunc', 'Uni2Func', 'ConchVitFunc',
            'ConvNeXtV2Func', 'TissueSegFunc', 'HestSegFunc', 'Uni2PcaSegFunc',
            'TissueMaskConfig', 'TileSampler', 'config',
            'stage1_estimation.KnnEstMpp', 'stage1_estimation.ClassifierEstMpp',
            'stage1_estimation.PrototypeEstMpp',
            'stage2_retrieval.SlidingWinSimRot')
_SUPERPATHPOINT = ('SuperPoint.Backbones', 'SuperPoint.Datasets',
                   'SuperPoint.Decoders', 'SuperPoint.EncoderBackbone',
                   'SuperPoint.Heads', 'SuperPoint.HomographicAdaptation',
                   'SuperPoint.KeypointNet', 'SuperPoint.Losses',
                   'SuperPoint.Teacher', 'SuperPoint.Trainer',
                   'common.HomographyConfig')


def _project_configs():
    import importlib
    for name in _MODULES:
        importlib.import_module(name)
    _paths.add_training_package('SuperPathPoint')
    for name in _SUPERPATHPOINT:
        importlib.import_module(name)
    seen, stack = [], list(CI.IdentifiedConfig.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if (cls not in seen and dataclasses.is_dataclass(cls)
                and cls.__module__ != __name__
                and not cls.__module__.startswith('test_')):
            seen.append(cls)
    return sorted(seen, key=lambda c: (c.__module__, c.__name__))


def _default(cls):
    try:
        return cls()
    except TypeError:
        make = _factories().get(cls.__name__)
        if make is None:
            raise AssertionError(
                f'{cls.__name__} has a field with no default; add it to '
                f'_factories() so this test can build one') from None
        return make(cls)


def _identity_fields(cls):
    skip = set(getattr(cls, 'NOT_IDENTITY', ()))
    return [f for f in dataclasses.fields(cls) if f.name not in skip]


def _has_default(f) -> bool:
    return (f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING)


def t_every_config_owns_a_complete_literal_baseline():
    """Rule 2 per class: its OWN BASELINE, literals only, naming every identity
    field that has a default and nothing else."""
    bad = []
    classes = _project_configs()
    for cls in classes:
        if 'BASELINE' not in vars(cls):
            bad.append(f'{cls.__name__}: no BASELINE of its own')
            continue
        try:
            CI._check_baseline(cls)
        except TypeError as e:
            bad.append(str(e))
            continue
        names = {f.name for f in _identity_fields(cls)}
        keys = set(cls.BASELINE)
        stray = keys - names
        missing = {f.name for f in _identity_fields(cls) if _has_default(f)} - keys
        if stray:
            bad.append(f'{cls.__name__}: BASELINE names no identity field {sorted(stray)}')
        if missing:
            bad.append(f'{cls.__name__}: no BASELINE entry for {sorted(missing)}')
    assert not bad, '\n        '.join([''] + bad)
    return f'{len(classes)} classes'


#: Legal values for fields whose vocabulary `__post_init__` checks, so the
#: generic `_variants` (which can only guess) is not the only candidate.
_ALTERNATIVES = {
    'head': ('trunk', 'attn_pool'), 'pooling': ('grid2x2', 'tokens', 'cls_avg'),
    'scorer': ('entropy',), 'bucket_frame': ('at_inherit',),
    'floor_frame': ('taken',), 'stack_kind': ('R',), 'on_incomplete': ('keep',),
    'candidates': ('random',),
}


def _variants(value, name=''):
    """Values of the same kind as `value`, in order of preference."""
    yield from (a for a in _ALTERNATIVES.get(name, ()) if a != value)
    if isinstance(value, bool):
        yield not value
    elif isinstance(value, int):
        yield value + 1
        if value > 0:
            yield value - 1
    elif isinstance(value, float):
        for c in (value + 0.01, value * 0.9 if value else 0.5, value - 0.01, 0.5):
            if c != value:
                yield c
    elif isinstance(value, str):
        yield value + '_x'
        if value:
            yield ''
    elif value is None:
        yield from (1.0, 1, 'x')
    elif isinstance(value, tuple):
        if value:
            for c in _variants(value[-1]):
                yield value[:-1] + (c,)
            for c in _variants(value[0]):
                yield (c,) + value[1:]
            if len(value) > 1:
                yield value[:-1]
        else:
            yield (0,)


def _perturbations(cfg, prefix=''):
    """(field path, changed config, whether the id must move): one per field
    for which some variant constructs and actually changes the stored value.
    A field none of `_variants` can change is reported as untested."""
    skip = set(getattr(cfg, 'NOT_IDENTITY', ()))
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        path = f'{prefix}{f.name}'
        if CI._is_config(value):
            for sub, inner, moves in _perturbations(value, path + '.'):
                if inner is None:
                    yield sub, None, None
                    continue
                try:
                    yield sub, dataclasses.replace(cfg, **{f.name: inner}), moves
                except (TypeError, ValueError, KeyError):
                    yield sub, None, None
            continue
        for candidate in _variants(value, f.name):
            try:
                changed = dataclasses.replace(cfg, **{f.name: candidate})
            except (TypeError, ValueError, KeyError):
                continue
            if CI._canon(getattr(changed, f.name)) != CI._canon(value):
                yield path, changed, f.name not in skip
                break
        else:
            yield path, None, None


def t_every_identity_field_moves_the_id():
    """Rule 1 per class: change any identity field and the id moves; change a
    NOT_IDENTITY field and it does not."""
    bad, untested, n = [], [], 0
    for cls in _project_configs():
        cfg = _default(cls)
        base = cfg.identity_id()
        for path, changed, moves in _perturbations(cfg):
            name = f'{cls.__name__}.{path}'
            if changed is None:
                untested.append(name)
                continue
            n += 1
            if (changed.identity_id() != base) != moves:
                bad.append(f'{name}: id {"did not move" if moves else "moved"}')
    assert not bad, '\n        '.join([''] + bad)
    note = f'{n} fields'
    if untested:
        note += f'; no variant constructs for {", ".join(untested)}'
    return note


# ── lint ──────────────────────────────────────────────────────────────────────

#: Rule 5's exceptions: identity hashing itself, the photo rng's seed (a seed,
#: not a name), and the frozen copy of that seed in its own test.
_HASHLIB_ALLOWED = {'utilities/ConfigIdentity.py', 'query_sim/camera.py',
                    'utilities/test_modules/test_fov_supply.py'}
_LINT_ROOTS = ('aiNNModel', 'utilities', 'query_sim', 'stage1_estimation',
               'stage2_retrieval', 'stage3_localization', 'training')


def t_nothing_else_hashes_for_identity():
    import re
    imports = re.compile(r'^\s*(import hashlib|from hashlib )', re.M)
    found = []
    for top in _LINT_ROOTS:
        for dirpath, _, files in os.walk(_ROOT / top):
            for name in files:
                if not name.endswith('.py'):
                    continue
                path = Path(dirpath) / name
                rel = path.relative_to(_ROOT).as_posix()
                if rel in _HASHLIB_ALLOWED:
                    continue
                if imports.search(path.read_text(errors='replace')):
                    found.append(rel)
    assert not found, (f'hashlib outside ConfigIdentity: {found}. An id is '
                       f'ConfigIdentity.short_id of parts (rule 5)')


# ── weights_id ────────────────────────────────────────────────────────────────

def t_weights_id_is_content():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    a = CI.weights_id(m)
    assert len(a) == CI.ID_HEX, a
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


def t_weights_id_ignores_the_wrapper():
    """DataParallel and torch.compile prefix every key (`module.`, `_orig_mod.`);
    the weights are the same, so the id must be. The decoy is a changed weight
    under the same wrappers: the unwrapping must not unwrap the difference."""
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    a = CI.weights_id(m)
    dp = torch.nn.DataParallel(m)
    assert list(dp.state_dict())[0].startswith('module.'), 'the pit is not there'
    assert CI.weights_id(dp) == a, 'DataParallel moved the id'
    compiled = torch.compile(m)
    assert CI.weights_id(compiled) == a, 'torch.compile moved the id'
    assert CI.weights_id(torch.nn.DataParallel(compiled)) == a, \
        'compiled, then split across cards, moved the id'
    assert CI.unwrapped(torch.nn.DataParallel(compiled)) is m
    other = torch.nn.Linear(4, 3)
    other.load_state_dict(m.state_dict())
    with torch.no_grad():
        other.weight[0, 0] += 0.5
    assert CI.weights_id(torch.nn.DataParallel(other)) != a, \
        'a changed weight under the same wrapper kept the id'


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
    """Under PEP 563 `f.type` is a string, so a nested config is only rebuilt
    if its annotation is resolved; otherwise it comes back as a plain DICT,
    the outer dataclass accepts it, and the failure surfaces far away."""
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
    `KeypointNetConfig.descriptor` is `Optional[DescriptorHeadConfig]`; both
    states have to survive the trip."""
    for value in (None, _Inner(a=5)):
        back = CI.config_from_json(CI.config_json(_Outer(maybe=value)))
        if back.maybe != value or (value is not None
                                   and not isinstance(back.maybe, _Inner)):
            raise AssertionError(
                f'Optional nested config came back as {back.maybe!r}, '
                f'expected {value!r}')


def t_a_dict_that_is_not_a_config_is_refused_loudly():
    """The decoy for the two above: the failure must be an ERROR, not a value.
    A field annotated as a plain dict cannot be a serialised config --
    `_as_plain` only ever writes one for a nested config -- so the honest
    answer is to refuse."""
    @CI.register('t_notaconfig')
    @dataclass(frozen=True)
    class _NotAConfig(CI.IdentifiedConfig):
        k: int = 3

    text = json.dumps({'name': 't_notaconfig', 'fields': {'k': {'a': 1}}})
    try:
        CI.config_from_json(text)
    except TypeError:
        return
    raise AssertionError('a dict in a non-config field was accepted')


def t_registry_names_what_it_has():
    """An unknown name must list the known ones: the usual failure is a module
    nobody imported, not a typo."""
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
    check('one value, one string',            t_enc_keeps_one_value_one_string)
    check('injective',                        t_enc_is_injective)

    print('parts_of')
    check('baseline values are omitted',      t_baseline_values_are_omitted)
    check('differences appear sorted',        t_differences_appear_sorted)
    check('NOT_IDENTITY is skipped',          t_not_identity_is_skipped)
    check('a field the baseline lacks counts', t_field_absent_from_baseline_always_counts)
    check('nested against its own baseline',  t_nested_is_against_its_own_baseline)
    check('nested class is identity',         t_nested_class_is_part_of_the_identity)
    check('a config in a baseline is refused', t_a_baseline_holding_a_config_is_refused)
    check('exclude drops by name',            t_exclude_drops_by_name)

    print('ids, versions, records')
    check('short_id tracks the parts',        t_short_id_tracks_the_parts)
    check('versions follow the owner',        t_versions_follow_the_owner)
    check('record_diff names the field',      t_record_names_what_differs)
    check('environment',                      t_environment_names_the_libraries)

    print('every config')
    check('own, complete, literal baseline',  t_every_config_owns_a_complete_literal_baseline)
    check('every identity field moves the id', t_every_identity_field_moves_the_id)

    print('lint')
    check('hashlib only in ConfigIdentity',   t_nothing_else_hashes_for_identity)

    print('weights_id')
    check('hashes content, not names',        t_weights_id_is_content)
    check('ignores device, not dtype',        t_weights_id_ignores_device_and_layout)
    check('ignores the wrapper, not a weight', t_weights_id_ignores_the_wrapper)
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
