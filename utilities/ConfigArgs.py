"""Command-line flags for a frozen dataclass config, made from its own fields.

    add_config_args(ap, base, 'richness')          # --richness-caps, --richness-floors, ...
    cfg = config_from_args(args, base, 'richness')  # base, with whatever was given

A config used to have a flag written by hand for each of the few fields somebody
had wanted to change, which left the rest fixed in a default nobody could reach
without editing the module. Here EVERY field is a flag, named after the field
(`floor_frame` -> `--richness-floor-frame`), and none of them has a default of its
own: a flag that is not given leaves the field at whatever `base` holds. So the
value a run uses is the one written where `base` is built, unless the command line
says otherwise -- which is what lets a script keep its configs in one visible
place and still take an override from sbatch.

WHAT A FLAG TAKES
    int, float, str      one value
    bool                 `--x` and `--no-x`
    Optional[X]          an X, or the word `none`
    Tuple[X, ...]        one or more X
    Tuple[X, X]          exactly two (any fixed length)
    Tuple[Tuple[A, B], ...]   pairs written `0.25,1.0 1.0,0.25`
    a dataclass          its fields, one level down: `--sampler-richness-caps`
                         (unless `skip` names it: a config with its own flags)

`config_from_args` goes through `dataclasses.replace`, so `__post_init__` runs on
the result: a caps tuple of the wrong length, a floor above its cap, a name that is
not a known scorer, is refused here, at parse time, with the config's own message.

A type this module does not know is an error when the flags are ADDED, not a field
that silently cannot be set.
"""

from __future__ import annotations

import argparse
import dataclasses
import typing
from typing import Any, Dict, Optional, Sequence, Tuple

#: What an unset flag holds. Not None, because `none` is a value an Optional field
#: can be given.
_UNSET = object()


def _flag(prefix: str, path: Sequence[str]) -> str:
    return '--' + '-'.join([prefix, *path]).replace('_', '-')


def _dest(prefix: str, path: Sequence[str]) -> str:
    return '_'.join([prefix, *path]).replace('-', '_')


def _fields(obj) -> list:
    return [f for f in dataclasses.fields(obj) if f.init]


def _hints(cls) -> Dict[str, Any]:
    return typing.get_type_hints(cls)


def _unwrap_optional(tp):
    """`(X, True)` for Optional[X], `(tp, False)` otherwise."""
    if typing.get_origin(tp) is typing.Union:
        rest = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(rest) == 1 and len(rest) != len(typing.get_args(tp)):
            return rest[0], True
    return tp, False


def _scalar(tp, optional: bool):
    """A parser for one token of type `tp`."""
    def parse(token: str):
        if optional and token.strip().lower() == 'none':
            return None
        if tp is bool:
            raise ValueError('bool has no value token')
        return tp(token)
    parse.__name__ = getattr(tp, '__name__', 'value')
    return parse


def _pair(inner: Tuple, optional: bool):
    """A parser for a token like `0.25,1.0`."""
    parsers = [_scalar(t, False) for t in inner]

    def parse(token: str):
        parts = token.split(',')
        if len(parts) != len(parsers):
            raise argparse.ArgumentTypeError(
                f'{token!r}: want {len(parsers)} values separated by commas')
        return tuple(p(v) for p, v in zip(parsers, parts))
    parse.__name__ = 'pair'
    return parse


def _spec(name: str, tp):
    """`(kind, parser, nargs, optional)` for one field, or a TypeError."""
    tp, optional = _unwrap_optional(tp)
    origin = typing.get_origin(tp)
    if tp in (int, float, str):
        return 'scalar', _scalar(tp, optional), None, optional
    if tp is bool:
        return 'bool', None, None, optional
    if origin is tuple:
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            item = args[0]
            if typing.get_origin(item) is tuple:
                return 'tuple', _pair(typing.get_args(item), False), '+', optional
            if item in (int, float, str):
                return 'tuple', _scalar(item, optional), '+', optional
        elif args and all(a in (int, float, str) for a in args) and len(set(args)) == 1:
            return 'tuple', _scalar(args[0], optional), len(args), optional
    raise TypeError(f'field {name!r}: no command-line form for type {tp!r}')


def _walk(base, prefix: str, skip: Sequence[str], path: Tuple[str, ...] = ()):
    """Yield `(path, field, hint, value)` for every settable field of `base`,
    descending into a field whose VALUE is a dataclass -- by the value's own
    class, so a config held under its base type still shows its own fields."""
    hints = _hints(type(base))
    for f in _fields(base):
        here = (*path, f.name)
        value = getattr(base, f.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            if here[0] in skip and len(here) == 1:
                continue
            yield from _walk(value, prefix, skip, here)
        else:
            if len(here) == 1 and here[0] in skip:
                continue
            yield here, f, hints.get(f.name), value


def add_config_args(ap: argparse.ArgumentParser, base, prefix: str, *,
                    skip: Sequence[str] = (), group: Optional[str] = None) -> None:
    """One flag per field of `base` (see the module docstring). `skip` names
    top-level fields left out, because another config owns them.

    The parser must be made with `allow_abbrev=False`. argparse otherwise takes any
    unique PREFIX of a flag for the flag, and these flags are prefixes of each
    other by construction: with `query_mpp` skipped, `--camera-query-mpp 0.5` would
    quietly set `--camera-query-mpp-jitter`. A typo that changes a different field
    and prints nothing is the failure this whole module exists to avoid."""
    if ap.allow_abbrev:
        raise ValueError('add_config_args needs a parser made with '
                         'allow_abbrev=False: a flag that is a prefix of another '
                         'would silently set the other')
    target = ap.add_argument_group(group or f'{prefix} config') if group != '' else ap
    for path, f, hint, value in _walk(base, prefix, skip):
        kind, parser, nargs, optional = _spec('.'.join(path), hint)
        flag, dest = _flag(prefix, path), _dest(prefix, path)
        help_text = f'{".".join(path)} (base: {value!r})'.replace('%', '%%')
        if kind == 'bool':
            target.add_argument(flag, dest=dest, default=_UNSET, help=help_text,
                                action=argparse.BooleanOptionalAction)
        else:
            target.add_argument(flag, dest=dest, default=_UNSET, type=parser,
                                nargs=nargs, help=help_text,
                                metavar=f.name.upper())


def _given(args, prefix: str, path: Sequence[str]):
    return getattr(args, _dest(prefix, path), _UNSET)


def config_from_args(args, base, prefix: str, *, skip: Sequence[str] = (),
                     _path: Tuple[str, ...] = ()):
    """`base` with every flag of this prefix that was given applied. `base` itself
    when none was, so a caller can tell nothing changed with `is`."""
    hints = _hints(type(base))
    changes: Dict[str, Any] = {}
    for f in _fields(base):
        here = (*_path, f.name)
        value = getattr(base, f.name)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            if len(here) == 1 and f.name in skip:
                continue
            sub = config_from_args(args, value, prefix, _path=here)
            if sub is not value:
                changes[f.name] = sub
            continue
        if len(here) == 1 and f.name in skip:
            continue
        given = _given(args, prefix, here)
        if given is _UNSET:
            continue
        kind = _spec('.'.join(here), hints.get(f.name))[0]
        if kind == 'tuple':
            # `none` on an Optional tuple parses to a one-element [None]
            given = None if list(given)[:1] == [None] else tuple(given)
        changes[f.name] = given
    return dataclasses.replace(base, **changes) if changes else base


def describe(cfg, name: str = '') -> list:
    """`name.field = value` lines for every field of `cfg`, nested ones dotted:
    what a run prints so its log says which values it really used."""
    lines = []

    def walk(obj, path):
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                walk(value, (*path, f.name))
            else:
                lines.append(f'{".".join((*path, f.name))} = {value!r}')
    walk(cfg, (name,) if name else ())
    return lines
