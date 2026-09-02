"""One implementation of "which configuration produced this".

GigaPathFunc and HESTSegFunc grew the same machinery independently: a frozen
baseline, an encoder for hashable strings, a walk over the non-baseline fields,
a lazy sha256 of a state dict, a short id and a json of the fields behind it.
The two copies of the string encoder were byte-identical, which is a bug with a
delay on it -- fix float formatting in one and the encoder and the segmenter
begin hashing by different rules, silently, in the one place this project has
decided correctness depends on.

So the scheme lives here once. What each config MEANS stays with the thing it
configures: this module holds no baseline of its own and knows nothing about
GigaPath, HEST or any other model.

The three rules
---------------
1. A BASELINE IS APPEND-ONLY AND A NAME IS NEVER REPURPOSED.
   Adding a field re-hashes nothing if the baseline gains a value that
   reproduces the previous behaviour; old stores stay valid and are RIGHT to,
   because nothing about their vectors changed. Adding one WITHOUT a baseline
   entry re-hashes everything, which is a recompute -- not a wrong answer.
   Giving an existing name a new meaning is the only change that fails
   silently: old and new hash the same and hold different things.

2. ONLY QUANTITIES WHOSE DEFINITION IS STABLE.
   The values vary freely; that is the point. What must not vary is what the
   NAME means. SafeSlide.base_mpp failed this on 2026-08-13 -- its definition
   changed from mpp-x to the mean of mpp-x and mpp-y, and every store took a
   new hash while the tiles and the vectors were unchanged.

3. THE HASH SEPARATES FILES; IT DOES NOT PROVE CORRECTNESS.
   Every id here is built from strings a caller composed. Getting the hash
   wrong costs a recompute, or at worst two configurations overwriting each
   other's file in turn -- a cache that never hits, which is visible. Getting a
   CHECK wrong costs a wrong answer. So the checks live with the domain and are
   made of the thing itself: recomputed grid geometry against stored
   coordinates, re-encoded tiles against stored vectors.

Why the baseline is not the field defaults
------------------------------------------
Hashing "fields that differ from their default" sounds equivalent and is not.
A default is a moving reference: change one and every hash taken before it
silently re-points. A store written when scale_size defaulted to 256 would
collide with a config built after the default became 224 -- same hash,
different vectors, no error. A frozen literal cannot move, so a changed default
splits new from old instead of merging them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from typing import Any, Dict, List, Optional, Type

import torch


# ── encoding ──────────────────────────────────────────────────────────────────

def enc(value: Any) -> str:
    """A value as the string a hash will see.

    Fixed precision for floats rather than repr(): repr is free to differ
    between Python builds, and this feeds a digest that has to come out the
    same on every machine. %.12g keeps the 0.04% between a requested ds and a
    level's own (4.00003 against 4.0) while dropping the noise below it.
    """
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, float):
        return f'{value:.12g}'
    if isinstance(value, (tuple, list)):
        return ','.join(enc(v) for v in value)
    if value is None:
        return ''
    return str(value)


def _is_config(value: Any) -> bool:
    return isinstance(value, IdentifiedConfig) and dataclasses.is_dataclass(value)


def parts_against(cfg, baseline: Dict[str, Any],
                  skip: Optional[tuple] = None) -> List[str]:
    """`name=value` for every field that differs from the frozen baseline.

    Sorted, so where a field is declared cannot move a hash.

    A field ABSENT from the baseline is always included. It was added after the
    baseline was frozen, so there is nothing for it to equal, and dropping it
    would let a new knob change the output without changing the name -- the
    failure the whole scheme exists to prevent, arriving through the door
    marked "backwards compatible".

    A field that is itself a config recurses and its parts are prefixed. Its
    zero point comes from the OUTER baseline, so a nested config never carries
    one of its own: the same TransformConfig can be baseline under one model and
    not under another, which is what lets the shape be shared while the numbers
    stay with whoever validated them.
    """
    # Read off the config when not told, because the nested branch below
    # already does and two rules for the same thing is one rule too many. The
    # test that found this called the free function directly and got a
    # NOT_IDENTITY field back in the parts.
    if skip is None:
        skip = getattr(cfg, 'NOT_IDENTITY', ())

    out: List[str] = []
    for f in sorted(dataclasses.fields(cfg), key=lambda f: f.name):
        if f.name in skip:
            continue
        value = getattr(cfg, f.name)
        want = baseline.get(f.name, _MISSING)

        if _is_config(value):
            nested = want if _is_config(want) else None
            inner = parts_against(
                value,
                {g.name: getattr(nested, g.name) for g in dataclasses.fields(nested)}
                if nested is not None else {})
            out.extend(f'{f.name}.{p}' for p in inner)
            continue

        if want is not _MISSING and value == want:
            continue
        out.append(f'{f.name}={enc(value)}')
    return out


class _Missing:
    def __repr__(self) -> str:
        return '<missing>'


_MISSING = _Missing()


def short_id(parts: List[str]) -> str:
    """The eight hex characters a store records.

    The caller's order is kept. parts_against already sorts, and re-sorting here
    would merge two genuinely different orderings.
    """
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:8]


def weights_id(module: Optional[torch.nn.Module]) -> str:
    """sha256 of the parameters a module actually holds. '' for no module.

    Hashing the STATE DICT and not a checkpoint file is what survives the cases
    a name cannot:

        a finetune saved beside the original    a path or a revision still
                                                describes the original
        best.pth overwritten in place           the path did not change and the
                                                weights did
        an adapter merged at load time          the file on disk is the base
                                                model; the weights are not

    Keys are sorted so dict order cannot leak in. Name, shape and dtype go into
    the digest with the bytes: two tensors can hold identical bytes under
    different shapes, and a .half() model is genuinely different numbers, so
    both have to move it. Device and stride are normalised away -- the same
    parameters on the GPU are the same parameters.

    memoryview and not .tobytes(): the latter copies every tensor a second time,
    and the largest in a foundation model is hundreds of MB.
    """
    if module is None:
        return ''
    h = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(memoryview(t.numpy()).cast('B'))
    return h.hexdigest()[:16]


def cfg_json(parts: List[str], provenance: Dict[str, Any]) -> str:
    """The fields behind an id, so a mismatch can name one.

    An id is eight hex characters and can only say that two configurations
    differ. Anything that falls back to recomputing has to be able to say WHICH
    field moved, or a permanently cold cache reads exactly like a correctly
    invalidated one.

    `provenance` carries the non-identity fields. They cannot move a hash by
    definition, so they cost nothing and answer what the hash cannot: a weights
    hash says the parameters changed, a path says which file they came from.
    """
    return json.dumps({'parts': parts,
                       'provenance': {k: enc(v) for k, v in provenance.items()}},
                      sort_keys=True)


# ── mixins ────────────────────────────────────────────────────────────────────

class IdentifiedConfig:
    """A frozen dataclass whose non-baseline fields form an identity.

    Subclasses declare NOT_IDENTITY -- fields that cannot change the output and
    so must not split a cache. batch_size is the standing example: a ViT
    normalises per sample, so batching cannot change a single vector, and it is
    also the most-tuned knob in the repo.

    No BASELINE attribute here on purpose. The zero point belongs to whoever
    owns the top-level config, and a nested one takes its zero point from the
    value the outer baseline holds.
    """

    NOT_IDENTITY: tuple = ()

    def identity_parts(self, baseline: Dict[str, Any]) -> List[str]:
        return parts_against(self, baseline, skip=self.NOT_IDENTITY)

    def provenance(self) -> Dict[str, Any]:
        return {n: getattr(self, n) for n in self.NOT_IDENTITY}


class IdentifiedBuild:
    """A built object: a config, a device, and possibly a loaded model.

    Subclasses set `cfg`, `device`, `model` (which may be None) and BASELINE,
    then get identity for free. `model` being None is a first-class state -- a
    segmentation method with no network still has to be able to name itself.
    """

    BASELINE: Dict[str, Any] = {}

    cfg: Any
    device: Any
    model: Optional[torch.nn.Module]

    @property
    def weights_id(self) -> str:
        """Computed on first use and cached. A caller that never asks never pays
        the few seconds a foundation model's state dict costs."""
        cached = getattr(self, '_weights_id', None)
        if cached is None:
            cached = weights_id(getattr(self, 'model', None))
            self._weights_id = cached
        return cached

    def identity_parts(self) -> List[str]:
        parts = self.cfg.identity_parts(self.BASELINE)
        wid = self.weights_id
        return parts + ([f'weights={wid}'] if wid else [])

    def identity_id(self) -> str:
        """Config plus loaded weights.

        Deliberately not derivable from the config alone: the config says what
        to build, the weights say what got built, and a finetune is where those
        two come apart.
        """
        return short_id(self.identity_parts())

    def identity_json(self) -> str:
        return cfg_json(self.identity_parts(), self.cfg.provenance())


# ── the model half ────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class ModelConfig(IdentifiedConfig):
    """Which network to construct, and with which parameters.

    Shared because the model-loading half genuinely is: GigaPath comes from
    timm and HEST's DeepLabV3 comes from torchvision, so two factories were
    already in use before anyone asked how to support a third.

    source names the factory, arch names the thing within it:

        'timm'         arch is a timm name, 'hf_hub:owner/repo@rev' included
        'torchvision'  arch is an attribute path under torchvision.models,
                       e.g. 'segmentation.deeplabv3_resnet50'
        'local'        arch is 'package.module:Class'

    weights is a local checkpoint to load over whatever the factory built, or
    None for the factory's own. It is in NOT_IDENTITY because weights_id already
    hashes what the file CONTAINS: putting the path in as well would invalidate
    a cache when a checkpoint moved between mounts without a number changing.
    Paths are provenance; content is identity.

    Construction kwargs are NOT here. num_classes=0 and global_pool='' are what
    make a model an encoder, and num_classes=2 is what makes one a tissue
    segmenter -- they are constants of the domain, not settings a caller varies,
    so they are passed by build() at each call site rather than hashed.
    """
    source:  str = 'timm'
    arch:    str = ''
    dtype:   str = 'fp16'
    weights: Optional[str] = None

    NOT_IDENTITY = ('weights',)

    def torch_dtype(self) -> torch.dtype:
        try:
            return {'fp16': torch.float16, 'fp32': torch.float32}[self.dtype]
        except KeyError:
            raise ValueError(
                f"dtype must be 'fp16' or 'fp32', got {self.dtype!r}") from None

    def build(self, **factory_kwargs) -> torch.nn.Module:
        """Construct on the CPU. Moving to a device is the caller's, because
        only the caller knows whether a checkpoint has to be loaded first."""
        if self.source == 'timm':
            import timm
            model = timm.create_model(self.arch, pretrained=self.weights is None,
                                      **factory_kwargs)
        elif self.source == 'torchvision':
            import torchvision
            target = torchvision.models
            for part in self.arch.split('.'):
                target = getattr(target, part)
            model = target(**factory_kwargs)
        elif self.source == 'local':
            import importlib
            module_name, _, class_name = self.arch.partition(':')
            if not class_name:
                raise ValueError(
                    f"source='local' needs arch='package.module:Class', "
                    f'got {self.arch!r}')
            model = getattr(importlib.import_module(module_name), class_name)(
                **factory_kwargs)
        else:
            raise ValueError(
                f"source must be 'timm', 'torchvision' or 'local', "
                f'got {self.source!r}')

        if self.weights:
            from pathlib import Path
            path = Path(self.weights)
            if not path.exists():
                raise FileNotFoundError(f'no such checkpoint: {path}')
            state = torch.load(path, map_location='cpu')
            if isinstance(state, dict):
                state = state.get('state_dict', state)
            model.load_state_dict(state)
        return model


# ── registry ──────────────────────────────────────────────────────────────────
#
# The forward direction -- config to object -- needs no registry: cfg.build()
# already dispatches, because which class the config is IS the choice. What
# needs one is the reverse, name to config, for two things a hash cannot do:
# a CLI flag or a jobscript naming an implementation, and reconstructing the
# configuration a store recorded so the store becomes reproducible rather than
# merely identifiable.

_REGISTRY: Dict[str, Type] = {}


def register(name: str):
    """Class decorator. The name is part of identity wherever it is stored --
    changing implementation always changes the output."""
    def deco(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(
                f'{name!r} is already registered to '
                f'{_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}; '
                f'one name, one claimant')
        _REGISTRY[name] = cls
        cls.REGISTERED_AS = name
        return cls
    return deco


def registered() -> List[str]:
    return sorted(_REGISTRY)


def config_from(name: str, **over):
    """The config class registered under `name`, constructed with `over`.

    A registry fills by import side effect, so the usual failure is not a typo
    but a module nobody imported. 'unknown: uni' cannot tell those two apart;
    listing what IS registered can.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        known = ', '.join(registered()) or \
            '(nothing -- no implementation module has been imported)'
        raise KeyError(
            f'no config registered as {name!r}. Registered: {known}') from None
    return cls(**over)


def config_json(cfg) -> str:
    """A config as json, keyed by its registered name so it can come back."""
    name = getattr(type(cfg), 'REGISTERED_AS', None)
    if name is None:
        raise ValueError(
            f'{type(cfg).__name__} is not registered, so it cannot be named in '
            f'json. Decorate it with @register("...")')
    return json.dumps({'name': name, 'fields': _as_plain(cfg)}, sort_keys=True)


def config_from_json(text: str):
    payload = json.loads(text)
    cls = _REGISTRY[payload['name']]
    return _from_plain(cls, payload['fields'])


def _as_plain(cfg) -> dict:
    out = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        out[f.name] = _as_plain(v) if _is_config(v) else v
    return out


def _field_types(cls) -> dict:
    """`{field: resolved type}`, working under `from __future__ import annotations`.

    THIS IS NOT A REFINEMENT, IT IS THE WHOLE NESTED PATH. Under PEP 563 --
    which every module in this project turns on -- `dataclasses.fields(cls)`
    reports `f.type` as the STRING `'HomographyConfig'`, never the class. The
    `isinstance(f.type, type)` test that used to guard the nested branch is
    therefore False for every config in the repo, and a nested config came back
    as a plain dict. Silently: `PairDatasetConfig(**...)` accepts it, and the
    failure surfaces later and elsewhere as `'dict' object has no attribute
    'kwargs'`, inside a DataLoader worker.

    `get_type_hints` re-evaluates the strings in the defining module's
    namespace. It can still fail for a class defined inside a function body, so
    the raw annotations are the fallback -- that path was already correct
    before PEP 563 and stays correct for anything it can resolve.
    """
    try:
        return typing.get_type_hints(cls)
    except Exception:                                             # noqa: BLE001
        return {f.name: f.type for f in dataclasses.fields(cls)}


def _config_type(hint):
    """The `IdentifiedConfig` a field holds, seen through `Optional[...]`.

    `KeypointNetConfig.descriptor` is `Optional[DescriptorHeadConfig]` -- None
    is MagicPoint, a detector with no descriptor head -- so a check for a bare
    class would miss it and hand back the dict again.
    """
    if isinstance(hint, type) and issubclass(hint, IdentifiedConfig):
        return hint
    for arg in typing.get_args(hint):
        if isinstance(arg, type) and issubclass(arg, IdentifiedConfig):
            return arg
    return None


def _from_plain(cls, fields: dict):
    kwargs = {}
    by_name = {f.name: f for f in dataclasses.fields(cls)}
    hints = _field_types(cls)
    for name, value in fields.items():
        if name not in by_name:
            continue          # a field this build no longer has; see rule 1
        if isinstance(value, dict):
            nested = _config_type(hints.get(name, by_name[name].type))
            if nested is None:
                # LOUD, because the silent version is what this cost. No
                # registered config has a plain dict field -- `_as_plain` only
                # ever writes one for a nested config -- so a dict whose type
                # cannot be resolved means the annotation did not come back,
                # and returning it unconverted builds a config that is wrong in
                # a way nothing downstream checks.
                raise TypeError(
                    f'{cls.__name__}.{name} was serialised as a nested config '
                    f'but its annotation {hints.get(name, by_name[name].type)!r} '
                    f'does not resolve to an IdentifiedConfig, so it cannot be '
                    f'rebuilt. Leaving it as a dict is how this failed before')
            kwargs[name] = _from_plain(nested, value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
