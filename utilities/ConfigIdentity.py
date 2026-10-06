"""Which configuration produced an artifact, and the proof that a stored one is it.

An artifact -- a cache entry, a checkpoint, a row of results -- has two
descriptions, kept apart on purpose:

    ADDRESS  `identity_id()`: 16 hex of the config's identity parts. It decides
             where a cache entry lives, so only config values go into it.
    RECORD   `record(obj, **upstream)`: the parts, every VERSION on the way,
             the ids of the direct upstream artifacts and the environment.
             Written beside the artifact and compared field by field on load
             (`record_diff`). A difference means the stored thing is not the
             asked-for thing: a cache recomputes, a checkpoint or a results
             pairing refuses, and either way the message names the field.

What each config MEANS stays with the thing it configures; this module knows
nothing about any model.

The rules
---------
1. Every dataclass field either changes the output -- identity -- or is named
   in NOT_IDENTITY. The generic registry test changes each identity field of
   every registered config and requires the id to move.

2. Each config class owns its zero point: `BASELINE`, a dict of LITERALS equal
   to the class's own defaults when it was frozen. A field equal to its
   baseline is left out of the parts, so a field added with a baseline that
   reproduces the old behaviour keeps every id; a field absent from the
   baseline is always in. A nested config field's baseline entry is the NAME
   of the class it holds, and the nested config contributes its own parts
   against its own BASELINE. Never a config instance and never computed from
   the defaults: a baseline that moves with the defaults lets a changed
   default keep the id over different output. `parts_of` refuses one that is
   not literal. Editing a baseline renames every id of that class, which is
   what it is for; editing a default splits new from old.

3. Code that changes an output without changing a config value bumps the
   `VERSION` of the class that owns that code, and the reason goes in
   log/TODO.log. Versions are in the record, not the address: the stale entry
   is caught on load and recomputed in place. The fingerprint tests pin each
   producer's output on synthetic input to its VERSION, so a behaviour change
   that forgets the bump fails a test.

4. The environment -- the versions of the libraries that make pixels and
   vectors (`environment`) -- is in every record, so an upgraded conda env
   recomputes instead of reading what the old one wrote.

5. `enc` is injective (canonical JSON) and `short_id` hashes the JSON of the
   parts list. Nothing else in the repo hashes for identity.

Why not hash the source
-----------------------
A python function declares no dependency set: hashing its own file misses what
it calls, hashing the repo invalidates every cache on every edit, and neither
sees a library upgrade. Rule 3 measures the behaviour instead and rule 4 the
libraries.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import numbers
import sys
import typing
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

# torch is imported where a model is BUILT, not here. Every config in the repo
# goes through this module -- the mask recipe, the sampler's, query_sim's -- and
# a module that only names and hashes configs has no reason to make all of them
# unimportable without torch. weights_id needs no import: it calls methods on
# the module it is handed.
if TYPE_CHECKING:
    import torch


# ── encoding ──────────────────────────────────────────────────────────────────

#: Hex characters in every id: `short_id`, `weights_id`, `file_fingerprint`.
ID_HEX = 16


def _canon(value: Any) -> Any:
    """`value` as plain JSON data, two spellings of one value made one.

    Numbers compare by value: ds 4 and ds 4.0 read the same pixels, so they
    must be the same id. Floats are cut to 12 significant digits, which keeps
    the 0.04% between a requested ds and a level's own (4.00003 against 4.0)
    and drops the noise below it. Types stay apart where they are different
    values: None, '' and 0, True and 1, ('a,b',) and ('a', 'b') are each two."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        x = float(f'{float(value):.12g}')
        return int(x) if x.is_integer() and abs(x) < 2 ** 53 else x
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return [_canon(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in value.items()}
    return str(value)


def enc(value: Any) -> str:
    """A value as the string a hash will see: canonical JSON of `_canon`.
    Injective, so two different values can never write the same part."""
    return json.dumps(_canon(value), sort_keys=True, separators=(',', ':'))


def _is_config(value: Any) -> bool:
    return isinstance(value, IdentifiedConfig) and dataclasses.is_dataclass(value)


_LITERAL = (type(None), bool, int, float, str)
_CHECKED: set = set()


def _check_baseline(cls) -> Dict[str, Any]:
    """`cls.BASELINE`, refused unless every value is a literal (rule 2)."""
    baseline = getattr(cls, 'BASELINE', {})
    if cls in _CHECKED:
        return baseline

    def literal(v) -> bool:
        if isinstance(v, (tuple, list)):
            return all(literal(x) for x in v)
        if isinstance(v, dict):
            return all(isinstance(k, str) and literal(x) for k, x in v.items())
        return isinstance(v, _LITERAL)

    if not isinstance(baseline, dict):
        raise TypeError(f'{cls.__name__}.BASELINE must be a dict, got '
                        f'{type(baseline).__name__}')
    bad = sorted(k for k, v in baseline.items() if not literal(v))
    if bad:
        raise TypeError(
            f'{cls.__name__}.BASELINE holds non-literal values for {bad}. A '
            f'baseline is literals only; for a nested config write the NAME of '
            f'its class, and the nested config is compared against its own '
            f'BASELINE (ConfigIdentity rule 2)')
    _CHECKED.add(cls)
    return baseline


_MISSING = object()


def parts_of(cfg, exclude: tuple = ()) -> List[str]:
    """`name=enc(value)` for every identity field of `cfg` that differs from
    its class's BASELINE, sorted by name so declaration order cannot move an
    id. A field absent from the baseline is always in.

    A nested config writes `name=<class>` when its class is not the one the
    baseline names, then its own parts prefixed `name.`, each against its own
    class's BASELINE. `exclude` drops fields by name, for an id that covers
    part of a config (`TissueMaskConfig.region_id`)."""
    baseline = _check_baseline(type(cfg))
    skip = set(getattr(cfg, 'NOT_IDENTITY', ())) | set(exclude)
    out: List[str] = []
    for f in sorted(dataclasses.fields(cfg), key=lambda f: f.name):
        if f.name in skip:
            continue
        value = getattr(cfg, f.name)
        want = baseline.get(f.name, _MISSING)
        if _is_config(value):
            name = type(value).__name__
            if want != name:
                out.append(f'{f.name}={enc(name)}')
            out.extend(f'{f.name}.{p}' for p in value.identity_parts())
            continue
        if want is not _MISSING and _canon(value) == _canon(want):
            continue
        out.append(f'{f.name}={enc(value)}')
    return out


def short_id(parts: List[str]) -> str:
    """`ID_HEX` hex of the JSON of `parts`. The caller's order is kept:
    `parts_of` already sorts, and re-sorting here would merge two orderings a
    caller meant to be different."""
    text = json.dumps([str(p) for p in parts], separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()[:ID_HEX]


def versions_of(obj) -> Dict[str, int]:
    """Every nonzero VERSION `obj` depends on: each class in its MRO that sets
    one itself, its `cfg` when it is a build, and every nested config field --
    keyed by the class that owns the code (rule 3). An inherited VERSION is
    the parent's code and is recorded under the parent's name, so a bump in a
    base class reaches every subclass."""
    out: Dict[str, int] = {}
    for cls in type(obj).__mro__:
        v = cls.__dict__.get('VERSION', 0)
        if v:
            out[cls.__qualname__] = int(v)
    cfg = getattr(obj, 'cfg', None)
    if _is_config(cfg):
        out.update(versions_of(cfg))
    if _is_config(obj):
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if _is_config(value):
                out.update(versions_of(value))
    return dict(sorted(out.items()))


#: Distributions whose versions decide pixels or vectors (rule 4). Absent ones
#: are skipped; openslide's C library is read off the module.
_ENV_DISTS = ('numpy', 'scipy', 'scikit-learn', 'opencv-python',
              'opencv-python-headless', 'opencv-contrib-python', 'Pillow',
              'openslide-python', 'openslide-bin', 'torch', 'torchvision',
              'timm', 'safetensors')
_ENV: Optional[Dict[str, str]] = None


def environment() -> Dict[str, str]:
    """`{name: version}` of the python and the libraries in `_ENV_DISTS`, read
    from package metadata (nothing heavy is imported), once per process."""
    global _ENV
    if _ENV is None:
        from importlib import metadata                              # noqa: PLC0415
        env = {'python': '.'.join(str(v) for v in sys.version_info[:3])}
        for name in _ENV_DISTS:
            try:
                env[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        try:
            import openslide                                        # noqa: PLC0415
            env['libopenslide'] = str(openslide.__library_version__)
        except Exception:                                           # noqa: BLE001
            pass
        _ENV = dict(sorted(env.items()))
    return dict(_ENV)


def record(obj, **upstream: str) -> Dict[str, Any]:
    """What is written beside an artifact made by `obj` (a config or a build)
    from the upstream artifacts named in `upstream` (`seg_id=...`)."""
    return {'id': obj.identity_id(), 'parts': list(obj.identity_parts()),
            'versions': versions_of(obj),
            'upstream': {k: str(v) for k, v in sorted(upstream.items())},
            'env': environment()}


def record_diff(stored: Optional[Dict[str, Any]], want: Dict[str, Any]) -> List[str]:
    """Why `stored` is not `want`, one line per difference; empty when it is.
    A missing record is a difference: an artifact that cannot say what made
    it is not the one asked for."""
    if not stored:
        return ['no identity record']
    out = []
    for key in ('id', 'upstream', 'versions', 'env'):
        a, b = stored.get(key), want.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    out.append(f'{key}.{k}: stored {a.get(k)!r}, now {b.get(k)!r}')
        else:
            out.append(f'{key}: stored {a!r}, now {b!r}')
    if stored.get('parts') != want.get('parts'):
        sa, sb = set(stored.get('parts') or ()), set(want.get('parts') or ())
        out += [f'part only in stored: {p}' for p in sorted(sa - sb)]
        out += [f'part only now: {p}' for p in sorted(sb - sa)]
    return out


def unwrapped(module: torch.nn.Module) -> torch.nn.Module:
    """The network inside whatever runs it: DataParallel and
    DistributedDataParallel hold it as `.module`, torch.compile as `._orig_mod`,
    and they nest (compiled, then split across cards). How many cards run a
    model, or whether it is compiled, does not change one of its weights, so
    nothing that names the weights may see the wrapper -- its `module.` and
    `_orig_mod.` key prefixes would make one encoder two ids.
    `TileEncoder._set_model` keeps `self.model` bare; this is the same rule for
    any module handed in from outside (a trainer's DDP copy)."""
    import torch                                                  # noqa: PLC0415
    wrappers = (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    while True:
        if isinstance(module, wrappers):
            module = module.module
        elif hasattr(module, '_orig_mod') and isinstance(module._orig_mod,
                                                           torch.nn.Module):
            module = module._orig_mod
        else:
            return module


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
    parameters on the GPU are the same parameters. So is the wrapper
    (`unwrapped`): the same parameters on four cards are the same parameters.

    memoryview and not .tobytes(): the latter copies every tensor a second time,
    and the largest in a foundation model is hundreds of MB.
    """
    if module is None:
        return ''
    h = hashlib.sha256()
    state = unwrapped(module).state_dict()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(memoryview(t.numpy()).cast('B'))
    return h.hexdigest()[:ID_HEX]


_FINGERPRINTS: Dict[tuple, str] = {}


def file_fingerprint(path) -> str:
    """sha256 of a file's bytes, `ID_HEX` hex -- for a key that has to see a
    checkpoint's CONTENT without building the model that would hash its
    parameters (`weights_id`).

    Memoised on (path, size, mtime): a cache lookup asks once per slide, and a
    few hundred MB read a hundred times is minutes spent on the same answer.
    Replacing the file changes its mtime, which is what makes the memo safe.
    """
    import os                                                 # noqa: PLC0415
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _FINGERPRINTS:
        h = hashlib.sha256()
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b''):
                h.update(chunk)
        _FINGERPRINTS[key] = h.hexdigest()[:ID_HEX]
    return _FINGERPRINTS[key]


# ── mixins ────────────────────────────────────────────────────────────────────

class IdentifiedConfig:
    """A frozen dataclass whose fields, against its class's BASELINE, form an
    identity.

    A subclass sets, without annotations (an annotation would make them
    dataclass fields):

        BASELINE      literal dict, the class's defaults when frozen (rule 2)
        NOT_IDENTITY  fields that cannot change the output and so must not
                      split a cache -- batch_size: a ViT normalises per sample,
                      so batching cannot change a single vector
        VERSION       the behaviour of the code this config drives (rule 3)
    """

    BASELINE: Dict[str, Any] = {}
    NOT_IDENTITY: tuple = ()
    VERSION: int = 0

    def identity_parts(self) -> List[str]:
        return parts_of(self)

    def identity_id(self) -> str:
        return short_id(self.identity_parts())

    def provenance(self) -> Dict[str, Any]:
        """The NOT_IDENTITY values, nested configs' prefixed: what a run
        records and an id must not see."""
        out = {n: getattr(self, n) for n in self.NOT_IDENTITY}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if _is_config(value):
                out.update({f'{f.name}.{k}': v
                            for k, v in value.provenance().items()})
        return out


class IdentifiedBuild:
    """A built object: a config, a device, and possibly a loaded model.

    Subclasses set `cfg`, `device` and `model` (which may be None) and get
    identity for free: the config's parts plus the loaded weights. `model`
    being None is a first-class state -- a segmentation method with no
    network still has to be able to name itself. A VERSION on the build class
    covers the code that runs the model.
    """

    VERSION: int = 0

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
        wid = self.weights_id
        return self.cfg.identity_parts() + ([f'weights={enc(wid)}'] if wid else [])

    def identity_id(self) -> str:
        """Config plus loaded weights.

        Deliberately not derivable from the config alone: the config says what
        to build, the weights say what got built, and a finetune is where those
        two come apart.
        """
        return short_id(self.identity_parts())


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

    BASELINE = {'source': 'timm', 'arch': '', 'dtype': 'fp16'}
    NOT_IDENTITY = ('weights',)

    def torch_dtype(self) -> torch.dtype:
        import torch                                          # noqa: PLC0415
        try:
            return {'fp16': torch.float16, 'fp32': torch.float32}[self.dtype]
        except KeyError:
            raise ValueError(
                f"dtype must be 'fp16' or 'fp32', got {self.dtype!r}") from None

    def build(self, **factory_kwargs) -> torch.nn.Module:
        """Construct on the CPU. Moving to a device is the caller's, because
        only the caller knows whether a checkpoint has to be loaded first."""
        import torch                                          # noqa: PLC0415
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
    reports `f.type` as the STRING `'HomographyConfig'`, never the class, so an
    `isinstance(f.type, type)` test is False for every config in the repo and a
    nested config would come back as a plain dict -- silently:
    `PairDatasetConfig(**...)` accepts it, and the failure surfaces later as
    `'dict' object has no attribute 'kwargs'`, inside a DataLoader worker.

    `get_type_hints` re-evaluates the strings in the defining module's
    namespace. It can still fail for a class defined inside a function body, so
    the raw annotations are the fallback.
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
                # LOUD. No registered config has a plain dict field -- `_as_plain` only
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
