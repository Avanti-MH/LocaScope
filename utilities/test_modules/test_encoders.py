#!/usr/bin/env python3
"""The three real encoders, asked the same questions.

    python utilities/test_modules/test_encoders.py                  # all three
    python utilities/test_modules/test_encoders.py --encoder uni2
    python utilities/test_modules/test_encoders.py --no-dual-load    # skip the 4.5 GB

test_tile_encoder checks the TEMPLATE against fake models, which is what makes
it a second. This file checks the implementations against the weights they
actually load, which no test did: ConchVitFunc went in with 923e9e6 without one
forward pass through it, and Uni2Func had only ever been exercised by a bench.

Every check is either two sources that must agree or a deliberately wrong
alternative that must be refused. There is one exception, marked where it
appears: dim and feat_hw are PINNED per checkpoint, because "which model
loaded" has no second source -- a wrong checkpoint that loads cleanly is
exactly the failure this catches.

Run on the trunk. GigaPath and UNI2 are bare ViTs and CONCH has head='trunk'
for the same shape, so one body asks all three the same thing and the answers
are comparable. CONCH's attentional pooler is a different exit and gets its own
section rather than a special case inside the shared one.

Ordered by cost. The config checks need no weights, the build needs one copy,
and the hub-against-local comparison needs two -- so a wrong answer costs
seconds before it costs 4.5 GB twice.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: BEFORE `import torch`, and that placement is the whole of why it works.
#: huggingface_hub reads HF_HOME into module constants at its own import, and
#: something in this environment reaches it during `import torch` -- the
#: transformers deprecation warning prints before any line of this file runs.
#: So the setdefault each encoder module does above its own `import timm` is
#: already too late: by then the constant is frozen to ~/.cache/huggingface,
#: and prov-gigapath came down a second time on 2026-08-22 with a complete
#: copy sitting in /work since April.
#:
#: Here it is early enough. Nothing has been imported yet except argparse, os
#: and sys, none of which touch the hub.
#:
#: setdefault, so jobscripts/_env.sh and an exported HF_HOME both still win.
os.environ.setdefault(
    'HF_HOME', os.environ.get('LOCASCOPE_MODEL_WEIGHTS',
                              '/work/u26130998/model_weights'))

# _paths holds the one definition of OUTPUT_ROOT for every package, so it lives
# in utilities/ rather than beside this file. That directory goes on sys.path
# here, because setup_import_paths -- which puts the rest there -- is inside it.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402
import numpy as np                                               # noqa: E402

from _paths import setup_import_paths                            # noqa: E402
setup_import_paths()

from ConfigIdentity import weights_id                            # noqa: E402
from TileEncoderFunc import (admissible_poolings, encoder_config,  # noqa: E402
                             encoder_names, pool_slots, pooling_kinds)

#: The same value the block above put into HF_HOME, read back rather than
#: respelled: if an exported HF_HOME won there, this must follow it, or the
#: check below would compare the cache against a directory nobody chose.
_WEIGHTS = Path(os.environ['HF_HOME'])

#: What each encoder IS, and how else its weights can be reached.
#:
#: `dim` and `feat_hw` are pinned rather than read back from the model, and
#: they are the only pinned numbers here. Everything else in this file compares
#: two sources; "the intended checkpoint loaded" has no second source, and a
#: substitution that loads cleanly is silent -- retrieval would simply get
#: worse. GigaPath is 14x14 of 1536, UNI2 is 16x16 of 1536 with eight register
#: tokens beside the CLS, the CONCH trunk is 28x28 of 768.
#:
#: `local` is a checkpoint file for ModelConfig.weights, which builds the
#: architecture and loads that file instead of the hub's. None means this
#: encoder has no second route: CONCH already reads a local file by default and
#: its hub path is gated, so there is one way in and the dual-load check has
#: nothing to compare.
ENCODERS = {
    'gigapath': dict(
        over={},
        dim=1536, feat_hw=(14, 14), num_prefix=1,
        local=_WEIGHTS / 'prov-gigapath' / 'pytorch_model.bin'),
    'uni2': dict(
        over={},
        dim=1536, feat_hw=(16, 16), num_prefix=9,
        local=_WEIGHTS / 'UNI2-h' / 'pytorch_model.bin'),
    'conch_vit': dict(
        over={'head': 'trunk'},
        dim=768, feat_hw=(28, 28), num_prefix=1,
        local=None),
}

#: The arms bench_slidewin_pooling asks for. Named here so the admissibility
#: check is asked about the set the sweeps actually use, rather than about a
#: list invented for a test.
WANTED = ('cls', 'cls_avg', 'cls_std', 'rings3',
          'grid2x2', 'grid4x4', 'grid7x7', 'grid8x8', 'grid14x14', 'grid16x16')

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


def rejects(fn, why: str):
    """`fn` must raise. The decoy half of every pair below."""
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f'accepted what it must refuse: {why}')


# ── the tiles every check shares ─────────────────────────────────────────────
#
# Random uint8 rather than real tissue. Nothing here asserts a similarity or a
# ranking -- only shapes, agreements and refusals -- and a fixed seed makes the
# two sides of every comparison see identical input, which is the property that
# matters.

def _tiles(n: int = 2, size: int = 256):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            for _ in range(n)]


def _free(enc):
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── config, before any weights load ──────────────────────────────────────────

def t_hub_cache_is_the_shared_one():
    """The cache huggingface_hub will actually use, read from its constants.

    FIRST, because a wrong answer here costs 4.5 GB and a progress bar and
    nothing else says anything -- the run succeeds, the numbers are right, and
    a second copy of every model lands in $HOME. That happened while this file
    was being written: prov-gigapath came down again on 2026-08-22 with a
    complete copy sitting in /work since April.

    HF_HOME is read into module constants at huggingface_hub's own import, so
    the setdefault each encoder module does above its `import timm` only works
    when that module is the first thing in the process to reach the hub. It
    frequently is not. Exporting HF_HOME before python starts is what actually
    decides, which is what jobscripts/_env.sh is for -- and this file is run by
    hand, outside any jobscript.

    Reading the frozen constant rather than os.environ is the point: the
    environment is what was asked for, the constant is what will happen.
    """
    from huggingface_hub import constants
    got = Path(constants.HF_HUB_CACHE)
    want = _WEIGHTS / 'hub'
    assert got == want, (
        f'huggingface_hub will use {got}, not {want}. Every weight not already '
        f'there will be downloaded again. Fix it before the interpreter starts:'
        f'\n          export HF_HOME={_WEIGHTS}\n'
        f'        setting it inside python is too late -- the constant above '
        f'was frozen at import.')
    return str(got)


def t_config_resolves(name, spec, dev, dtype):
    """--encoder <name> reaches the module that registers it.

    encoder_config imports by the _IMPLEMENTATIONS table; config_from alone
    would say `not registered` for a module nobody had imported, which reads as
    a missing encoder rather than a missing import.
    """
    cfg = encoder_config(name, **spec['over'])
    assert cfg.POOLINGS, f'{type(cfg).__name__} has no POOLINGS table'
    return f'{type(cfg).__name__}  pooling={cfg.pooling!r}'


def t_admissible_matches_the_table(name, spec, dev, dtype):
    """What admissible_poolings keeps is exactly what POOLINGS holds.

    Both sides are cheap and neither needs the model, which is the point:
    admissible_poolings exists so a sweep drops an arm BEFORE loading weights,
    instead of dying inside pooling_kinds with a shape message afterwards. If
    the two disagree, a bench silently compares a different set of arms than
    the one it printed.
    """
    cfg = encoder_config(name, **spec['over'])
    keep, drop = admissible_poolings(cfg, WANTED)
    assert set(keep) | set(drop) == set(WANTED), 'an arm went missing'
    assert not (set(keep) & set(drop)), 'an arm is in both'
    for m in keep:
        assert m in cfg.POOLINGS, f'{m} kept but not in POOLINGS'
    for m in drop:
        assert m not in cfg.POOLINGS, f'{m} dropped but IS in POOLINGS'
    return f'keeps {len(keep)}: {" ".join(keep)}'


def t_grid_family_divides_the_token_grid(name, spec, dev, dtype):
    """A kept gridNxN must divide feat_hw; a dropped one must not.

    This is the arithmetic behind the previous check rather than a restatement
    of it: there the two sides were one table read twice, here the claim is
    about the numbers. GigaPath 14x14 admits grid2x2 and grid7x7 and refuses
    grid4x4; UNI2 16x16 is the other way round. A table edited without the grid
    in mind passes the first check and fails this one.
    """
    cfg = encoder_config(name, **spec['over'])
    h, w = spec['feat_hw']
    bad = []
    for mode in cfg.POOLINGS:
        if not mode.startswith('grid'):
            continue
        side = int(mode[4:].split('x')[0])
        if h % side or w % side:
            bad.append(f'{mode} does not divide {h}x{w}')
    for mode in WANTED:
        if not mode.startswith('grid') or mode in cfg.POOLINGS:
            continue
        side = int(mode[4:].split('x')[0])
        if h % side == 0 and w % side == 0 and side not in (h,):
            bad.append(f'{mode} divides {h}x{w} yet is not offered')
    assert not bad, '; '.join(bad)
    return f'{h}x{w}'


def t_pool_slots_counts_are_consistent(name, spec, dev, dtype):
    """pool_slots names n entries for a gridNxN and n is 1 + N*N.

    Names only, no tensor and no weights. pooling_kinds is checked against
    these names later, once there is a model; getting the count wrong here
    would make that comparison agree with itself.
    """
    fake = {'dim': spec['dim'], 'feat_hw': spec['feat_hw'],
            'num_prefix': spec['num_prefix']}
    seen = {}
    for mode in encoder_config(name, **spec['over']).POOLINGS:
        if mode in ('', 'tokens', 'identity'):
            continue
        slots, layout = pool_slots(mode, fake)
        seen[mode] = len(slots)
        if mode.startswith('grid'):
            side = int(mode[4:].split('x')[0])
            assert len(slots) == 1 + side * side, \
                f'{mode}: {len(slots)} slots, expected 1 + {side}*{side}'
            assert layout == f'grid:{side}x{side}', f'{mode}: layout {layout}'
    assert seen.get('cls') == 1, f"cls has {seen.get('cls')} slots"
    return '  '.join(f'{m}={n}' for m, n in sorted(seen.items()))


# ── the model ────────────────────────────────────────────────────────────────

def t_spec_is_this_checkpoint(enc, name, spec):
    """dim, feat_hw and num_prefix are the ones this checkpoint should have.

    THE PINNED CHECK, and the only one. Every other comparison here would pass
    just as happily against the wrong model: shapes would still agree with each
    other, poolings would still reduce. What a substituted checkpoint changes
    is retrieval quality, weeks later, with nothing pointing back here.
    """
    got = enc.model_spec
    assert got.dim == spec['dim'], f'dim {got.dim}, expected {spec["dim"]}'
    assert tuple(got.feat_hw) == spec['feat_hw'], \
        f'feat_hw {tuple(got.feat_hw)}, expected {spec["feat_hw"]}'
    assert got.num_prefix == spec['num_prefix'], \
        f'num_prefix {got.num_prefix}, expected {spec["num_prefix"]}'
    return f'dim={got.dim} feat_hw={tuple(got.feat_hw)} prefix={got.num_prefix}'


def t_token_count_agrees_with_spec(enc, name, spec):
    """The forward pass returns feat_hw[0]*feat_hw[1] + num_prefix tokens.

    Two sources: the length of the tensor the model produced, and what
    model_token_spec read off its config. Agreement is evidence -- either alone
    would be a claim. UNI2 is where this earns its cost: eight register tokens
    sit between the CLS and the patches, and getattr(m, num_prefix_tokens, 1)
    would have said 1.
    """
    toks = enc.tokens(_tiles())
    h, w = enc.model_spec.feat_hw
    want = (2, h * w + enc.model_spec.num_prefix, enc.model_spec.dim)
    assert tuple(toks.shape) == want, f'{tuple(toks.shape)}, expected {want}'
    return f'{tuple(toks.shape)} = {h}x{w} + {enc.model_spec.num_prefix}'


def t_features_are_the_cls_slot(enc, name, spec):
    """features() must BE pooling 'cls', not merely resemble it.

    Production ships features(); the pooling bench compares arms against
    pooling_kinds(..., 'cls'). If those two are not the same vector, every arm
    in that bench was scored against a baseline that is not what production
    uses, and the whole comparison means nothing while looking complete.

    Both sides are normalised because that is how the window score consumes
    them, and both run at the same dtype -- comparing an fp16 pooled feature
    against an fp32 shipped one measures precision and blames the pooling.
    """
    tiles = _tiles()
    shipped = F.normalize(enc(tiles).float(), dim=-1)
    slots = pooling_kinds(enc.tokens(tiles), 'cls', enc.model_spec)
    pooled = F.normalize(slots[:, 0].float(), dim=-1)
    cos = float((shipped * pooled).sum(-1).min())
    assert cos > 1 - 1e-5, f'cos={cos:.7f}: features() is not the cls slot'
    return f'cos={cos:.7f}'


def t_pooled_matches_pool_slots(enc, name, spec):
    """The tensor pooled() returns has as many slots as pool_slots names.

    Written by different code -- one reduces, the other names -- so their
    agreement is evidence. A pooling that quietly produced four slots where
    five were named would store vectors under the wrong labels, and a rotated
    query would then be matched slot-for-slot against the wrong cells.
    """
    tiles = _tiles()
    cfg = enc.cfg
    checked = []
    for mode in ('cls', 'rings3', 'grid2x2'):
        if mode not in cfg.POOLINGS:
            continue
        slots, layout = pool_slots(mode, enc.model_spec)
        out = enc.pooled(tiles, mode)
        n = 1 if out.ndim == 2 else out.shape[1]
        assert n == len(slots), \
            f'{mode}: tensor has {n} slots, pool_slots names {len(slots)}'
        assert out.shape[-1] == enc.model_spec.dim, \
            f'{mode}: width {out.shape[-1]}, dim {enc.model_spec.dim}'
        checked.append(f'{mode}={n}')
    assert checked, 'no pooling was checked'
    return '  '.join(checked)


def t_an_inadmissible_pooling_is_refused(enc, name, spec):
    """A dropped arm is dropped for one of two reasons, and they differ.

    admissible_poolings reads a table; this asks the arithmetic. But the table
    leaves out more than the arithmetic forbids, and conflating the two is a
    mistake this check made on its first run:

      the side does not divide feat_hw    pooling_kinds MUST raise. grid4x4 on
                                          14x14 has no blocks to form.
      the side IS feat_hw                 pooling_kinds must ACCEPT. grid14x14
                                          on 14x14 is one slot per cell, which
                                          is what 'tokens' already is, and the
                                          table omits it so that one reduction
                                          does not get two names and two ids
                                          over one set of vectors.

    Only GigaPath exercises the second case out of WANTED: 14 divides 14, while
    UNI2 and CONCH drop nothing whose side equals their grid. A test written
    against the other two alone would have passed and said nothing.
    """
    cfg = encoder_config(name, **spec['over'])
    _, drop = admissible_poolings(cfg, WANTED)
    h, w = spec['feat_hw']
    toks = enc.tokens(_tiles())

    refused, accepted = [], []
    for mode in (m for m in drop if m.startswith('grid')):
        side = int(mode[4:].split('x')[0])
        if h % side or w % side:
            rejects(lambda m=mode: pooling_kinds(toks, m, enc.model_spec),
                    f'{mode} on a {h}x{w} grid, which it does not divide')
            refused.append(mode)
        else:
            assert (side, side) == (h, w), (
                f'{mode} divides {h}x{w} without being one slot per cell, so '
                f'it is a real arm this encoder simply does not offer -- the '
                f'table and the grid family have drifted apart')
            out = pooling_kinds(toks, mode, enc.model_spec)
            assert out.shape[1] == 1 + h * w, (
                f'{mode} gave {out.shape[1]} slots, expected 1 + {h}*{w}')
            accepted.append(mode)

    parts = []
    if refused:
        parts.append(f'refused {" ".join(refused)}')
    if accepted:
        parts.append(f'{" ".join(accepted)} == tokens, offered under that name')
    return '; '.join(parts) or 'nothing dropped for this grid'


# ── hub against local ────────────────────────────────────────────────────────

def t_hub_and_local_are_the_same_weights(name, spec, dev, dtype):
    """Two routes to the weights must give the same parameters, exactly.

    weights_id is a sha256 of the state dict the module actually holds, so this
    compares what was loaded rather than where it came from. Cheaper and
    stricter than comparing features: no forward pass, and a half-loaded
    state_dict that still produces plausible vectors cannot hide in it.

    Two things nobody had checked are inside this one:

      ModelConfig.build passes pretrained=False when weights is set and then
      load_state_dict(...) with the default strict=True, against a model built
      with num_classes=0 and global_pool=''. A checkpoint carrying head keys
      raises here -- which is the right place for it.

      timm resolves an hf-hub: arch through the hub even with pretrained=False,
      because the architecture is named in the repo config. If that is so,
      `local` is not offline, it merely skips the gigabytes.

    Both sides run at the same dtype: the hash is over the tensors as held, so
    fp16 against fp32 would differ for a reason that is not a difference.
    """
    path = spec['local']
    assert path is not None, 'no second route'
    assert path.exists(), f'no such checkpoint: {path}'

    hub = encoder_config(name, **spec['over']).with_model(dtype=dtype).build(dev)
    hub_id = weights_id(hub.model)
    _free(hub)

    loc = encoder_config(name, **spec['over'])\
        .with_model(dtype=dtype, weights=str(path)).build(dev)
    loc_id = weights_id(loc.model)
    _free(loc)

    assert hub_id == loc_id, (
        f'hub gave {hub_id} and {path.name} gave {loc_id}. The two routes '
        f'load different parameters, so whichever one a run took is part of '
        f'its result and is not recorded anywhere')
    return f'both {hub_id}'


# ── CONCH's other exit ───────────────────────────────────────────────────────

def t_conch_attn_pool_is_a_different_vector(dev, dtype):
    """head='attn_pool' is 512-d out of the pooler, not 768-d out of the trunk.

    The two heads are vectors of different widths in different spaces, which is
    why head is a config field rather than a call-site argument. 768 here would
    mean the pooler was skipped and the id would still look right.
    """
    enc = encoder_config('conch_vit', head='attn_pool')\
        .with_model(dtype=dtype).build(dev)
    feats = enc(_tiles())
    toks = enc.tokens(_tiles())
    assert tuple(feats.shape) == (2, 512), \
        f'{tuple(feats.shape)}, expected (2, 512)'
    assert tuple(toks.shape) == (2, 785, 768), \
        f'trunk {tuple(toks.shape)}, expected (2, 785, 768)'
    _free(enc)
    return 'feats (2, 512), trunk (2, 785, 768)'


def t_conch_attn_pool_refuses_a_token_pooling(dev, dtype):
    """head='attn_pool' with grid2x2 dies at CONFIG, not at the forward.

    One 512-d vector has no token axis. Raising in __post_init__ is what makes
    the failure cost nothing; the same mistake caught at the first forward
    costs a gigabyte of weights first.
    """
    rejects(lambda: encoder_config('conch_vit', head='attn_pool',
                                   pooling='grid2x2'),
            'attn_pool + grid2x2')
    return 'refused at config time'


def t_conch_prefixes_and_decoys():
    """The three strict=True prefixes match, and wrong ones raise.

    Upstream's factory calls load_state_dict(strict=False) and DISCARDS what
    was missing, so a prefix matching nothing gives a randomly initialised
    module and a number that looks like a number. _split raises on empty
    instead, and this is the decoy that proves it does.

    'visual.trun' is deliberately NOT among the decoys: it is a proper prefix
    of the real one, so every trunk key starts with it and _split is right to
    return them. A decoy the code correctly accepts is a broken decoy.
    """
    import ConchVitFunc as C
    state = torch.load(C._download(), map_location='cpu')
    state = state.get('state_dict', state)
    good = ('visual.trunk.', 'visual.attn_pool_contrast.', 'visual.ln_contrast.')
    bad = ('visual.trunk_typo.', 'vision.trunk.', 'trunk.',
           'visual.trunk.blocks.999.')
    for prefix in good:
        assert C._split(state, prefix), f'{prefix} matched nothing'
    for prefix in bad:
        rejects(lambda p=prefix: C._split(state, p), prefix)
    return f'{len(state)} keys, {len(good)} prefixes, {len(bad)} decoys refused'


def t_conch_weights_stay_local():
    """_download returns the file on disk and never reaches the hub.

    First thing to fail if the weights move again: the hub path needs a token
    scope this account does not have, so falling through to it is not a slower
    success, it is a GatedRepoError with nothing to say about paths.
    """
    import ConchVitFunc as C
    path = C._download()
    assert path.exists(), f'{path} does not exist'
    assert str(path).startswith(str(C.CONCH_LOCAL)), \
        f'{path} is not under CONCH_LOCAL ({C.CONCH_LOCAL})'
    return f'{path.name}  {path.stat().st_size:,} B'


# ── driver ───────────────────────────────────────────────────────────────────

_CONFIG_CHECKS = (
    ('config resolves',                 t_config_resolves),
    ('admissible matches POOLINGS',     t_admissible_matches_the_table),
    ('grid family divides feat_hw',     t_grid_family_divides_the_token_grid),
    ('pool_slots counts are consistent', t_pool_slots_counts_are_consistent),
)

_MODEL_CHECKS = (
    ('spec is this checkpoint',         t_spec_is_this_checkpoint),
    ('token count agrees with spec',    t_token_count_agrees_with_spec),
    ('features() is the cls slot',      t_features_are_the_cls_slot),
    ('pooled matches pool_slots',       t_pooled_matches_pool_slots),
    ('an inadmissible pooling raises',  t_an_inadmissible_pooling_is_refused),
)


def run_encoder(name, dev, dtype, dual_load: bool):
    spec = ENCODERS[name]
    print(f'\n{"=" * 78}\n{name}\n{"=" * 78}')

    print('config  (no weights)')
    for label, fn in _CONFIG_CHECKS:
        check(f'{name}: {label}', lambda f=fn: f(name, spec, dev, dtype))

    if name == 'conch_vit':
        print('weights  (local only -- the hub repo is gated)')
        check(f'{name}: weights stay local', t_conch_weights_stay_local)
        check(f'{name}: prefixes and decoys', t_conch_prefixes_and_decoys)

    print('model')
    enc = encoder_config(name, **spec['over']).with_model(dtype=dtype).build(dev)
    for label, fn in _MODEL_CHECKS:
        check(f'{name}: {label}', lambda f=fn: f(enc, name, spec))
    _free(enc)

    if name == 'conch_vit':
        print('head=attn_pool')
        check(f'{name}: attn_pool is 512-d',
              lambda: t_conch_attn_pool_is_a_different_vector(dev, dtype))
        check(f'{name}: attn_pool refuses grid2x2',
              lambda: t_conch_attn_pool_refuses_a_token_pooling(dev, dtype))

    if spec['local'] is not None:
        if dual_load:
            print('hub against local  (builds the model twice)')
            check(f'{name}: same weights either way',
                  lambda: t_hub_and_local_are_the_same_weights(
                      name, spec, dev, dtype))
        else:
            print('hub against local: skipped, drop --no-dual-load to run it')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--encoder', nargs='+', default=list(ENCODERS),
                    choices=list(ENCODERS),
                    help='which to check (default: all three)')
    ap.add_argument('--no-dual-load', action='store_true',
                    help='skip building each model twice to compare weights_id')
    ap.add_argument('--dtype', choices=['fp16', 'fp32'], default=None,
                    help='default: fp16 on a GPU, fp32 on the CPU')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # fp16 is unsupported for several of these ops on the CPU. Nothing here
    # asserts a NUMBER -- shapes, agreements and refusals only -- so the
    # precision can follow the device without changing what is being tested.
    dtype = args.dtype or ('fp16' if dev.type == 'cuda' else 'fp32')
    print(f'device={dev}  dtype={dtype}  weights={_WEIGHTS}')
    assert set(args.encoder) <= set(encoder_names()), \
        f'unknown encoder; registered: {", ".join(encoder_names())}'

    print('\nhub cache')
    check('points at the shared directory', t_hub_cache_is_the_shared_one)

    for name in args.encoder:
        run_encoder(name, dev, dtype, not args.no_dual_load)

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{"=" * 78}\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed:\n  ' + '\n  '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
