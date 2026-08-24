#!/usr/bin/env python3
"""GigaPath's current path against the free functions it replaced.

    python utilities/test_modules/test_gigapath_equivalence.py

Loads GigaPath. There is no cheap half any more and no --with-model flag:
every check here needs the real weights, because what is under test is that
two ways of reaching the same number reach the same number.

The reference is GigaPathFunc_old.py, kept for exactly this. 923e9e6 turned a
set of free functions into a TileEncoder with three implementations behind it,
and a refactor of that size is believed rather than known until the old code
is still there to disagree.

Why this matters more than a refactor usually does. Every pooling in the
experiment is scored against `cls` as the baseline, so if `cls` is not exactly
what gigapath_encode produced, every conclusion is measured against the wrong
reference and nothing raises -- the tables still print, the arms still rank.
It is an equality test rather than a spot check because the thing that would
break it is switching global_pool to 'avg', one of the pooling variants under
consideration, which swaps which of timm's `norm` / `fc_norm` is the real
LayerNorm.

The pooling arithmetic that used to share this file went to test_tile_encoder
with its subject: 923e9e6 took model_token_spec, _ring_bins and pool_tokens out
of GigaPathFunc, which now holds none of them, and a test named after one
encoder was the wrong place to keep the only coverage of a shared module. The
three real encoders are in test_encoders.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: BEFORE `import torch`. huggingface_hub freezes HF_HOME into module constants
#: at its own import, and something here reaches it during `import torch`, so
#: the setdefault inside GigaPathFunc runs too late to decide anything. Without
#: this line the 4.5 GB checkpoint is fetched again into ~/.cache/huggingface.
#: setdefault, so jobscripts/_env.sh and an exported HF_HOME both still win.
os.environ.setdefault(
    'HF_HOME', os.environ.get('LOCASCOPE_MODEL_WEIGHTS',
                              '/work/u26130998/model_weights'))

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('aiNNModel', 'utilities'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402
import torch.nn.functional as F                             # noqa: E402
from PIL import Image                                       # noqa: E402

# Through GigaPathFunc and not TileEncoderFunc, on purpose. These three are
# re-exported there, and the re-export is the claim under test as much as the
# numbers are: every caller spells `from GigaPathFunc import pooling_kinds`,
# and 923e9e6 was meant to change where the definition lives, not what anyone
# imports. Reaching past it here would stop checking that.
from GigaPathFunc import pool_slots, pooling_kinds             # noqa: E402

_RESULTS = []


def check(name, fn):
    try:
        fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}')
    except Exception as e:                                   # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n        {type(e).__name__}: {e}')


# ── the checks ────────────────────────────────────────────────────────────────
#
# No fake spec and no `rejects` helper any more: both belonged to the synthetic
# half, and nothing here refuses anything. Every check below builds the model
# and compares two computations of one number.

def t_global_pool_empty_gives_the_same_tokens(n_tiles=4):
    """`global_pool=''` must hand back exactly what we hand-assemble today.

    gigapath_encode_tokens reaches past the public API:

        m = getattr(model, 'module', model)
        toks = m.fc_norm(m.forward_features(batch))

    which is timm's forward_head rebuilt by hand, minus the pooling. timm has a
    supported way to say the same thing -- global_pool='' -- and if the two are
    the same tensor then the hand assembly can go, and with it the private
    attribute access, the DataParallel unwrap, and the comment explaining which
    of norm/fc_norm is the real LayerNorm.

    Why they SHOULD be identical, from timm 1.0.28:

        forward(x)      = forward_head(forward_features(x))
        forward_head(x) = head(head_drop(fc_norm(pool(x))))
        pool(x)         = x when pool_type is ''   (pool1d.py:21-22, returns
                          before the prefix-dropping slice that avg/max take)
        head_drop       = Dropout, identity under .eval()
        head            = Identity at num_classes=0

    so forward(x) collapses to fc_norm(forward_features(x)). This asserts each
    of those preconditions rather than trusting the chain.

    Equality is checked EXACTLY, not against a decoy, and that is deliberate:
    this is not two ways of computing a thing, it is the same ops in the same
    order reached by two spellings. A difference of 1e-7 would mean run-to-run
    nondeterminism between the two forwards, not a design difference -- so the
    message reports the magnitude, which is what tells the two apart.

    reset_classifier is used rather than a second 4.5 GB load. Safe HERE because
    use_fc_norm is decided in __init__ (vision_transformer.py:800) from
    global_pool in ('avg','avgmax','max') -- neither 'token' nor '' is in that
    set, so flipping between those two cannot change which norm is real. It
    would NOT be safe for 'avg', and the assertions below pin that.
    """
    import torch as _t
    from GigaPathFunc_old import gigapath_model, gigapath_encode

    dev = _t.device('cuda' if _t.cuda.is_available() else 'cpu')
    model = gigapath_model(dev)
    m = getattr(model, 'module', model)

    assert not m.training, 'model must be in eval() or head_drop is not identity'
    assert m.global_pool == 'token', f'expected the production setting, got {m.global_pool!r}'
    assert isinstance(m.fc_norm, torch.nn.Identity), (
        f'fc_norm is {type(m.fc_norm).__name__}, not Identity -- use_fc_norm is '
        f'true, so this model was built for avg/max pooling and the equality '
        f'below is not the one described')
    assert isinstance(m.head, torch.nn.Identity), (
        f'head is {type(m.head).__name__}, not Identity -- num_classes != 0')

    rng = np.random.default_rng(0)
    tiles = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
             for _ in range(n_tiles)]
    from TileEncoderFunc import TransformConfig, _to_pil
    tf = TransformConfig().build()
    batch = torch.stack([tf(_to_pil(t)) for t in tiles]).to(dev)

    with torch.no_grad():
        hand = m.fc_norm(m.forward_features(batch))       # what we do today
        cls_hand = F.normalize(hand[:, 0].float(), dim=-1)

        m.reset_classifier(0, '')
        try:
            assert m.global_pool == '', 'reset_classifier did not take'
            assert isinstance(m.fc_norm, torch.nn.Identity), (
                'reset_classifier rebuilt fc_norm; the flip is not free after all')
            public = model(batch)                          # the supported spelling
            pooled = m.pool(public, pool_type='token')     # per-call, no state change
        finally:
            m.reset_classifier(0, 'token')

    assert tuple(public.shape) == tuple(hand.shape), (
        f"global_pool='' gave {tuple(public.shape)}, hand assembly gave "
        f"{tuple(hand.shape)} -- '' is supposed to drop nothing")
    assert public.shape[1] == 1 + 14 * 14, (
        f"expected 197 tokens, got {public.shape[1]} -- '' dropped something")

    gap = float((public - hand).abs().max())
    print(f'        tokens {tuple(public.shape)}   max|Δ| {gap:.3e}')
    assert gap == 0.0, (
        f"global_pool='' and fc_norm(forward_features(...)) differ by {gap:.3e}. "
        f'At ~1e-7 this is nondeterminism between two forwards, not a design '
        f'difference; anything larger means the op sequences are not the same.')

    # And the production vector survives the move: pool(...,'token') off the ''
    # output is the CLS, which is what gigapath_encode returns today.
    # .cpu() on both sides: gigapath_encode ends each batch with feats.cpu()
    # (GigaPathFunc.py:193), so `shipped` is on the host while everything above
    # is still on the model's device.
    shipped = F.normalize(gigapath_encode(tiles, model, dev).float(), dim=-1).cpu()
    cos_pub = float(F.cosine_similarity(
        F.normalize(pooled.float(), dim=-1).cpu(), shipped, dim=-1).min())
    cos_hand = float(F.cosine_similarity(cls_hand.cpu(), shipped, dim=-1).min())
    print(f'        cos(pool(.,token), production) {cos_pub:.8f}   '
          f'cos(hand cls, production) {cos_hand:.8f}')
    assert cos_pub > 1 - 1e-6, (
        f"pool(global_pool='' output, 'token') is not the production feature: "
        f'worst cos {cos_pub:.8f}')


#: GigaPath ships an input and the tensor its own inference code produces from
#: it (demo/3_load_tile_encoder.py:29-34). Anchoring the baseline here rather
#: than on a vector of our own recording is the difference between "the same as
#: last week" and "the same as what the authors validated".
_GIGAPATH_REPO = Path('/work/u26130998/prov-gigapath')
_REF_PNG = _GIGAPATH_REPO / 'images' / 'prov_normal_000_1.png'
_REF_PT = _GIGAPATH_REPO / 'images' / 'prov_normal_000_1.pt'


def t_baseline_matches_gigapath_reference():
    """The frozen baselines must reproduce the author's own numbers.

    _TRANSFORM_BASELINE and _ENCODER_BASELINE are what every hash is measured
    against, and the one rule holding that scheme up is that a value added to a
    baseline reproduces the previous behaviour. A rule kept by hand needs
    something that fails when it is not kept, and this is it.

    It also settles 256/224 on evidence rather than on config.json, which
    declares crop_pct 1.0 while all four of GigaPath's own code paths resize to
    256 and centre-crop 224. If the metadata were right and this test's
    transform wrong, the reference vector would not come out.

    fp32 and no autocast, matching the demo. atol is the demo's own 1e-2, which
    is loose because it has to cover other people's torch and cuDNN builds; the
    printed max|delta| is the number worth reading.
    """
    import torch as _t
    from GigaPathFunc import GigaPathEncoderConfig, _GIGAPATH_BASELINE
    from TileEncoderFunc import TransformConfig

    if not _REF_PNG.exists() or not _REF_PT.exists():
        print(f'        [SKIP] no reference at {_GIGAPATH_REPO}')
        return

    # The baseline is the claim; TransformConfig()'s defaults only happen to
    # match it today. Build from the baseline itself so a changed default is
    # caught by the NEXT check rather than silently passing this one.
    assert GigaPathEncoderConfig().transform == _GIGAPATH_BASELINE['transform'], (
        "GigaPathEncoderConfig's transform default has drifted from the "
        'baseline. That is allowed -- but then every store written with the old '
        'default is reachable only by spelling the baseline values out, so make '
        'sure that is what you meant.')

    dev = _t.device('cuda' if _t.cuda.is_available() else 'cpu')
    encoder = GigaPathEncoderConfig().with_model(dtype='fp32').build(dev)

    got = encoder.features([Image.open(_REF_PNG).convert('RGB')])
    expected = _t.load(_REF_PT, map_location='cpu').squeeze().float()

    # gigapath_encode and .features() both L2-normalise; the demo does not, so
    # compare directions rather than magnitudes and report both.
    got = got.squeeze().float().cpu()
    cos = float(F.cosine_similarity(got, F.normalize(expected, dim=-1), dim=-1))
    print(f'        ref {tuple(expected.shape)}  ours {tuple(got.shape)}  '
          f'cos {cos:.8f}  |ref| {float(expected.norm()):.4f}')
    assert got.shape == expected.shape, (
        f'reference is {tuple(expected.shape)}, we produced {tuple(got.shape)}')
    assert cos > 1 - 1e-4, (
        f'the baseline transform + CLS reduction does not reproduce GigaPath\'s '
        f'own reference output: cos {cos:.6f}. Either a baseline value changed '
        f'or the 256/224 decision did.')


def t_encoder_matches_the_old_path(n_tiles=8):
    """GigaPathEncoder must compute what the free functions computed.

    One model, flipped, rather than two 4.5 GB loads. Safe for this pair because
    use_fc_norm is decided in __init__ from global_pool in ('avg','avgmax','max')
    -- neither '' nor 'token' is in that set, so which of norm/fc_norm is real
    cannot change under the flip. t_global_pool_empty_gives_the_same_tokens
    proves the token halves agree at exactly zero; this one covers the two
    things built on top of that: the pooled feature and the batch loop.

    batch_size below n_tiles on purpose -- the loop has to run more than once,
    or an off-by-one in the batching never shows.
    """
    import torch as _t
    from GigaPathFunc import GigaPathEncoderConfig
    from GigaPathFunc_old import gigapath_encode, gigapath_encode_tokens

    dev = _t.device('cuda' if _t.cuda.is_available() else 'cpu')
    encoder = GigaPathEncoderConfig(batch_size=3).build(dev)
    m = getattr(encoder.model, 'module', encoder.model)

    rng = np.random.default_rng(0)
    tiles = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
             for _ in range(n_tiles)]

    new_tokens = encoder.tokens(tiles)
    new_feats = encoder.features(tiles)

    m.reset_classifier(0, 'token')
    try:
        old_tokens = gigapath_encode_tokens(tiles, encoder.model, dev,
                                            batch_size=3, dtype=_t.float16)
        old_feats = F.normalize(
            gigapath_encode(tiles, encoder.model, dev,
                            batch_size=3, dtype=_t.float16).float(), dim=-1)
    finally:
        m.reset_classifier(0, '')

    d_tok = float((new_tokens - old_tokens).abs().max())
    d_feat = float((new_feats.cpu() - old_feats.cpu()).abs().max())
    # Decoy: the same features rolled by one tile. A batching or ordering slip
    # produces a difference of that size, not of the size of float noise.
    decoy = float((new_feats.cpu() - old_feats.roll(1, dims=0).cpu()).abs().max())
    print(f'        tokens max|Δ| {d_tok:.3e}   features max|Δ| {d_feat:.3e}   '
          f'rolled decoy {decoy:.3e}')

    assert d_tok == 0.0, (
        f'.tokens() differs from gigapath_encode_tokens by {d_tok:.3e}; both '
        f'run the same blocks, so this is not precision')
    assert d_feat * 1000 < decoy, (
        f'.features() differs from gigapath_encode by {d_feat:.3e}, not clear '
        f'of the rolled decoy at {decoy:.3e}')

    # .pooled() is the third encoding path and the one bench_slidewin_pooling
    # will move onto. It reduces INSIDE the batch loop, so its result is
    # assembled from per-batch pieces while the reference pools one whole token
    # tensor at the end -- concatenation order and slot bookkeeping are what can
    # go wrong, and neither raises.
    # Scored against decoys and not against zero. The first version asserted
    # exact equality on the grounds that both sides run the same arithmetic on
    # the same values -- they do, but not on the same DEVICE: .pooled() reduces
    # on the GPU inside the loop while the reference pools the tokens
    # gigapath_encode_tokens has already moved to the host. 1536 floats summed
    # in two orders differ around 1e-8, which is the same 1.49e-08 the
    # reduce-vs-host check below reports independently.
    for mode in ('cls', 'cls_avg', 'rings3', 'grid2x2'):
        new_p = encoder.pooled(tiles, mode)
        fs = encoder.pooled_spec(new_p, mode)
        slots, layout = fs.slots, fs.slot_layout
        # Two calls, not one unpacking. 923e9e6 split pool_tokens -- which
        # returned the tensor and its slot names together -- into pooling_kinds
        # that reduces and pool_slots that names, precisely so the names stop
        # being a byproduct of a reduction. This line still unpacked three
        # values and had done since that commit: it sits behind --with-model,
        # which nothing passed, so it was never run.
        ref_p = pooling_kinds(old_tokens, mode, encoder.model_spec).cpu()
        ref_slots, ref_layout = pool_slots(mode, encoder.model_spec)
        d = float((new_p - ref_p).abs().max())
        # Two decoys, because neither covers every mode. Rolling the TILES
        # catches a batching or concatenation slip and works at any slot count;
        # rolling the SLOTS catches a reduce that returns them in another order,
        # and only exists when there is more than one.
        decoys = {'tile-roll': float((new_p - ref_p.roll(1, dims=0)).abs().max())}
        if ref_p.shape[1] > 1:
            decoys['slot-roll'] = float((new_p - ref_p.roll(1, dims=1)).abs().max())
        worst = min(decoys.values())
        print(f'        pooled {mode:8s} {tuple(new_p.shape)}  max|Δ| {d:.3e}   '
              + '  '.join(f'{k} {v:.2e}' for k, v in decoys.items()))
        assert (slots, layout) == (ref_slots, ref_layout), (
            f'{mode}: .pooled() reports {(slots, layout)}, pooling_kinds says '
            f'{(ref_slots, ref_layout)}')
        assert d * 1000 < worst, (
            f'{mode}: .pooled() differs from pooling the tokens afterwards by '
            f'{d:.3e}, not clear of the nearest decoy at {worst:.3e}')

    # And the identity surface answers, without a store being involved.
    parts = encoder.identity_parts()
    print(f'        identity_id {encoder.identity_id()}   parts {parts}')
    assert any(p.startswith('weights=') for p in parts), \
        'identity_parts must carry the loaded weights, not just the config'
    assert not any(p.startswith('batch_size') for p in parts), \
        'batch_size must stay out of identity, or tuning throughput drops the cache'


def t_cls_equals_production(n_tiles=8):
    """pooling_kinds(...,'cls') must BE gigapath_encode, not merely resemble it."""
    import torch as _t
    from GigaPathFunc_old import (gigapath_model, gigapath_encode,
                                   gigapath_encode_tokens)

    dev = _t.device('cuda' if _t.cuda.is_available() else 'cpu')
    model = gigapath_model(dev)                       # single card on purpose
    # Written out rather than read off the model, and that is right for THIS
    # file. Both sides of every comparison here would otherwise be described by
    # the same reader, so a reader that lied would make them agree. These three
    # numbers are facts about prov-gigapath: 1536 wide, a 14x14 grid at 224 with
    # patch 16, one CLS and no registers.
    spec = {'dim': 1536, 'feat_hw': (14, 14), 'num_prefix': 1}
    print(f'        spec = {spec}')

    # State the precondition instead of failing obscurely on it. "Production" is
    # gigapath_encode's output, and which token that is depends on global_pool:
    # at 'token' it is the CLS, at 'avg' it is the mean of the patches. The
    # equality below is cls-vs-cls only while the model pools on the token.
    gp = getattr(getattr(model, 'module', model), 'global_pool', None)
    assert gp == 'token', (
        f"this test compares the cls slot against gigapath_encode, which is only "
        f"the same quantity when global_pool='token'; model says {gp!r}. If the "
        f"model was deliberately switched, compare against the matching pooling "
        f"mode instead of loosening this.")

    rng = np.random.default_rng(0)
    tiles = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
             for _ in range(n_tiles)]

    a = gigapath_encode(tiles, model, dev)
    b = pooling_kinds(gigapath_encode_tokens(tiles, model, dev), 'cls', spec)
    slots, _ = pool_slots('cls', spec)
    b = b[:, 0]
    cos = F.cosine_similarity(a, b, dim=-1)
    print(f'        cos(production, cls slot): min {cos.min():.8f}')
    assert (1 - cos).max() < 1e-4, \
        f'cls slot is not the production feature: worst cos {cos.min():.6f}'


POOLINGS = ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2')


def t_reduce_equals_pooling_afterwards(n_tiles=8):
    """Pooling inside the encoder must give what pooling after it gave.

    `gigapath_encode_tokens(reduce=...)` exists so the tokens never cross to the
    host: 1.21 MB per tile becomes 86 KB. That moves the arithmetic onto the GPU
    and packs five poolings into one tensor, so two new things can go wrong and
    neither raises.

      dtype   under autocast the tokens arrive fp16. Pooling means .mean(1) and
              .std(1) over 196 of them, and doing that in fp16 degrades the
              averaged slots while leaving cls -- one token, no arithmetic --
              exactly right. The encoder calls .float() before reduce for this
              reason; move it after and the experiment reports that averaging
              poolings do not help.
      packing five descriptors of different widths share one tensor and are
              split by offset. A wrong offset hands each mode another mode's
              numbers, in the right shape.

    Both are scored against decoys rather than a tolerance, because the right
    tolerance here is unknown and the wrong answers are not:

      fp16     the same pooling done in fp16, i.e. the dtype bug itself
      rolled   the mode's own slots rotated by one, i.e. the packing bug

    The agreement has to beat both by 1000x. It is not asserted as exact even
    for cls: the host path reduces on the CPU and this one on the GPU, and two
    sums of 1536 floats in different orders are not obliged to match bit for
    bit. What they cannot do is differ by as much as a decoy.
    """
    import torch as _t
    from GigaPathFunc import GigaPathEncoderConfig

    # The new API on both sides, and not GigaPathFunc_old: this compares two
    # routes THROUGH THE CURRENT ENCODER -- tokens crossing to the host and
    # pooled after, against pooled inside the batch loop. The frozen module has
    # no reduce hook, because the hook is the thing under test.
    dev = _t.device('cuda' if _t.cuda.is_available() else 'cpu')
    encoder = GigaPathEncoderConfig(batch_size=4).build(dev)
    spec = encoder.model_spec
    dim = int(spec['dim'])

    rng = np.random.default_rng(0)
    tiles = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
             for _ in range(n_tiles)]

    # Every token crosses, pooling afterwards -- what the bench used to do, and
    # still the default. fp16 autocast, so the tokens are the ones a run sees.
    tokens = encoder.tokens(tiles)
    assert tokens.dtype == _t.float32 and tokens.device.type == 'cpu'
    host = {m: pooling_kinds(tokens, m, spec) for m in POOLINGS}

    # Pooled on the model's device, inside the batch loop. batch_size below
    # n_tiles on purpose: reduce runs more than once and the parts are
    # concatenated, which is where an off-by-one batch would show.
    widths = {}

    def reduce(t):
        assert t.dtype == _t.float32, f'reduce got {t.dtype}, not fp32'
        assert t.device.type == dev.type, 'reduce ran off the model device'
        parts = []
        for m in POOLINGS:
            flat = pooling_kinds(t, m, spec).flatten(1)
            widths[m] = flat.shape[1]
            parts.append(flat)
        return _t.cat(parts, dim=1).cpu()

    packed = encoder.tokens(tiles, reduce=reduce)
    assert packed.shape == (n_tiles, sum(widths.values())), \
        f'packed is {tuple(packed.shape)}, expected {(n_tiles, sum(widths.values()))}'

    start = 0
    for m in POOLINGS:
        got = packed[:, start:start + widths[m]].reshape(n_tiles, -1, dim)
        start += widths[m]
        want = host[m]
        assert got.shape == want.shape, f'{m}: {tuple(got.shape)} vs {tuple(want.shape)}'

        same = (got - want).abs().max().item()
        fp16 = (pooling_kinds(tokens.half(), m, spec).float()
                - want).abs().max().item()
        decoys = {'fp16': fp16}
        if want.shape[1] > 1:                     # cls has one slot to roll
            decoys['rolled'] = (got - want.roll(1, dims=1)).abs().max().item()

        worst = min(decoys.values())
        print(f'        {m:8s} n={want.shape[1]}  diff {same:.2e}   '
              + '  '.join(f'{k} {v:.2e}' for k, v in decoys.items()))
        assert same * 1000 < worst, (
            f'{m}: device pooling differs from host pooling by {same:.3e}, '
            f'not clear of the nearest decoy at {worst:.3e}')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles', type=int, default=8,
                    help='tiles per comparison; more is slower, not stricter')
    args = ap.parse_args()

    # --with-model is gone. It selected the half of this file that needed the
    # weights, and the other half now lives in test_tile_encoder -- so a flag
    # that turns everything off is not an option worth having.
    print('equivalence against GigaPathFunc_old  (loads GigaPath)')
    check("global_pool='' == hand-assembled tokens",
          lambda: t_global_pool_empty_gives_the_same_tokens())
    check('baseline reproduces GigaPath reference',
          t_baseline_matches_gigapath_reference)
    check('GigaPathEncoder == the free functions',
          lambda: t_encoder_matches_the_old_path(args.tiles))
    check('cls slot == gigapath_encode',
          lambda: t_cls_equals_production(args.tiles))
    check('reduce == pooling afterwards',
          lambda: t_reduce_equals_pooling_afterwards(args.tiles))

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
