#!/usr/bin/env python3
"""Unit test for the token/pooling additions in aiNNModel/GigaPathFunc.

Two halves, because they cost very different amounts:

    python utilities/test_modules/test_gigapath_pooling.py
        Pure tensor maths against a fake spec. No model, no GPU, no WSI --
        runs on a login node in a second and catches most mistakes.

    python utilities/test_modules/test_gigapath_pooling.py --with-model
        Loads GigaPath and asserts the two things the cheap half cannot:
        pooling_kinds(..., 'cls') is bit-for-bit the production feature, and
        pooling inside the encoder gives what pooling after it gave.

The first check is why this file exists. Every pooling in the experiment is
compared against `cls` as the baseline, so if `cls` is not exactly what
gigapath_encode produces today, every conclusion is measured against the wrong
reference and nothing raises. It is written as an equality test rather than a
one-off spot check because the thing that would break it -- switching
global_pool to 'avg', one of the pooling variants under consideration -- swaps
which of timm's `norm` / `fc_norm` is the real LayerNorm.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

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

from GigaPathFunc import (model_token_spec, pooling_kinds,     # noqa: E402
                          pool_slots,
                          _ring_bins)

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

SPEC = {'dim': 8, 'feat_hw': (14, 14), 'num_prefix': 1}


def toks(n=3, spec=SPEC, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = spec['num_prefix'] + spec['feat_hw'][0] * spec['feat_hw'][1]
    return torch.randn(n, t, spec['dim'], generator=g)


# ── shape / slot bookkeeping ──────────────────────────────────────────────────

def t_shapes_and_slots():
    x = toks()
    for mode, n, layout in (('cls', 1, 'none'),
                            ('cls_avg', 2, 'none'),
                            ('cls_std', 2, 'none'),
                            ('rings3', 4, 'ring:3'),
                            ('grid2x2', 5, 'grid:2x2'),
                            ('tokens', 197, 'grid:14x14')):
        f = pooling_kinds(x, mode, SPEC)
        slots, lay = pool_slots(mode, SPEC)
        assert f.shape == (3, n, SPEC['dim']), f'{mode}: shape {tuple(f.shape)}'
        assert len(slots) == n, f'{mode}: {len(slots)} names for {n} slots'
        assert lay == layout, f'{mode}: layout {lay!r} != {layout!r}'
        assert slots[0] == 'cls', f'{mode}: slot 0 must be cls'


def t_every_slot_is_unit_norm():
    # Each slot separately, not the flattened vector -- otherwise the weight
    # between slots is fixed at write time, and that weighting is the open
    # question the experiment exists to answer.
    for mode in ('cls', 'cls_avg', 'cls_std', 'rings3', 'grid2x2', 'tokens'):
        f = pooling_kinds(toks(), mode, SPEC)
        n = f.norm(dim=-1)
        assert torch.allclose(n, torch.ones_like(n), atol=1e-5), \
            f'{mode}: slot norms range {n.min():.4f}..{n.max():.4f}'


def t_cls_slot_is_just_the_cls_token():
    x = toks()
    f = pooling_kinds(x, 'cls_avg', SPEC)
    assert torch.allclose(f[:, 0], F.normalize(x[:, 0], dim=-1), atol=1e-6)


def t_registers_are_dropped():
    # DINOv2 _reg4 has num_prefix=5. Patches must start after the registers; the
    # failure mode otherwise is four junk vectors averaged into every pooling,
    # with no error and only a slightly worse score to show for it.
    spec = {'dim': 8, 'feat_hw': (4, 4), 'num_prefix': 5}
    x = torch.zeros(1, 5 + 16, 8)
    x[0, 0] = 1.0                       # cls
    x[0, 1:5] = 99.0                    # registers -- must not reach any slot
    x[0, 5:] = 2.0                      # patches
    f = pooling_kinds(x, 'cls_avg', spec)
    assert torch.allclose(f[0, 1], F.normalize(torch.full((8,), 2.0), dim=-1)), \
        'the average picked up the register tokens'


def t_no_prefix_makes_slot_0_the_average():
    # A CNN or a U-Net has no CLS: num_prefix is 0 and its summary vector is the
    # global average, which is also what its features() returns. Reading
    # tokens[:, 0] regardless would put the TOP-LEFT CELL in slot 0 and label it
    # 'cls'. Nothing downstream raises -- the token-count check passes because
    # 0 + gh*gw == T, the store validates, the vectors look like vectors. It
    # would show up only as a pooling that scores worse than it should.
    #
    # Scored against that exact decoy rather than a tolerance: the corner is
    # what the old code returned, so if it ever comes back this fails at the
    # decoy and not at the margin.
    spec = {'dim': 4, 'feat_hw': (2, 2), 'num_prefix': 0}
    x = torch.zeros(1, 4, 4)
    x[0, 0] = torch.tensor([9.0, 0.0, 0.0, 0.0])        # the corner cell
    x[0, 1:] = 1.0
    f = pooling_kinds(x, 'cls', spec)
    slots, _ = pool_slots('cls', spec)
    want = F.normalize(x[0].mean(0), dim=-1)
    corner = F.normalize(x[0, 0], dim=-1)
    d_want = float((f[0, 0] - want).abs().max())
    d_corner = float((f[0, 0] - corner).abs().max())
    assert d_want * 1000 < d_corner, (
        f'slot 0 is {d_want:.3e} from the average and {d_corner:.3e} from the '
        f'corner cell -- it took tokens[:, 0] with no prefix to take')
    # 'gap' and not 'cls': StoreMeta.slots is the only record of what slot 0
    # holds, so naming a global average after a token this model does not have
    # would make the store say something false about itself.
    assert slots[0] == 'gap', f"slot 0 of a prefix-free model is {slots[0]!r}"


def t_no_prefix_refuses_the_mode_that_would_duplicate():
    # cls_avg pairs the summary with the patch mean. With no prefix the summary
    # IS the patch mean, so both slots would be the same vector -- bit-identical
    # after per-slot normalisation. Nothing raises on its own: the store holds
    # two slots, every similarity is exactly doubled, and the result reads as
    # "cls_avg performs like cls" rather than as a broken configuration.
    #
    # cls_std is the control. std is not mean, so it stays legal, and if this
    # ever starts raising the refusal has grown past what it was for.
    spec = {'dim': 4, 'feat_hw': (2, 2), 'num_prefix': 0}
    x = torch.randn(2, 4, 4, generator=torch.Generator().manual_seed(1))
    rejects(lambda: pooling_kinds(x, 'cls_avg', spec), 'num_prefix=0')
    f = pooling_kinds(x, 'cls_std', spec)
    slots, _ = pool_slots('cls_std', spec)
    assert slots == ('gap', 'std'), f'slots {slots}'
    assert not torch.allclose(f[:, 0], f[:, 1]), 'std collapsed onto the summary'

    # Every remaining mode names slot 0 for what it actually holds.
    for mode in ('cls', 'cls_std', 'rings2', 'grid2x2', 'tokens'):
        pooling_kinds(x, mode, spec)
        slots, _ = pool_slots(mode, spec)
        assert slots[0] == 'gap', f'{mode}: slot 0 is {slots[0]!r}'


def t_no_prefix_keeps_every_cell_as_a_patch():
    # The other half of the same rule: with num_prefix 0 the summary is computed
    # FROM the cells, so no cell may be consumed by it. A slice of tokens[:, 1:]
    # would drop one and still produce a plausible answer for every mode.
    spec = {'dim': 2, 'feat_hw': (2, 2), 'num_prefix': 0}
    x = torch.zeros(1, 4, 2)
    x[0, :, 0] = torch.arange(4, dtype=torch.float32)
    f = pooling_kinds(x, 'tokens', spec)
    slots, layout = pool_slots('tokens', spec)
    assert layout == 'grid:2x2'
    assert len(slots) == 5, f'1 summary + 4 cells, got {len(slots)}'
    for k in range(4):
        assert torch.allclose(f[0, 1 + k], F.normalize(x[0, k], dim=-1),
                              atol=1e-5), f'cell {k} is not slot {1 + k}'


def t_outputspec_and_storemeta_subscript_the_same():
    # pooling_kinds takes either supplier by name. The point of the test is that
    # nothing converts between them: the same three keys reach it whether it was
    # handed a live encoder's model_spec or a StoreMeta off disk, so a rename on
    # one side fails here rather than at some consumer months later.
    from TileEncoderFunc import ModelOutputSpec
    spec = ModelOutputSpec(kind='tokens', dim=8, feat_hw=(4, 4), num_prefix=1)
    for k, want in (('dim', 8), ('feat_hw', (4, 4)), ('num_prefix', 1),
                    ('kind', 'tokens')):
        assert spec[k] == want, f'spec[{k!r}] == {spec[k]!r}, wanted {want!r}'
    x = toks(spec={'dim': 8, 'feat_hw': (4, 4), 'num_prefix': 1})
    a = pooling_kinds(x, 'rings3', spec)
    b = pooling_kinds(x, 'rings3', {'dim': 8, 'feat_hw': (4, 4),
                                        'num_prefix': 1})
    assert torch.equal(a, b), 'ModelOutputSpec and the plain dict disagree'
    rejects(lambda: spec['token_grid'], 'token_grid')


def t_grid_blocks_average_the_right_cells():
    spec = {'dim': 1, 'feat_hw': (4, 4), 'num_prefix': 1}
    x = torch.zeros(1, 17, 1)
    x[0, 1:] = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    # 4x4 grid, 2x2 blocks -> top-left block holds cells 0,1,4,5 -> mean 2.5
    g = x[0, 1:].reshape(4, 4)
    want = [g[:2, :2].mean(), g[:2, 2:].mean(), g[2:, :2].mean(), g[2:, 2:].mean()]
    f = pooling_kinds(x, 'grid2x2', spec)
    slots, _ = pool_slots('grid2x2', spec)
    assert slots[1:] == ('g00', 'g01', 'g10', 'g11')
    # after per-slot normalization a 1-D vector is just its sign, so compare the
    # pre-normalized means through a second call on a widened dim
    x2 = x.repeat(1, 1, 2)
    f2 = pooling_kinds(x2, 'grid2x2', {**spec, 'dim': 2})
    for k, w in enumerate(want):
        got = f2[0, 1 + k]
        assert torch.allclose(got, F.normalize(torch.tensor([w, w]), dim=-1), atol=1e-5), \
            f'block {slots[1+k]}: expected mean {w}'


def t_rings_hold_equal_counts():
    bins = _ring_bins(14, 14, 3)
    counts = [int((bins == k).sum()) for k in range(3)]
    assert max(counts) - min(counts) <= 1, f'ring sizes {counts} are not balanced'
    assert sorted(set(bins.tolist())) == [0, 1, 2]


def t_rings_follow_the_tokens_to_the_gpu():
    """A CPU ring mask cannot index CUDA tokens, and rings3 is the only mode
    that indexes at all -- the others are reshape and slice, which never notice.

    So pooling moved onto the GPU (bench_slidewin_pooling's `reduce`) breaks
    exactly one of the five and leaves four working. It raises rather than
    corrupting, but it raises deep inside a run that has already read a whole
    tissue region, which is minutes and tens of GB from here.
    """
    assert _ring_bins(14, 14, 3).device.type == 'cpu'
    if not torch.cuda.is_available():
        return                                   # nothing to check without one
    dev = torch.device('cuda')
    assert _ring_bins(14, 14, 3, dev).device.type == 'cuda'
    assert torch.equal(_ring_bins(14, 14, 3, dev).cpu(), _ring_bins(14, 14, 3))
    f = pooling_kinds(toks().to(dev), 'rings3', SPEC)
    assert f.device.type == 'cuda', 'ring pooling left the tokens\' device'


def t_bad_inputs_are_refused():
    rejects(lambda: pooling_kinds(torch.zeros(3, 8), 'cls', SPEC), '[N, T, D]')
    rejects(lambda: pooling_kinds(torch.zeros(3, 5, 8), 'cls', SPEC), 'disagree')
    rejects(lambda: pooling_kinds(toks(), 'nonsense', SPEC), 'unknown pooling mode')
    rejects(lambda: pooling_kinds(toks(), 'grid3x3', SPEC), 'does not divide')


# ── config fields are honoured ────────────────────────────────────────────────
#
# A config field that is declared, documented and hashed but never read is the
# failure this section exists for. It happened here: weights, tome_r and compile
# were added to EncoderConfig with comments explaining their effect and none of
# them reached __init__. Setting weights='ft.ckpt' loaded the HF weights.
#
# The shape of that failure is worse than doing nothing. An inert field that
# differs from its baseline still MOVES the hash -- two stores, two filenames,
# identical vectors, and every reason to believe they hold different things.
#
# tome_r was one of the three and is gone; `pooling` took its place in the
# checks below, and it is the better example because reading it is what
# features() now does rather than something build() does once.
#
# vit_tiny rather than GigaPath: the loading path does not depend on the
# architecture, and 5 MB keeps this in the half that runs on a login node.

_TINY = 'vit_tiny_patch16_224'


def _tiny_encoder(**over):
    from GigaPathFunc import GigaPathEncoderConfig
    # arch, dtype and weights live on the nested ModelConfig now, so they go
    # through with_model rather than being spelled at the top level.
    model_over = {k: over.pop(k) for k in ('weights',) if k in over}
    return (GigaPathEncoderConfig(batch_size=2, **over)
            .with_model(arch=_TINY, dtype='fp32', **model_over))


def t_weights_field_is_actually_loaded():
    """GigaPathEncoderConfig(weights=...) must change the model, not just the hash.

    The second checkpoint is the first one with a single tensor perturbed. A
    real finetune is not needed and would not test anything more: what is under
    test is whether load_state_dict is reached, and any two different parameter
    sets answer that.

    weights_id is the probe because it is derived from the parameters actually
    resident. If cfg.weights were ignored, the second encoder would hold the
    pretrained values, the two ids would match, and this fails.
    """
    import tempfile
    from GigaPathFunc import GigaPathEncoderConfig

    dev = torch.device('cpu')
    base = _tiny_encoder().build(dev)
    id_a = base.weights_id

    state = {k: v.clone() for k, v in base.model.state_dict().items()}
    # A matrix, and noise rather than a constant. The first attempt added 0.25
    # to sorted(state)[0], which was blocks.0.attn.proj.bias -- a constant added
    # to a bias shifts every token by the same vector, and every LayerNorm
    # downstream subtracts the per-token mean, so 0.25 arrived at the output as
    # 3.6e-07. The id moved and the features did not, which says something true
    # about both: weights_id is strictly more sensitive than behaviour, and a
    # perturbation has to survive normalisation to test anything.
    victim = next(k for k in sorted(state)
                  if state[k].ndim >= 2 and state[k].is_floating_point())
    g = torch.Generator().manual_seed(0)
    t = state[victim]
    state[victim] = t + torch.randn(t.shape, generator=g) * 0.1 * float(t.std())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'perturbed.pt'
        torch.save(state, path)
        other = _tiny_encoder(weights=str(path)).build(dev)
        id_b = other.weights_id

        assert id_a != id_b, (
            f'weights_id is {id_a} both with and without weights= -- the '
            f'checkpoint was not loaded, so the field is inert')

        # And the vectors move, which is the thing the id is standing in for.
        rng = np.random.default_rng(0)
        tiles = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
                 for _ in range(2)]
        gap = float((base.features(tiles) - other.features(tiles)).abs().max())
        print(f'        {victim}  +10% noise   ids {id_a[:8]} -> {id_b[:8]}   '
              f'max|Δfeat| {gap:.3e}')
        assert gap > 1e-4, f'weights changed but features did not: {gap:.3e}'

    # The path is provenance, not identity: the same content under another name
    # must not split the cache.
    parts = other.identity_parts()
    assert not any(p.startswith('weights=/') or 'perturbed' in p for p in parts), \
        f'the checkpoint PATH leaked into identity: {parts}'


def t_identity_moves_only_where_it_should():
    """Which fields reach the hash, checked without loading anything."""
    from GigaPathFunc import GigaPathEncoderConfig, _GIGAPATH_BASELINE

    B = _GIGAPATH_BASELINE
    base = GigaPathEncoderConfig().identity_parts(B)
    assert base == [], f'a default config should be all-baseline, got {base}'

    from ConfigIdentity import ModelConfig
    M = B['model']
    import dataclasses as _dc
    for over, why in ((dict(batch_size=999), 'batch_size'),
                      (dict(compile=True), 'compile'),
                      (dict(model=_dc.replace(M, weights='/tmp/x.ckpt')), 'weights'),
                      (dict(pooling='cls'), "pooling at its baseline, spelled")):
        assert GigaPathEncoderConfig(**over).identity_parts(B) == [], \
            f'{why} must not reach the hash: ' \
            f'{GigaPathEncoderConfig(**over).identity_parts(B)}'

    for over, why in ((dict(pooling='grid2x2'), 'pooling away from baseline'),
                      (dict(model=_dc.replace(M, dtype='fp32')), 'dtype'),
                      (dict(model=_dc.replace(M, arch='timm:other')), 'arch')):
        assert GigaPathEncoderConfig(**over).identity_parts(B) != [], \
            f'{why} must reach the hash'

    # The transform rides in under its own prefix, and only when it differs.
    assert GigaPathEncoderConfig(transform=B['transform']).identity_parts(B) == []
    grey = GigaPathEncoderConfig(
        transform=_dc.replace(B['transform'], preprocess='grey')).identity_parts(B)
    assert grey == ['transform.preprocess=grey'], grey


# ── model_token_spec ──────────────────────────────────────────────────────────

class _Fake(torch.nn.Module):
    def __init__(self, head=None):
        super().__init__()
        self.embed_dim = 8
        self.num_prefix_tokens = 1
        self.head = head if head is not None else torch.nn.Identity()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.grid_size = (4, 4)


def t_spec_unwraps_dataparallel():
    m = _Fake()
    dp = torch.nn.DataParallel(m)
    # The pit itself: DataParallel defines no __getattr__, so nn.Module's is
    # used and it searches only _parameters / _buffers / _modules.
    rejects(lambda: dp.embed_dim, 'embed_dim')
    assert model_token_spec(dp) == model_token_spec(m) == {
        'dim': 8, 'feat_hw': (4, 4), 'num_prefix': 1}


def t_spec_refuses_a_classifier():
    rejects(lambda: model_token_spec(_Fake(head=torch.nn.Linear(8, 5))),
            'num_classes=0')


# ── the expensive half ────────────────────────────────────────────────────────

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
        ref_p, ref_slots, ref_layout = pooling_kinds(old_tokens, mode,
                                                   encoder.model_spec)
        ref_p = ref_p.cpu()
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
    spec = model_token_spec(model)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-model', action='store_true',
                    help='also load GigaPath and check cls == gigapath_encode')
    ap.add_argument('--tiles', type=int, default=8)
    args = ap.parse_args()

    print('pooling_kinds')
    check('shapes and slot names agree',        t_shapes_and_slots)
    check('every slot is unit norm',            t_every_slot_is_unit_norm)
    check('cls slot is the cls token',          t_cls_slot_is_just_the_cls_token)
    check('register tokens are dropped',        t_registers_are_dropped)
    check('no prefix -> slot 0 is the average', t_no_prefix_makes_slot_0_the_average)
    check('no prefix -> cls_avg is refused',    t_no_prefix_refuses_the_mode_that_would_duplicate)
    check('no prefix -> every cell is kept',    t_no_prefix_keeps_every_cell_as_a_patch)
    check('ModelOutputSpec subscripts like the dict', t_outputspec_and_storemeta_subscript_the_same)
    check('grid blocks average the right cells', t_grid_blocks_average_the_right_cells)
    check('rings hold equal counts',            t_rings_hold_equal_counts)
    check('ring mask follows the tokens',       t_rings_follow_the_tokens_to_the_gpu)
    check('bad inputs are refused',             t_bad_inputs_are_refused)

    print('model_token_spec')
    check('unwraps DataParallel',               t_spec_unwraps_dataparallel)
    check('refuses a classifier head',          t_spec_refuses_a_classifier)

    print('EncoderConfig fields are honoured')
    check('identity moves only where it should', t_identity_moves_only_where_it_should)
    check('weights= is actually loaded',        t_weights_field_is_actually_loaded)

    if args.with_model:
        print('equivalence (loads GigaPath)')
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
    else:
        print('equivalence: skipped, pass --with-model to run it')

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
