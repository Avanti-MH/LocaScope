#!/usr/bin/env python3
"""Tests of CoAE's parts that need the slides, the FM and a GPU (`locascope/training/CoAE`): a batch of the supply (shapes, a place's
views contiguous, slot outside and source inside, twins), the targets of a real batch, the network frozen (gigapath's 14 x 14 tokens,
latent 7 x 7 x 128, Co-figure 28 x 28, the output shapes, the bottleneck noise only in training, the checkpoint without the FM's weights) and with
the residual cls', which loss reaches what, and the probe (slides never trained on, fixed, and its measures). The training slide is one;
the two Supplies of the tests are shared (`base_supply`, `quad_supply`).

    python tests/TestCoAE/test_coae_slide.py
"""
from __future__ import annotations

import itertools
from concurrent.futures import Future
import os
import sys
import time
import traceback
from dataclasses import replace
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch                                                                         # noqa: E402
import torch.nn.functional as TF                                                     # noqa: E402

from locascope.training.CoAE.CoFigure import Targets, n_channels, part_names   # noqa: E402
from locascope.training.CoAE.Config import CoaeConfig                               # noqa: E402
from locascope.training.CoAE.Losses import ChannelScale, recon_pooled, recon_point  # noqa: E402
from locascope.training.CoAE.Model import CoaeNet                                   # noqa: E402
from locascope.core.slides.AccessDatasets import by_source, list_names             # noqa: E402
from locascope.training.CoAE.Sims import GREY_WEIGHTS, Geometry, Shot, centre, frame   # noqa: E402
from locascope.training.CoAE.Supply import (BLANK, SCREEN_PAD, USABLE, Batch, Flight, Made, Supply, groups_of, read_spot,   # noqa: E402
                                            training_slides)
from locascope.core.slides.ReadGeometry import ReadSpec                             # noqa: E402
from locascope.core.slides.SafeSlide import SafeSlide                               # noqa: E402
from locascope.core.slides.SlideReader import SlideReader                           # noqa: E402
from locascope.training.CoAE.Validate import pooled_sides, probe_measures                # noqa: E402
from locascope.training.CoAE import LogLayout as LL                                 # noqa: E402
from locascope.training.CoAE.train import Views, own_state, targets_of, views_of    # noqa: E402

DEV: torch.device = torch.device('cuda')
# the UNI2-PCA masks are in the shared cache for every @pure-train slide; the draws go to this test's own cache
CFG: CoaeConfig = replace(CoaeConfig(), data=replace(CoaeConfig().data, places=8, mask='uni2_pca', draw_cache='TestCoae', slides=1))
_RESULTS: List[Tuple[str, Optional[BaseException]]] = []
STATE: dict = {}
CFG4: CoaeConfig = replace(CFG, data=replace(CFG.data, cell_parts=4))               # a cell lists a quarter of its pool an epoch: `seen` decides the next epoch
_SHARED: Dict[str, Supply] = {}                                                      # the two Supplies of the tests: built at first use, closed at the end


def check(name: str, fn: Callable[[], str]) -> None:
    t0: float = time.time()
    try:
        msg: str = fn()
        _RESULTS.append((name, None))
        print(f'PASS  {name}: {msg} ({time.time() - t0:.0f} s)', flush=True)
    except BaseException as exc:                                                     # noqa: BLE001
        _RESULTS.append((name, exc))
        print(f'FAIL  {name}: {type(exc).__name__}: {exc}', flush=True)
        traceback.print_exc()


def shared(key: str, cfg: CoaeConfig) -> Supply:
    """The tests' Supply of `cfg` (rank 0 of 1), built at its first use, handed out with `seen` / `last_seen` as a new Supply has
    them (the only state a listing reads beside the config; a batch is a function of the listing, step, epoch and rank)."""
    if key not in _SHARED:
        _SHARED[key] = Supply(cfg, 0, 1, DEV, workers=8)
    sup: Supply = _SHARED[key]
    sup.seen[:] = 0
    sup.last_seen[:] = -1
    return sup


def base_supply() -> Supply:
    """CFG: t_a5, t_a6, t_a_supply_batch, t_ab_screen, t_p_probe."""
    return shared('base', CFG)


def quad_supply() -> Supply:
    """CFG4 (cell_parts 4): t_a4, t_a7."""
    return shared('quad', CFG4)


def t_a_supply_batch() -> str:
    sup: Supply = base_supply()
    E: int = sup.steps_an_epoch
    assert E >= 6, (E, sup.budget)
    steps_a: List = list(sup.epoch_listing(0))                 # the whole epoch (no reading; cheap)
    sup.seen[:] = 0                                            # the same state as the first call: nothing seen yet
    sup.last_seen[:] = -1
    steps_b: List = list(itertools.islice(sup.epoch_listing(0), 6))
    assert steps_a[:6] == steps_b, 'the listing is not deterministic'
    assert sup.listed == CFG.data.places and all(len(s) == sup.listed for s in steps_a), 'a step lists places'
    assert E - 1 <= len(steps_a) <= E, (len(steps_a), E)      # a full epoch, at most one step short
    flat: List = [p for s in steps_a for p in s]
    assert len(set(flat)) == len(flat), 'a place listed twice in one epoch'
    per_cell: Counter = Counter((p.slide, p.ds) for p in flat)
    for key, bud in zip(sup.cell_keys, sup.budget):
        assert per_cell[key] <= bud, (key, per_cell[key], bud)
        assert per_cell[key] >= bud - 2 or len(steps_a) < E, (key, per_cell[key], bud)    # the budget is listed unless the epoch ended short
    for s in steps_a:                                          # no two footprints of a step meet, on any slide
        for i, p in enumerate(s):
            assert not any(p.meets(q) for q in s[i + 1:]), 'two places of a step meet'
    # the mix is stationary: every step carries each rung within (the rung's cells + 1) of its rate (budget / E), summed over the rung's cells
    rate: Counter = Counter()
    cells_of: Counter = Counter()
    for (slide, ds), bud in zip(sup.cell_keys, sup.budget):
        rate[ds] += bud / E
        cells_of[ds] += 1
    for s in steps_a[:E - 1]:
        got: Counter = Counter(p.ds for p in s)
        for ds, r in rate.items():
            assert abs(got[ds] - r) <= cells_of[ds] + 1, (ds, got[ds], r)       # each cell is off by less than 1 a step
    first: List = steps_a[0]
    b: Batch = sup.batch(first, 0, 0)
    refs0: torch.Tensor = b.refs.clone()
    assert b.blank == 0 and len(b.slides) == CFG.data.places, (b.blank, len(b.slides))        # the screen at start leaked nothing
    s_n: int = len(sup.sources)
    v: int = s_n * CFG.data.slots
    g_n: int = groups_of(CFG.data, s_n)
    assert v == 8 and b.slots == 2 and g_n == sup.groups == 2, (v, b.slots, g_n)
    n: int = CFG.data.places * v
    assert b.refs.shape == (CFG.data.places, sup.read, sup.read, 3) and b.refs.dtype == torch.uint8, b.refs.shape
    assert len(b.frames) == CFG.data.places * g_n and b.base.shape == (n,) and len(b.shots) == n, (len(b.frames), b.base.shape, len(b.shots))
    for i in range(CFG.data.places):                           # a ref view has no shot, a simulated one has
        for j in range(v):
            assert (b.shots[i * v + j] is None) == (j % s_n == 0), (i, j)
            assert b.shots[i * v + j] is None or isinstance(b.shots[i * v + j], Shot), (i, j)
    views: Views = views_of(b, CFG, DEV)
    assert views.tiles.shape == (n, 256, 256, 3) and views.tiles.dtype == torch.uint8 and views.tiles.device.type == DEV.type, views.tiles.shape
    again_v: Views = views_of(b, CFG, DEV)
    assert torch.equal(again_v.tiles, views.tiles) and torch.equal(again_v.grey, views.grey), 'views_of twice differs (the noise is by seed)'
    assert views.grey.shape[0] == CFG.data.places * CFG.data.slots * CFG.canvases, views.grey.shape
    grey_w: torch.Tensor = torch.tensor(GREY_WEIGHTS, device=DEV)
    worst: float = 0.0
    for i in range(CFG.data.places):
        ref_i: torch.Tensor = b.refs[i].to(DEV)
        a0: int = sup.read // 2 - 128
        assert torch.equal(views.tiles[i * v], ref_i[a0:a0 + 256, a0:a0 + 256]), f'place {i}: the canonical ref\'s input is the scan\'s own centre'
        p0: int = sup.read // 2 - CFG.data.photo_px // 2
        want_grey: torch.Tensor = ref_i[p0:p0 + CFG.data.photo_px, p0:p0 + CFG.data.photo_px].float() @ grey_w
        # the uint8 canvas is the crop bit for bit (the tile above; test_sim_gpu t_k_frame (a)); its grey is three float32 products summed by
        # a matmul whose last bit may depend on the batch's size (the BLAS kernel is chosen by it): 1e-3 grey levels, 60 ulp at 255
        d_grey: float = float((views.grey[i * CFG.data.slots * CFG.canvases] - want_grey).abs().max())
        assert d_grey <= 1e-3, f'place {i}: vref reads the scan\'s own grey canvas (max |d| {d_grey:.2e})'
        worst = max(worst, d_grey)
        cpu_c: torch.Tensor = torch.from_numpy(centre(frame(b.refs[i].numpy(), b.geos[i * v + s_n], CFG.data.photo_px), 256)).to(DEV)
        d: float = float((views.tiles[i * v + s_n].float() - cpu_c.float()).abs().mean())
        assert d <= 1.0, f'place {i}: slot 1 ref tile differs from the CPU framing by mean {d:.3f}'
    assert torch.equal(b.place, torch.arange(CFG.data.places).repeat_interleave(v)), 'a place\'s views contiguous'
    assert torch.equal(b.source, torch.arange(s_n).repeat(2 * CFG.data.places)), 'the sources in order, slot outside'
    for i in range(CFG.data.places):
        gs = b.geos[i * v:(i + 1) * v]
        assert all(g == Geometry() for g in gs[:s_n]), 'slot 0: canonical'
        assert all(g == gs[s_n] for g in gs[s_n:]) and gs[s_n] != Geometry(), 'slot 1: twins, one perturbation'
    banned = set(list_names(dataset='ki67_with_photo'))
    assert not banned & set(sup.names), 'a ki67_with_photo slide is among the training slides'
    pure_train = {n for _, names in by_source('@pure-train', stain='Ki67') for n in names}
    assert set(sup.names) <= pure_train, 'a training slide outside @pure-train Ki67'
    try:
        training_slides(replace(CFG.data, dataset='ki67_with_photo', stain=None))
        raise AssertionError('ki67_with_photo was accepted as training data')
    except ValueError:
        pass
    # a step's rung weights: w sqrt(m) the same on every rung present, mean 1 over the step's places, the absent rungs absent
    counts = {1.0: 100, 4.0: 9, 32.0: 1}
    w = sup.weights_of(counts)
    root = {ds: w[ds] * m ** 0.5 for ds, m in counts.items()}
    assert max(root.values()) - min(root.values()) < 1e-9 * max(root.values()), root
    assert abs(sum(m * w[ds] for ds, m in counts.items()) - 110) < 1e-9 and set(w) == set(counts)
    per_place = [float(b.weight[i]) for i in range(0, len(b.weight), v)]                 # b = sup.batch(first, 0, 0): a weight a place
    assert min(per_place) > 0 and len(set(per_place)) <= 6
    fifth: List = steps_a[5]
    sup.world = 2                                             # a Supply's rank and world are fixed once it made a batch
    try:
        sup.batch(fifth, 5, 0)
        raise AssertionError('a Supply whose world changed after its first batch was accepted')
    except ValueError:
        pass
    sup.world = 1
    # the boards are reused: the same listing, step and epoch again gives the same batch, and the first batch's reads did not change
    def same(x: Batch, y: Batch) -> bool:
        """Whether two batches are the same: the reads, the frames, the views' frames and shots, the geometry."""
        return (torch.equal(x.refs, y.refs) and x.frames == y.frames and torch.equal(x.base, y.base) and x.shots == y.shots
                and x.geos == y.geos)
    again: Batch = sup.batch(first, 0, 0)
    assert same(again, b), 'a reused board changed the batch'
    assert torch.equal(again.source, b.source) and torch.equal(again.place, b.place)
    assert torch.equal(b.refs, refs0), 'the first batch\'s reads changed when the board was reused (the batch must own a copy)'
    for key in ('cand', 'wall', 'pack', 'read', 'work', 'cpu', 'pids'):
        assert key in b.timing, key
    assert b.timing['pids'] >= 1 and b.timing['work'] > 0, b.timing
    # two flights in the air at once: each batch equals the one made alone (the two board sets do not disturb each other)
    second: List = steps_a[1]
    f1: Flight = sup.launch(first, 0, 0)
    f2: Flight = sup.launch(second, 1, 0)
    c1: Batch = sup.land(f1)
    c2: Batch = sup.land(f2)
    alone: Batch = sup.batch(second, 1, 0)
    assert same(c1, b), 'flight 1 differs from the batch alone'
    assert same(c2, alone), 'flight 2 differs from the batch alone'
    STATE['batch'] = b
    STATE['views'] = views
    return (f'{CFG.data.places} places x {v} views {sup.sources}; {len(sup.names)} training slides, {len(sup.places)} places in {len(sup.cells)} cells, '
            f'all @pure-train Ki67, none ki67_with_photo (refused when asked); {E} steps an epoch; '
            f'slot 0 grey max |d| {worst:.1e}')


def t_a4_seen() -> str:
    sup: Supply = quad_supply()
    steps0: List = list(sup.epoch_listing(0))
    e0: List = [p for s in steps0 for p in s]
    seen0 = sup.seen.copy()
    assert seen0.max() == 1 and int(seen0.sum()) == len(steps0) * CFG.data.places, (seen0.max(), seen0.sum())
    e1: List = [p for s in sup.epoch_listing(1) for p in s]
    big = {k for k, idx, bud in zip(sup.cell_keys, sup.cells, sup.budget) if len(idx) >= 2 * bud}
    seen_places = {sup.places[i] for i in range(len(seen0)) if seen0[i] > 0}                # the places counted as seen in epoch 0
    a0 = {p for p in e0 if p in seen_places and (p.slide, p.ds) in big}
    a1 = {p for p in e1 if (p.slide, p.ds) in big}
    assert big and not (a0 & a1), f'{len(a0 & a1)} seen places of a cell with twice its budget listed again in epoch 1'
    assert sup.seen.max() <= 2 and (sup.last_seen[sup.seen > 0] >= 0).all()
    for cell, bud in zip(sup.cells, sup.budget):
        assert bud == -(-len(cell) // 4), (len(cell), bud)
    return f'cell_parts 4: budget ceil(pool / 4); epoch 0 saw {int(seen0.sum())} places once; epoch 1 took unseen ones in {len(big)} big cells'


def t_a5_step_sizes() -> str:
    """Regression: every step an epoch yields lists exactly `listed` places (a negative ask or a short trim once made some larger)."""
    sup: Supply = base_supply()
    steps_a: List = list(sup.epoch_listing(0))
    sizes: Counter = Counter(len(s) for s in steps_a)
    assert set(sizes) == {sup.listed}, dict(sizes)
    assert len(steps_a) >= sup.steps_an_epoch - 2, (len(steps_a), sup.steps_an_epoch)
    return f'{len(steps_a)} of {sup.steps_an_epoch} steps, every one of {sup.listed} places'


def t_a6_pool() -> str:
    """The pool: at most `pool_cap` places a cell, and all training lists come from it."""
    sup: Supply = base_supply()
    draw: Counter = Counter((p.slide, p.ds) for p in sup.places)
    for key, cell in zip(sup.cell_keys, sup.cells):
        assert 0 < len(cell) <= min(128, draw[key]), (key, len(cell), draw[key])
        assert len(set(cell)) == len(cell) and all((sup.places[i].slide, sup.places[i].ds) == key for i in cell), key
    pool_places: Set[Any] = {sup.places[i] for cell in sup.cells for i in cell}
    assert len(pool_places) == sum(len(c) for c in sup.cells), 'a place in two cells'
    listed: Set[Any] = {p for s in sup.epoch_listing(0) for p in s}
    assert listed <= pool_places, 'an epoch listed a place outside the pool'
    return f'{len(sup.cells)} cells, pools {sum(len(c) for c in sup.cells)} of {len(sup.places)} drawn places, every epoch inside the pool'


def t_a7_state() -> str:
    """A Supply's state (seen, last_seen), saved and loaded again (by the same Supply and by a newly built one), gives the same next epoch."""
    a: Supply = quad_supply()
    list(a.epoch_listing(0))
    st: Dict[str, Any] = a.state()
    e1: List = list(a.epoch_listing(1))
    b: Supply = Supply(CFG4, 0, 1, DEV, workers=8)                                   # a resumed run: a new object takes the saved state
    try:
        b.load_state(st)
        assert list(b.epoch_listing(1)) == e1, 'a new Supply with a loaded state gives another listing'
    finally:
        b.close()
    a.load_state(st)                                                                 # back to the end of epoch 0
    assert list(a.epoch_listing(1)) == e1, 'a loaded state gives another listing'
    a.seen[:] = 0                                                                    # without the state
    a.last_seen[:] = -1
    assert list(a.epoch_listing(1)) != e1, 'without the state the listing is the same: the state does not matter'
    short: Dict[str, torch.Tensor] = {k: torch.zeros(3, dtype=torch.int32) for k in ('seen', 'last_seen')} | {'pool': st['pool']}
    try:
        a.load_state(short)
        raise AssertionError('a state of 3 places was accepted')
    except ValueError:
        pass
    no_pool: Dict[str, torch.Tensor] = {k: v for k, v in st.items() if k != 'pool'}
    try:
        a.load_state(no_pool)
        raise AssertionError('a state without the pool checksum was accepted')
    except ValueError:
        pass
    other_pool: Dict[str, torch.Tensor] = st | {'pool': st['pool'] + 1}
    try:
        a.load_state(other_pool)
        raise AssertionError('a state of another pool was accepted')
    except ValueError:
        pass
    return ('load_state(state()) of epoch 0 gives the epoch 1 listing again, on the same Supply and on a new one; without the state another; '
            'a wrong length, a missing or another pool checksum are refused')


def t_ab_screen() -> str:
    """The screen at start: its verdicts, that its centre tile is the full read's, the leak path (`land`)."""
    sup: Supply = base_supply()
    everyone: List[int] = [i for c in sup.cells for i in c]
    sample: List[int] = everyone[::max(1, len(everyone) // 64)][:64]
    assert len({sup.places[i].ds for i in sample}) > 1, 'the sample covers one rung only'
    assert sup.screen(sample) == [USABLE] * len(sample), 'a place of the pool does not screen as usable'
    # the screen's small square is the centre of the full read (the resampling rungs are where it could differ)
    tile: int = CFG.cofig.tile
    side: int = tile + 2 * SCREEN_PAD
    readers: Dict[int, SlideReader] = {}
    worst: Dict[float, int] = {}
    for i in sample:
        p = sup.places[i]
        if p.slide not in readers:                                  # one reader a slide (setdefault would open one a place)
            readers[p.slide] = SlideReader(SafeSlide(sup.paths[p.slide]))
        r: SlideReader = readers[p.slide]
        s = sup.spot(p)
        full: Optional[Any] = read_spot(r, s, sup.read)
        off: int = int(round((sup.read - side) // 2 * s.ds))
        small: Optional[Any] = r.read(s.x0 + off, s.y0 + off, ReadSpec(side, side), s.ds, level=s.level)
        assert full is not None and small is not None, (p.ds, i)
        a, b_ = centre(small, tile), centre(full, tile)
        if not (a == b_).all():
            worst[p.ds] = max(worst.get(p.ds, 0), int(abs(a.astype('int16') - b_.astype('int16')).max()))
    levels: Dict[int, List[float]] = {s_: r_.level_downsamples for s_, r_ in readers.items()}
    assert not worst, f'the screen\'s centre tile differs from the full read\'s: max pixel difference by rung {worst}; level_downsamples by slide {levels}'
    # a place that comes back blank or off the slide after all: left out and counted, nothing raised
    first: List = next(sup.epoch_listing(0))
    n_p: int = CFG.data.places
    v: int = sup.versions
    base: Batch = sup.batch(first, 0, 0)
    f: Flight = sup.launch(first, 0, 0)
    for x in f.futs:
        x.result()
    gone: List[int] = [1, 4, 6]
    for i in gone:
        done: Future = Future()
        done.set_result(Made(BLANK, [], [], 0, 0.0, 0.0, 0.0))
        f.futs[i] = done
    b: Batch = sup.land(f)
    keep: List[int] = [i for i in range(n_p) if i not in gone]
    assert len(b.slides) == n_p - 3 and b.blank == 3, (len(b.slides), b.blank)
    assert torch.equal(b.place, torch.arange(n_p - 3).repeat_interleave(v))
    rows: List[int] = [i * v + j for i in keep for j in range(v)]
    assert torch.equal(b.refs, base.refs[keep]), 'the other places\' reads changed'
    assert [base.geos[r] for r in rows] == b.geos and [base.shots[r] for r in rows] == b.shots, 'the other places\' views changed'
    g_n: int = sup.groups
    assert b.frames == [(n_, g) for n_, i in enumerate(keep) for _, g in base.frames[i * g_n:(i + 1) * g_n]], 'the frames of the other places changed'
    assert torch.equal(b.base, torch.tensor([n_ * g_n + int(base.base[i * v + j]) - i * g_n for n_, i in enumerate(keep) for j in range(v)])), 'a view\'s frame'
    w_mean: float = float(b.weight[::v].mean())
    assert abs(w_mean - 1.0) < 1e-5, w_mean
    f = sup.launch(first, 0, 0)
    for x in f.futs:                                                # the workers are done with the board before their results are replaced
        x.result()
    for i in range(n_p):
        done = Future()
        done.set_result(Made(BLANK, [], [], 0, 0.0, 0.0, 0.0))
        f.futs[i] = done
    b0: Batch = sup.land(f)
    assert b0.refs.shape == (0, sup.read, sup.read, 3) and b0.base.shape == (0,) and b0.frames == [] and b0.shots == [], b0.refs.shape
    assert b0.source.dtype == torch.int64 and b0.place.dtype == torch.int64 and b0.base.dtype == torch.int64 and b0.weight.shape == (0,)
    assert b0.blank == n_p and b0.slides == []
    v0: Views = views_of(b0, CFG, DEV)
    assert v0.tiles.shape == (0, 256, 256, 3) and v0.grey.shape == (0, CFG.data.photo_px, CFG.data.photo_px), (v0.tiles.shape, v0.grey.shape)
    return (f'{len(sample)} pool places screen as usable and their screened tile is the full read\'s centre; 3 of {n_p} places back blank: '
            f'left out, counted, the rest unchanged; all {n_p}: an empty batch')


def t_b_targets() -> str:
    b: Batch = STATE['batch']
    views: Views = STATE['views']
    tg, p_mean, p_w = targets_of(b, views.grey, CFG, DEV, 224)
    n: int = len(b.shots)
    oc: int = CFG.cofig.out_cells                                     # the Co-figure's cells, from the encoder (gigapath 28, uni2 32)
    nch: int = n_channels(CFG.cofig)
    assert tg.mean.shape == (n, nch, oc, oc) and torch.isfinite(tg.mean).all() and bool(tg.valid.all()), tg.mean.shape
    v: int = CFG.data.slots * len(b.sources)
    x = tg.mean.view(-1, CFG.data.slots, len(b.sources), nch, oc, oc)
    assert torch.equal(x, x[:, :, :1].expand_as(x)), 'vref: the 4 views of a slot share the target'
    assert p_mean.shape == (n, 29, 4, 4) and torch.isfinite(p_mean).all(), p_mean.shape
    pv = p_mean.view(-1, v, 29, 4, 4)
    assert torch.equal(pv, pv[:, :1].expand_as(pv)), 'mix: the 8 views of a place share the descriptor'
    STATE['tg'] = tg
    return f'mean {tuple(tg.mean.shape)} finite, valid all True, a slot\'s views share it; descriptor {tuple(p_mean.shape)}, a place\'s {v} views share it'


def t_c_net_frozen() -> str:
    b: Batch = STATE['batch']
    tiles: torch.Tensor = STATE['views'].tiles
    net = CoaeNet(CFG, DEV, n_sources=len(b.sources)).to(DEV)
    assert net.field_px == 224, net.field_px
    assert not any(n.startswith('disc') for n, _ in net.named_parameters()), 'w_adv 0: no discriminator'
    assert net.fm.depth == 40 and net.fm.blocks == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39) and net.fm.last == 40, (net.fm.depth, net.fm.blocks)
    net.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = net(tiles[:8])
        o_b = net(tiles[:8])
    oc: int = CFG.cofig.out_cells
    side: int = CFG.latent_side
    assert o.cofig.shape == (8, n_channels(CFG.cofig), oc, oc) and net.cls_dim == net.fm.dim and o.cls.shape == (8, net.fm.dim)
    assert o.latent.shape == (8, side, side, CFG.model.latent_dim), o.latent.shape
    assert o.pooled.shape == (8, 29, 4, 4) and o.cls_fm.shape == (8, 1536) and o.avg_fm.shape == (8, 1536)
    assert torch.equal(o.cofig, o_b.cofig) and torch.equal(o.latent, o_b.latent), 'eval: no noise, two forwards alike'
    net.train()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        t1, t2 = net(tiles[:8]), net(tiles[:8])
    assert not torch.equal(t1.cofig, t2.cofig) and torch.equal(t1.latent, t2.latent), 'train: the noise is after the latent'
    net.eval()
    raw, cls_fm, avg_fm = net.fm(tiles[:2])
    assert cls_fm.shape == (2, net.fm.dim) and avg_fm.shape == (2, net.fm.dim), cls_fm.shape
    assert raw[0].shape[1:3] == (14, 14), raw[0].shape
    head, lora = net.trainable()
    assert not lora and all(not p.requires_grad for p in net.fm.vit.parameters())
    st = own_state(net)
    assert not any(k.startswith('fm.vit.') for k in st), 'the frozen FM is not in the checkpoint'
    n_ck: int = sum(v.numel() for v in st.values())
    STATE['net'] = net
    return (f'gigapath tokens {tuple(raw[0].shape[1:3])}, latent {tuple(o.latent.shape[1:])}; cofig {tuple(o.cofig.shape)}, cls {tuple(o.cls.shape)}; '
            f'checkpoint {n_ck / 1e6:.1f} M values, {sum(p.numel() for p in head) / 1e6:.1f} M trainable')


def t_cc_residual() -> str:
    b: Batch = STATE['batch']
    tiles: torch.Tensor = STATE['views'].tiles
    cfg = replace(CFG, model=replace(CFG.model, cls_residual=True))
    net = CoaeNet(cfg, DEV, n_sources=len(b.sources)).to(DEV).eval()
    with torch.no_grad():
        o = net(tiles[:8])
        _, cls_fm, _ = net.fm(tiles[:8])
        start = torch.nn.functional.cosine_similarity(o.cls.float(), cls_fm.float(), dim=1)
    assert float(start.min()) > 0.999, f'cls\' does not start as exactly the FM CLS: {start.tolist()}'
    del net
    torch.cuda.empty_cache()
    return f'cls\' starts as exactly the FM CLS, no projection (cos >= {float(start.min()):.4f})'


def t_cd_losses_reach() -> str:
    """Which loss reaches what: recon_point the latent (not cls'); recon_pooled cls' AND the latent (nothing detached in between)."""
    b: Batch = STATE['batch']
    net: CoaeNet = STATE['net']                                  # t_c's network, in training mode for this test only
    net.train()
    try:
        k: int = 8
        tiles: torch.Tensor = STATE['views'].tiles
        tg, p_mean, p_w = targets_of(b, STATE['views'].grey, CFG, DEV, net.field_px)
        sub = Targets(tg.mean[:k], tg.obs[:k], tg.valid[:k])
        sc = ChannelScale(n_channels(CFG.cofig)).to(DEV)
        sc.update(sub.mean, sub.valid)
        net.see_places(p_mean[:k])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = net(tiles[:k])
        l_rec = recon_point(o.cofig.float(), sub, sc)
        l_pool = recon_pooled(o.pooled.float(), p_mean[:k], p_w[:k], net.pool_sd)
        gl, gc = torch.autograd.grad(l_rec, [o.latent, o.cls], retain_graph=True, allow_unused=True)
        assert gl is not None and torch.isfinite(gl).all() and float(gl.abs().sum()) > 0, 'recon_point does not reach the latent'
        assert gc is None or float(gc.abs().sum()) == 0, 'recon_point reaches cls\''
        pl, pc = torch.autograd.grad(l_pool, [o.latent, o.cls], retain_graph=True, allow_unused=True)
        assert pc is not None and torch.isfinite(pc).all() and float(pc.abs().sum()) > 0, 'recon_pooled does not reach cls\''
        assert pl is not None and torch.isfinite(pl).all() and float(pl.abs().sum()) > 0, 'recon_pooled does not reach the latent'
        msg: str = (f'recon_point -> latent {float(gl.norm()):.2e}, cls\' none; recon_pooled -> cls\' {float(pc.norm()):.2e} and latent '
                    f'{float(pl.norm()):.2e}')
    finally:
        net.eval()
    torch.cuda.empty_cache()
    return msg


def t_p_probe() -> str:
    sup: Supply = base_supply()
    net: CoaeNet = STATE['net']
    val: Supply = Supply(replace(CFG, data=replace(CFG.data, dataset='@pure-half-val', slides=1)), 0, 1, DEV, workers=8, pool=False)
    assert sum(len(c) for c in val.cells) == len(val.places), 'the probe\'s Supply is limited by a pool'
    try:                                                                             # a places-alone Supply lists no epoch
        val.steps_an_epoch
        raise AssertionError('a Supply for its places alone listed an epoch')
    except RuntimeError:
        pass
    assert not set(val.names) & set(sup.names), 'a probe slide is a training slide'
    assert not set(list_names(dataset='ki67_with_photo')) & set(val.names), 'a probe slide has a real photo'
    pcfg = replace(CFG, probe=replace(CFG.probe, places=4, rungs=(4.0, 8.0)))
    pr = val.probe_set(pcfg, 224, DEV)
    p: int = len(pr.cell)
    labs = len(val.probe_labs)
    assert pr.tiles.shape == (p, 4 + 2 * labs, 256, 256, 3) and pr.kinds[0] == 'anchor' and len(pr.kinds) == 14 == len(set(pr.kinds)), pr.kinds
    assert pr.direct.shape == (p, 14, 29, 4, 4) and torch.isfinite(pr.direct).all(), pr.direct.shape
    oc: int = CFG.cofig.out_cells
    sides: List[int] = list(pooled_sides(CFG))                          # the pooled Co-figure columns: 2, 4 and the latent's side
    pooled_names: List[str] = [f'cofig-direct{g}' for g in sides]
    assert pr.maps.shape == (p, 14, n_channels(CFG.cofig), oc, oc) and torch.isfinite(pr.maps.float()).all(), pr.maps.shape
    again = val.probe_set(pcfg, 224, DEV)
    assert torch.equal(pr.tiles, again.tiles), 'the probe is not the same twice'
    val.close()
    sc = ChannelScale(n_channels(CFG.cofig)).to(DEV)
    pm = probe_measures(net, pr, pcfg, sc, DEV)
    feats = pm['features']
    n_feats: int = 2 * (19 + len(sides))                                # 15 (head, fm+head among them) + the pooled columns + 4 signals, each as it is and -m
    assert len(feats) == n_feats and all(len(v) == 13 for v in feats.values()), {k: len(v) for k, v in feats.items()}
    assert all(0.0 <= s['hit'] <= 1.0 for v in feats.values() for s in v.values())
    assert set(pm['map_direct']) == {kd for kd in pr.kinds if kd.startswith('twin:')}, set(pm['map_direct'])
    for kd, v in pm['map_direct'].items():
        assert set(v) == set(part_names(CFG.cofig)) and all(-1.0 - 1e-4 <= c <= 1.0 + 1e-4 for c in v.values()), (kd, v)
    assert len(pm['groups']) == 3 and {n for _, g_ in pm['groups'] for n in g_} == {n for n in feats if not n.endswith('-m')}, pm['groups']
    assert len(pm['groups'][1][1]) == 6 + len(sides) and set(pooled_names) <= set(pm['groups'][1][1]), pm['groups'][1]
    assert set(pm['anchors']) == set(feats), set(pm['anchors']) ^ set(feats)
    for name, s in pm['anchors'].items():
        assert -1.0 - 1e-4 <= s['neg'] <= s['nn'] <= 1.0 + 1e-4, (name, s)
    assert pm['anchors']['direct']['nn'] < 0.9999, pm['anchors']['direct']
    if CFG.data.dilate_px == 0 and CFG.head.pool_cells == 4:
        pooled: torch.Tensor = TF.avg_pool2d(pr.maps.float().flatten(0, 1), oc // 4)
        gap: float = float((pooled - pr.direct.flatten(0, 1)[:, :n_channels(CFG.cofig)]).abs().max())
        assert gap < 2e-3, gap
        gap_line: str = f'the 4 x 4 mean of the maps against direct\'s first moments: max|d| {gap:.2e}'
    else:
        gap_line = 'the 4 x 4 mean of the maps against direct: skipped (dilate_px or pool_cells not 0 / 4)'
    print(gap_line, flush=True)
    sib: float = feats['cofig-direct']['sib:interp']['hit']
    assert sib >= 0.75, ('cofig-direct on sib:interp', sib)
    table: str = LL.probe_table(pm)
    print(table, flush=True)
    print(f'the table\'s widest line: {max(len(x) for x in table.splitlines())} characters', flush=True)
    torch.cuda.empty_cache()
    hits: str = ', '.join(f'{kd} {feats["cofig-direct"][kd]["hit"]:.2f}' for kd in ('sib:interp', 'sib:large', 'twin:ood-heavy'))
    return (f'{p} places in {len(pr.cells)} cells x 14 views, disjoint from the training slides and the photo slides, the same twice; '
            f'40 features x 13 kinds; cofig-direct hit {hits}; '
            f'{gap_line}')


_TESTS = [t for n, t in sorted(globals().items()) if n.startswith('t_')]


def main() -> int:
    try:
        for fn in _TESTS:
            check(fn.__name__[4:].replace('_', ' '), fn)
    finally:
        for sup in _SHARED.values():                                                 # the two shared Supplies: their workers and boards
            sup.close()
        _SHARED.clear()
    failed: List[str] = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
