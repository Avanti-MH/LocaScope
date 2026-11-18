'''Synthetic FoVs through the first phase, made as they are used.

The FoVs are not stored. A slide is opened, its tissue mask read, a draw of FoV
positions made over every level, and -- one LEVEL at a time, a visit -- the level's
features are taken from the host store (`MemoryFeatureStore`, filled by `precompute_levels`
before the first epoch: read from the disk cache `LevelDiskCache`, or encoded and written to it), photographs are
rendered, and each goes through the retriever, whose candidates and features are
handed to the caller one FoV after the next. The next slide replaces it all.

    stream = fov_stream(slides, make_retriever, fov, mask_job, device, label_cfg, ...)
    for res in stream:                       # a FovResult
        data = prepare_fov(res, ...)         # BEFORE asking for the next one

A `FovResult` is only valid until the next one is asked for: `res.source` (the
retriever's `FEATURES` extra, a `GridSource`) holds this query's features.
`prepare_fov` copies what a step needs.

WHAT A "VISIT" COSTS. Encoding a level is the expensive part (about two to three
minutes for a level of a few hundred thousand tiles; one slide's levels, about
four), and everything after is cheap, so `per_visit` FoVs are taken at once and a
caller trains on them several times before the level is dropped.
'''
from __future__ import annotations

import re
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from locascope.core.cache.Store import FeatureMapCache, FeatureMeta, FeatureStore, _ds_name, to_store_tensors
from locascope.core.config.ConfigIdentity import record as identity_record
from locascope.core.tiles.PatchingLib import FeaturesMap, WsiFeaturesMap, region_grids
from locascope.core.slides.SafeSlide import SafeSlide
from locascope.core.slides.SlideReader import SlideReader
from locascope.core.tiles.TileSampler import PlanSpec
from locascope.core.tiles.TissueMaskConfig import MASK_RECIPES, MaskMaker
from locascope.core.query_sim.FovSupply import FovRecipe, FovSupply
from locascope.core.query_sim.camera import Render
from locascope.cli.bench.BenchCommon import (Answers, Truth, answers, matching_rotation, placed_estimate, query_grid_centre,
                                             stage_hexid)
from locascope.core.stage2_retrieval.Context import WindowBatch, window_boxes
from locascope.core.stage2_retrieval.RerankerNeural import NeuralInputs, neural_inputs, pair_features
from locascope.core.stage2_retrieval.Retriever import RetrievalOutput
from locascope.core.stage2_retrieval.StageInterface import CandidateSet

from locascope.training.RerankHead.Labels import Boxes, LabelConfig, Labels, choose_subset, label_windows, truth_boxes


# ══════════════════════════════════════════════════════════════════════════════
#  where the time goes, and what is kept between epochs
# ══════════════════════════════════════════════════════════════════════════════

_TIMERS: Dict[str, float] = defaultdict(float)
_TIMERS_LOCK: threading.Lock = threading.Lock()


def tick(name: str, seconds: float) -> None:
    """Add `seconds` to the timer `name` (any thread)."""
    with _TIMERS_LOCK:
        _TIMERS[name] += float(seconds)


def take_timers() -> Dict[str, float]:
    """The timers since the last call, and zero them."""
    with _TIMERS_LOCK:
        out: Dict[str, float] = dict(_TIMERS)
        _TIMERS.clear()
    return out


FP16_MAX: float = 65504.0


class LevelDiskCache(FeatureMapCache):
    """The levels of ONE slide on disk, `result/cache/<job>/slide=/seg=/region=/grid=t<tile>-o<0|1>/features/`: the repo's
    `FeatureMapCache` (the same entry, record, atomic write -- a staged directory moved into place, the record last --
    and geometry check), with the retriever's CONFIG in the encoder's place. A variant is named
    `ds<d>-<feature>-<encoder>-<stage id>`: the stage id (`stage_hexid`) is the config's identity and the CONTENT of
    the checkpoint it reads, so another checkpoint or another usage is another file and the same ones are read by
    every training that uses them. Held in fp16, as the memory store holds them; `load` hands them back in fp16."""

    def __init__(self, job: str, wsi_path: str, cfg: Any, mask_cfg: Any) -> None:
        super().__init__(job, wsi_path, cfg, mask_cfg, mode='rw', verbose=False)
        named: str = str(getattr(cfg, 'arm_encoder', '') or getattr(cfg.encoder, 'REGISTERED_AS', '') or 'enc')
        self._tag: str = re.sub(r'[^A-Za-z0-9._,-]', '_', named)
        self._stage: str = stage_hexid(cfg)

    @property
    def pooling(self) -> str:
        return str(getattr(self.encoder, 'arm_pooling', '') or self.encoder.encoder.pooling)

    def record(self) -> Dict[str, Any]:
        if self._record is None:
            self._record = identity_record(self.encoder, also=(FeatureStore, SlideReader, self.mask_cfg),
                                           seg_id=self.seg_id, region_id=self.region_id, stage=self._stage)
        return self._record

    def fid(self, ds: float, pooling: Optional[str] = None) -> str:
        return '-'.join([_ds_name(ds), pooling or self.pooling, self._tag, self._stage])

    def _storage_dtype(self) -> torch.dtype:
        return torch.float16

    def _meta(self, *, level: int, ds: float, tile_size: int, overlap: bool, pooling: str, slots: Tuple[str, ...],
              slot_layout: str, dim: int, n: int) -> FeatureMeta:
        return FeatureMeta(
            wsi_stem=self.wsi_stem, wsi_path=self.wsi_path, level=int(level), ds=float(ds), mpp=0.0, base_mpp=0.0,
            tile_size=int(tile_size), overlap=bool(overlap), pooling=pooling, slots=tuple(slots),
            slot_layout=slot_layout, dim=int(dim), feat_hw=None, num_prefix=0, encoder_id=self._stage,
            seg_id=self.seg_id, region_id=self.region_id, coverage='grid', n_available=int(n), n_tiles=int(n))

    def load(self, regions: Sequence[Any], *, ds: float, level: int, tile_size: int, overlap: bool
             ) -> Optional[WsiFeaturesMap]:
        """The level in fp16 on the host (the base class would make an fp32 copy of it first), or None on a miss."""
        geo: Dict[str, Any] = dict(ds=ds, level=level, tile_size=tile_size, overlap=overlap)
        if self.check(regions, self.pooling, **geo) is None:
            return None
        tensors, _ = FeatureStore.load(self.path(ds=ds, tile_size=tile_size, overlap=overlap), keys=('features',))
        flat: torch.Tensor = tensors['features']
        grids: List[Any] = region_grids(regions, **geo)
        want: int = sum(len(g) for g in grids)
        if flat.ndim != 3 or flat.shape[1] != 1 or flat.dtype != torch.float16 or flat.shape[0] != want:
            raise ValueError(f'{self.path(ds=ds, tile_size=tile_size, overlap=overlap)}: features '
                             f'{tuple(flat.shape)} {flat.dtype}, wanted [{want}, 1, D] float16')
        flat = flat[:, 0]
        maps: List[FeaturesMap] = []
        at: int = 0
        for grid in grids:
            maps.append(FeaturesMap(grid, flat[at:at + len(grid)], source='LevelDiskCache'))
            at += len(grid)
        return WsiFeaturesMap(list(regions), maps, **geo)

    def file_bytes(self, *, ds: float, tile_size: int, overlap: bool) -> int:
        """Size of the level's file, 0 when there is none."""
        p: Path = self.path(ds=ds, tile_size=tile_size, overlap=overlap)
        return int(p.stat().st_size) if p.is_file() else 0


class MemoryFeatureStore:
    """The encoded levels of ONE slide, kept on the host in fp16 between epochs: the retriever's `feature_store`
    (`Context.encode_level`). The features of a level do not change from one epoch to the next (the slide, the
    encoder and the pooling are the same; only the FoVs are new), and encoding them is most of the time. `save`
    copies a level's map to the host in fp16 (refusing a value that is not finite or not an fp16 number) and, when
    a `disk` cache is given, writes it there; `load` hands it back when the level, ds, tile cut, lattices and regions
    are the same -- from memory, else from `disk` -- and the retriever moves it to the device and scores it in fp32.
    It says it holds the levels of one retriever (`single_retriever`: its key has no encoder in it), so a retriever
    whose features are not its encoder's (CoAE's) may keep its levels in it."""

    single_retriever: bool = True

    def __init__(self, disk: Optional[LevelDiskCache] = None, who: str = '') -> None:
        self._maps: Dict[Tuple[Any, ...], Any] = {}
        self.disk: Optional[LevelDiskCache] = disk
        self.who: str = who
        self.hits: int = 0                  # a level found (in memory or on disk)
        self.misses: int = 0                # a level not found: it is encoded
        self.from_disk: int = 0
        self.written: int = 0
        self.disk_bytes: int = 0            # the files this store read or wrote

    @staticmethod
    def _key(regions: Sequence[Any], ds: float, level: int, tile_size: int, overlap: bool) -> Tuple[Any, ...]:
        return (int(level), float(ds), int(tile_size), bool(overlap),
                tuple((int(r.x), int(r.y), int(r.w), int(r.h)) for r in regions))

    def load(self, regions: Sequence[Any], *, ds: float, level: int, tile_size: int, overlap: bool) -> Optional[Any]:
        key: Tuple[Any, ...] = self._key(regions, ds, level, tile_size, overlap)
        found: Optional[Any] = self._maps.get(key)
        if found is None and self.disk is not None:
            found = self.disk.load(regions, ds=ds, level=level, tile_size=tile_size, overlap=overlap)
            if found is not None:
                self._maps[key] = found
                self.from_disk += 1
                self.disk_bytes += self.disk.file_bytes(ds=ds, tile_size=tile_size, overlap=overlap)
        self.hits += found is not None
        self.misses += found is None
        return found

    def save(self, features: Any) -> None:
        key: Tuple[Any, ...] = self._key(features.regions, features.ds, features.level, features.tile_size,
                                         features.overlap)
        for m in features.maps:             # on the device the encoder left them on: no copy made for the check
            if m.features.numel():
                lo, hi = m.features.aminmax()
                if not (float(lo) >= -FP16_MAX and float(hi) <= FP16_MAX):
                    raise FloatingPointError(f'{self.who} L{features.level}: features outside fp16 or not finite '
                                             f'(min {float(lo)}, max {float(hi)})')
        host: Any = features.to('cpu', torch.float16)
        self._maps[key] = host
        if self.disk is not None and host.n_patches():
            self.disk.save(host)
            self.written += 1
            self.disk_bytes += self.disk.file_bytes(ds=features.ds, tile_size=features.tile_size,
                                                    overlap=features.overlap)

    def nbytes(self) -> int:
        return sum(int(m.features.numel()) * int(m.features.element_size())
                   for maps in self._maps.values() for m in maps.maps)


@dataclass
class FovResult:
    """One FoV and what the retriever made of it. Valid until the next one."""
    slide: str
    dataset: str
    level: int
    ds: float
    index: int
    img: np.ndarray
    cs: CandidateSet
    retriever: Any
    out: Any                      # what the retriever handed back (a chain's: its last phase's)
    source: Any                   # the first phase's features (its `FEATURES` extra): this query's and this level's features
    where: Boxes
    boxes: np.ndarray             # [K, 4] candidate boxes
    labels: Labels
    queries: Dict[int, torch.Tensor]   # rotation -> the query's main tile grid, copied


def shape_fn(source: Any) -> Callable[[int], Tuple[int, int]]:
    """`(rows, cols)` of a window at a rotation, from the query as the retriever cut it (`source`, a
    `GridSource`)."""
    def shape_of(rotation: int) -> Tuple[int, int]:
        g: Any = source.query.containers[rotation].grid
        return int(g.grid_rows), int(g.grid_cols)
    return shape_of


@dataclass(frozen=True)
class Placed:
    """Where a photo stands on its slide, whatever says so (a render's geometry, a real photo's homography)."""
    truth: Truth                               # level, ds, rot, centre(tile)
    fov_centre: Tuple[float, float]            # the photo's centre, level-0
    fov_w0: float                              # the photo's sides, level-0 px
    fov_h0: float


def searched(retriever: Any, wsi: SafeSlide, slide: str, dataset: str, index: int, img: np.ndarray, placed: Placed,
             label_cfg: LabelConfig, topk: Optional[int], say: Callable[[str], None],
             encoded: set) -> Optional[FovResult]:
    """One photo through the retriever, as a FovResult; None when it is passed over (the retriever could not search
    it, its list is empty or at another level, or no main window is near enough to be an answer)."""
    truth: Truth = placed.truth
    t0: float = time.perf_counter()
    try:
        out: RetrievalOutput = retriever.retrieve(
            placed_estimate(truth.ds, truth.level, wsi.base_mpp), img, topk=topk)
    except Exception as exc:                                  # noqa: BLE001
        say(f'  [{index}] L{truth.level}: {type(exc).__name__}: {exc} -- skipped')
        return None
    finally:
        tick('retrieve', time.perf_counter() - t0)
    for lv, (seconds, tiles) in dict(getattr(retriever, 'level_encodes', {}) or {}).items():
        if lv not in encoded:                     # the encoding (or the read back) is inside `retrieve`
            encoded.add(lv)
            tick('level_encode', seconds)
            tick('level_tiles', tiles)
    cs: CandidateSet = out.candidates
    if not len(cs) or cs.level != truth.level:
        return None
    # the first phase's features: the frame and the query the list was made in (a chain's rerankers
    # may offer features of their own, nearer)
    src: Any = out.find(type(retriever).FEATURES)[-1].value
    shape_of: Callable[[int], Tuple[int, int]] = shape_fn(src)
    tile: int = int(cs.grids[0].tile_size)
    rows, cols = shape_of(truth.rot)
    # the answers, as the benches define them; a FoV with none (no main window near enough) is passed
    # over, as the bench passes it
    found: Optional[Answers] = answers(cs, (rows, cols), truth.centre(tile), truth.rot)
    if found is None:
        return None
    where: Boxes = truth_boxes(truth.centre(tile), truth.rot, rows, cols, tile, cs.ds,
                               placed.fov_centre, placed.fov_w0, placed.fov_h0,
                               label_cfg, [(a.key, a.origin) for a in found])
    boxes: np.ndarray = window_boxes(cs, shape_of)
    tol_px: float = float(getattr(retriever.cfg, 'min_sep_tiles', 1.0)) * tile * cs.ds
    labels: Labels = label_windows(boxes, np.array([c.rotation for c in cs.candidates]),
                                   [c.key() for c in cs.candidates], where, tile * cs.ds, label_cfg,
                                   tol_px)
    queries: Dict[int, torch.Tensor] = {
        rot: src.query_grid(rot, src.pooling).detach().clone() for rot in (0, 90, 180, 270)}
    return FovResult(slide=slide, dataset=dataset, level=truth.level, ds=float(cs.ds), index=index, img=img,
                     cs=cs, retriever=retriever, out=out, source=src, where=where, boxes=boxes,
                     labels=labels, queries=queries)


def draw_fovs(reader: SlideReader, fov: FovRecipe, mask: Any) -> Tuple[FovSupply, List[Tuple[int, Any]]]:
    """The FoV supply of a slide and its drawn positions `(index, meta)` over every level, by `fov`'s sampler (its seed
    is the draw's). Raises RuntimeError when the mask holds no position at any level."""
    microscope: Render = Render(reader, fov.sensor, fov.gap, ds=1.0, seed=fov.sampler.seed)
    plan: PlanSpec = PlanSpec('ladder', fov.rungs_for(reader.level_downsamples), camera=microscope.spec)
    supply: FovSupply = FovSupply(microscope, plan, fov.sampler, mask)
    return supply, [(i, s.meta) for i, s in enumerate(supply.sampler)]


def drawn_levels(reader: SlideReader, fov: FovRecipe, mask: Any) -> List[int]:
    """The levels `fov`'s draw puts a FoV at, on this slide."""
    return sorted({int(meta.level) for _, meta in draw_fovs(reader, fov, mask)[1]})


def precompute_levels(slides: Sequence[Tuple[str, str, str]], make_retriever: Callable[[str], Any], mask_job: str,
                      device: Any, levels_of: Callable[[str, SlideReader, Any], Sequence[int]],
                      say: Callable[[str], None] = print) -> Tuple[int, int]:
    """Before the first epoch: every level `levels_of(name, reader, mask)` names, of each slide in `slides`, into the
    slide's `MemoryFeatureStore` (the retriever's `feature_store`), read from the disk cache or encoded (and written
    there). Prints each level's source, tiles, seconds and size. Returns `(levels read from the cache, levels
    encoded)`. A level the retriever cannot read is passed over (its visit would find the same); a value that is not
    finite stops the run."""
    masks: MaskMaker = MaskMaker(MASK_RECIPES['hest'], mask_job, device)
    cached: int = 0
    encoded: int = 0
    for dataset, name, path in slides:
        started: float = time.perf_counter()
        wsi: SafeSlide = SafeSlide(path)
        mask: Any = masks.mask(wsi)[0]
        reader: SlideReader = SlideReader(wsi)
        try:
            levels: List[int] = sorted({int(lv) for lv in levels_of(name, reader, mask)})
        except RuntimeError as exc:
            say(f'precompute {dataset} {name}: no FoV position at any level -- skipped ({str(exc).splitlines()[0]})')
            continue
        retriever: Any = make_retriever(name)
        retriever.build(wsi, mask)
        store: MemoryFeatureStore = retriever.search.feature_store
        for level in levels:
            disk0: int = store.from_disk
            bytes0: int = store.nbytes()
            seconds: float
            tiles: int
            try:
                seconds, tiles = retriever.search.precompute(level)
            except FloatingPointError:
                raise
            except Exception as exc:                              # noqa: BLE001
                say(f'precompute {dataset} {name} L{level}: {type(exc).__name__}: {exc} -- skipped')
                continue
            was_cached: bool = store.from_disk > disk0
            cached += was_cached
            encoded += not was_cached
            say(f'precompute {dataset} {name} L{level}: {"read from the cache" if was_cached else "newly encoded"}, '
                f'{tiles} tiles, {seconds:.1f} s, +{(store.nbytes() - bytes0) / 1e9:.2f} GB')
        say(f'precompute {dataset} {name}: {len(levels)} levels, {store.nbytes() / 1e9:.2f} GB in memory, '
            f'{store.disk_bytes / 1e9:.2f} GB of cache files, {time.perf_counter() - started:.0f} s')
        del retriever
    return cached, encoded


def fov_stream(slides: Sequence[Tuple[str, str, str]], make_retriever: Callable[[str], Any],
               fov: FovRecipe, mask_job: str, device: Any, label_cfg: LabelConfig,
               per_visit: int, seed: int, epoch: int = 0, levels: Optional[Sequence[int]] = None,
               workers: int = 1, say: Callable[[str], None] = print,
               topk: Optional[int] = None) -> Iterator[FovResult]:
    """`slides` is `(dataset, name, path)`. For each, in the order given: the
    slide's FoVs of each level (all of them in a random order per level, at most
    `per_visit` each), through a fresh retriever, its list cut to `topk` (None:
    the whole list). FoVs whose level the retriever could not search (it went
    finer: the query did not fit) are passed over, as the bench passes them: their
    truth was never on offer."""
    masks: MaskMaker = MaskMaker(MASK_RECIPES['hest'], mask_job, device)
    # the photos of a level are taken by `workers` threads at once (`FovSupply.photo` is the same picture on
    # any thread) while the first of them is searched
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        yield from _fov_stream(slides, make_retriever, fov, masks, pool, label_cfg, per_visit, seed, epoch, levels,
                               say, topk)


def _fov_stream(slides: Sequence[Tuple[str, str, str]], make_retriever: Callable[[str], Any], fov: FovRecipe,
                masks: MaskMaker, pool: ThreadPoolExecutor, label_cfg: LabelConfig, per_visit: int, seed: int,
                epoch: int, levels: Optional[Sequence[int]], say: Callable[[str], None],
                topk: Optional[int]) -> Iterator[FovResult]:
    """`fov_stream` once its tools are made."""
    for dataset, name, path in slides:
        say(f'== {dataset} {name}')
        t0: float = time.perf_counter()
        wsi: SafeSlide = SafeSlide(path)
        mask: Any = masks.mask(wsi)[0]
        reader: SlideReader = SlideReader(wsi)
        tick('open_and_mask', time.perf_counter() - t0)
        t0 = time.perf_counter()
        try:
            supply: FovSupply
            positions: List[Tuple[int, Any]]
            supply, positions = draw_fovs(reader, fov, mask)
        except RuntimeError as exc:
            say(f'  no FoV position at any level -- skipped ({str(exc).splitlines()[0]})')
            continue
        tick('draw_positions', time.perf_counter() - t0)
        retriever: Any = make_retriever(name)
        retriever.build(wsi, mask)
        encoded: set = set()                                  # the levels whose encoding has been timed
        by_level: Dict[int, List[Tuple[int, Any]]] = {}
        for i, meta in positions:
            by_level.setdefault(int(meta.level), []).append((i, meta))
        order: List[int] = sorted(by_level)
        rng: np.random.Generator = np.random.default_rng([seed, epoch, zlib.crc32(name.encode())])
        rng.shuffle(order)
        for level in order:
            if levels is not None and level not in levels:
                continue
            chosen: List[int] = rng.permutation(len(by_level[level]))[:per_visit].tolist()
            photos: Dict[int, Any] = {j: pool.submit(supply.photo, by_level[level][j][1]) for j in chosen}
            for j in chosen:
                index, meta = by_level[level][j]
                t0 = time.perf_counter()
                img, params = photos[j].result()
                tick('photo_wait', time.perf_counter() - t0)
                geom: Dict[str, float] = supply.geometry(meta)
                truth: Truth = Truth(
                    level=int(geom['level']), ds=float(meta.ds), rot=matching_rotation(int(params['rot_deg'])),
                    centre=partial(query_grid_centre, supply.camera_for(meta.ds), geom['x0'], geom['y0'],
                                   params, img.shape))
                res: Optional[FovResult] = searched(
                    retriever, wsi, name, dataset, index, img,
                    Placed(truth, (geom['center_x0'], geom['center_y0']), geom['w0'], geom['h0']),
                    label_cfg, topk, say, encoded)
                if res is not None:
                    yield res


@dataclass
class FovData:
    """What a training step needs of one FoV, copied out of the retriever: the
    tensors of a SUBSET of its candidates and their labels, on the device."""
    ni: NeuralInputs
    pos: torch.Tensor
    neg: torch.Tensor
    covers: torch.Tensor
    rot_ok: torch.Tensor
    loc: torch.Tensor
    loc_mask: torch.Tensor
    near: torch.Tensor
    near_target: torch.Tensor
    pair: Optional[torch.Tensor]
    slide: str
    level: int
    index: int
    n_total: int


def _t(a: np.ndarray, device: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    return torch.as_tensor(a, device=device, dtype=dtype)


def prepare_fov(res: FovResult, pooling: str, n_cand: int, hard_share: float,
                rng: np.random.Generator, need_pair: bool = False) -> Optional[FovData]:
    """The training tensors of one FoV, or None when no candidate in its list is
    a positive (the truth was never proposed: nothing to rank)."""
    lab: Labels = res.labels
    if not lab.pos.any():
        return None
    sel: np.ndarray = choose_subset(lab, n_cand, hard_share, rng)
    cands: List[Any] = [res.cs.candidates[i] for i in sel]
    batch: WindowBatch = WindowBatch.build(res.cs, res.source, pooling, candidates=cands)
    device: Any = batch.groups[0].q.device
    ni: NeuralInputs = neural_inputs(
        batch, torch.tensor([c.score for c in cands], device=device),
        torch.tensor(sel + 1, device=device), [c.rotation for c in cands], [c.lattice for c in cands])
    pair: Optional[torch.Tensor] = None
    if need_pair:
        centres: np.ndarray = np.stack([(res.boxes[sel, 0] + res.boxes[sel, 2]) / 2.0,
                                        (res.boxes[sel, 1] + res.boxes[sel, 3]) / 2.0], axis=1)
        pair = pair_features(_t(centres, device, torch.float32),
                             _t(np.array([c.rotation for c in cands]), device),
                             float(res.cs.grids[0].tile_size) * float(res.cs.ds))
    return FovData(ni=ni, pos=_t(lab.pos[sel], device), neg=_t(lab.neg[sel], device),
                   covers=_t(lab.covers[sel], device), rot_ok=_t(lab.rot_ok[sel], device),
                   loc=_t(lab.loc[sel], device, torch.float32), loc_mask=_t(lab.loc_mask[sel], device),
                   near=_t(lab.near[sel], device), near_target=_t(lab.soft[sel], device, torch.float32),
                   pair=pair, slide=res.slide, level=res.level, index=res.index, n_total=len(res.cs))


def full_inputs(res: FovResult, pooling: str) -> Tuple[NeuralInputs, Optional[torch.Tensor]]:
    """The tensors of ALL of a FoV's candidates (evaluation scores the whole
    list), and the pair features for a model that wants them."""
    cands: List[Any] = list(res.cs.candidates)
    batch: WindowBatch = WindowBatch.build(res.cs, res.source, pooling)
    device: Any = batch.groups[0].q.device
    ni: NeuralInputs = neural_inputs(batch, torch.tensor([c.score for c in cands], device=device),
                                     torch.arange(1, len(cands) + 1, device=device),
                                     [c.rotation for c in cands], [c.lattice for c in cands])
    centres: np.ndarray = np.stack([(res.boxes[:, 0] + res.boxes[:, 2]) / 2.0,
                                    (res.boxes[:, 1] + res.boxes[:, 3]) / 2.0], axis=1)
    pair: torch.Tensor = pair_features(_t(centres, device, torch.float32),
                                       _t(np.array([c.rotation for c in cands]), device),
                                       float(res.cs.grids[0].tile_size) * float(res.cs.ds))
    return ni, pair


__all__ = ['FovResult', 'FovData', 'Placed', 'LevelDiskCache', 'MemoryFeatureStore', 'draw_fovs', 'drawn_levels',
           'precompute_levels', 'fov_stream', 'searched', 'prepare_fov', 'full_inputs', 'shape_fn']
