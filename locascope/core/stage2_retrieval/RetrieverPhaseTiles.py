"""PhaseTiles: a first-phase retriever -- the photo's tiles matched to the level as MAPS, each tile free to move. The map is a
`MAPS` provider: LOCAL PHASE maps (`localphase`, `congruency`: closed form, no encoder, no weights) or the latent cells of a CoAE
checkpoint stitched into a map (`coae-latent`).

    r   = RetrieverPhaseTiles(PHASETILES_RECIPES['localphase'], device).build(wsi, mask)
    out = r.retrieve(est_mpp_result, shot_img)

Every step is a fixed transform, every number of it a field of the config:

  1. the level stage 1 routes to (`SlideLevels.route`: or the nearest finer one that holds the query);
  2. each tissue region of it, its box grown by `edge_px` level px (so a window whose centre is in the box is whole inside the map),
     read and made a map by the `rep`'s provider. `phase_map`: grey, mean off, a Tukey taper, FFT, log-Gabor bands at `freqs`
     (cycles per 256 px) with their Riesz pair, back to space, each pixel's vector made unit (the contrast dropped, the phase kept:
     `localphase`), or the phase congruency of `pc_scales` (`congruency`); averaged on `cell` px cells. `coae-latent`: the latent
     of every tile of the region's `grid_stride` px grid, stitched (`stitch`, weights `tent`) on `cell` px cells. Made once a level and kept;
  3. the query, as it is (no scale is tried), at every rotation the level fits: tiles of `query_tile` px every `tile_stride` px, the
     grid centred on the photo, each made the same map by the provider;
  4. each tile against the region map, the provider's own correlation at every place (`localphase`, `congruency`: multi-channel
     ZNCC by FFT; `coae-latent`: cosine by `conv2d`), then the best within +-`pool_px` px of each place:
     a tile may move that far from where a window puts it (what absorbs the scale and the turn the query has against the level);
  5. a window's score = the mean over the tiles of that best, the window placed by its centre (the photo's own size, unscaled);
  6. every window of the slide's lattices (`main`, `offset`: the windows every retriever lists, the query's tile grid) takes the best
     window score within half a tile of its centre, and the shift to it (`PhaseCandidate.shift`: the window moved off its lattice by
     that many level px); the list is `top_candidates`' (the best `raw_topk`, `min_sep_tiles` apart a rotation).

The windows are the lattice windows every stage-2 method lists, so stage 3 reads them as any other; a window's shift places it where
the tiles found it. The analysis behind it: scratch/diag/a1_phase_localize.py (ExpA1Full), WinEmbedGapReport.md."""

from __future__ import annotations

import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as TF

from locascope.core.backbones.TileEncoder import TileEncoderConfig
from locascope.core.config.ConfigIdentity import IdentifiedConfig, register
from locascope.core.slides.ReadGeometry import ReadSpec
from locascope.core.slides.SafeSlide import SafeSlide
from locascope.core.slides.SlideReader import SlideReader
from locascope.core.stage1_estimation.StageInterface import EstMppResult
from locascope.core.stage2_retrieval.ArmSpec import COAE_MAPS, coae_encoder_name
from locascope.core.stage2_retrieval.Context import SlideLevels, query_array, top_candidates
from locascope.core.stage2_retrieval.Retriever import RetrievalOutput, Retriever, RetrieverConfig
from locascope.core.stage2_retrieval.RetrieverCoae import CoaeTilesConfig, coae_tile_encoder
from locascope.core.stage2_retrieval.StageInterface import Candidate, CandidateSet


# ══════════════════════════════════════════════════════════════════════════════
#  the maps a retriever can search: the registry and what an encoded one is stitched with
# ══════════════════════════════════════════════════════════════════════════════

class MapProvider(Protocol):
    """What differs between the maps of PhaseTiles: how a block of pixels and a query tile become a map, and how a tile's map is held
    against a block of a region's. Everything else of the search is the retriever's."""
    margin_px: int            # level px read round a block's core before its map is made
    step_px: int              # a block's core is a multiple of this many px
    tile_cells: int           # the side of a query tile's map, in cells
    dtype: torch.dtype        # what a region's map is kept in on the host
    model: Optional[Any]      # what IdentifiedBuild.weights_id hashes (None: no network)

    def block_map(self, rgb: np.ndarray) -> torch.Tensor:
        """RGB [h, w, 3] -> [C, h // cell, w // cell] on the device."""

    def tile_maps(self, img: np.ndarray, tl: np.ndarray) -> torch.Tensor:
        """The turned query and its tiles' top-lefts (y, x) -> [n, C, t, t]."""

    def query(self, tiles: torch.Tensor) -> Any:
        """What `match` takes of a rotation's tiles; it has `forget(but)`."""

    def block(self, sub: torch.Tensor) -> Any:
        """The query-free half of a region block [C, H, W]; it has `nbytes()`."""

    def match(self, block: Any, query: Any) -> torch.Tensor:
        """[n, H - t + 1, W - t + 1] float32."""


@dataclass(frozen=True)
class MapKind:
    """A registered map: the class that builds its provider, and whether the map is read from an encoder (`cfg.source` is needed) or
    from the pixels (`cfg.source` is None)."""
    build: Callable[['RetrieverPhaseTilesConfig', torch.device], MapProvider]
    encoded: bool


#: `rep` -> its kind; declared BEFORE the config class (which checks `rep` against it)
MAPS: Dict[str, MapKind] = {}


def register_map(name: str, *, encoded: bool) -> Callable[[type], type]:
    """Class decorator: `MAPS[name]` = a `MapKind` that builds the class."""
    def add(cls: type) -> type:
        MAPS[name] = MapKind(cls, encoded)               # type: ignore[arg-type]
        return cls
    return add


def tent(side: int, step: int) -> torch.Tensor:
    """float32 [side]: 1 - |a - (side - 1) / 2| / step. ValueError when a weight is <= 0. (7, 4): .25 .5 .75 1 .75 .5 .25; two tiles a step
    apart sum to 1."""
    a: torch.Tensor = torch.arange(side, dtype=torch.float32)
    w: torch.Tensor = 1.0 - (a - (side - 1) / 2).abs() / step
    if bool((w <= 0).any()):
        raise ValueError(f'tent({side}, {step}): a weight is <= 0 (the side must be less than 2 * step + 1)')
    return w


#: how tiles that overlap are merged into one cell of a map
MERGES: Dict[str, Callable[[int, int], torch.Tensor]] = {'tent': tent}


@register('coae_map')
@dataclass(frozen=True)
class CoaeMapConfig(IdentifiedConfig):
    """The source of an encoded map: the FM the checkpoint reads (`encoder`, pooling 'cls': it names the FM and gives the batch), the
    checkpoint and the usage that is the map (`tiles`; its weights' content is identity through the stage's id), the spacing of the
    region's tiles (`grid_stride`, level px) and how tiles that overlap are merged (`merge`, a key of MERGES)."""
    encoder:     TileEncoderConfig
    tiles:       CoaeTilesConfig
    grid_stride: int = 128
    merge:       str = 'tent'

    BASELINE = {'grid_stride': 128, 'merge': 'tent'}
    VERSION = 1

    def __post_init__(self) -> None:
        if self.merge not in MERGES:
            raise ValueError(f'merge must be one of {tuple(MERGES)}, got {self.merge!r}')
        if self.tiles.feature not in COAE_MAPS:
            raise ValueError(f'a map is made of one of {COAE_MAPS}, got {self.tiles.feature!r}')
        if str(self.encoder.pooling) != 'cls':
            raise ValueError(f'the CoAE checkpoint reads the FM\'s CLS: the encoder\'s pooling must be \'cls\', got {self.encoder.pooling!r}')
        if self.grid_stride < 1:
            raise ValueError(f'grid_stride must be >= 1, got {self.grid_stride}')


@register('retriever_phasetiles')
@dataclass(frozen=True)
class RetrieverPhaseTilesConfig(RetrieverConfig):
    """The map (`rep`, a key of `MAPS`; for `localphase` and `congruency` its bands `freqs` or `pc_scales`, `lg_sigma`, the taper
    `taper_px`; for an encoded one its `source`; `cell` for all), the query's tiles (`query_tile`
    every `tile_stride`), how far a tile may move (`pool_px`), how far each region's box is grown on every side before the map and the
    search (`edge_px`, level px, a multiple of `tile_size`; for every window centred in the box to be scored it must be at least half
    the query's long side plus `pool_px`), the region blocks the maps are made and searched in (`block_px`, memory
    only: not identity), the GPU memory the blocks' spectra may be kept in (`spec_cache_gb`, memory only: not identity), the slide's lattices (`tile_size`, `overlap`) and the list (`raw_topk`, `min_sep_tiles`). Lists of values are
    tuples: `freqs=(8.0, 24.0)`."""
    rep:           str                 = 'localphase'
    freqs:         Tuple[float, ...]   = (8.0, 24.0)
    pc_scales:     Tuple[float, ...]   = (4.0, 8.0, 16.0, 32.0)
    lg_sigma:      float               = 0.55
    cell:          int                 = 8
    taper_px:      int                 = 32
    query_tile:    int                 = 256
    tile_stride:   int                 = 384
    pool_px:       int                 = 96
    edge_px:       int                 = 0
    tile_size:     int                 = 256
    overlap:       bool                = True
    raw_topk:      int                 = 1000
    min_sep_tiles: float               = 1.0
    block_px:      int                 = 4096
    spec_cache_gb: float               = 64.0
    source:        Optional[CoaeMapConfig] = None

    BASELINE = {'rep': 'localphase', 'freqs': (8.0, 24.0), 'pc_scales': (4.0, 8.0, 16.0, 32.0), 'lg_sigma': 0.55, 'cell': 8,
                'taper_px': 32, 'query_tile': 256, 'tile_stride': 384, 'pool_px': 96, 'edge_px': 0, 'tile_size': 256, 'overlap': True,
                'raw_topk': 1000, 'min_sep_tiles': 1.0, 'source': None}
    NOT_IDENTITY = ('block_px', 'spec_cache_gb')
    VERSION = 1

    @property
    def arm_encoder(self) -> str:
        """The encoder name the arm writes: `<Fm>CoAE+<arm>` for an encoded map, '' for a map of the pixels."""
        if self.source is None:
            return ''
        return coae_encoder_name(str(getattr(self.source.encoder, 'REGISTERED_AS', '')), self.source.tiles.weights)

    @property
    def arm_pooling(self) -> str:
        """The arm's pooling: the usage that is the map ('' for a map of the pixels: the arm writes `raw`)."""
        return '' if self.source is None else self.source.tiles.feature

    @property
    def arm_op(self) -> str:
        """The arm's operation: the `rep` of a map of the pixels (what the arm writes after `+`), '' for an encoded one (none to choose)."""
        return self.rep if self.source is None else ''

    def __post_init__(self) -> None:
        kind: Optional[MapKind] = MAPS.get(self.rep)
        if kind is None:
            raise ValueError(f'rep must be one of {tuple(MAPS)}, got {self.rep!r}')
        if kind.encoded != (self.source is not None):
            raise ValueError(f'rep {self.rep!r} {"reads an encoder: it needs a source" if kind.encoded else "reads the pixels: it takes no source"}')
        if self.cell < 1 or self.query_tile % self.cell or self.pool_px % self.cell or self.tile_size % (2 * self.cell):
            raise ValueError('cell must divide query_tile, pool_px and half of tile_size')
        if self.source is not None:
            if self.rep != f'coae-{self.source.tiles.feature}':
                raise ValueError(f'rep {self.rep!r} is not the map of the usage {self.source.tiles.feature!r}')
            if self.source.grid_stride % self.cell or self.tile_stride % self.cell:
                raise ValueError('cell must divide source.grid_stride and tile_stride')
            if self.query_tile != self.tile_size:
                raise ValueError('query_tile must be tile_size (the checkpoint reads one tile size)')
        if self.edge_px < 0 or self.edge_px % self.tile_size:
            raise ValueError('edge_px must be a non-negative multiple of tile_size')
        if self.raw_topk < 1 or self.block_px < 4 * self.query_tile:
            raise ValueError('raw_topk >= 1 and block_px >= 4 query tiles')
        if self.spec_cache_gb < 0:
            raise ValueError('spec_cache_gb must be >= 0')


def coae_latent_config(encoder: TileEncoderConfig, weights: str, *, feature: str, overlap: bool, raw_topk: int,
                       min_sep_tiles: float) -> RetrieverPhaseTilesConfig:
    """THE settings of the latent-map retriever, every field written out; what `<Fm>CoAE+<arm>:PhaseTiles::latent` builds."""
    return RetrieverPhaseTilesConfig(
        rep=f'coae-{feature}', freqs=(8.0, 24.0), pc_scales=(4.0, 8.0, 16.0, 32.0), lg_sigma=0.55, cell=32, taper_px=32,
        query_tile=256, tile_stride=384, pool_px=96, edge_px=1024, tile_size=256, overlap=overlap, raw_topk=raw_topk,
        min_sep_tiles=min_sep_tiles, block_px=8192, spec_cache_gb=64.0,
        source=CoaeMapConfig(encoder=encoder, tiles=CoaeTilesConfig(weights=weights, feature=feature), grid_stride=128, merge='tent'))


@dataclass(frozen=True)
class PhaseCandidate(Candidate):
    """A lattice window and the shift (dx, dy) level px that puts its origin on the top-left of the turned query where the tiles found it
    (within half a tile). The origin is the turned query's top-left, as every retriever's is."""
    shift: Tuple[int, int] = (0, 0)

    def shift_px(self) -> Tuple[int, int]:
        return (int(self.shift[0]), int(self.shift[1]))

    def extra(self) -> Dict[str, Any]:
        return dict(dx=int(self.shift[0]), dy=int(self.shift[1]))


# ══════════════════════════════════════════════════════════════════════════════
#  the map
# ══════════════════════════════════════════════════════════════════════════════

def _taper(n: int, ramp: int, device: torch.device) -> torch.Tensor:
    m: int = min(int(ramp), n // 2)
    w: torch.Tensor = torch.ones(n, device=device)
    if m > 0:
        r: torch.Tensor = 0.5 * (1.0 - torch.cos(np.pi * (torch.arange(m, device=device) + 0.5) / m))
        w[:m] = r
        w[n - m:] = r.flip(0)
    return w


def _box_sum(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """The sum of every h x w window of x [b, H, W] (valid positions): [b, H - h + 1, W - w + 1]."""
    c: torch.Tensor = TF.pad(x.cumsum(1).cumsum(2), (1, 0, 1, 0))
    return c[:, h:, w:] - c[:, :-h, w:] - c[:, h:, :-w] + c[:, :-h, :-w]


def _local_mean(x: torch.Tensor, size: int) -> torch.Tensor:
    """The mean of x [1, H, W] over the size x size px round each pixel (edges replicated)."""
    _, h, w = x.shape
    p: int = size // 2
    xp: torch.Tensor = TF.pad(x[None], (p, p, p, p), mode='replicate')[0].double()
    return (_box_sum(xp, size, size)[:, :h, :w] / (size * size)).float()


def phase_map(grey: torch.Tensor, cfg: RetrieverPhaseTilesConfig) -> torch.Tensor:
    """The map of a grey image [H, W] (float, 0-255): [C, H // cell, W // cell]. Frequencies are in cycles per 256 px."""
    h, w = grey.shape
    dev: torch.device = grey.device
    d: torch.Tensor = (grey - grey.mean()) * _taper(h, cfg.taper_px, dev)[:, None] * _taper(w, cfg.taper_px, dev)[None, :]
    f: torch.Tensor = torch.fft.fft2(d)
    fy: torch.Tensor = (torch.fft.fftfreq(h, device=dev) * 256.0)[:, None].expand(h, w)
    fx: torch.Tensor = (torch.fft.fftfreq(w, device=dev) * 256.0)[None, :].expand(h, w)
    r: torch.Tensor = torch.hypot(fy, fx).clamp(min=1e-6)

    def mono(f0: float) -> torch.Tensor:
        lg: torch.Tensor = torch.exp(-torch.log(r / f0) ** 2 / (2 * np.log(cfg.lg_sigma) ** 2))
        lg[0, 0] = 0.0
        fg: torch.Tensor = f * lg
        return torch.stack([torch.fft.ifft2(fg).real, torch.fft.ifft2(fg * (1j * fx / r)).real, torch.fft.ifft2(fg * (1j * fy / r)).real])

    if cfg.rep == 'localphase':
        parts: List[torch.Tensor] = []
        for f0 in cfg.freqs:
            m: torch.Tensor = mono(float(f0))
            parts.append(m / m.norm(dim=0, keepdim=True).clamp(min=1e-6))
        out: torch.Tensor = torch.cat(parts)
    else:
        acc: torch.Tensor = torch.zeros(3, h, w, device=dev)
        amp_sum: torch.Tensor = torch.zeros(1, h, w, device=dev)
        for f0 in cfg.pc_scales:
            m = mono(float(f0))
            acc += m
            amp_sum += m.norm(dim=0, keepdim=True)
        energy: torch.Tensor = acc.norm(dim=0, keepdim=True)
        pc: torch.Tensor = (energy - 0.05 * _local_mean(amp_sum, 256)).clamp(min=0) / (amp_sum + 1e-6)
        theta: torch.Tensor = torch.atan2(acc[2:3], acc[1:2])
        out = torch.cat([pc, pc * torch.cos(2 * theta), pc * torch.sin(2 * theta)])
    hc, wc = h // cfg.cell, w // cfg.cell
    return TF.avg_pool2d(out[None, :, :hc * cfg.cell, :wc * cfg.cell], cfg.cell)[0]


@dataclass(frozen=True)
class RegionSpectrum:
    """What `zncc_maps` needs of a region block [C, H, W] for query tiles of h x w cells, and does not depend on the query."""
    spec:  torch.Tensor          # [C, H, W // 2 + 1] complex128: rfft2 of the block in float64
    sd:    torch.Tensor          # [1, H - h + 1, W - w + 1] float64: sqrt of the window variance (all channels), clamped at 1e-12
    valid: torch.Tensor          # the same shape, bool: variance > 1e-12
    shape: Tuple[int, int]       # (H, W)

    def nbytes(self) -> int:
        return int(self.spec.numel()) * 16 + int(self.sd.numel()) * 8 + int(self.valid.numel())


def region_spectrum(region: torch.Tensor, h: int, w: int) -> RegionSpectrum:
    """The query-free part of `zncc_maps` for a region block [C, H, W] and query tiles of h x w cells."""
    _, big_h, big_w = region.shape
    r64: torch.Tensor = region.double()
    spec: torch.Tensor = torch.fft.rfft2(r64)
    s1: torch.Tensor = _box_sum(r64, h, w)
    s2: torch.Tensor = _box_sum(r64 * r64, h, w)
    var: torch.Tensor = (s2 - s1.pow(2) / (h * w)).sum(0, keepdim=True)
    return RegionSpectrum(spec=spec, sd=var.clamp(min=1e-12).sqrt(), valid=var > 1e-12, shape=(int(big_h), int(big_w)))


class QuerySpectra:
    """The tiles of one turned query [n, C, h, w]: mean off, float64, their norms, and their zero-padded spectrum at each block
    shape asked for (made once a shape)."""

    def __init__(self, queries: torch.Tensor) -> None:
        self.hw: Tuple[int, int] = (int(queries.shape[2]), int(queries.shape[3]))
        self.q: torch.Tensor = (queries - queries.mean(dim=(2, 3), keepdim=True)).double()
        self.qn: torch.Tensor = self.q.pow(2).sum(dim=(1, 2, 3)).sqrt().clamp(min=1e-12)
        self.kept: Dict[Tuple[int, int], torch.Tensor] = {}

    def at(self, big_h: int, big_w: int) -> torch.Tensor:
        """rfft2 of the tiles padded to (big_h, big_w): [n, C, big_h, big_w // 2 + 1]. NOT conjugated."""
        if (big_h, big_w) not in self.kept:
            self.kept[(big_h, big_w)] = torch.fft.rfft2(self.q, s=(big_h, big_w))
        return self.kept[(big_h, big_w)]

    def forget(self, but: Tuple[int, int]) -> None:
        """Drop every shape except `but` (a region's edge blocks have shapes no other region has)."""
        self.kept = {k: v for k, v in self.kept.items() if k == but}


def zncc_of(rs: RegionSpectrum, qs: QuerySpectra) -> torch.Tensor:
    """`zncc_maps` from its two halves: the region block's spectrum and the query's."""
    big_h, big_w = rs.shape
    h, w = qs.hw
    corr: torch.Tensor = torch.fft.irfft2((rs.spec[None] * qs.at(big_h, big_w).conj()).sum(1), s=(big_h, big_w))
    corr = corr[:, :big_h - h + 1, :big_w - w + 1]
    out: torch.Tensor = torch.where(rs.valid, corr / (rs.sd * qs.qn[:, None, None]), torch.zeros_like(corr))
    return out.float()


def zncc_maps(region: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """ZNCC of region [C, H, W] with queries [n, C, h, w], each channel's mean off on its own, all channels one vector: [n, H-h+1, W-w+1].

    The cross-correlation runs in float64: in an all-zero stretch of the map the float32 FFT rounding error, divided by a
    near-zero variance, becomes a fake score above 1 (see ExpPhaseAttractor). Windows with zero variance (no content at
    all, var <= 1e-12) score 0 (no evidence).
    """
    qs: QuerySpectra = QuerySpectra(queries)
    return zncc_of(region_spectrum(region, qs.hw[0], qs.hw[1]), qs)


def _sibling(slide: SafeSlide) -> SafeSlide:
    """A read thread's own handle on the slide's file (`SafeSlide.sibling`: a failed read replaces its handle, which must not be one
    another thread is reading)."""
    return slide.sibling()


def check_edge(edge_px: int, pool_px: int, h: int, w: int) -> None:
    """Raise when a query of h x w px is longer than `edge_px` supports (a window centred in a region box would leave the map).
    `edge_px = 0` (the boxes as they are) is not checked."""
    need: int = max(h, w) // 2 + pool_px
    if 0 < edge_px < need:
        raise ValueError(f'query {w}x{h}: edge_px {edge_px} < {need} (half its long side + pool_px); '
                         f'a window centred in a region box would leave the map')


def centre_cells(extent: int, cell: int) -> int:
    """Cells from a window's top-left to the cell its score is kept at, along one axis: half the turned query's side (`extent` px).
    THE one formula: `window_scores` writes with it and `run` reads with it."""
    return int(round(extent / 2 / cell))


def lattice_cells(starts: Sequence[int], lo: int, extent: int, cell: int) -> np.ndarray:
    """The score-map cells, along one axis, of the lattice windows whose top-lefts are `starts + lo` px from the region's origin, for
    a turned query `extent` px long on that axis. `starts + lo` are multiples of `cell` (the config holds tile_size to 2 * cell)."""
    return (np.asarray(starts, dtype=np.int64) + int(lo)) // int(cell) + centre_cells(extent, cell)


def _grey(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    t: torch.Tensor = torch.as_tensor(np.ascontiguousarray(rgb), device=device).float()
    return t[..., 0] * 0.299 + t[..., 1] * 0.587 + t[..., 2] * 0.114


# ══════════════════════════════════════════════════════════════════════════════
#  the providers: a map of the pixels, a map of a checkpoint's latent
# ══════════════════════════════════════════════════════════════════════════════

@register_map('congruency', encoded=False)        # `phase_map` reads cfg.rep itself: one class, two names
@register_map('localphase', encoded=False)
class PixelMap:
    """The map of the pixels (`phase_map`), a tile held against a region by multi-channel ZNCC (`zncc_of`)."""

    def __init__(self, cfg: RetrieverPhaseTilesConfig, device: torch.device) -> None:
        self.cfg: RetrieverPhaseTilesConfig = cfg
        self.device: torch.device = device
        self.margin_px: int = 8 * cfg.cell                # the filters' support
        self.step_px: int = cfg.cell
        self.tile_cells: int = cfg.query_tile // cfg.cell
        self.dtype: torch.dtype = torch.float16
        self.model: Optional[Any] = None

    def block_map(self, rgb: np.ndarray) -> torch.Tensor:
        return phase_map(_grey(rgb, self.device), self.cfg)

    def tile_maps(self, img: np.ndarray, tl: np.ndarray) -> torch.Tensor:
        g: torch.Tensor = _grey(img, self.device)
        qt: int = self.cfg.query_tile
        return torch.stack([phase_map(g[y:y + qt, x:x + qt], self.cfg) for y, x in tl])

    def query(self, tiles: torch.Tensor) -> QuerySpectra:
        return QuerySpectra(tiles)

    def block(self, sub: torch.Tensor) -> RegionSpectrum:
        return region_spectrum(sub, self.tile_cells, self.tile_cells)

    def match(self, block: RegionSpectrum, query: QuerySpectra) -> torch.Tensor:
        return zncc_of(block, query)


def stitch(lat: torch.Tensor, gy: torch.Tensor, gx: torch.Tensor, profile: torch.Tensor, step: int, shape: Tuple[int, int]) -> torch.Tensor:
    """lat float32 [n, s, s, L] (row, column, channel) of the tiles at lattice (gy, gx) int64 -> [L, shape[0], shape[1]] float32:
    cell (step * gy + a, step * gx + b) is the weighted mean of the tiles that cover it, 0 where none does."""
    n, s, _, dim = lat.shape
    cells: torch.Tensor = lat.new_zeros(shape[0], shape[1], dim)
    wsum: torch.Tensor = lat.new_zeros(shape[0], shape[1])
    for a in range(s):
        for b in range(s):                                    # within one (a, b) no two tiles write one cell
            w: torch.Tensor = profile[a] * profile[b]
            cells.index_put_((step * gy + a, step * gx + b), w * lat[:, a, b], accumulate=True)
            wsum.index_put_((step * gy + a, step * gx + b), w.expand(n), accumulate=True)
    cells = torch.where(wsum[..., None] > 0, cells / wsum.clamp(min=1e-12)[..., None], torch.zeros_like(cells))
    return cells.permute(2, 0, 1).contiguous()


@dataclass(frozen=True)
class CosineBlock:
    """What `cosine_of` needs of a region block [C, H, W] for query tiles of t x t cells, and does not depend on the query."""
    cells: torch.Tensor       # [1, C, H, W] float32 on the device
    norm:  torch.Tensor       # [1, H - t + 1, W - t + 1] float32: the L2 norm of every t x t window over all channels
    valid: torch.Tensor       # the same shape, bool: norm > 1e-6 (a window nothing covers scores 0: no evidence)

    def nbytes(self) -> int:
        return 4 * int(self.cells.numel()) + 4 * int(self.norm.numel()) + int(self.valid.numel())


def cosine_block(sub: torch.Tensor, t: int) -> CosineBlock:
    """The query-free part of `cosine_of` for a region block [C, H, W] and query tiles of t x t cells."""
    x: torch.Tensor = sub.float()[None]
    norm: torch.Tensor = _box_sum((x[0] * x[0]).sum(0, keepdim=True).double(), t, t).clamp(min=0).sqrt().float()
    return CosineBlock(x, norm, norm > 1e-6)


class CosineKernels:
    """A rotation's tiles [n, C, t, t], each made unit over all its entries (the plain cosine of the `latent` usage: no mean taken off)."""

    def __init__(self, tiles: torch.Tensor) -> None:
        t: torch.Tensor = tiles.float()
        self.k: torch.Tensor = t / t.flatten(1).norm(dim=1).clamp(min=1e-12)[:, None, None, None]
        self.hw: Tuple[int, int] = (int(t.shape[2]), int(t.shape[3]))

    def forget(self, but: Tuple[int, int]) -> None:
        """Nothing is kept a block shape."""


#: kernels a `conv2d` call (memory)
COSINE_KERNELS: int = 64


def cosine_of(cb: CosineBlock, ck: CosineKernels) -> torch.Tensor:
    """[n, H - t + 1, W - t + 1] float32: the cosine of each tile's map with every window of the block; 0 where `valid` is False."""
    with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=False):   # cudnn's TF32 is on by default: 10 mantissa bits
        corr: torch.Tensor = torch.cat([TF.conv2d(cb.cells, ck.k[i:i + COSINE_KERNELS])[0] for i in range(0, len(ck.k), COSINE_KERNELS)])   # a correlation: no flip
    return torch.where(cb.valid, corr / cb.norm.clamp(min=1e-6), torch.zeros_like(corr))


@register_map('coae-latent', encoded=True)
class CoaeLatentMap:
    """The map of a CoAE checkpoint's latent: the cells of every tile of the region's `grid_stride` px grid, stitched (`stitch`) on
    `cell` px cells; a tile held against a region by the cosine of all its cells (`cosine_of`)."""

    def __init__(self, cfg: RetrieverPhaseTilesConfig, device: torch.device) -> None:
        assert cfg.source is not None
        src: CoaeMapConfig = cfg.source
        self.cfg: RetrieverPhaseTilesConfig = cfg
        self.device: torch.device = device
        self.enc: Any = coae_tile_encoder(src.tiles.weights, src.tiles.feature, src.encoder, cfg.query_tile, device)   # refuses another FM, another tile, TF32 matmul
        net: Any = self.enc.net
        side: int = int(net.latent_side)
        if int(net.field_px) != side * cfg.cell:
            raise ValueError(f'the checkpoint\'s latent cell is {int(net.field_px) / side} px and the config says {cfg.cell}')
        self.dim: int = int(net.cfg.model.latent_dim)
        self.step: int = src.grid_stride // cfg.cell
        self.profile: torch.Tensor = MERGES[src.merge](side, self.step).to(device)
        self.tile_cells: int = side
        self.margin_px: int = cfg.query_tile
        self.step_px: int = src.grid_stride
        self.dtype: torch.dtype = torch.float32
        self.model: Optional[Any] = net.fm.vit

    def block_map(self, rgb: np.ndarray) -> torch.Tensor:
        t: torch.Tensor = torch.as_tensor(np.ascontiguousarray(rgb), device=self.device)                # [h, w, 3] uint8
        h, w = int(t.shape[0]), int(t.shape[1])
        tile, stride = self.cfg.query_tile, self.step_px
        shape: Tuple[int, int] = (h // self.cfg.cell, w // self.cfg.cell)
        if h < tile or w < tile:                                                                        # no tile fits: nothing covers
            return torch.zeros(self.dim, shape[0], shape[1], device=self.device)
        tiles: torch.Tensor = t.unfold(0, tile, stride).unfold(1, tile, stride)                         # [ny, nx, 3, T, T]: the tile at (stride * gy, stride * gx)
        ny, nx = int(tiles.shape[0]), int(tiles.shape[1])
        lat: torch.Tensor = self.enc.cells(tiles.permute(0, 1, 3, 4, 2).reshape(ny * nx, tile, tile, 3))   # [ny * nx, s, s, L] float32
        gy, gx = (g.flatten() for g in torch.meshgrid(torch.arange(ny, device=self.device), torch.arange(nx, device=self.device), indexing='ij'))
        return stitch(lat, gy, gx, self.profile, self.step, shape)

    def tile_maps(self, img: np.ndarray, tl: np.ndarray) -> torch.Tensor:
        tile: int = self.cfg.query_tile
        return self.enc.cells(np.stack([img[y:y + tile, x:x + tile] for y, x in tl])).permute(0, 3, 1, 2).contiguous()     # [n, L, s, s]

    def query(self, tiles: torch.Tensor) -> CosineKernels:
        return CosineKernels(tiles)

    def block(self, sub: torch.Tensor) -> CosineBlock:
        return cosine_block(sub, self.tile_cells)

    def match(self, block: CosineBlock, query: CosineKernels) -> torch.Tensor:
        return cosine_of(block, query)


#: Named retrievers, every field written out; `--stage2 phasetiles:<name>`. After the providers: a config checks its `rep` against `MAPS`.
PHASETILES_RECIPES: Dict[str, RetrieverPhaseTilesConfig] = {
    'localphase': RetrieverPhaseTilesConfig(rep='localphase', freqs=(8.0, 24.0), pc_scales=(4.0, 8.0, 16.0, 32.0), lg_sigma=0.55, cell=8,
                                            taper_px=32, query_tile=256, tile_stride=384, pool_px=96, edge_px=1024, tile_size=256, overlap=True,
                                            raw_topk=1000, min_sep_tiles=1.0, block_px=4096, spec_cache_gb=64.0, source=None),
    'congruency': RetrieverPhaseTilesConfig(rep='congruency', freqs=(8.0, 24.0), pc_scales=(4.0, 8.0, 16.0, 32.0), lg_sigma=0.55, cell=8,
                                            taper_px=32, query_tile=256, tile_stride=384, pool_px=96, edge_px=1024, tile_size=256, overlap=True,
                                            raw_topk=1000, min_sep_tiles=1.0, block_px=4096, spec_cache_gb=64.0, source=None),
}


# ══════════════════════════════════════════════════════════════════════════════
#  the retriever
# ══════════════════════════════════════════════════════════════════════════════

class RetrieverPhaseTiles(Retriever):
    """See the module docstring. The WSI interface only (it lists the slide's lattice windows itself)."""

    UPSTREAM = (EstMppResult,)

    def __init__(self, cfg: RetrieverPhaseTilesConfig, device: Optional[Any] = None, *,
                 encoders: Optional[Any] = None, multi_gpu: bool = False, read_workers: int = 0,
                 feature_store: Optional[Any] = None) -> None:
        """`read_workers` and `feature_store` are what the pipeline hands every first phase (`MultiPhases`): this one reads its level's
        blocks with `read_workers` threads (each its own slide handle) and keeps no features, so it takes `feature_store` and does not
        use it."""
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: RetrieverPhaseTilesConfig = cfg
        self.read_workers: int = int(read_workers)
        self.levels: Optional[SlideLevels] = None
        self.reader: Optional[SlideReader] = None
        self.maps: Dict[int, List[Optional[torch.Tensor]]] = {}                       # level -> per region its map [C, Hc, Wc] (cpu)
        self.spectra: Dict[Tuple[int, int, int, int, int], Any] = {}                   # (level, region, by, bx, blk) -> the provider's block, on self.device
        self.spectra_bytes: int = 0
        self.spectra_budget: int = int(cfg.spec_cache_gb * 2 ** 30)
        self.spectra_hits: int = 0                                                       # blocks taken from the cache
        self.spectra_skipped: int = 0                                                    # blocks made and not kept: the budget was full
        self._provider: Optional[MapProvider] = None

    @property
    def provider(self) -> MapProvider:
        """The `rep`'s map provider (`MAPS`), built on first use: naming a stage loads nothing."""
        if self._provider is None:
            self._provider = MAPS[self.cfg.rep].build(self.cfg, torch.device(self.device))
        return self._provider

    @property
    def model(self) -> Any:
        """What IdentifiedBuild.weights_id hashes: the provider's (as RetrieverCoae: the network's FM; None for a map of the pixels)."""
        return self.provider.model

    @model.setter
    def model(self, value: Any) -> None:
        """The base class sets `model` to None; the provider's is the model here."""

    def build(self, wsi: Union[str, Any], mask: Any) -> 'RetrieverPhaseTiles':
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        super().build(wsi, mask)
        self.levels = SlideLevels(wsi, mask, self.cfg.tile_size, self.cfg.overlap, self.cfg.edge_px)
        self.reader = SlideReader(wsi)
        self.maps = {}
        self.spectra = {}
        self.spectra_bytes = 0
        self.spectra_hits = 0
        self.spectra_skipped = 0
        self.spectra_budget = int(self.cfg.spec_cache_gb * 2 ** 30)
        dev: torch.device = torch.device(self.device)
        if dev.type == 'cuda':                                   # mem_get_info wants an index ('cuda' alone has none)
            index: int = torch.cuda.current_device() if dev.index is None else int(dev.index)
            self.spectra_budget = min(self.spectra_budget, int(torch.cuda.mem_get_info(index)[0]) // 2)
        return self

    def frame(self, level: int) -> CandidateSet:
        assert self.levels is not None
        return self.levels.frame(level)

    # ── the level's maps, made once ─────────────────────────────────────────
    def prepare(self, level: int) -> None:
        if level in self.maps:
            return
        assert self.levels is not None and self.reader is not None
        cfg: RetrieverPhaseTilesConfig = self.cfg
        ds: float = self.levels.ds(level)
        grids = self.levels.geometry(level)[1]
        provider: MapProvider = self.provider
        margin: int = provider.margin_px                                                 # read round a block: the map's support
        step: int = (cfg.block_px - 2 * margin) // provider.step_px * provider.step_px
        made: List[Optional[torch.Tensor]] = [None] * len(grids)
        sizes: List[Tuple[int, int]] = [(int(g.height) // cfg.cell, int(g.width) // cfg.cell) for g in grids]       # per region (hc, wc)
        jobs: List[Tuple[int, int, int, int, int, int, int]] = []                       # (ri, y0, x0, ry0, rx0, ry1, rx1), in the order made
        for ri in range(len(grids)):
            hc, wc = sizes[ri]
            if hc < 1 or wc < 1:
                continue
            for y0 in range(0, hc * cfg.cell, step):
                for x0 in range(0, wc * cfg.cell, step):
                    ry0, rx0 = max(0, y0 - margin), max(0, x0 - margin)
                    ry1, rx1 = min(hc * cfg.cell, y0 + step + margin), min(wc * cfg.cell, x0 + step + margin)
                    jobs.append((ri, y0, x0, ry0, rx0, ry1, rx1))

        local: threading.local = threading.local()
        siblings: List[SafeSlide] = []                                                   # list.append is thread-safe

        def read(job: Tuple[int, int, int, int, int, int, int]) -> Optional[np.ndarray]:
            if not hasattr(local, 'reader'):
                s: SafeSlide = _sibling(self.reader.slide)
                siblings.append(s)
                local.reader = SlideReader(s, resize=self.reader.resize)
            ri, _, _, ry0, rx0, ry1, rx1 = job
            return local.reader.read(*grids[ri].local_to_l0(rx0, ry0), ReadSpec(rx1 - rx0, ry1 - ry0), ds, level=level)

        workers: int = max(1, self.read_workers)
        ahead: int = 2 * workers                                                         # blocks read before the map takes them
        pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=workers)
        try:
            pending: deque[Future] = deque()
            nxt: int = 0
            for job in jobs:                                                             # the results are taken in the order of `jobs`
                while nxt < len(jobs) and len(pending) < ahead:
                    pending.append(pool.submit(read, jobs[nxt]))
                    nxt += 1
                rgb: Optional[np.ndarray] = pending.popleft().result()
                if rgb is None:
                    continue
                ri, y0, x0, ry0, rx0, _, _ = job
                hc, wc = sizes[ri]
                m: torch.Tensor = provider.block_map(rgb)
                full: Optional[torch.Tensor] = made[ri]
                if full is None:
                    full = torch.zeros(m.shape[0], hc, wc, dtype=provider.dtype)
                    made[ri] = full
                oy, ox = (y0 - ry0) // cfg.cell, (x0 - rx0) // cfg.cell
                ny, nx = min(step // cfg.cell, hc - y0 // cfg.cell, m.shape[1] - oy), min(step // cfg.cell, wc - x0 // cfg.cell, m.shape[2] - ox)
                full[:, y0 // cfg.cell:y0 // cfg.cell + ny, x0 // cfg.cell:x0 // cfg.cell + nx] = m[:, oy:oy + ny, ox:ox + nx].to(provider.dtype).cpu()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            for s in siblings:
                self.reader.slide.reopens += s.reopens
                self.reader.slide.holes.extend(s.holes)
                s.close()
        self.maps[level] = made

    # ── one query ────────────────────────────────────────────────────────────
    def query_tiles(self, query: np.ndarray, rotation: int) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """The query turned by `rotation` (np.rot90, as every retriever turns it): its tiles' maps [n, C, t, t] (t = the provider's
        `tile_cells`), their top-lefts (y, x) in px, and the turned query's (h, w)."""
        cfg = self.cfg
        arr: np.ndarray = query_array(query)
        img: np.ndarray = arr if rotation == 0 else np.rot90(arr, k=rotation // 90)
        h, w = int(img.shape[0]), int(img.shape[1])
        ys: np.ndarray = np.arange(0, h - cfg.query_tile + 1, cfg.tile_stride)
        xs: np.ndarray = np.arange(0, w - cfg.query_tile + 1, cfg.tile_stride)
        ys, xs = ys + (h - cfg.query_tile - ys[-1]) // 2, xs + (w - cfg.query_tile - xs[-1]) // 2
        tl: np.ndarray = np.array([(int(y), int(x)) for y in ys for x in xs])
        maps: torch.Tensor = self.provider.tile_maps(img, tl)
        return maps, tl, (h, w)

    def block_spectrum(self, region: torch.Tensor, by: int, bx: int, blk: int, key: Optional[Tuple[int, int]]) -> Any:
        """The provider's block (`MapProvider.block`: the spectrum of a phase map) of block (by, bx) of a region map. `key` = (level, region
        index) keeps it for the slide's other queries while the budget (`spec_cache_gb`) lasts; None (a map that is not a level's
        region: a crop) keeps nothing."""
        full: Optional[Tuple[int, int, int, int, int]] = None if key is None else (key[0], key[1], by, bx, blk)
        if full is not None and full in self.spectra:
            self.spectra_hits += 1
            return self.spectra[full]
        sub: torch.Tensor = region[:, by:by + blk, bx:bx + blk].to(self.device).float()
        rs: Any = self.provider.block(sub)
        if full is not None:
            if self.spectra_bytes + rs.nbytes() <= self.spectra_budget:
                self.spectra[full] = rs
                self.spectra_bytes += rs.nbytes()
            else:
                self.spectra_skipped += 1
        return rs

    def window_scores(self, region: torch.Tensor, tiles: torch.Tensor, tl: np.ndarray, hw: Tuple[int, int]) -> torch.Tensor:
        """The score of a window of the query's size centred at every cell of a region map [C, Hc, Wc]: [Hc, Wc] (-inf where the window
        and its tiles' moves do not fit). The uncached entry: any map, e.g. a crop."""
        return self.scores(region, self.provider.query(tiles), tl, hw, None)

    def scores(self, region: torch.Tensor, spectra: Any, tl: np.ndarray, hw: Tuple[int, int],
               key: Optional[Tuple[int, int]]) -> torch.Tensor:
        """`window_scores` with the query's tiles as `spectra` (`MapProvider.query`: for a phase map their padded spectra, made once a
        block shape) and the region's blocks kept under `key` (see `block_spectrum`). Searched in blocks of `block_px` (the region can be
        large)."""
        cfg = self.cfg
        c_ = cfg.cell
        hc, wc = region.shape[1], region.shape[2]
        h, w = hw
        pr: int = cfg.pool_px // c_
        oy, ox = np.round(tl[:, 0] / c_).astype(int), np.round(tl[:, 1] / c_).astype(int)
        th: int = self.provider.tile_cells
        ext_h, ext_w = int(oy.max()) + th + 2 * pr, int(ox.max()) + th + 2 * pr        # a window's cells with its tiles' moves
        blk: int = max(cfg.block_px // c_, max(ext_h, ext_w) + 1)
        out: torch.Tensor = torch.full((hc, wc), float('-inf'), device=self.device)      # on the GPU: the lattice pooling runs on it too
        stride_y, stride_x = blk - ext_h + 1, blk - ext_w + 1
        spectra.forget(but=(blk, blk))
        for by in range(0, max(hc - ext_h, 0) + 1, max(stride_y, 1)):
            for bx in range(0, max(wc - ext_w, 0) + 1, max(stride_x, 1)):
                sh, sw = min(blk, hc - by), min(blk, wc - bx)                          # the block's size
                if sh < ext_h or sw < ext_w:
                    continue
                m: torch.Tensor = self.provider.match(self.block_spectrum(region, by, bx, blk, key), spectra)   # tile top-left at every cell
                m = TF.max_pool2d(m[:, None], kernel_size=(1, 2 * pr + 1), stride=1, padding=(0, pr))
                m = TF.max_pool2d(m, kernel_size=(2 * pr + 1, 1), stride=1, padding=(pr, 0))[:, 0]
                hs: int = min(sh - ext_h + 1, m.shape[1] - pr - int(oy.max()))
                ws: int = min(sw - ext_w + 1, m.shape[2] - pr - int(ox.max()))
                if hs <= 0 or ws <= 0:
                    continue
                sc: torch.Tensor = torch.zeros(hs, ws, device=m.device)
                for j in range(len(tl)):
                    sc += m[j, pr + oy[j]:pr + oy[j] + hs, pr + ox[j]:pr + ox[j] + ws]
                sc /= len(tl)
                # cell (i, k) of sc: the window's top-left at cell (by + pr + i, bx + pr + k); its centre h/2, w/2 px further
                cy0, cx0 = by + pr + centre_cells(h, c_), bx + pr + centre_cells(w, c_)
                ny, nx = min(hs, hc - cy0), min(ws, wc - cx0)
                if ny <= 0 or nx <= 0:
                    continue
                cur = out[cy0:cy0 + ny, cx0:cx0 + nx]
                out[cy0:cy0 + ny, cx0:cx0 + nx] = torch.maximum(cur, sc[:ny, :nx])
        return out

    def run(self, upstream: EstMppResult, query: np.ndarray) -> RetrievalOutput:
        assert self.levels is not None
        cfg = self.cfg
        q_shape: Tuple[int, ...] = tuple(query_array(query).shape)                       # the query is searched unscaled: one size at every level
        check_edge(cfg.edge_px, cfg.pool_px, int(q_shape[0]), int(q_shape[1]))
        dims: Dict[int, Tuple[int, int]] = self.levels.query_dims(query)

        def search(level: int) -> CandidateSet:
            assert self.levels is not None
            frame: CandidateSet = self.levels.frame(level)
            grids = frame.grids
            regions: List[Optional[torch.Tensor]] = self.maps[level]
            maps: Dict[Tuple[int, int, str], torch.Tensor] = {}
            shifts: Dict[Tuple[int, int, str], np.ndarray] = {}
            half: int = cfg.tile_size // 2 // cfg.cell                                  # half a lattice step, in cells
            for rot in self.levels.fitting_rotations(level, dims):
                tiles, tl, hw = self.query_tiles(query, rot)
                spectra: Any = self.provider.query(tiles)                                # one a rotation, shared by the level's regions
                rows_q, cols_q = dims[rot]
                for ri, (g, rmap) in enumerate(zip(grids, regions)):
                    if rmap is None:
                        continue
                    score: torch.Tensor = self.scores(rmap, spectra, tl, hw, (int(level), ri))  # on self.device
                    # the dense +-half pooling on the GPU (on the CPU it was 73 % of stage 2: ExpProfilePhaseTiles, 2026-11-07)
                    pooled, idx = TF.max_pool2d(score[None, None], kernel_size=2 * half + 1, stride=1, padding=half, return_indices=True)
                    pooled, idx = pooled[0, 0], idx[0, 0]
                    wc: int = score.shape[1]
                    for lattice in (('main', 'offset') if cfg.overlap else ('main',)):
                        n_r, n_c = g.lattice_dims(lattice)
                        n_r, n_c = n_r - rows_q + 1, n_c - cols_q + 1
                        if n_r < 1 or n_c < 1:
                            continue
                        lo: int = g._half(lattice)
                        yi: np.ndarray = np.clip(lattice_cells(g._main_row_starts[:n_r], lo, hw[0], cfg.cell), 0, score.shape[0] - 1)
                        xi: np.ndarray = np.clip(lattice_cells(g._main_col_starts[:n_c], lo, hw[1], cfg.cell), 0, score.shape[1] - 1)
                        yt, xt = torch.as_tensor(yi, device=pooled.device), torch.as_tensor(xi, device=pooled.device)
                        s: torch.Tensor = pooled[yt][:, xt].cpu()
                        at: torch.Tensor = idx[yt][:, xt].cpu()
                        dy = ((at // wc).numpy() - yi[:, None]) * cfg.cell
                        dx = ((at % wc).numpy() - xi[None, :]) * cfg.cell
                        maps[(int(rot), ri, lattice)] = s.float()
                        shifts[(int(rot), ri, lattice)] = np.stack([dx, dy], -1)

            def make(region: int, lattice: str, row: int, col: int, rotation: int, score_: float) -> PhaseCandidate:
                dxy = shifts[(int(rotation), region, lattice)][row, col]
                return PhaseCandidate(region, lattice, row, col, rotation, score_, shift=(int(dxy[0]), int(dxy[1])))

            cs: CandidateSet = top_candidates(maps, frame, cfg.raw_topk, cfg.min_sep_tiles, make)
            kept = tuple(c for c in cs.candidates if np.isfinite(c.score))               # windows that never fit are not candidates
            return CandidateSet(candidates=kept, level=cs.level, ds=cs.ds, grids=cs.grids, irretrievable_lvl=cs.irretrievable_lvl,
                                alter_lvl=cs.alter_lvl)

        cs: CandidateSet = self.levels.route(int(upstream.chosen_level), dims, self.prepare, search)
        return RetrievalOutput(candidates=cs)


__all__ = ['MAPS', 'MapKind', 'MapProvider', 'PHASETILES_RECIPES', 'PhaseCandidate', 'CoaeMapConfig', 'CoaeLatentMap', 'PixelMap',
           'RetrieverPhaseTiles', 'RetrieverPhaseTilesConfig', 'coae_latent_config', 'phase_map', 'register_map', 'stitch', 'tent', 'zncc_maps']
