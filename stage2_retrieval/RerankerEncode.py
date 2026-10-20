"""Rerankers that read pixels and encode them now. Training-free, and dear: they are for the candidates that
survived the cheap rerankers, a few dozen per query, not the hundreds the first phase proposed.

    RerankerShift       the window moved by less than a tile, re-encoded
    RerankerTokenMatch  patch tokens matched across the window, one transform
    RerankerXEnc        the candidates scored again by another encoder

Each reads tiles through its own `OnDemandTiles` (`Context.RerankFeatures.tiles`), with the encoder its
config names: neighbouring windows share tiles, and a tile asked about twice is encoded once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from ConfigIdentity import register
from TileEncoderFunc import TileEncoderConfig
from stage2_retrieval.Context import ScoringInput, WindowBatch, rotated_query_tiles, tile_vectors
from stage2_retrieval.Retriever import RetrieverConfig
from stage2_retrieval.RerankerScorers import ColumnRanker
from stage2_retrieval.StageInterface import Candidate


def _tile_origins(inp: ScoringInput, c: Candidate, shape: Tuple[int, int],
                  dx: int = 0, dy: int = 0) -> List[Tuple[int, int]]:
    """Level-0 top-left corners of a window's tiles, row-major, the window moved
    by `(dx, dy)` level px from where the candidate says it is."""
    ox, oy = inp.cs.origin_l0(c)
    ds: float = float(inp.cs.ds)
    step: float = float(inp.cs.grids[c.region_index].tile_size) * ds
    bx: int = ox + int(round(dx * ds))
    by: int = oy + int(round(dy * ds))
    return [(bx + int(round(j * step)), by + int(round(i * step)))
            for i in range(shape[0]) for j in range(shape[1])]


# ══════════════════════════════════════════════════════════════════════════════
#  S3  a window between the lattice's windows
# ══════════════════════════════════════════════════════════════════════════════

@register('reranker_shift')
@dataclass(frozen=True)
class RerankerShiftConfig(RetrieverConfig):
    """The encoder, tile cut, lattices, `topk_in` and `reuse` as `RerankerDualConfig`. Shifts are
    `stride_px * {-radius .. radius}` level px in x and in y, from where the candidate is. Ordered by
    `shift`."""
    encoder:   TileEncoderConfig
    tile_size: int = 256
    overlap:   bool = True
    topk_in:   int = 1000
    stride_px: int = 64
    radius:    int = 1
    reuse:     Tuple[str, ...] = ()

    BASELINE = {'tile_size': 256, 'overlap': True, 'topk_in': 1000, 'stride_px': 64, 'radius': 1}
    NOT_IDENTITY = ('reuse',)
    VERSION = 1

    def __post_init__(self) -> None:
        if self.stride_px < 1 or self.radius < 1:
            raise ValueError('stride_px >= 1 and radius >= 1 are required')


class RerankerShift(ColumnRanker):
    """A photograph lands wherever it lands, and the retriever only looks at the windows of two lattices half
    a tile apart: a window whose truth is a quarter of a tile off scores less than the truth would at the
    right place. Here each candidate is also tried at the shifts around it -- new tiles, read and encoded --
    and keeps the best.

    Columns: `shift` (the best score of the window and its shifts), `shift_dx` and `shift_dy` (the position it
    was best at, level px from the lattice window: what makes the candidate a `ShiftedCandidate`) and
    `shift_gain` (best minus the window unshifted)."""

    READS_TILES = True

    @property
    def column(self) -> str:
        return 'shift'

    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        pooling: str = inp.pooling()
        batch: WindowBatch = inp.batch(pooling)
        stride, radius = int(self.cfg.stride_px), int(self.cfg.radius)
        shifts: List[Tuple[int, int]] = [(dx * stride, dy * stride)
                                         for dy in range(-radius, radius + 1)
                                         for dx in range(-radius, radius + 1) if (dx, dy) != (0, 0)]
        table: List[Tuple[int, int]] = [(0, 0)] + shifts
        best: List[torch.Tensor] = []
        gain: List[torch.Tensor] = []
        dxs: List[torch.Tensor] = []
        dys: List[torch.Tensor] = []
        for grp in batch.groups:
            cands: List[Candidate] = [inp.cs.candidates[i] for i in grp.index.tolist()]
            origins: List[Tuple[int, int]] = []
            for c in cands:
                for dx, dy in shifts:
                    origins += _tile_origins(inp, c, grp.shape, dx, dy)
            vec: torch.Tensor = inp.encode_tiles(torch.tensor(origins, dtype=torch.long), pooling)
            vec = vec.reshape(len(cands), len(shifts), grp.shape[0], grp.shape[1], -1)
            moved: torch.Tensor = (vec.float() * grp.q.float().unsqueeze(1)).sum(-1).mean(dim=(2, 3))   # [G, S]
            here: torch.Tensor = grp.cos.mean(dim=(1, 2))                                                 # [G]
            every: torch.Tensor = torch.cat([here[:, None], moved], dim=1)                                # [G, S + 1]
            top, arg = every.max(dim=1)
            at: torch.Tensor = torch.tensor(table, dtype=torch.float32, device=arg.device)[arg]           # [G, 2]
            base: torch.Tensor = torch.tensor([c.shift_px() for c in cands], dtype=torch.float32,
                                              device=arg.device)
            best.append(top)
            gain.append(top - here)
            dxs.append(base[:, 0] + at[:, 0])
            dys.append(base[:, 1] + at[:, 1])
        return {'shift': batch.scatter(best), 'shift_gain': batch.scatter(gain),
                'shift_dx': batch.scatter(dxs), 'shift_dy': batch.scatter(dys)}


# ══════════════════════════════════════════════════════════════════════════════
#  S4  tokens, one transform
# ══════════════════════════════════════════════════════════════════════════════

@register('reranker_tokenmatch')
@dataclass(frozen=True)
class RerankerTokenMatchConfig(RetrieverConfig):
    """The encoder whose patch tokens are matched and the tile cut, `topk_in`; a token match is a pair of patch
    tokens, one of the query and one of the window, that are each other's nearest neighbour with cosine at
    least `sim_min`. At most `top_matches` (best first) are kept. `hypotheses` transforms (a scale and a
    shift, solved from two matches that are at least `min_span_px` apart, with the scale inside
    `scale_range`) are tried, seeded; a match agrees with one within `tol_px` level px. `chunk` is memory.
    Ordered by `tokinl`. The tokens are the model's own, so the encoder's pooling plays no part."""
    encoder:     TileEncoderConfig
    tile_size:   int = 256
    topk_in:     int = 1000
    sim_min:     float = 0.5
    tol_px:      float = 24.0
    hypotheses:  int = 512
    top_matches: int = 1500
    min_span_px: float = 128.0
    scale_range: Tuple[float, float] = (0.8, 1.25)
    seed:        int = 0
    chunk:       int = 25

    BASELINE = {'tile_size': 256, 'topk_in': 1000, 'sim_min': 0.5, 'tol_px': 24.0, 'hypotheses': 512,
                'top_matches': 1500, 'min_span_px': 128.0, 'scale_range': (0.8, 1.25), 'seed': 0}
    NOT_IDENTITY = ('chunk',)
    VERSION = 1

    def __post_init__(self) -> None:
        if self.hypotheses < 1 or self.top_matches < 2 or self.chunk < 1:
            raise ValueError('hypotheses >= 1, top_matches >= 2 and chunk >= 1 are required')
        if len(self.scale_range) != 2 or not 0 < self.scale_range[0] <= self.scale_range[1]:
            raise ValueError(f'scale_range must be (low, high) with 0 < low <= high, got {self.scale_range}')


class RerankerTokenMatch(ColumnRanker):
    """The retriever compares tile vectors: one number per tile. Inside a tile a pair of windows that really
    are the same place agree on WHERE things are -- every patch token that finds its match finds it at the same
    displacement -- and a pair that only look alike do not. This looks for that: the model's own patch tokens,
    matched across the whole window, and the largest set of matches that one transform (a scale and a shift;
    the rotation is the candidate's) explains.

    Columns: `tokinl` (inliers over the tokens compared), `n_inliers` and `inlier_ratio` (what makes the
    candidate a `TokenMatchCandidate`)."""

    READS_FEATURES = False
    READS_TILES = True

    @property
    def column(self) -> str:
        return 'tokinl'

    @staticmethod
    def _cells(enc: Any, tile: int) -> Tuple[int, torch.Tensor]:
        """`(prefix, cell centres)`: how many tokens come before the patch grid and each patch cell's centre
        in level px from its tile's corner."""
        spec: Any = enc.model_spec
        prefix: int = int(spec['num_prefix'])
        gh, gw = (int(v) for v in spec['feat_hw'])
        ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing='ij')
        centre: torch.Tensor = torch.stack([(xs.reshape(-1) + 0.5) * tile / gw,
                                            (ys.reshape(-1) + 0.5) * tile / gh], dim=1)
        return prefix, centre

    @staticmethod
    def _positions(shape: Tuple[int, int], tile: int, cells: torch.Tensor) -> torch.Tensor:
        """`[rows * cols * P, 2]` (x, y) of every patch token of a window, in the window's frame: tile `(i, j)`
        starts at `(j * tile, i * tile)`."""
        corner: torch.Tensor = torch.tensor([(j * tile, i * tile) for i in range(shape[0])
                                             for j in range(shape[1])], dtype=torch.float32)
        return (corner[:, None, :] + cells[None, :, :]).reshape(-1, 2)

    def _ransac(self, pq: torch.Tensor, pw: torch.Tensor, gen: torch.Generator) -> int:
        """The most matches one `(scale, shift)` explains."""
        m: int = int(pq.shape[0])
        if m < 2:
            return 0
        h: int = int(self.cfg.hypotheses)
        ia: torch.Tensor = torch.randint(0, m, (h,), generator=gen).to(pq.device)
        ib: torch.Tensor = torch.randint(0, m, (h,), generator=gen).to(pq.device)
        dq: torch.Tensor = pq[ib] - pq[ia]
        dw: torch.Tensor = pw[ib] - pw[ia]
        span2: torch.Tensor = (dq * dq).sum(dim=1)
        scale: torch.Tensor = (dw * dq).sum(dim=1) / span2.clamp(min=1e-6)
        lo, hi = (float(v) for v in self.cfg.scale_range)
        ok: torch.Tensor = (span2 >= float(self.cfg.min_span_px) ** 2) & (scale >= lo) & (scale <= hi)
        shift: torch.Tensor = pw[ia] - scale[:, None] * pq[ia]
        pred: torch.Tensor = scale[:, None, None] * pq[None] + shift[:, None, :]            # [H, M, 2]
        near: torch.Tensor = (pred - pw[None]).norm(dim=2) <= float(self.cfg.tol_px)
        counts: torch.Tensor = (near & ok[:, None]).sum(dim=1)
        return int(counts.max())

    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        enc: Any = self.encoder
        tile: int = int(inp.tile_size)
        prefix, cells = self._cells(enc, tile)
        per_tile: int = int(cells.shape[0])
        gen: torch.Generator = torch.Generator().manual_seed(int(self.cfg.seed))
        kept: Dict[int, torch.Tensor] = {}                       # rotation -> [n * P, D] unit tokens
        for rot in sorted({c.rotation for c in inp.cs.candidates}):
            tiles, _ = rotated_query_tiles(inp.query, rot, tile)
            tok: torch.Tensor = enc.tokens(tiles)[:, prefix:]
            kept[rot] = F.normalize(tok, dim=-1).reshape(-1, tok.shape[-1]).half()
        inliers: List[float] = []
        ratio: List[float] = []
        cands: List[Candidate] = list(inp.cs.candidates)
        for start in range(0, len(cands), int(self.cfg.chunk)):
            group: List[Candidate] = cands[start:start + int(self.cfg.chunk)]
            # the tiles of the whole chunk are read in one call: a read starts workers that each open the
            # slide, which costs more than the tiles do when it is made once a candidate
            origins: List[Tuple[int, int]] = []
            for c in group:
                origins += _tile_origins(inp, c, inp.shape_of(c.rotation))
            raw_all: torch.Tensor = inp.encode_tiles(torch.tensor(origins, dtype=torch.long), '', raw=True)
            taken: int = 0
            for c in group:
                shape: Tuple[int, int] = inp.shape_of(c.rotation)
                n_tok: int = shape[0] * shape[1] * per_tile
                raw: torch.Tensor = raw_all[taken:taken + shape[0] * shape[1]]
                taken += shape[0] * shape[1]
                win: torch.Tensor = F.normalize(raw[:, prefix:], dim=-1).reshape(-1, raw.shape[-1]).half()
                sim: torch.Tensor = kept[c.rotation] @ win.T                                    # [n_tok, n_tok]
                best_q: torch.Tensor = sim.argmax(dim=1)
                best_w: torch.Tensor = sim.argmax(dim=0)
                idx: torch.Tensor = torch.arange(n_tok, device=sim.device)
                value: torch.Tensor = sim[idx, best_q].float()
                mutual: torch.Tensor = (best_w[best_q] == idx) & (value >= float(self.cfg.sim_min))
                pick: torch.Tensor = idx[mutual]
                if len(pick) > int(self.cfg.top_matches):
                    pick = pick[value[pick].topk(int(self.cfg.top_matches)).indices]
                qpos: torch.Tensor = self._positions(shape, tile, cells).to(sim.device)
                n_in: int = self._ransac(qpos[pick], qpos[best_q[pick]], gen)
                inliers.append(float(n_in))
                ratio.append(n_in / float(n_tok))
        dev: Any = inp.table['in'].device
        r: torch.Tensor = torch.tensor(ratio, dtype=torch.float32, device=dev)
        return {'tokinl': r, 'n_inliers': torch.tensor(inliers, dtype=torch.float32, device=dev),
                'inlier_ratio': r}


# ══════════════════════════════════════════════════════════════════════════════
#  another encoder
# ══════════════════════════════════════════════════════════════════════════════

@register('reranker_xenc')
@dataclass(frozen=True)
class RerankerXEncConfig(RetrieverConfig):
    """`encoder` (pooling included) scores the candidates again; `tile_size` its cut; `topk_in` read. Ordered
    by `xenc`."""
    encoder:   TileEncoderConfig
    tile_size: int = 256
    topk_in:   int = 1000

    BASELINE = {'tile_size': 256, 'topk_in': 1000}
    VERSION = 1


class RerankerXEnc(ColumnRanker):
    """Two encoders disagree in different places. The first phase picked these candidates; another encoder
    reads the same tiles and scores each window as the sliding window does -- the mean cosine of the tiles
    under the query's -- and the two scores are evidence for a stacker. Column `xenc`."""

    READS_FEATURES = False
    READS_TILES = True

    @property
    def column(self) -> str:
        return 'xenc'

    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        enc: Any = self.encoder
        pooling: str = enc.feature_pooling
        queries: Dict[int, torch.Tensor] = {}
        for rot in sorted({c.rotation for c in inp.cs.candidates}):
            images, shape = rotated_query_tiles(inp.query, rot, int(inp.tile_size))
            queries[rot] = tile_vectors(enc, images, pooling).reshape(shape[0], shape[1], -1)
        out: List[float] = []
        assert inp.tiles is not None
        for c in inp.cs.candidates:
            q: torch.Tensor = queries[c.rotation]
            shape = (int(q.shape[0]), int(q.shape[1]))
            w: torch.Tensor = inp.tiles.vectors(_tile_origins(inp, c, shape), pooling).reshape(q.shape)
            out.append(float((q.float() * w.float()).sum(-1).mean()))
        return {'xenc': torch.tensor(out, dtype=torch.float32, device=inp.table['in'].device)}


__all__ = ['RerankerShift', 'RerankerShiftConfig', 'RerankerTokenMatch', 'RerankerTokenMatchConfig',
           'RerankerXEnc', 'RerankerXEncConfig']
