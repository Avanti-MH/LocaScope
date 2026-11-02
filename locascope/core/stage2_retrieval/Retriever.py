"""What every stage-2 method has to be, and nothing more.

    ret = SomeRetriever(cfg, device, encoders=pool).build(wsi, mask)
    out = ret.retrieve(upstream, query)              # RetrievalOutput -- always
    out = ret.retrieve(upstream, query, topk=10)     # the first 10 of the same list

A retriever eats the output of what came before it and hands back a list of candidate windows. That much is
universal, and it is the whole interface. A FIRST-PHASE retriever (the sliding window, BagOfTiles, TileVote,
ClusterBow, WinEmbed) eats stage 1's `EstMppResult` and searches a level; a RERANKER (Dual, CSLS, Corr, ...)
eats another retriever's `RetrievalOutput` and reorders the list it was handed. Nothing else tells them apart:
a reranker is a retriever whose upstream is a list. `MultiPhases` chains them.

TWO WAYS TO BE HANDED THE WINDOWS
---------------------------------
A retriever is given its windows in one of two ways, and a class says which it takes (`takes_wsi`,
`takes_candidates`, read off `UPSTREAM`):

    WSI             `retrieve(est, query)` with stage 1's `EstMppResult`. The retriever finds the level, and
                    lists the slide's whole grid of windows by itself -- no candidate set is made, so it may
                    share work between the overlapping windows (a box filter, vectors computed once for the
                    slide). It only makes sense as the first phase.
    candidate set   `retrieve(out, query)` with the `RetrievalOutput` of the phase before. It scores the
                    windows it is given and nothing else. It works at any phase. As the FIRST phase it is
                    handed every window of the slide (`all_windows`), which costs a candidate per window.

A class may take both (set `UPSTREAM = (EstMppResult, RetrievalOutput)` and implement `run_wsi` and
`run_candidates`; `run` dispatches). Which phase a method may stand at is `ArmSpec.METHODS`'s, and the
chain is refused when a method is put where it cannot work (`MultiPhases`).

What a retriever reads comes in three layers:

    1. `upstream`              stage 1's estimate, or the upstream retriever's output     (`retrieve`)
    2. the slide and the query the wsi and mask `build` binds, the query `retrieve` is given
    3. the upstream's extras   what an upstream made on the way and offers (`RetrievalOutput.extras`)

and it hands back a `RetrievalOutput`: the candidates (required) and the extras it chooses to offer.

Every parameter that changes the list is a field of the config, so the id is fixed before anything is built.
The one exception is `topk`, chosen by the consumer when it calls: the first `topk` of the list the config
makes (`raw_topk` long), a prefix, which changes no candidate's rank.

TWO WAYS TO SAVE WORK, NEITHER ALLOWED TO CHANGE A NUMBER
---------------------------------------------------------
An encoder can be handed in (`EncoderPool`): a retriever asks the pool for the encoder its config names and
gets the loaded one when the trunk is the same -- one model for uni2 read through `cls` and through
`cls_avg`. A trunk that differs is never substituted: `check_encoder` raises, and the pipeline stops.

An upstream's extras can be reused (the bridge lock): a retriever whose config says `reuse=('level_features',)`
takes the upstream's features instead of encoding them. If the upstream offers that kind under another key
(another encoder, pooling, tile, lattice), `BridgeError`: the two were meant to be the same and are not. If
the upstream offers none (its list was read back from a cache, which keeps no tensors), the retriever makes
what it needs itself, for the list it was handed -- that was its job all along; reuse is its short cut.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from locascope.core.config.ConfigIdentity import IdentifiedBuild, IdentifiedConfig, enc, parts_of
from locascope.core.stage2_retrieval.StageInterface import CandidateSet

if TYPE_CHECKING:       # annotations only
    import numpy as np
    import torch


# ══════════════════════════════════════════════════════════════════════════════
#  what a retriever is configured by and hands back
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RetrieverConfig(IdentifiedConfig):
    """The config of a retriever. No fields: each method's config defines its own. By convention the length
    of the list a first-phase method makes is `raw_topk`, and the length of the list a reranker reads (and
    hands back, reordered) is `topk_in`; both are identity.

    The class is identity too: the fields say how a method is set, not which method it is, and two methods
    each at their own baseline (BagOfTiles' `gap`, TileVote's `exact`, Dual's `dual`) differ from it in the
    same fields only -- one id, so one entry and one `_after-` for everything downstream."""

    BASELINE = {}

    def identity_parts(self) -> List[str]:
        return [f'retriever={enc(type(self).__name__)}', *parts_of(self)]


@dataclass(frozen=True)
class Extra:
    """One thing a retriever made on the way and offers downstream. `key` says exactly what it is (for level
    features: the encoder's trunk, the pooling, the tile size, whether the offset lattice is there), so a
    retriever that wants to reuse it can tell whether it is the thing it would have made."""
    kind: str
    key: Tuple[Any, ...]
    value: Any


@dataclass
class RetrievalOutput:
    """What a retriever hands back: the candidates, best first, in the frame they mean something in; the
    extras it offers; the output it was made from (`parent`, None for a first-phase retriever); and who made
    it (`producer`, the entry name when one is known)."""
    candidates: CandidateSet
    extras: Dict[str, Extra] = field(default_factory=dict)
    parent: Optional['RetrievalOutput'] = None
    producer: str = ''

    def chain(self) -> List['RetrievalOutput']:
        """This output and every output upstream of it, nearest first."""
        out: List[RetrievalOutput] = []
        node: Optional[RetrievalOutput] = self
        while node is not None:
            out.append(node)
            node = node.parent
        return out

    def find(self, kind: str) -> List[Extra]:
        """Every extra of `kind` this output and its upstream offer, nearest first."""
        return [o.extras[kind] for o in self.chain() if kind in o.extras]


@dataclass(frozen=True)
class Uses:
    """A retriever's statement that it reads an upstream extra of `kind`, and which one (`key`)."""
    kind: str
    key: Tuple[Any, ...]


class BridgeError(RuntimeError):
    """An upstream offers what a retriever said it would reuse, but not the thing it would have made."""


def take(upstream: Any, use: Uses) -> Optional[Any]:
    """The value of the upstream extra `use` names: the nearest one of its kind whose key is `use.key`.
    None when the upstream offers none of that kind (the retriever then makes it itself). An upstream
    that offers the kind only under other keys raises: the retriever said it would reuse it, so the two
    configs were meant to agree."""
    if not isinstance(upstream, RetrievalOutput):
        return None
    offered: List[Extra] = upstream.find(use.kind)
    if not offered:
        return None
    for extra in offered:
        if extra.key == use.key:
            return extra.value
    raise BridgeError(f'{use.kind}: wanted {use.key}, the upstream offers '
                      + '; '.join(str(e.key) for e in offered))


# ══════════════════════════════════════════════════════════════════════════════
#  encoders, shared by trunk
# ══════════════════════════════════════════════════════════════════════════════

def trunk_of(cfg: Any) -> Any:
    """The encoder config with its pooling taken out: what decides the loaded model. Two configs with one
    trunk differ only in how the model's tokens are reduced, so one loaded model serves both."""
    return dataclasses.replace(cfg, pooling='')


def trunk_id(cfg: Any) -> str:
    return trunk_of(cfg).identity_id()


def pooling_view(encoder: Any, pooling: str) -> Any:
    """`encoder` read through another pooling: a second view over the SAME loaded weights. The pooling is
    a reduction after the model, so changing it changes nothing loaded. A model with a learned head cannot
    be read this way (the head's output is the feature)."""
    cfg: Any = dataclasses.replace(encoder.cfg, pooling=pooling)
    if cfg.pooling == encoder.cfg.pooling:
        return encoder
    if str(getattr(encoder.cfg, 'head', '')):
        raise ValueError('an encoder with a head cannot be read through another pooling')
    clone: Any = object.__new__(type(encoder))
    clone.__dict__.update(encoder.__dict__)
    clone.cfg = cfg
    return clone


def check_encoder(built: Any, wanted: Any) -> None:
    """Raise unless `built` is the trunk `wanted` names: the same model config (precision included), the
    same transform and head. The pooling may differ (`pooling_view`)."""
    got: str = trunk_id(built.cfg)
    want: str = trunk_id(wanted)
    if got != want:
        raise ValueError(f'the encoder handed in is {type(built.cfg).__name__} {got}, the config names '
                         f'{type(wanted).__name__} {want}: a retriever never uses an encoder its config '
                         f'does not name')


class EncoderPool:
    """Loaded encoders by trunk. `get(cfg)` hands back the encoder `cfg` describes, read through its pooling:
    the one already loaded when the trunk is, built once otherwise. `offer(built)` adds one built elsewhere;
    two different loaded models under one trunk (a finetune loaded into one) are refused."""

    def __init__(self, device: Any, multi_gpu: bool = False) -> None:
        self.device: Any = device
        self.multi_gpu: bool = multi_gpu
        self._built: Dict[str, Any] = {}

    def offer(self, built: Any) -> None:
        key: str = trunk_id(built.cfg)
        held: Optional[Any] = self._built.get(key)
        if held is None:
            self._built[key] = built
            return
        if held is not built and held.model is not built.model and held.weights_id != built.weights_id:
            raise ValueError(f'two encoders under trunk {key} hold different weights')

    def get(self, cfg: Any) -> Any:
        key: str = trunk_id(cfg)
        if key not in self._built:
            self._built[key] = cfg.build(self.device, multi_gpu=self.multi_gpu)
        built: Any = self._built[key]
        check_encoder(built, cfg)
        return pooling_view(built, cfg.pooling)


# ══════════════════════════════════════════════════════════════════════════════
#  the retriever
# ══════════════════════════════════════════════════════════════════════════════

class Retriever(IdentifiedBuild):
    """Subclass, set `UPSTREAM`, implement `run` (or `run_wsi` and/or `run_candidates`, one for each interface
    the class takes). `retrieve` is the call everyone makes and no subclass
    overrides: it checks the upstream's type, runs, and cuts the list to the consumer's `topk`.

    `encoders` is a pool to take encoders from (shared with the caller and the other retrievers of a run);
    without one the retriever builds its own. `model` is None unless a subclass loads one of its own."""

    #: The upstream types `run` accepts: `(EstMppResult,)` for the WSI interface, `(RetrievalOutput,)` for the
    #: candidate-set interface, both for a method that takes either. Set by every subclass.
    UPSTREAM: Tuple[type, ...] = ()
    #: The tables `tables` writes besides the list, by name: what an entry of this retriever holds and a cache
    #: hit asks for (`BenchCommon.phase_roles`). None by default.
    TABLES: Tuple[str, ...] = ()

    def __init__(self, cfg: IdentifiedConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False) -> None:
        import torch                                                  # noqa: PLC0415
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg: IdentifiedConfig = cfg
        self.device: Any = torch.device(device)
        self.encoders: EncoderPool = encoders if encoders is not None else EncoderPool(self.device, multi_gpu)
        self.model: Optional[Any] = None
        self.wsi: Optional[Any] = None
        self.mask: Optional[Any] = None

    # ── what it needs ────────────────────────────────────────────────────────
    def encoder_needs(self) -> Dict[str, Any]:
        """The encoder configs this retriever reads with, by role. A caller that holds a pool can say which
        will be shared; the default is the config's `encoder`, when it has one."""
        enc: Optional[Any] = getattr(self.cfg, 'encoder', None)
        return {'encoder': enc} if enc is not None else {}

    def uses(self) -> Tuple[Uses, ...]:
        """The upstream extras this retriever reuses (the bridge lock). None by default."""
        return ()

    def list_limit(self) -> int:
        """The longest list this retriever hands back: the config's `raw_topk` (first phase) or `topk_in`
        (reranker). A consumer asking for more is refused."""
        for name in ('raw_topk', 'topk_in'):
            value: Optional[int] = getattr(self.cfg, name, None)
            if value is not None:
                return int(value)
        raise NotImplementedError(f'{type(self.cfg).__name__} has neither raw_topk nor topk_in')

    # ── one slide ────────────────────────────────────────────────────────────
    def build(self, wsi: Any, mask: Any) -> 'Retriever':
        """Bind a slide and its mask; forget what was kept for the last one."""
        self.wsi, self.mask = wsi, mask
        return self

    # ── one query ────────────────────────────────────────────────────────────
    def retrieve(self, upstream: Any, query: 'np.ndarray', *, topk: Optional[int] = None) -> RetrievalOutput:
        """The list, cut to the consumer's `topk` (None: the whole list). Raises on an upstream this
        retriever does not eat and on a `topk` longer than the list its config makes."""
        if self.wsi is None:
            raise RuntimeError(f'{type(self).__name__}: call build(wsi, mask) first')
        if self.UPSTREAM and not isinstance(upstream, self.UPSTREAM):
            raise TypeError(f'{type(self).__name__} eats {" or ".join(t.__name__ for t in self.UPSTREAM)}, '
                            f'not {type(upstream).__name__}')
        if topk is not None and int(topk) > self.list_limit():
            raise ValueError(f'topk {topk} is longer than the {self.list_limit()} this config lists')
        out: RetrievalOutput = self.run(upstream, query)
        return cut(out, topk)

    @classmethod
    def takes_wsi(cls) -> bool:
        """True when the class takes stage 1's estimate and finds the slide's windows itself (the WSI
        interface)."""
        from locascope.core.stage1_estimation.StageInterface import EstMppResult     # noqa: PLC0415
        return EstMppResult in cls.UPSTREAM

    @classmethod
    def takes_candidates(cls) -> bool:
        """True when the class takes the list of the phase before (the candidate-set interface)."""
        return RetrievalOutput in cls.UPSTREAM

    def run(self, upstream: Any, query: 'np.ndarray') -> RetrievalOutput:
        """The whole list (`list_limit` long, or every window there is when fewer exist). By default the
        upstream's type picks `run_wsi` or `run_candidates`; a subclass with one interface may override `run`
        instead."""
        from locascope.core.stage1_estimation.StageInterface import EstMppResult     # noqa: PLC0415
        if isinstance(upstream, EstMppResult):
            return self.run_wsi(upstream, query)
        return self.run_candidates(upstream, query)

    def run_wsi(self, upstream: Any, query: 'np.ndarray') -> RetrievalOutput:
        """The WSI interface: search the slide's whole grid of windows (the list as `run` describes it)."""
        raise NotImplementedError(f'{type(self).__name__} does not take stage 1\'s estimate: it needs a list')

    def run_candidates(self, upstream: RetrievalOutput, query: 'np.ndarray') -> RetrievalOutput:
        """The candidate-set interface: score the windows of `upstream` (the list as `run` describes it)."""
        raise NotImplementedError(f'{type(self).__name__} does not take a list')

    def all_windows(self, upstream: Any, query: 'np.ndarray') -> RetrievalOutput:
        """Every window of the slide the query fits, as a list of candidates scored 0, for a method that
        takes candidates and stands first (`MultiPhases` hands it this). The level is the one the WSI interface
        would search (`SlideLevels.route`: stage 1's, or the nearest finer one that holds the query), at every
        rotation the query fits at, on both lattices. The list carries no extras and no order. It is a
        candidate per window -- some millions for a large slide -- which is what the WSI interface exists to
        avoid."""
        from locascope.core.stage2_retrieval.Context import SlideLevels              # noqa: PLC0415
        from locascope.core.stage2_retrieval.StageInterface import Candidate         # noqa: PLC0415
        tile: int = int(getattr(self.cfg, 'tile_size', 256))
        overlap: bool = bool(getattr(self.cfg, 'overlap', True))
        if self.wsi is None:
            raise RuntimeError(f'{type(self).__name__}: call build(wsi, mask) first')
        levels: SlideLevels = SlideLevels(self.wsi, self.mask, tile, overlap)
        dims: Dict[int, Tuple[int, int]] = levels.query_dims(query)

        def search(level: int) -> CandidateSet:
            frame: CandidateSet = levels.frame(level)
            found: List[Any] = []
            for rotation in levels.fitting_rotations(level, dims):
                rows, cols = dims[rotation]
                for region, grid in enumerate(frame.grids):
                    for lattice, (g_rows, g_cols) in (('main', (grid.grid_rows, grid.grid_cols)),
                                                      ('offset', (grid.overlap_rows, grid.overlap_cols))):
                        found += [Candidate(region, lattice, i, j, int(rotation), 0.0)
                                  for i in range(int(g_rows) - rows + 1) for j in range(int(g_cols) - cols + 1)]
            return dataclasses.replace(frame, candidates=tuple(found))

        return RetrievalOutput(candidates=levels.route(int(upstream.chosen_level), dims,
                                                       lambda level: None, search))

    def tables(self, out: RetrievalOutput, qc: Any) -> Dict[str, List[Dict[str, Any]]]:
        """The extras of `out` this retriever writes as tables of its entry, by table name. None by default."""
        return {}


def cut(out: RetrievalOutput, topk: Optional[int]) -> RetrievalOutput:
    """`out` with its first `topk` candidates (all of them for None). The extras stay: they describe the
    search, not the cut."""
    if topk is None or len(out.candidates) <= int(topk):
        return out
    return dataclasses.replace(out, candidates=dataclasses.replace(
        out.candidates, candidates=out.candidates.candidates[:int(topk)]))


__all__ = ['BridgeError', 'EncoderPool', 'Extra', 'RetrievalOutput', 'Retriever', 'RetrieverConfig', 'Uses',
           'check_encoder', 'cut', 'pooling_view', 'take', 'trunk_id', 'trunk_of']
