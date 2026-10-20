"""The arms of the stage 2 retriever bench that are not the streamed sliding window: the phases they run, each
the stage 2 entry it writes.

An arm is a retriever and then any number of rerankers, named `[encoder:]Method::pooling[+op][@K]` joined by
`::` (`stage2_retrieval/ArmSpec.py`). It is the chain `ArmSpec.config_of_arm` makes -- the same configs a
pipeline run of `--stage2 arm:<arm>` holds -- and every phase of it is the stage 2 entry the pipeline files
that phase under (`BenchCommon.phase_ids`): a bench arm and a pipeline run of one config are one file.

The entries are all the bench keeps. Its rows -- each arm's ranks, in two MODES -- are read off them
(`bench_stage2_retriever.slide_rows`) by the one definition of what an answer is and where it ranks
(`BenchCommon.answers`, `ranked_ranks`, `listed_places`):

    ranked       every window of the level ranked by the arm's score (its `window_scores`), no suppression
                 (NMS off) and no list in the way: the search itself, judged strictly -- only the answer's
                 own window counts. A first phase's `truth` row holds these ranks.
    listed       what the phase hands on: its list (`raw_topk` windows, a window dropped when a better one
                 of its rotation lies within `min_sep_tiles`: NMS on), cut to the arm's `@K` when it has one;
                 an answer's place in it, the window that dropped it standing for it.

A reranker reorders a list it is given, so it has a listed mode only, judged on its chain's first phase's
answers.

HOW THE PHASES ARE RUN (`ArmEngine`). The arms form a tree of phases, a phase being shared by every arm
whose chain starts with the same phases (the node is the phase's entry). Every retriever is built from its
config (`stage2_retrieval.retriever_class`) with one `EncoderPool` for the run, so a trunk is loaded once,
and one `LevelStore`, so a level is encoded once per (encoder, pooling, cut) for all the arms that read it
and dropped when the level is done. A reranker that reuses its upstream's features (`reuse`) reads them
through the bridge lock. Rerankers that differ only in the column they order by are scored once and
reordered (`Context.reordered`): the same numbers. A sliding-window first phase is the streamed pass's
entry (`bench_stage2_retriever`), so the engine runs it for the lists after it and writes none of it.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import stage2_retrieval
from BenchCommon import ORACLE, TRUTH_VERSION, Truth, phase_ids, phase_record, phase_roles, write_phase
from PatchingLib import QueryPatchContainer
from stage2_retrieval.ArmSpec import SEP, ArmSpec, parse_arm, phase_configs
from stage2_retrieval.Context import LevelFeatures, feature_key, reordered
from stage2_retrieval.MultiPhases import MultiPhasesConfig
from stage2_retrieval.Retriever import EncoderPool, RetrievalOutput, Retriever

MODE_RANKED, MODE_LISTED = 'ranked', 'listed'
#: a sliding window scored in one streamed pass; any other retriever; a reranker
FAMILY_POOLING, FAMILY_METHOD, FAMILY_RERANK = 'pooling', 'method', 'reranker'

#: The tile every arm is cut at.
TILE: int = 256


# ── what a query is, and what a level gave ────────────────────────────────────

@dataclass(frozen=True)
class ListQuery:
    """One FoV as these arms take it: its photo, the estimate it is routed by, the level and ds it was placed
    at, `centre` the level-0 point the query's tile grid stands for and `rotation` the one that sets it
    upright -- what its stage 2 entries' truth is made of."""
    fov_id: int
    image: np.ndarray
    estimate: Any
    centre: Tuple[float, float]
    rotation: int
    level: int
    ds: float

    def truth(self) -> Truth:
        """Where the FoV was placed, as the stage 2 entry's truth table needs it."""
        centre: Tuple[float, float] = self.centre
        return Truth(level=int(self.level), ds=float(self.ds), rot=int(self.rotation), centre=lambda tile: centre)


@dataclass
class LevelResult:
    """What one (slide, level) gave: how many searches were made, and how many skipped (the first phase
    searched another level: no entry of this level holds the FoV)."""
    searched: int = 0
    skipped: int = 0


def arm_texts(specs: Sequence[ArmSpec]) -> List[str]:
    """The text of every arm the specs make, a prefix before the arms that extend it. A plain sliding window
    is the streamed pass's and is left out; one cut at @K is a list of its own row."""
    out: List[str] = []
    for spec in specs:
        for n in range(1, len(spec.segments) + 1):
            prefix: ArmSpec = spec.prefix(n)
            if prefix.retriever.method != 'SlidingWindow' or n > 1 or prefix.retriever.cut:
                if prefix.text() not in out:
                    out.append(prefix.text())
    return out



# ── the levels a run's retrievers share ───────────────────────────────────────

class LevelStore:
    """The levels the run's retrievers have encoded, by what they are (`Context.feature_key`: trunk, pooling,
    tile, lattices) and level: a level is encoded once for every arm that reads it. Only the level being run
    is kept (`keep`); the bench's alone -- a pipeline run keeps its levels in its retriever."""

    def __init__(self) -> None:
        self._levels: Dict[Tuple[Tuple[Any, ...], int], LevelFeatures] = {}
        self._level: Optional[int] = None

    def keep(self, level: Optional[int]) -> None:
        """Drop every level but `level` (None: all)."""
        self._level = level
        self._levels = {k: v for k, v in self._levels.items() if level is not None and k[1] == level}

    def view(self, key: Tuple[Any, ...]) -> '_StoreView':
        return _StoreView(self, key)


class _StoreView:
    """A `LevelStore` as one retriever's `feature_store`: the levels of its own features."""

    def __init__(self, store: LevelStore, key: Tuple[Any, ...]) -> None:
        self.store: LevelStore = store
        self.key: Tuple[Any, ...] = key

    def shared_level(self, level: int, make: Callable[[], LevelFeatures]) -> LevelFeatures:
        slot: Tuple[Tuple[Any, ...], int] = (self.key, int(level))
        if slot not in self.store._levels:
            self.store._levels[slot] = make()
        return self.store._levels[slot]


# ── the arms ──────────────────────────────────────────────────────────────────

#: failures of one arm on one slide after which it is skipped for the rest of the slide
STRIKES: int = 3


@dataclass
class _Node:
    """One phase of one or more arms: its config, its entry, and what follows it."""
    cfg: Any
    entry: str
    own: str
    after: str
    retriever: str                                # the key of the object that runs it
    texts: List[str] = field(default_factory=list)   # the arms whose last phase this is
    children: Dict[str, '_Node'] = field(default_factory=dict)


def _sibling_key(cfg: Any) -> Tuple[Any, ...]:
    """What a reranker is besides the column it orders by: two configs equal here are one scoring."""
    return (type(cfg).__name__, tuple((f.name, repr(getattr(cfg, f.name))) for f in dataclasses.fields(cfg)
                                      if f.name != 'order_by'))


class ArmEngine:
    """Every arm of a run that is not the streamed sliding window, as the tree of the phases they run (see
    the module docstring). `seg` and `region` are the mask's ids, part of every entry's id."""

    def __init__(self, specs: Sequence[ArmSpec], *, raw_topk: int, min_sep: float, device: Any,
                 encoders: EncoderPool, seg: str, region: str, read_workers: int = 0) -> None:
        self.specs: Tuple[ArmSpec, ...] = tuple(specs)
        self.raw_topk: int = int(raw_topk)
        self.min_sep: float = float(min_sep)
        self.device: Any = device
        self.encoders: EncoderPool = encoders
        self.store: LevelStore = LevelStore()
        self.read_workers: int = read_workers
        self.roots: Dict[str, _Node] = {}
        self._nodes: Dict[str, _Node] = {}           # entry name -> node
        self._of_text: Dict[str, _Node] = {}         # arm text -> its last phase
        self._objects: Dict[str, Retriever] = {}
        self._slide: Optional[Tuple[Any, Any]] = None
        for text in arm_texts(self.specs) + [s.text() for s in self.specs]:
            spec: ArmSpec = parse_arm(text)
            configs: List[Any] = phase_configs(spec, raw_topk=self.raw_topk, min_sep_tiles=self.min_sep)
            ids: List[Tuple[str, str]] = phase_ids(MultiPhasesConfig(phases=tuple(configs)), (seg, region))
            level: Dict[str, _Node] = self.roots
            after: str = ''
            node: Optional[_Node] = None
            for cfg, (name, own) in zip(configs, ids):
                if own not in level:
                    level[own] = _Node(cfg=cfg, entry=name, own=own, after=after, retriever=cfg.identity_id())
                    self._nodes[name] = level[own]
                node = level[own]
                level, after = node.children, own
            assert node is not None
            if text not in node.texts and (len(spec.segments) > 1 or spec.retriever.method != 'SlidingWindow'
                                           or spec.retriever.cut):
                node.texts.append(text)
            self._of_text[text] = node
        self._records: Dict[str, Dict[str, Any]] = {
            name: dict(phase_record(n.cfg, name, n.after, stage1=ORACLE, seg=seg, region=region),
                       truth=TRUTH_VERSION, limit=0)
            for name, n in self._nodes.items() if not self._streamed(n)}
        #: arm text -> the first error it raised. An arm that fails writes no entry and the others go on: a run
        #: of hours does not end for one method; the bench says so loudly and exits with an error.
        self.failures: Dict[str, str] = {}
        self._strikes: Dict[str, int] = {}          # arm text -> failures on the current slide
        self._sink: Optional[Any] = None

    # ── which arms and entries there are ─────────────────────────────────────
    @staticmethod
    def _streamed(node: _Node) -> bool:
        """A sliding-window first phase: the streamed pass writes its entry."""
        return not node.after and type(node.cfg).__name__ == 'RetrieverSlidingWinSimRotConfig'

    def arm_names(self) -> List[str]:
        return arm_texts(self.specs)

    def entry_of(self, text: str) -> str:
        """The stage 2 entry of the arm `text`'s last phase: the list its listed rows are read off."""
        return self._of_text[text].entry

    def first_entry_of(self, text: str) -> str:
        """The stage 2 entry of the arm `text`'s first phase: the one whose truth its rows are judged on."""
        node: _Node = self._of_text[text]
        while node.after:
            node = next(n for n in self._nodes.values() if any(c is node for c in n.children.values()))
        return node.entry

    def wanted(self) -> Dict[str, Dict[str, Any]]:
        """`{entry name: record}` of every entry the engine writes (not the streamed sliding windows')."""
        return dict(self._records)

    def roles(self) -> Dict[str, Tuple[str, ...]]:
        """`{entry name: the tables it holds}` of every entry the engine writes (`phase_roles`)."""
        return {name: phase_roles(stage2_retrieval.retriever_class(self._nodes[name].cfg),
                                  not self._nodes[name].after, True)
                for name in self._records}

    # ── one slide ────────────────────────────────────────────────────────────
    def _object(self, node: _Node) -> Retriever:
        if node.retriever not in self._objects:
            cfg: Any = node.cfg
            store: Optional[Any] = None
            if getattr(cfg, 'encoder', None) is not None and hasattr(cfg, 'tile_size'):
                store = self.store.view(feature_key(cfg.encoder, int(cfg.tile_size), bool(getattr(cfg, 'overlap', True))))
            obj: Retriever = stage2_retrieval.retriever_class(cfg)(
                cfg, self.device, encoders=self.encoders, read_workers=self.read_workers, feature_store=store)
            if self._slide is not None:
                obj.build(*self._slide)
            self._objects[node.retriever] = obj
        return self._objects[node.retriever]

    def build(self, wsi: Any, mask: Any) -> None:
        self._strikes = {}
        self._slide = (wsi, mask)
        self.store.keep(None)
        for obj in self._objects.values():
            obj.build(wsi, mask)

    def _failed(self, texts: Sequence[str], exc: BaseException) -> None:
        for arm in texts:
            self._strikes[arm] = self._strikes.get(arm, 0) + 1
            first: bool = arm not in self.failures
            self.failures.setdefault(arm, f'{type(exc).__name__}: {exc}')
            if first:
                print(f'  FAILED  {arm}   {self.failures[arm]}', flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _under(self, node: _Node) -> List[str]:
        """Every arm text that runs `node`."""
        out: List[str] = list(node.texts)
        for child in node.children.values():
            out += self._under(child)
        return out

    def _needed(self, node: _Node) -> bool:
        """Does `node`, or a phase after it, have an entry to write."""
        if self._sink is not None and self._sink.has(node.entry):
            return True
        return any(self._needed(c) for c in node.children.values())

    # ── one level ────────────────────────────────────────────────────────────
    def run(self, queries: Sequence[ListQuery], level: int, sink: Any) -> LevelResult:
        """Every phase whose entry `sink` (`ArmTables.SlideTables`) holds, and the phases before it, on every
        FoV of a level: each one's tables into its entry."""
        res: LevelResult = LevelResult()
        self._sink = sink
        self.store.keep(level)
        roots: List[_Node] = [n for n in self.roots.values() if self._needed(n)]
        for q in queries:
            qc: QueryPatchContainer = QueryPatchContainer(q.image)
            qc.extract_all(TILE, overlap=True)
            counted: bool = False
            for root in roots:
                if all(self._strikes.get(t, 0) >= STRIKES for t in self._under(root)):
                    continue
                t0: float = time.perf_counter()
                try:
                    out: RetrievalOutput = self._object(root).retrieve(q.estimate, q.image)
                except Exception as exc:                              # noqa: BLE001
                    self._failed(self._under(root), exc)
                    continue
                seconds: float = time.perf_counter() - t0
                if not len(out.candidates) or out.candidates.level != level:
                    res.skipped += 1
                    continue
                if not counted:
                    res.searched += 1
                    counted = True
                self._phase_done(root, out, q, qc, seconds)
                self._descend(root, out, q, qc)
        return res

    def _phase_done(self, node: _Node, out: RetrievalOutput, q: ListQuery, qc: QueryPatchContainer,
                    seconds: float) -> None:
        """A phase's list: its entry's tables, when the sink holds them."""
        if self._sink is not None and self._sink.has(node.entry) and not self._streamed(node):
            write_phase(self._sink.tables(node.entry), q.fov_id, out, self._object(node), q.image, qc, q.truth(),
                        TILE, True, round(seconds, 3))

    def _descend(self, node: _Node, upstream: RetrievalOutput, q: ListQuery, qc: QueryPatchContainer) -> None:
        """Run the phases after `node` on its list: one scoring per reranker, one reordering per column."""
        groups: Dict[Tuple[Any, ...], List[_Node]] = {}
        for child in node.children.values():
            if not self._needed(child) or all(self._strikes.get(t, 0) >= STRIKES for t in self._under(child)):
                continue
            groups.setdefault(_sibling_key(child.cfg), []).append(child)
        for siblings in groups.values():
            t0: float = time.perf_counter()
            try:
                first: RetrievalOutput = self._object(siblings[0]).retrieve(upstream, q.image)
            except Exception as exc:                                  # noqa: BLE001
                self._failed([t for s in siblings for t in self._under(s)], exc)
                continue
            seconds: float = time.perf_counter() - t0
            # every column's list is made from this scoring BEFORE anything below runs
            outs: List[Tuple[_Node, RetrievalOutput]] = [(siblings[0], first)] + [
                (s, reordered(upstream, first, str(s.cfg.order_by))) for s in siblings[1:]]
            for child, out in outs:
                self._phase_done(child, out, q, qc, seconds)
            for child, out in outs:
                if child.children:
                    self._descend(child, out, q, qc)


__all__ = ['ArmEngine', 'FAMILY_METHOD', 'FAMILY_POOLING', 'FAMILY_RERANK', 'LevelResult', 'LevelStore',
           'ListQuery', 'MODE_LISTED', 'MODE_RANKED', 'SEP', 'arm_texts']
