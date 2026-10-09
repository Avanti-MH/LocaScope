"""LocaScope end-to-end 3-stage pipeline glue.

Strings the stages together over one WSI. It knows each stage only by its
interface, so any method of a stage drops in and no concrete class is named
here:

    Stage 1 -- mpp estimation   MppEstimator   stage1_estimation/StageInterface.py
                estimate(query)                      -> EstMppResult
    Stage 2 -- retrieval        Retriever, then zero or more Rerankers
                                               stage2_retrieval/StageInterface.py
                retrieve(query, EstMppResult)        -> CandidateSet
                rerank(query, CandidateSet)          -> CandidateSet
    Stage 3 -- localization     Localizer      stage3_localization/StageInterface.py
                localize(query, CandidateSet, topk)  -> LocalizationResultSet

Each stage is handed the query image and the previous stage's output.

What this class does, and the only things it does:

* build(): the mask, from `masks` (a `TissueMaskConfig.MaskMaker`: its recipe
  and, when it has one, its cache -- a slide whose mask is cached is never
  segmented again), then binds every stage to the slide and that ONE mask, so
  stage 1's reference bank and stage 2's windows agree on what tissue is.
* run(): the three stages in a row, each timed, each error kept with the
  results of the stages before it.

What it does NOT do: choose a method or a parameter of one, build a pyramid
level, or read a stage's internals. Which level stage 2 searches, and what it
does when the routed one cannot hold the query, is the retriever's own business
(`CandidateSet.level`, `irretrievable_lvl`, `alter_lvl`); a set with no
candidates means no level could be searched, and stage 3 is then not run.

A STAGE comes as a spec, or already built:

* "method:recipe" -- a recipe of that stage's package (`stage1_estimation.
  recipe`, ...), built here on `build()`;
* `(config, class)` -- the same, with the config already resolved (a recipe
  with flags applied);
* an object -- a stage already built, used as it is. A loop over slides builds
  its stages once, passes them to a pipeline per slide, and no model is loaded
  twice; a stage not needed (its results are cached) is None.

The runtime arguments of a build (`device`, `multi_gpu`, `read_workers`) are
given to a stage only if its constructor takes them. Rerankers have no recipe
table yet: they come as `(config, class)` or built.

Usage:

    masks = MaskMaker(MASK_RECIPES['hest'], job, device)   # job's mask cache
    pl = LocaScopePipeline(wsi, masks, stage1='knn:gigapath',
                           stage2='slidewin:gigapath', stage3='sift:default',
                           device=device).build()
    result = pl.run(shot_img)       # result.stage1, .stage2, .stage3

    r1 = pl.stage1(shot_img)        # or each stage alone, for a caller that
    cs = pl.stage2(shot_img, r1)    # keeps what each made
    rs = pl.stage3(shot_img, cs)
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import stage1_estimation
import stage2_retrieval
import stage3_localization
from SafeSlide                          import SafeSlide                      # noqa: E402
from TissueMask                         import TissueMask                     # noqa: E402
from stage1_estimation.StageInterface   import EstMppResult, MppEstimator     # noqa: E402
from stage2_retrieval.StageInterface    import CandidateSet, Reranker, Retriever  # noqa: E402
from stage3_localization.StageInterface import Localizer, LocalizationResultSet   # noqa: E402

if TYPE_CHECKING:       # annotations only
    from TissueMaskConfig import MaskMaker, TissueMaskConfig

__all__ = ['LocaScopePipeline', 'LocaScopeQueryResult', 'StageSpec',
           'construct_stage']

#: "method:recipe", a resolved `(config, class)`, or a stage already built.
StageSpec = Union[str, Tuple[Any, type], object]

#: The package that holds each stage's recipe tables.
_PACKAGES: Dict[int, Any] = {1: stage1_estimation, 2: stage2_retrieval,
                             3: stage3_localization}


def construct_stage(cls: type, cfg: Any, runtime: Dict[str, Any]) -> Any:
    """A stage built from its class and config. `runtime` is how it is built
    (`device`, `multi_gpu`, `read_workers`), not what it is; a constructor takes
    the ones it declares and no others, so a stage that needs no device (stage
    3's SIFT) is not handed one. The one place a stage is built: the pipeline
    builds its string and `(config, class)` specs here, and the bench's `Stage`
    builds its own the same way."""
    accepted: Dict[str, Any] = {
        k: v for k, v in runtime.items()
        if k in inspect.signature(cls.__init__).parameters}
    return cls(cfg, **accepted)


@dataclass
class LocaScopeQueryResult:
    """Per-shot pipeline output: each stage's own result, as its interface
    defines it, and what went wrong.

    A stage that did not run is None. `stage2` with no candidates means no
    level could be searched; `stage3` is then None. `error` names the stage
    that raised and why, the earlier stages' results kept.
    """
    stage1:     Optional[EstMppResult]
    stage2:     Optional[CandidateSet]
    stage3:     Optional[LocalizationResultSet]
    error:      Optional[str]
    # Wall seconds per stage of THIS shot; None for a stage that did not run.
    t_stage1_s: Optional[float]
    t_stage2_s: Optional[float]
    t_stage3_s: Optional[float]


class LocaScopePipeline:
    """3-stage LocaScope pipeline over one WSI."""

    def __init__(
        self,
        wsi:            Union[str, SafeSlide],
        masks:          MaskMaker,
        stage1:         Optional[StageSpec] = None,   # None: not run here
        stage2:         Optional[StageSpec] = None,
        stage3:         Optional[StageSpec] = None,
        rerankers:      Sequence[StageSpec] = (),
        topk:           int = 10,                      # candidates stage 3 verifies
        bank_cache_job: Optional[str] = None,          # stage 1's bank; None: in memory
        device:         Optional[torch.device] = None,
        multi_gpu:      bool = False,
        read_workers:   int = 0,
    ) -> None:
        # SafeSlide, not OpenSlide: a MIRAX read that lands on a cell the
        # scanner never wrote raises, and that raise latches on the handle, so
        # every later call fails -- metadata included. Since this one object is
        # handed to every stage, the recovery has to live inside it; healing
        # swaps the native handle in place and every holder keeps working.
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi: SafeSlide = wsi
        # Where the mask comes from; its recipe is how the mask was built.
        self.masks: MaskMaker = masks
        self.mask_cfg: TissueMaskConfig = masks.cfg
        self.topk: int = topk
        self.bank_cache_job: Optional[str] = bank_cache_job
        # SafeSlide.base_mpp: the one definition of the slide's scale.
        self.base_mpp: float = wsi.base_mpp   # raises if the slide carries no mpp

        self.mask: Optional[TissueMask] = None
        self._specs: Dict[int, Optional[StageSpec]] = {1: stage1, 2: stage2, 3: stage3}
        self._rerank_specs: Tuple[StageSpec, ...] = tuple(rerankers)
        self._runtime: Dict[str, Any] = dict(device=device, multi_gpu=multi_gpu,
                                             read_workers=read_workers)
        # Built by build(), each from its spec; None where no spec was given.
        self.estimator: Optional[MppEstimator] = None
        self.retriever: Optional[Retriever] = None
        self.rerankers: Tuple[Reranker, ...] = ()
        self.localizer: Optional[Localizer] = None
        self._built: bool = False

    # ── One-time setup ────────────────────────────────────────────────────────
    def _make(self, n: int, spec: Optional[StageSpec]) -> Optional[Any]:
        """The stage object of `spec` (see the module docstring). `n` is the
        stage whose recipe table a "method:recipe" string is looked up in; 0
        for a reranker, which has none."""
        if spec is None:
            return None
        cfg: Any
        cls: type
        if isinstance(spec, str):
            if n not in _PACKAGES:
                raise ValueError(f'a reranker has no recipe table: {spec!r}; '
                                 f'give (config, class) or a built one')
            cfg, cls = _PACKAGES[n].recipe(spec)[2:]
        elif isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], type):
            cfg, cls = spec
        else:
            return spec
        return construct_stage(cls, cfg, self._runtime)

    def build(self) -> 'LocaScopePipeline':
        """Build the mask, then build and bind every stage to this slide and
        that mask (per-WSI one-time). No pyramid level is touched here."""
        # Segment (or read the cached raw mask), filter and merge in one place
        # and in one order: `MaskMaker.mask`.
        self.mask, _ = self.masks.mask(self.wsi)

        self.estimator = self._make(1, self._specs[1])
        self.retriever = self._make(2, self._specs[2])
        self.rerankers = tuple(self._make(0, s) for s in self._rerank_specs)
        self.localizer = self._make(3, self._specs[3])

        # The same mask to every stage that takes one, so stage 1's reference
        # bank and stage 2's windows agree on which regions are tissue (a method
        # that samples nothing ignores it -- StageInterface.MppEstimator).
        if self.estimator is not None:
            self.estimator.build(self.wsi, mask=self.mask, masks=self.masks,
                                 cache_job=self.bank_cache_job)
        if self.retriever is not None:
            self.retriever.build(self.wsi, self.mask)
        reranker: Reranker
        for reranker in self.rerankers:
            reranker.build(self.wsi, self.mask)
        if self.localizer is not None:
            self.localizer.build(self.wsi)
        self._built = True
        return self

    # ── One stage at a time ───────────────────────────────────────────────────
    #
    # A caller that keeps each stage's output (the bench) runs them one by one:
    # a stage it already has is read back instead, and the next one is handed
    # what was read. `run` is the three in a row.

    def stage1(self, img: np.ndarray) -> EstMppResult:
        """Stage 1: the mpp and the level it routes to."""
        if self.estimator is None:
            raise RuntimeError('stage 1 was not given to this pipeline')
        return self.estimator.estimate(img)

    def stage2(self, img: np.ndarray, r1: EstMppResult) -> CandidateSet:
        """Stage 2 on stage 1's output: the retriever, then every reranker in
        order. No candidates means no level could be searched."""
        if self.retriever is None:
            raise RuntimeError('stage 2 was not given to this pipeline')
        cs: CandidateSet = self.retriever.retrieve(img, r1)
        reranker: Reranker
        for reranker in self.rerankers:
            cs = reranker.rerank(img, cs)
        return cs

    def stage3(self, img: np.ndarray, cs: CandidateSet) -> LocalizationResultSet:
        """Stage 3 on stage 2's output: the first `topk` candidates verified."""
        if self.localizer is None:
            raise RuntimeError('stage 3 was not given to this pipeline')
        return self.localizer.localize(img, cs, self.topk)

    # ── Per-shot end-to-end ───────────────────────────────────────────────────
    def run(self, img: np.ndarray) -> LocaScopeQueryResult:
        """Run all 3 stages on one shot image, each on the previous one's
        output: EstMppResult -> CandidateSet -> LocalizationResultSet."""
        if not self._built:
            raise RuntimeError('LocaScopePipeline not built; call .build() first.')

        # Each stage ends in host values (an mpp, topk lists, a position), so
        # the GPU has finished by the time a clock is read.
        t0: float = time.perf_counter()
        try:
            r1: EstMppResult = self.stage1(img)
        except Exception as e:                                       # noqa: BLE001
            return LocaScopeQueryResult(
                None, None, None, f'stage1 failed: {type(e).__name__}: {e}',
                None, None, None)
        t1: float = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            cs: CandidateSet = self.stage2(img, r1)
        except Exception as e:                                       # noqa: BLE001
            return LocaScopeQueryResult(
                r1, None, None, f'stage2 failed: {type(e).__name__}: {e}',
                t1, None, None)
        t2: float = time.perf_counter() - t0
        if not len(cs):
            return LocaScopeQueryResult(r1, cs, None, None, t1, t2, None)

        t0 = time.perf_counter()
        try:
            rs: LocalizationResultSet = self.stage3(img, cs)
        except Exception as e:                                       # noqa: BLE001
            return LocaScopeQueryResult(
                r1, cs, None, f'stage3 failed: {type(e).__name__}: {e}',
                t1, t2, None)
        t3: float = time.perf_counter() - t0

        return LocaScopeQueryResult(r1, cs, rs, None, t1, t2, t3)
