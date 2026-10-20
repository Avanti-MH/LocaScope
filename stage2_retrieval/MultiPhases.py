"""Retrievers in a row: the output of each is the upstream of the next.

    cfg = MULTIPHASES_RECIPES['uni2-avg-dual']        # the sliding window, then Dual on its list
    r   = MultiPhases(cfg, device).build(wsi, mask)
    out = r.retrieve(est_mpp_result, shot_img)        # the last phase's RetrievalOutput

The first phase eats stage 1's estimate and lists windows; every later one eats the output before it
(`Retriever.UPSTREAM` says which a class eats, and the config is refused when the chain cannot be run). An
arm of the bench, `A::B::C`, is the same thing (`ArmSpec.config_of_arm`).

Every phase is an entry of its own: its id is its config's, with the upstream phase's hash among its parts
and in its name (`_after-`), so a phase run alone and the same phase in a chain are one file, and two chains
that share a prefix share its entries. `run` takes a `cached` hook for that: a phase whose list the caller
already has is not run, and the list goes on as the next phase's upstream (with no extras: the next phase
makes what it needs itself).

All phases share one `EncoderPool`, so a trunk is loaded once for the chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ConfigIdentity import register
from TileEncoderFunc import encoder_config
from stage1_estimation.StageInterface import EstMppResult
from stage2_retrieval.RerankerScorers import RerankerCslsConfig, RerankerDualConfig
from stage2_retrieval.Retriever import EncoderPool, RetrievalOutput, Retriever, RetrieverConfig
from stage2_retrieval.RetrieverSlidingWinSimRot import RetrieverSlidingWinSimRotConfig
from stage2_retrieval.StageInterface import CandidateSet


@register('multiphases')
@dataclass(frozen=True)
class MultiPhasesConfig(RetrieverConfig):
    """`phases` the retrievers in order: the first eats stage 1's estimate, each later one the list before
    it, and reads no more of it than that list holds (`topk_in` <= the previous list's length)."""
    phases: Tuple[RetrieverConfig, ...]

    BASELINE = {}
    VERSION = 1

    def __post_init__(self) -> None:
        from stage2_retrieval import retriever_class                 # noqa: PLC0415
        if not self.phases:
            raise ValueError('a chain has at least one phase')
        length: Optional[int] = None
        for i, cfg in enumerate(self.phases):
            cls: type = retriever_class(cfg)
            wants: type = EstMppResult if i == 0 else RetrievalOutput
            if wants not in getattr(cls, 'UPSTREAM', ()):
                raise ValueError(f'phase {i} ({type(cfg).__name__}) does not eat '
                                 f'{"stage 1" if i == 0 else "a list"}')
            if i and length is not None and int(getattr(cfg, 'topk_in', 0)) > length:
                raise ValueError(f'phase {i} reads {cfg.topk_in} candidates of a list {length} long')
            length = int(getattr(cfg, 'raw_topk', getattr(cfg, 'topk_in', 0)))


class MultiPhases(Retriever):
    """See the module docstring. `phases` are the built retrievers, in order."""

    UPSTREAM = (EstMppResult,)

    def __init__(self, cfg: MultiPhasesConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False, read_workers: int = 0,
                 feature_store: Optional[Any] = None) -> None:
        from stage2_retrieval import retriever_class                 # noqa: PLC0415
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: MultiPhasesConfig = cfg
        self.phases: List[Retriever] = [
            retriever_class(c)(c, self.device, encoders=self.encoders, multi_gpu=multi_gpu,
                               read_workers=read_workers, feature_store=feature_store)
            for c in cfg.phases]

    def list_limit(self) -> int:
        return self.phases[-1].list_limit()

    @property
    def level_encodes(self) -> Dict[int, Tuple[float, int]]:
        """The first phase's `{level: (seconds, tiles)}`: the levels the chain encoded to search."""
        return dict(getattr(self.phases[0], 'level_encodes', {}) or {})

    def encoder_needs(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for i, p in enumerate(self.phases):
            out.update({f'{i}.{role}': c for role, c in p.encoder_needs().items()})
        return out

    def build(self, wsi: Any, mask: Any) -> 'MultiPhases':
        super().build(wsi, mask)
        for p in self.phases:
            p.build(wsi, mask)
        return self

    def frame(self, level: int) -> CandidateSet:
        """The first phase's frame at `level` (the frame every later phase's list is in)."""
        return self.phases[0].frame(level)

    def run(self, upstream: EstMppResult, query: np.ndarray,
            cached: Optional[Callable[[int, Any], Optional[CandidateSet]]] = None) -> RetrievalOutput:
        """The last phase's output. `cached(i, upstream)`, when given, is asked first for phase `i`'s list; a
        list it has is not made again. A phase handed an empty list hands it on (nothing to reorder)."""
        out: Any = upstream
        for i, p in enumerate(self.phases):
            hit: Optional[CandidateSet] = cached(i, out) if cached is not None else None
            if hit is not None:
                out = RetrievalOutput(candidates=hit, parent=out if isinstance(out, RetrievalOutput) else None)
            elif i and not len(out.candidates):
                out = RetrievalOutput(candidates=out.candidates, parent=out)
            else:
                out = p.retrieve(out, query)
        return out

    def outputs(self, out: RetrievalOutput) -> List[RetrievalOutput]:
        """The output of every phase of the chain `out` ends, first phase first."""
        chain: List[RetrievalOutput] = out.chain()[:len(self.phases)]
        return list(reversed(chain))

    def tables(self, out: RetrievalOutput, qc: Any) -> Dict[str, List[Dict[str, Any]]]:
        """The last phase's tables (each phase's entry gets its own: `outputs`)."""
        return self.phases[-1].tables(out, qc)


#: Named chains, every field written out (test_config_identity's recipe lint); `--stage2 multiphases:<name>`.
#: The reranker reads the sliding window's features (`reuse`): the same encoder, pooling and cut.
MULTIPHASES_RECIPES: Dict[str, MultiPhasesConfig] = {
    'uni2-avg-dual': MultiPhasesConfig(phases=(
        RetrieverSlidingWinSimRotConfig(
            encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
            tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, score='mean'),
        RerankerDualConfig(
            encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
            tile_size=256, overlap=True, topk_in=1000, order_by='dual', reuse=('level_features',)))),
    'uni2-avg-csls': MultiPhasesConfig(phases=(
        RetrieverSlidingWinSimRotConfig(
            encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
            tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, score='mean'),
        RerankerCslsConfig(
            encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
            tile_size=256, overlap=True, topk_in=1000, mode='csls', n_ref=20000, k=10, seed=0, chunk=8192,
            reuse=('level_features',)))),
}


__all__ = ['MULTIPHASES_RECIPES', 'MultiPhases', 'MultiPhasesConfig']
