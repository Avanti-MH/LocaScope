'''Stage 1 (mpp estimation) by a K-nearest-neighbour vote over a WSI's own
pyramid, in the shape `StageInterface.MppEstimator` describes.

    cfg = KnnEstMppConfig(encoder='gigapath')
    est = KnnEstMpp(cfg, device).build(wsi)
    result = est.estimate(query_img)   # query_img: RGB uint8 np.ndarray

NOT `GigaPathKnnEstiMpp`/`GigapathKnnEstiMppResult` any more. There was never
anything GigaPath-specific in the KNN vote itself, only in which encoder a
caller happened to build and pass in -- `encoder` is a config field now
(`TileEncoderFunc`'s own registered name), built internally, the same as
`ClassifierEstMpp.py`'s. `KnnEstMpp` says what this method actually is: a KNN
vote, over whichever backbone `cfg.encoder` names.

WHY `build(wsi)` IS THE EXPENSIVE STEP HERE, unlike `ClassifierEstMpp`'s.
This method's reference bank is sampled and encoded FROM THE TARGET WSI
itself (`build_samples`/`build_ref_features`, kept as their own methods below
-- same reason the original file kept them: intermediate state worth
inspecting while debugging). That is the opposite of a trained classifier,
whose model exists independently of any one slide -- see `ClassifierEstMpp`'s
own docstring for the contrasting case, and why its `build(wsi)` is cheap.

PROCEDURE (`estimate`)
-----------------------
1. Cut `query` into `cfg.sampler_cfg.tile` patches -- main patches only,
   `overlap=True` for coverage, unchanged from before.
2. Encode every patch with the SAME encoder the reference bank was built
   from.
3. `KnnClassifier.predict`: median-of-medians vote over the k nearest
   reference tiles -- unchanged, see its own docstring.
4. The vote already comes out as an ABSOLUTE mpp -- reference labels are this
   WSI's own `base_mpp * level_downsample` (`build_ref_features`) -- so
   `estimated_ds` here is the reverse DIVISION (`estimated_mpp / base_mpp`),
   not a table lookup the way `ClassifierEstMpp`'s `rungs` lookup is: the two
   methods answer in different native units and only agree once both are
   converted to the same one.
5. Snap to a level this WSI actually has -- same inlined
   `wsi.coarser_level_for_downsample` call, same reasoning, as
   `ClassifierEstMpp.estimate` (see `StageInterface`'s docstring for why it
   is inlined rather than a shared function).
'''
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openslide
import torch
from PIL import Image

# _paths holds the one definition of every package's sys.path entry
# (setup_import_paths) -- utilities/ goes on the path here, by hand, because
# that function is INSIDE it and this is the one step nothing else can do
# for this file. Same idiom every test_modules/cli entry point uses.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'utilities'))
from _paths import setup_import_paths                                   # noqa: E402
setup_import_paths()

from ConfigIdentity import IdentifiedBuild, IdentifiedConfig, register  # noqa: E402
from PatchingLib import QueryPatchContainer, FeaturesMap                 # noqa: E402
from SafeSlide import SafeSlide                                         # noqa: E402
from TileEncoderFunc import encoder_config                              # noqa: E402
from TissueMaskConfig import MASK_RECIPES, TissueMaskConfig              # noqa: E402
from TissueMask import TissueMask                        # noqa: E402
from TileSampler import (OverlapConfig, RichnessConfig, SamplerConfig,  # noqa: E402
                         TileSampler, native_plans)

from StageInterface import EstMppResult                                 # noqa: E402


# ── config ───────────────────────────────────────────────────────────────────

#: The reference bank's own richness policy: any tile under 50% background,
#: no preference between the buckets that admits -- NOT `RichnessConfig()`'s
#: own general-purpose default (non-zero floors, meant for a training corpus
#: that wants specific proportions of busy/quiet tiles). Spelled out directly
#: rather than through `TileSampler.caps_for_tissue_ratio` -- that function's
#: own docstring calls it "the RETIRED tissue_ratio gate", kept only so two
#: OLD callers could reproduce their pre-richness-buckets behaviour without
#: each owning the translation. A NEW config going through a function that
#: names itself retired is the same mistake in a fresh coat -- this describes
#: the policy directly instead. These are the exact caps `tissue_ratio=0.5`
#: used to produce. Public (no leading underscore) so a caller building its
#: own `SamplerConfig` for this estimator -- `LocaScopePipeline`, when it
#: wants its own `--knn-samples`/`--knn-seed`-style overrides without
#: re-deriving this policy -- reuses this object rather than a second copy of
#: the tuple that could drift from it.
REFERENCE_BANK_RICHNESS = RichnessConfig(
    floors=(0.0,) * 7, caps=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0))


def _default_sampler_cfg() -> SamplerConfig:
    return SamplerConfig(richness=REFERENCE_BANK_RICHNESS, overlap=OverlapConfig())


@register('knn_est_mpp')
@dataclass(frozen=True)
class KnnEstMppConfig(IdentifiedConfig):
    '''Which encoder to vote with, which regions count as tissue, and how the
    reference bank is sampled -- three sub-configs, not loose parameters.

    `encoder` is `TileEncoderFunc`'s own registered name (`'gigapath'`,
    `'uni2'`, ...) -- built internally by `KnnEstMpp.__init__`
    (`encoder_config(cfg.encoder).build(device)`), not accepted as an
    already-built `Callable` the way the old `GigaPathKnnEstiMpp` took one.
    The two happened to share one encoder across pipeline stages before, but
    that sharing was incidental, never a requirement of this method.

    `mask_cfg` is `TissueMaskConfig` -- the same recipe object
    `LocaScopePipeline` already builds its own mask from, so a caller that
    wants THIS estimator's tissue definition to match the pipeline's just
    passes the same `TissueMaskConfig` to both. Defaults to
    `MASK_RECIPES['hest']`, named rather than the bare dataclass: `seg` has
    no default any more, because the old one was hsv without saying so.

    `sampler_cfg` is `TileSampler.SamplerConfig` -- tile size, n per rung,
    seed, richness caps/floors, overlap: everything that decides which tiles
    the vote runs against, already hashed by its own `sampler_id()`. The
    query side reads its tile size from here too (`sampler_cfg.tile`), not a
    second field of its own -- reference tiles and query patches have to be
    cut to the SAME size for the vote to compare like with like, and one
    field that could disagree with a second is exactly the failure mode this
    avoids.

    Every field is identity -- a different mask or sampling recipe changes
    which tiles the vote sees -- so `NOT_IDENTITY` stays empty.
    '''
    encoder: str
    mask_cfg: TissueMaskConfig = field(default_factory=lambda: MASK_RECIPES['hest'])
    sampler_cfg: SamplerConfig = field(default_factory=_default_sampler_cfg)
    k: int = 5


# ── result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KnnEstMppResult(EstMppResult):
    '''No fields beyond `EstMppResult`'s own five.

    The old `GigapathKnnEstiMppResult` carried `base_mpp`/`tile_size`/
    `samples_per_level`/`k`/`query_patch_count` -- every one of those is
    redundant under this design, for the same reasons `ClassifierEstMppResult`
    gives for its own dropped fields: `base_mpp` is no longer needed to
    reconstruct anything, now that both ds and mpp are stored directly on
    `EstMppResult`; `tile_size` is already on `cfg`; `samples_per_level`/`k`
    are this method's own BUILD-time hyperparameters (now on
    `KnnEstMppConfig`), not a property of one estimate; `query_patch_count`
    is already knowable from the input query at the call site.

    This subclass exists at all only so `type(result)` can say which method
    produced a Result without a `method: str` field -- see
    `StageInterface.EstMppResult`'s own docstring for why that field does not
    exist either.
    '''


# ── KNN classifier ────────────────────────────────────────────────────────────

class KnnClassifier:
    '''
    Generic KNN regression: predicts the median label of the k nearest neighbours.

    Debug state is stored after predict() for visualization:
        last_indices      [M, k] — which reference tiles each query patch matched
        last_patch_labels [M]    — per-patch median label before global median
    '''

    def __init__(self, ref_feats: torch.Tensor, ref_labels: np.ndarray, k: int = 5):
        self.ref_feats = ref_feats       # [N, D] L2-normalized
        self.ref_labels = ref_labels     # [N]
        self.k = k
        self.last_indices: Optional[torch.Tensor] = None    # [M, k]
        self.last_patch_labels: Optional[np.ndarray] = None # [M]

    def predict(self, query_feats: torch.Tensor) -> float:
        '''query_feats: [M, D] L2-normalized. Returns median-of-medians label.'''
        k = min(self.k, self.ref_feats.shape[0])
        sims = query_feats @ self.ref_feats.T          # [M, N]
        topk = sims.topk(k, dim=1)
        self.last_indices = topk.indices               # [M, k]
        self.last_patch_labels = np.median(
            self.ref_labels[self.last_indices.numpy()], axis=1
        )                                              # [M]
        return float(np.median(self.last_patch_labels))


# ── estimator ────────────────────────────────────────────────────────────────

class KnnEstMpp(IdentifiedBuild):
    '''Builds an encoder once (`__init__`), samples + encodes a reference
    bank per WSI (`build`), answers per query (`estimate`) -- see this
    module's docstring for the procedure and for why `build` is expensive
    here specifically, in contrast to `ClassifierEstMpp`'s.

    All intermediate state (`sampler`, `ref_feats`, `knn`, ...) is stored on
    self for debugging and visualisation, same as the original file.
    '''

    #: Append-only zero point for `identity_id`/`identity_parts` -- empty
    #: until a specific (encoder, mask_cfg, sampler_cfg, k) combination is
    #: adopted as the project's own default, per `ConfigIdentity`'s rule 1.
    BASELINE: Dict[str, Any] = {}

    def __init__(self, cfg: KnnEstMppConfig,
                device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.cfg = cfg
        self.device = torch.device(device)
        self.encoder = encoder_config(cfg.encoder).build(self.device)
        self.model = self.encoder.model   # for IdentifiedBuild.weights_id

        self.wsi = None
        self.mask: Optional[TissueMask] = None
        self.sampler: Optional[TileSampler] = None
        self.ref_feats: Optional[torch.Tensor] = None    # [N, D]
        self.ref_mpps: Optional[List[float]] = None
        self.knn: Optional[KnnClassifier] = None
        self.qc: Optional[QueryPatchContainer] = None
        self.qfm: Optional[FeaturesMap] = None
        self.query_feats: Optional[torch.Tensor] = None  # [M, D]

    # ── build: reference bank, sampled from the target WSI itself ───────────

    def build(self, wsi: Union[openslide.OpenSlide, str],
             mask: Optional[TissueMask] = None) -> 'KnnEstMpp':
        '''Bind `wsi` and build its reference bank. Expensive -- see this
        module's docstring for why, in contrast to `ClassifierEstMpp.build`.'''
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi = wsi
        self.mask = mask
        self.sampler = None
        self.ref_feats = None
        self._build_samples()
        self._build_ref_features()
        return self

    def _build_samples(self) -> TileSampler:
        '''Sample `cfg.sampler_cfg` tiles per WSI level within tissue regions
        (`cfg.mask_cfg`, unless a `mask` was already passed to `build()` --
        see there for why a caller sharing one mask across stages needs that
        override).

        One rung per PYRAMID level -- this bank's labels ARE the levels'
        mpps, so the magnifications are the slide's own and not a fixed
        ladder. Why the default richness is as strict as it is: this bank is
        the reference every query's mpp is voted against, so a tile that is
        mostly background carries no scale information and only adds a wrong
        neighbour -- see `REFERENCE_BANK_RICHNESS`.
        '''
        if self.mask is None:
            self.mask = self.cfg.mask_cfg.build(self.wsi, self.device)
        self.sampler = TileSampler(self.wsi, self.mask, self.cfg.sampler_cfg)
        self.sampler.sample(native_plans(self.wsi, self.cfg.sampler_cfg.tile))
        return self.sampler

    def _build_ref_features(self) -> torch.Tensor:
        '''Encode all sampled tiles; build the KnnClassifier.'''
        if self.sampler is None:
            self._build_samples()
        images = self.sampler.materialise(self.wsi).images()
        self.ref_feats = self.encoder(images)               # [N, D]
        # mpp is derived here rather than carried on the tile. It was a copy
        # of `base_mpp * level_downsample` stored at sampling time, and a
        # stored copy of a derived number is one that can go stale against
        # the handle it came from -- which is exactly what a KNN's LABELS
        # must not do.
        self.ref_mpps = [self.wsi.base_mpp
                         * self.wsi.level_downsamples[s.meta.level]
                         for s in self.sampler]
        self.knn = KnnClassifier(
            self.ref_feats, np.array(self.ref_mpps), k=self.cfg.k)
        return self.ref_feats

    # ── per-query ────────────────────────────────────────────────────────

    def build_query_features(
        self, query: Union[QueryPatchContainer, Image.Image, np.ndarray],
        overlap: bool = True,
    ) -> FeaturesMap:
        '''Encode query patches into a FeaturesMap. Kept as its OWN method,
        not folded into `estimate`, so a subclass that changes what happens
        to the vote (`SubspaceKnnEstMpp` in `bench_mpp_feature_
        decomposition.py`: project both sides into a fitted subspace before
        voting) can override `estimate` alone and still reuse this stage
        unchanged -- the same reason `_build_samples`/`_build_ref_features`
        stayed split out above.'''
        if isinstance(query, (Image.Image, np.ndarray)):
            query = QueryPatchContainer(query)
        if query.grid is None:
            query.extract_all(self.cfg.sampler_cfg.tile, overlap=overlap)
        self.qc = query
        self.qfm = self.qc.to_features(self.encoder)
        self.query_feats = torch.stack(list(self.qfm.iter_main_features()))  # [M, D]
        return self.qfm

    def estimate(self, query: Union[QueryPatchContainer, Image.Image, np.ndarray, None] = None,
                overlap: bool = True) -> KnnEstMppResult:
        if self.wsi is None:
            raise RuntimeError(
                'call build(wsi) before estimate() -- chosen_ds/chosen_mpp '
                'need the target WSI\'s own pyramid and base_mpp, and the '
                'reference bank the vote runs against is built there too')
        if query is not None:
            self.build_query_features(query, overlap=overlap)
        if self.knn is None:
            self._build_ref_features()
        if self.query_feats is None:
            raise RuntimeError(
                'no query features -- call build_query_features() or pass '
                'query to estimate()')

        base_mpp = self.wsi.base_mpp
        estimated_mpp = self.knn.predict(self.query_feats)
        # The vote's own native unit is ABSOLUTE mpp (reference labels are
        # THIS wsi's `base_mpp * level_downsample`), so ds is the reverse
        # division -- contrast `ClassifierEstMpp`, whose native unit is a
        # relative rung and multiplies by base_mpp instead.
        estimated_ds = estimated_mpp / base_mpp

        chosen_level = self.wsi.coarser_level_for_downsample(estimated_ds)
        chosen_ds = float(self.wsi.level_downsamples[chosen_level])
        chosen_mpp = base_mpp * chosen_ds

        return KnnEstMppResult(
            estimated_ds=estimated_ds, estimated_mpp=estimated_mpp,
            chosen_ds=chosen_ds, chosen_mpp=chosen_mpp,
            chosen_level=chosen_level)
