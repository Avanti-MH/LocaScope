'''Stage 1 (mpp estimation) by a K-nearest-neighbour vote over a WSI's own
pyramid, in the shape `StageInterface.MppEstimator` describes.

    est = KnnEstMpp(KNN_RECIPES['gigapath'], device).build(wsi, mask)
    result = est.estimate(query_img)   # query_img: RGB uint8 np.ndarray

A KNN vote over whichever backbone `cfg.encoder` names (`TileEncoderFunc`'s
registered name), built internally, the same as `ClassifierEstMpp.py`'s.

WHY `build(wsi)` IS THE EXPENSIVE STEP HERE, unlike `ClassifierEstMpp`'s.
This method's reference bank is sampled and encoded FROM THE TARGET WSI
itself (`build_samples`/`build_ref_features`, kept as their own methods below
-- intermediate state worth inspecting while debugging). That is the opposite of a trained classifier,
whose model exists independently of any one slide -- see `ClassifierEstMpp`'s
own docstring for the contrasting case, and why its `build(wsi)` is cheap.

PROCEDURE (`estimate`)
-----------------------
1. Cut `query` into `cfg.tile_size` patches -- main patches only,
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
5. Snap to a level this WSI actually has: `StageInterface.routed_level`.
'''
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openslide
import torch
from PIL import Image


from ConfigIdentity import IdentifiedBuild, IdentifiedConfig, register  # noqa: E402
from PatchingLib import QueryPatchContainer, FeaturesMap                 # noqa: E402
from SafeSlide import SafeSlide                                         # noqa: E402
from TileEncoderFunc import encoder_config                              # noqa: E402
from TissueMaskConfig import MASK_RECIPES, TissueMaskConfig              # noqa: E402
from TissueMask import TissueMask                        # noqa: E402
from TileSampler import (SAMPLER_RECIPES, SamplerConfig,           # noqa: E402
                         TileSampler)

from stage1_estimation.StageInterface import (EstMppResult, reference_bank,  # noqa: E402
                                              routed_level)
from ReadGeometry import ReadSpec                                        # noqa: E402
from SlideReader import SlideReader                                     # noqa: E402


# ── config ───────────────────────────────────────────────────────────────────

@register('knn_est_mpp')
@dataclass(frozen=True)
class KnnEstMppConfig(IdentifiedConfig):
    '''Which encoder to vote with, which regions count as tissue, and how the
    reference bank is sampled -- three sub-configs, not loose parameters.

    `encoder` is `TileEncoderFunc`'s own registered name (`'gigapath'`,
    `'uni2'`, ...) -- built internally by `KnnEstMpp.__init__`
    (`encoder_config(cfg.encoder).build(device)`), not accepted as an
    already-built `Callable`: each stage builds its own encoder.

    `mask_cfg` is `TissueMaskConfig` -- the same recipe object
    `LocaScopePipeline` already builds its own mask from, so a caller that
    wants THIS estimator's tissue definition to match the pipeline's just
    passes the same `TissueMaskConfig` to both. Defaults to
    `MASK_RECIPES['hest']`, named: `seg` has no default.

    `sampler_cfg` is `TileSampler.SamplerConfig` -- n per rung, seed,
    richness caps/floors, overlap: everything that decides WHERE the
    reference tiles are, already named by its own `identity_id()`.

    `tile_size` is the ONE size both sides are cut to: the reference bank's
    camera (a plain `tile_size` px tile) and the query's patches. Reference
    tiles and query patches have to be the same size for the vote to compare
    like with like, and one field that could disagree with a second is
    exactly the failure mode this avoids.

    Every field is identity -- a different mask or sampling recipe changes
    which tiles the vote sees -- so `NOT_IDENTITY` stays empty.
    '''
    encoder: str
    mask_cfg: TissueMaskConfig = field(default_factory=lambda: MASK_RECIPES['hest'])
    sampler_cfg: SamplerConfig = field(
        default_factory=lambda: SAMPLER_RECIPES['reference-bank'])
    k: int = 5
    tile_size: int = 256
    #: How many of each level's drawn tiles the vote uses -- the first ones of
    #: the draw. None: every one. The draw (`sampler_cfg.n_per_rung`) is what is
    #: cached; this picks from it, so several bank sizes share one draw.
    n_per_level: Optional[int] = None

    BASELINE = {'mask_cfg': 'TissueMaskConfig', 'sampler_cfg': 'SamplerConfig',
                'k': 5, 'tile_size': 256, 'n_per_level': None}

    def __post_init__(self):
        if self.n_per_level is not None and self.n_per_level < 1:
            raise ValueError(f'n_per_level must be positive or None, got '
                             f'{self.n_per_level}')


# ── result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KnnEstMppResult(EstMppResult):
    '''No fields beyond `EstMppResult`'s own five: `tile_size` is on `cfg`,
    the bank size and `k` are BUILD-time hyperparameters on `KnnEstMppConfig`,
    and ds and mpp are both on `EstMppResult`.

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
        last_sims         [M, k] — their cosines
    '''

    def __init__(self, ref_feats: torch.Tensor, ref_labels: np.ndarray, k: int = 5):
        self.ref_feats = ref_feats       # [N, D] L2-normalized
        self.ref_labels = ref_labels     # [N]
        self.k = k
        self.last_indices: Optional[torch.Tensor] = None    # [M, k]
        self.last_patch_labels: Optional[np.ndarray] = None # [M]
        self.last_sims: Optional[torch.Tensor] = None       # [M, k]

    def predict(self, query_feats: torch.Tensor) -> float:
        '''query_feats: [M, D] L2-normalized. Returns median-of-medians label.'''
        k = min(self.k, self.ref_feats.shape[0])
        sims = query_feats @ self.ref_feats.T          # [M, N]
        topk = sims.topk(k, dim=1)
        self.last_indices = topk.indices               # [M, k]
        self.last_sims = topk.values                   # [M, k]
        self.last_patch_labels = np.median(
            self.ref_labels[self.last_indices.cpu().numpy()], axis=1
        )                                              # [M]
        return float(np.median(self.last_patch_labels))


# ── estimator ────────────────────────────────────────────────────────────────

class KnnEstMpp(IdentifiedBuild):
    '''Builds an encoder once (`__init__`), samples + encodes a reference
    bank per WSI (`build`), answers per query (`estimate`) -- see this
    module's docstring for the procedure and for why `build` is expensive
    here specifically, in contrast to `ClassifierEstMpp`'s.

    All intermediate state (`sampler`, `ref_feats`, `knn`, ...) is stored on
    self for debugging and visualisation.
    '''

    def __init__(self, cfg: KnnEstMppConfig,
                device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.cfg = cfg
        self.device = torch.device(device)
        self.encoder = encoder_config(cfg.encoder).build(self.device)
        self.model = self.encoder.model   # for IdentifiedBuild.weights_id

        self.wsi = None
        self.mask: Optional[TissueMask] = None
        self.sampler: Optional[TileSampler] = None
        self.plan = None
        self.masks, self.cache_job = None, None
        self.ref_feats: Optional[torch.Tensor] = None    # [N, D]
        self.ref_mpps: Optional[List[float]] = None
        self.ref_index: List[int] = []
        self.knn: Optional[KnnClassifier] = None
        self.qc: Optional[QueryPatchContainer] = None
        self.qfm: Optional[FeaturesMap] = None
        self.query_feats: Optional[torch.Tensor] = None  # [M, D]

    # ── build: reference bank, sampled from the target WSI itself ───────────

    def build(self, wsi: Union[openslide.OpenSlide, str],
              mask: Optional[TissueMask] = None, *, masks=None,
              cache_job: Optional[str] = None) -> 'KnnEstMpp':
        '''Bind `wsi` and build its reference bank. Expensive -- see this
        module's docstring for why, in contrast to `ClassifierEstMpp.build`.

        With `cache_job` (and `masks`, the MaskMaker `mask` came from) the
        bank's draw is a cache entry and so are its encoded features, beside
        it: `.../draw=<sampler>/features/features_<pooling>-<encoder>`. A
        second build of the slide reads both and encodes nothing.'''
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi = wsi
        self.mask = mask
        self.masks, self.cache_job = masks, cache_job
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
        ladder (`StageInterface.reference_bank`). Why the default richness is
        as strict as it is: this bank is the reference every query's mpp is
        voted against, so a tile that is mostly background carries no scale
        information and only adds a wrong neighbour -- see
        `SAMPLER_RECIPES['reference-bank']`.
        '''
        if self.mask is None and not self.cache_job:
            self.mask = self.cfg.mask_cfg.build(self.wsi, self.device)
        if self.masks is not None and self.masks.cfg != self.cfg.mask_cfg:
            raise ValueError('the MaskMaker passed to build() is not this '
                             "estimator's mask recipe (cfg.mask_cfg)")
        self.sampler, self.plan = reference_bank(
            self.wsi, self.mask, self.cfg.sampler_cfg, self.cfg.tile_size,
            masks=self.masks, cache_job=self.cache_job)
        return self.sampler

    def _features_entry(self):
        '''`(entry, id, record)` of the bank's features in the cache job's
        tree, beside the draw; None without a cache job.'''
        if not getattr(self, 'cache_job', None):
            return None
        from ConfigIdentity import record                             # noqa: PLC0415
        from Store import FeatureStore, feature_id                    # noqa: PLC0415
        sid = self.cfg.sampler_cfg.identity_id()
        entry = TileSampler.draw_address(
            self.cache_job, self.sampler.slide, self.masks.cfg, self.plan
        ).at(draw=sid).entry('features')
        rec = record(self.encoder, also=(FeatureStore, SlideReader), draw=sid)
        return entry, feature_id(self.encoder, self.encoder.feature_pooling), rec

    def _build_ref_features(self) -> torch.Tensor:
        '''Encode all sampled tiles -- or read them back from the cache --
        and build the KnnClassifier.'''
        if self.sampler is None:
            self._build_samples()
        cached = self._features_entry()
        self.ref_feats = None
        if cached is not None:
            from Store import FeatureStore                            # noqa: PLC0415
            entry, fid, rec = cached
            state, diff = entry.status(fid, rec)
            if state == 'hit':
                tensors, _ = FeatureStore.load(FeatureStore.path(entry, fid))
                self.ref_feats = tensors['features'][:, 0].to(self.device)
            elif state == 'stale':
                print(f'  [knn bank] {entry.record_path(fid)} is stale, encoding '
                      f'again: ' + '; '.join(diff), flush=True)
        if self.ref_feats is None:
            # Straight off the slide through SlideReader: the level rule, level
            # px and filter every other read in the project uses. Native plans
            # need no resampling; 'area' is the ladder's filter where one would.
            images = SlideReader(self.wsi, resize='area').read_samples(
                self.sampler, ReadSpec(self.cfg.tile_size, self.cfg.tile_size))
            self.ref_feats = self.encoder(images)               # [N, D]
            if cached is not None:
                self._save_ref_features(*cached)
        # mpp is derived here rather than carried on the tile. It was a copy
        # of `base_mpp * level_downsample` stored at sampling time, and a
        # stored copy of a derived number is one that can go stale against
        # the handle it came from -- which is exactly what a KNN's LABELS
        # must not do.
        metas = [s.meta for s in self.sampler]
        #: The draw rows the vote uses: the first `n_per_level` of each level.
        self.ref_index = _first_per_level(metas, self.cfg.n_per_level)
        self.ref_mpps = [self.wsi.base_mpp
                         * self.wsi.level_downsamples[metas[j].level]
                         for j in self.ref_index]
        self.knn = KnnClassifier(
            self.ref_feats[self.ref_index], np.array(self.ref_mpps), k=self.cfg.k)
        return self.ref_feats

    def _save_ref_features(self, entry, fid, rec) -> None:
        from Store import FeatureMeta, FeatureStore, encoder_names    # noqa: PLC0415
        feats = self.ref_feats.detach()
        if feats.dtype not in (torch.float16, torch.float32):
            feats = feats.float()
        metas = [s.meta for s in self.sampler]
        pooling = self.encoder.feature_pooling
        spec = self.encoder.model_spec
        n = len(metas)
        meta = FeatureMeta(
            wsi_stem=self.sampler.slide,
            wsi_path=str(getattr(self.wsi, '_filename', '') or ''),
            level=-1, ds=0.0, mpp=0.0, base_mpp=float(self.wsi.base_mpp),
            tile_size=self.cfg.tile_size, overlap=False, pooling=pooling,
            slots=(pooling,), slot_layout='none', dim=int(feats.shape[-1]),
            feat_hw=spec.feat_hw, num_prefix=spec.num_prefix,
            encoder_id=encoder_names(self.encoder)[1],
            seg_id=self.masks.cfg.seg_id(), region_id=self.masks.cfg.region_id(),
            coverage='sample', n_available=n, n_tiles=n,
            sampler_id=self.cfg.sampler_cfg.identity_id(), plan=self.plan.key(),
            sample_seed=int(self.cfg.sampler_cfg.seed))
        FeatureStore.save(
            entry, fid, rec, meta=meta, features=feats.unsqueeze(1),
            x=torch.tensor([m.x for m in metas], dtype=torch.int32),
            y=torch.tensor([m.y for m in metas], dtype=torch.int32),
            extra={'level': torch.tensor([m.level for m in metas],
                                         dtype=torch.int16)})

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
            query.extract_all(self.cfg.tile_size, overlap=overlap)
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

        chosen_level, chosen_ds, chosen_mpp = routed_level(
            self.wsi.level_downsamples, base_mpp, estimated_ds)

        return KnnEstMppResult(
            estimated_ds=estimated_ds, estimated_mpp=estimated_mpp,
            chosen_ds=chosen_ds, chosen_mpp=chosen_mpp,
            chosen_level=chosen_level)

    def neighbour_rows(self) -> List[dict]:
        '''The last `estimate`'s vote as table rows, one per (query patch, k):
        the reference tile it matched -- `ref_index` is the tile's row in this
        slide's bank draw, `ref_level` and `ref_ds` its label -- and their
        cosine. Patches in the order `estimate` voted them.'''
        if self.knn is None or self.knn.last_indices is None:
            return []
        metas = [s.meta for s in self.sampler]
        idx = self.knn.last_indices.cpu().numpy()
        sims = self.knn.last_sims.float().cpu().numpy()
        out = []
        for p in range(idx.shape[0]):
            for k in range(idx.shape[1]):
                j = self.ref_index[int(idx[p, k])]
                m = metas[j]
                out.append(dict(patch=p, k=k + 1, ref_index=j,
                                ref_level=int(m.level),
                                ref_ds=float(self.wsi.level_downsamples[m.level]),
                                ref_x=int(m.x), ref_y=int(m.y),
                                cosine=float(sims[p, k])))
        return out


#: Named KNN estimators, every field written out (test_config_identity's recipe
#: lint); `--stage1 knn:<name>`. The reference bank is 40 tiles per pyramid
#: level under the reference-bank recipe; `mask_cfg` is the recipe of the mask
#: build() is handed, so a caller searching another mask replaces it.
KNN_RECIPES: Dict[str, KnnEstMppConfig] = {
    'gigapath': KnnEstMppConfig(
        encoder='gigapath', mask_cfg=MASK_RECIPES['hest'],
        sampler_cfg=replace(SAMPLER_RECIPES['reference-bank'],
                            n_per_rung=40, seed=42),
        k=5, tile_size=256, n_per_level=None),
    'uni2': KnnEstMppConfig(
        encoder='uni2', mask_cfg=MASK_RECIPES['hest'],
        sampler_cfg=replace(SAMPLER_RECIPES['reference-bank'],
                            n_per_rung=40, seed=42),
        k=5, tile_size=256, n_per_level=None),
}


def _first_per_level(metas, n: Optional[int]) -> List[int]:
    """Rows of `metas` (a draw, in its order) keeping the first `n` of each
    level; every row when `n` is None."""
    if n is None:
        return list(range(len(metas)))
    seen: Dict[int, int] = {}
    out = []
    for j, m in enumerate(metas):
        seen[m.level] = seen.get(m.level, 0) + 1
        if seen[m.level] <= n:
            out.append(j)
    return out
