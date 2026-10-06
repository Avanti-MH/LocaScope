'''Stage 1 (mpp estimation) by a trained `training/PrototypicalRoutingHead`
metric-based checkpoint, in the shape `StageInterface.MppEstimator` describes.

    cfg = PrototypeEstMppConfig.from_checkpoint(
        '/work/u26130998/result/PrototypicalRoutingHead/weights/'
        'uni2_frozen_cls_identity_identity_set_transformer_cosine_tau_none_6rung.pt')
    est = PrototypeEstMpp(cfg).build(wsi)
    result = est.estimate(query_img)   # query_img: RGB uint8 np.ndarray

Loading the model (`__init__`) is cheap; binding a WSI (`build`) is the
EXPENSIVE step here, same reason as `KnnEstMpp.build`, not `ClassifierEstMpp.
build`: this method has no fixed, trained set of class anchors the way a
plain classifier does (`ClassifierEstMpp`'s checkpoint IS the decision
boundary already). Here, `build(wsi)` samples and encodes a REFERENCE bank
straight off the target WSI -- the SUPPORT SIDE of the episodic pipeline
`cli/train.py` trained, now playing that same role against a real slide
instead of another training episode -- and runs it once through the
checkpoint's own Stage 2 (support_context/collapse) so every subsequent
`estimate(query)` call is cheap.

INDEPENDENT of `KnnEstMpp.py` ON PURPOSE: `StageInterface.py`'s
own "tree, not list of alternatives" docstring -- each branch owns its own
build-time setup. `native_plans`/`TileSampler`/`TissueMaskConfig` are shared
UTILITIES both methods happen to call the same way, but this file's own
richness/overlap tuning (`_REFERENCE_BANK_RICHNESS`/`_default_sampler_cfg`
below) is its OWN copy, not imported from `KnnEstMpp.py` -- re-tuning one
method's reference-bank recipe must not silently move the other's.

REFERENCE BANK = NATIVE PYRAMID LEVELS, NOT `Datasets.RUNGS`. The trained
model has no fixed rung ladder baked into it anywhere: `Episodes.sample_
episode` draws a different N-way rung SUBSET every episode (`--n-choices`),
and `CosineTauHead`/`AttnScoreHead`'s own `forward` takes however many
prototypes/support groups they are handed -- no layer's weight shape is
tied to a specific K (the "dynamic-K-safe" invariant `training/
PrototypicalRoutingHead/plan.md`'s own design keeps everywhere). So there is
no requirement to force a target WSI's reference bank to cover six specific
magnifications: `native_plans(wsi, tile)` (`utilities/TileSampler.py`'s own
"the ds ladder and the pyramid are different questions... this answers the
second, which is what a reference bank wants") gives one rung per pyramid
level THIS slide actually has, and the model classifies among however many
of those `cfg.levels` (or all of them, by default) selects -- a 4-way result
on a slide with 4 native levels is not a degraded answer, it is spec.md's
own N-way generality doing exactly what it was built for.

SUPPORT TILES ARE READ RAW, NEVER AUGMENTED -- `SlideReader.read_samples` is
a direct WSI read (no `Render`, no augment chain), the same
choice `KnnEstMpp` makes for its own reference bank and for the same reason: a
reference/support tile is not a photograph, at real deployment it never was
one. `training/MppRoutingHead/Datasets.CAMERA_GEOMETRY_ONLY` (`--support-
native`, `cli/train.py`) exists specifically so a checkpoint can be trained
against support pixels shaped like THIS, rather than like `CAMERA_FULL`'s
own simulated photograph -- whether a specific checkpoint actually was is a
fact about how it was trained, not something this file needs to branch on:
it just reads whatever pixels are here.

PROCEDURE
---------
`build(wsi)`:
1. Segment tissue (`cfg.mask_cfg`), sample one rung per native pyramid level
  (`native_plans`, filtered to `cfg.levels` if given) via `TileSampler`.
2. Read every sampled tile as a raw crop (`read_samples`), encode
  + pool each ONE AT A TIME PER LEVEL (never across levels -- same invariant
  `PrototypeGenerators.py`'s own docstring states for training).
3. Run each level's own pooled support set through `support_context` (G),
  caching the result (`self.g_output_by_level`) -- `query_context` (F) reads
  this UN-collapsed output on every `estimate()` call regardless of what
  Collapse does next, the same dedicated channel `episode_forward` uses.
4. Run `collapse` over each level's own `g_output_by_level` entry: caches
  EITHER `self.prototypes` (`[K, D]`, one row per level, `collapse.COLLAPSES`
  true) OR `self.raw_support` (`{level: [K_level, D]}`, `COLLAPSES` false) --
  never both.

`estimate(query)`:
1. Cut `query` into `cfg.tile_size` patches (`QueryPatchContainer`,
  `overlap=True`, scored on MAIN patches only -- same convention
  `ClassifierEstMpp.estimate`/`KnnEstMpp.estimate` both use).
2. Encode + pool, then `query_context` (F) against `self.g_output_by_level`.
3. `head(query_vecs, self.prototypes or self.raw_support)` -> logits ->
  softmax -- the SAME routing head `cli/train.py` trained, unchanged.
4. `FoVVote.vote(cfg.vote, probs, rungs=self.level_ds)` aggregates every
  query patch's own distribution into one level.
5. `self.level_ds[predicted_idx]` -> `estimated_ds`/`estimated_mpp` (THIS
  slide's own downsample/mpp at that level, not a value off an external
  ladder), then snapped to a level the WSI actually has
  (`wsi.coarser_level_for_downsample`) exactly like the other two methods.
'''
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import openslide
import torch

# _paths holds the one definition of every package's sys.path entry
# (setup_import_paths) -- utilities/ goes on the path here, by hand, because
# that function is INSIDE it and this is the one step nothing else can do
# for this file. Same idiom every test_modules/cli entry point uses.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'utilities'))
from _paths import setup_import_paths                                   # noqa: E402
setup_import_paths()

from ConfigIdentity import IdentifiedBuild, IdentifiedConfig, register  # noqa: E402
from TileSampler import (OverlapConfig, RichnessConfig, SamplerConfig,  # noqa: E402
                         TileSampler, native_plans)
from TissueMaskConfig import MASK_RECIPES, TissueMaskConfig              # noqa: E402
from PatchingLib import QueryPatchContainer                              # noqa: E402
from SafeSlide import SafeSlide                                         # noqa: E402
from Checkpoints import build_prototype_from_checkpoint                  # noqa: E402
from Features import encode_raw, trunk_raw                              # noqa: E402

from stage1_estimation.StageInterface import EstMppResult                                 # noqa: E402
from stage1_estimation.FoVVote import vote as fov_vote                                     # noqa: E402
from ReadGeometry import ReadSpec                                        # noqa: E402
from SlideReader import SlideReader                                     # noqa: E402


# ── reference-bank sampling recipe -- THIS FILE'S OWN, not KnnEstMpp's ─────

#: Own copy of the same SHAPE `KnnEstMpp.REFERENCE_BANK_RICHNESS` uses
#: (floors all 0, first three richness buckets effectively uncapped, the
#: bottom four excluded entirely) -- not imported from `KnnEstMpp.py`, see
#: this module's own docstring for why the two methods stay independent.
_REFERENCE_BANK_RICHNESS = RichnessConfig(
    floors=(0.0,) * 7, caps=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0))


def _default_sampler_cfg() -> SamplerConfig:
    '''`n_per_rung=5`, NOT `KnnEstMpp`'s own 500: those tiles are fed
    directly into `support_context`/`collapse` here, the same modules
    `cli/train.py` trained against support sets of size `--n-support`
    (default 5) -- `SetTransformerPrototype`'s self-attention and
    `BiLstmSupportContext`'s LSTM were both fit at THAT sequence length,
    not at KnnEstMpp's own "as many neighbours as possible" scale, which
    has no analogue here: nothing downstream keeps individual reference
    tiles for a per-example vote the way a KNN does. A caller that wants a
    different size passes its OWN `sampler_cfg` (this is a plain
    dataclass default, like `mask_cfg`'s -- see `PrototypeEstMppConfig`'s
    own docstring for why `from_checkpoint` does not try to read
    `--n-support` back off the checkpoint to set this automatically).
    '''
    return SamplerConfig(n_per_rung=5, richness=_REFERENCE_BANK_RICHNESS,
                         overlap=OverlapConfig())


# ── config ───────────────────────────────────────────────────────────────────

@register('prototype_est_mpp')
@dataclass(frozen=True)
class PrototypeEstMppConfig(IdentifiedConfig):
    '''Which trained checkpoint to run, and how to sample its reference bank.

    `encoder`/`pooling`/`support_context`/`query_context`/`collapse`/
    `routing_head`/`tile_size` are explicit fields for the same two reasons
    `ClassifierEstMppConfig` gives for its own four: a caller comparing
    several checkpoints wants them to read straight out of `identity_id`,
    and a config built by hand that DISAGREES with what `weights` actually
    contains is exactly the mismatch `PrototypeEstMpp.__init__` checks for.
    `from_checkpoint` is the usual way to get one -- it reads all seven
    straight off the checkpoint itself.

    `mask_cfg`/`sampler_cfg` are PLAIN defaults (`MASK_RECIPES['hest']`
    / `_default_sampler_cfg()`), not derived from the checkpoint by
    `from_checkpoint` -- unlike the seven architecture fields above, neither
    is something the checkpoint's own training necessarily recorded in a
    form worth depending on (`--n-support` happens to exist in `ckpt['args']`
    today, but building `sampler_cfg.n_per_rung` FROM it would make this
    config's own correctness depend on a training-script CLI field it has
    no other reason to read). A caller that wants the reference bank sized
    to match a specific checkpoint's own `--n-support` passes `sampler_cfg=
    replace(default_sampler_cfg(), n_per_rung=<that value>)` explicitly.

    `tile_size` is the ONE size both the reference bank and the query
    patches are cut to. `SamplerConfig` has no tile size any more (the
    footprint belongs to the camera), so there is no second field that could
    disagree with it.

    `levels`: `None` (default) samples EVERY native pyramid level
    (`native_plans`' own full list); given, restricts to just those level
    indices. Not identity-exempt: two runs differing only in which levels
    were sampled measure a different question.

    `weights` is NOT_IDENTITY, same reasoning `ClassifierEstMppConfig` gives:
    `IdentifiedBuild.weights_id` hashes the loaded state dicts directly.
    '''
    encoder: str
    pooling: str
    support_context: str
    query_context: str
    collapse: str
    routing_head: str
    tile_size: int
    weights: str
    mask_cfg: TissueMaskConfig = field(default_factory=lambda: MASK_RECIPES['hest'])
    sampler_cfg: SamplerConfig = field(default_factory=_default_sampler_cfg)
    levels: Optional[Sequence[int]] = None
    #: `FoVVote.VOTE_CHOICES` key -- an INFERENCE-time choice, never recorded
    #: on the checkpoint, same reasoning `ClassifierEstMppConfig.vote` gives.
    vote: str = 'mean_probability'

    NOT_IDENTITY = ('weights',)

    @classmethod
    def from_checkpoint(cls, weights: str, *,
                        mask_cfg: Optional[TissueMaskConfig] = None,
                        sampler_cfg: Optional[SamplerConfig] = None,
                        levels: Optional[Sequence[int]] = None,
                        vote: str = 'mean_probability') -> 'PrototypeEstMppConfig':
        ckpt = torch.load(weights, map_location='cpu')
        run_args = ckpt['args']
        return cls(
            encoder=ckpt['encoder'], pooling=run_args.get('pooling', ''),
            support_context=ckpt['support_context_name'],
            query_context=ckpt['query_context_name'],
            collapse=ckpt['collapse_name'],
            routing_head=ckpt['routing_head_name'],
            tile_size=int(run_args['tile']), weights=weights,
            mask_cfg=mask_cfg if mask_cfg is not None else MASK_RECIPES['hest'],
            sampler_cfg=(sampler_cfg if sampler_cfg is not None
                        else _default_sampler_cfg()),
            levels=levels, vote=vote)


# ── result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PrototypeEstMppResult(EstMppResult):
    '''`EstMppResult` plus what is specific to how this method arrived at its
    answer.

    `predicted_level` indexes THIS estimator's OWN `self.level_order` (built
    fresh per `build(wsi)` call, from whichever native levels that WSI
    actually has) -- NOT a class index into `Datasets.RUNGS` the way
    `ClassifierEstMppResult.predicted_class` is. See this module's own
    docstring for why there is no fixed external rung ladder here at all.

    `vote_extra`: same shape/reasoning as `ClassifierEstMppResult.vote_extra`
    -- whichever `FoVVote` function `cfg.vote` names filled in.
    '''
    predicted_level: int
    vote_extra: Dict[str, float]


# ── estimator ────────────────────────────────────────────────────────────────

class PrototypeEstMpp(IdentifiedBuild):
    '''Loads once (`__init__`), builds a reference bank once per WSI
    (`build`), answers per query (`estimate`) -- see this module's docstring
    for the procedure and for why `build` is expensive here, unlike
    `ClassifierEstMpp.build`.
    '''

    #: Append-only zero point for `identity_id`/`identity_parts` -- empty
    #: until a specific combination is adopted as the project's own
    #: default, per `ConfigIdentity`'s rule 1.
    BASELINE: Dict[str, Any] = {}

    def __init__(self, cfg: PrototypeEstMppConfig,
                device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
                encode_batch: int = 256):
        self.cfg = cfg
        self.device = torch.device(device)
        (self.pooling, self.support_context, self.query_context, self.collapse,
        self.head, self.encoder, self._ckpt) = build_prototype_from_checkpoint(
            cfg.weights, self.device)

        # A config built BY HAND (not via from_checkpoint) can name an
        # architecture that disagrees with what `weights` actually holds --
        # caught here rather than a wrong estimate surfacing later with no
        # error at all. Same check ClassifierEstMpp.__init__ runs.
        run_args = self._ckpt['args']
        actual_tile = int(run_args['tile'])
        mismatches = [
            (name, want, got) for name, want, got in
            (('encoder', cfg.encoder, self._ckpt['encoder']),
             ('pooling', cfg.pooling, run_args.get('pooling', '')),
             ('support_context', cfg.support_context, self._ckpt['support_context_name']),
             ('query_context', cfg.query_context, self._ckpt['query_context_name']),
             ('collapse', cfg.collapse, self._ckpt['collapse_name']),
             ('routing_head', cfg.routing_head, self._ckpt['routing_head_name']),
             ('tile_size', cfg.tile_size, actual_tile))
            if want != got]
        if mismatches:
            detail = '; '.join(f'{n}: cfg={w!r} checkpoint={g!r}'
                               for n, w, g in mismatches)
            raise ValueError(
                f'{cfg.weights} disagrees with cfg: {detail}. Build the '
                f'config with PrototypeEstMppConfig.from_checkpoint(weights) '
                f'rather than by hand if this was not deliberate')

        #: Whether `collapse` reduces a level's own support set to one
        #: prototype (`True`, the default every registry entry but `off`
        #: carries) or leaves it un-collapsed (`False`, `PassthroughCollapse`
        #: only) -- decides which of `self.prototypes`/`self.raw_support`
        #: `build(wsi)` fills in, same flag `episode_forward` reads.
        self.collapses = getattr(self.collapse, 'COLLAPSES', True)

        # Combined so IdentifiedBuild.weights_id hashes every trained
        # module's actual parameters in one pass -- a config naming the
        # same architecture but loaded from a DIFFERENT trained checkpoint
        # must not collide with this one. Same idiom ClassifierEstMpp uses.
        self.model = torch.nn.ModuleDict({
            'encoder': self.encoder.model, 'pooling': self.pooling,
            'support_context': self.support_context,
            'query_context': self.query_context,
            'collapse': self.collapse, 'head': self.head})

        frozen = bool(self._ckpt['frozen'])
        self._num_prefix = (int(self.encoder.model_spec.num_prefix) if frozen
                           else 0)
        self._encode_batch = encode_batch
        self._raw_of = (
            (lambda p: encode_raw(self.encoder, p, self._encode_batch, self.device))
            if frozen else (lambda p: trunk_raw(self.encoder, p, self.device)))

        self.wsi = None
        self.mask = None
        #: Filled by `build(wsi)`; `None` until then, same convention
        #: `ClassifierEstMpp.wsi`/`KnnEstMpp.sampler` use to mean "call
        #: build() first".
        self.level_order: Optional[List[int]] = None
        self.level_ds: Optional[List[float]] = None
        self.g_output_by_level: Optional[Dict[int, torch.Tensor]] = None
        self.prototypes: Optional[torch.Tensor] = None
        self.raw_support: Optional[Dict[int, torch.Tensor]] = None

    def _encode_pool(self, images: List[np.ndarray]) -> torch.Tensor:
        '''`[tile,tile,3]` uint8 crops -> `[N, D]` pooled features -- the ONE
        place both `build`'s reference tiles and `estimate`'s query patches
        go through encode+pool, so the two sides can never drift apart on
        how a tile becomes a vector (same invariant this project states
        elsewhere as "same Pooling instance for support and query").'''
        patches = torch.from_numpy(np.stack(images))
        raw = self._raw_of(patches)
        return self.pooling(raw, self._num_prefix)

    def build(self, wsi: Union[openslide.OpenSlide, str],
             mask=None) -> 'PrototypeEstMpp':
        '''Sample + encode the reference bank, then run it through this
        checkpoint's own Stage 2 ONCE -- see this module's docstring for the
        four-step procedure. Overwrites whatever a previous `build()` call
        (on a different WSI) left cached, same as `ClassifierEstMpp.build`/
        `KnnEstMpp.build` -- one estimator instance is reusable across many
        WSIs, one at a time.
        '''
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi = wsi
        self.mask = mask if mask is not None else self.cfg.mask_cfg.build(wsi, self.device)

        plans = native_plans(wsi, self.cfg.tile_size)
        if self.cfg.levels is not None:
            wanted = set(self.cfg.levels)
            plans = [p for p in plans if p.level in wanted]

        sampler = TileSampler(wsi, self.mask, self.cfg.sampler_cfg)
        sampler.sample(plans)
        images = SlideReader(wsi, resize='area').read_samples(
            sampler, ReadSpec(self.cfg.tile_size, self.cfg.tile_size))
        levels_of_sample = [s.meta.level for s in sampler]

        with torch.no_grad():
            pooled = self._encode_pool(images)                       # [N, D]

        by_level: Dict[int, List[torch.Tensor]] = {}
        for level, vec in zip(levels_of_sample, pooled):
            by_level.setdefault(level, []).append(vec)

        self.level_order = sorted(by_level)
        self.level_ds = [float(wsi.level_downsamples[lv]) for lv in self.level_order]

        self.g_output_by_level = {}
        prototypes = []
        raw_support = {}
        with torch.no_grad():
            for level in self.level_order:
                support = torch.stack(by_level[level])                # [K_level, D]
                g_out = self.support_context(support)
                self.g_output_by_level[level] = g_out
                out = self.collapse(g_out)
                if self.collapses:
                    prototypes.append(out)
                else:
                    raw_support[level] = out
        self.prototypes = torch.stack(prototypes) if self.collapses else None
        self.raw_support = raw_support if not self.collapses else None
        return self

    def estimate(self, query: np.ndarray) -> PrototypeEstMppResult:
        return self.from_probs(self.patch_probs(query), self.cfg.vote)

    @property
    def classes_ds(self) -> List[float]:
        '''The ds each class of `patch_probs` stands for, in column order:
        this WSI's own native levels, set by `build`.'''
        return self.level_ds

    def query_patches(self, query: np.ndarray) -> list:
        '''The patches the vote is over: the query's MAIN tiles only, the
        convention ClassifierEstMpp/KnnEstMpp both use. Public so a caller
        weighting patches (FoVVote.QUALITY_SIGNALS) weights these very
        tiles.'''
        qc = QueryPatchContainer(query)
        qc.extract_all(self.cfg.tile_size, overlap=True)
        return list(qc.iter_main())

    def patch_probs(self, query: np.ndarray) -> torch.Tensor:
        '''`[N, K]`: each main patch's softmax over `classes_ds`. The half of
        `estimate` that runs the model -- a caller comparing vote rules
        computes this once and hands it to `from_probs` per rule.'''
        if self.wsi is None:
            raise RuntimeError(
                'call build(wsi) before estimate() -- the reference bank '
                'and chosen_ds/chosen_mpp both need the target WSI')

        images = self.query_patches(query)

        with torch.no_grad():
            query_vecs = self._encode_pool(images)                    # [N, D]
            query_vecs = self.query_context(query_vecs, self.g_output_by_level)
            support = self.prototypes if self.collapses else self.raw_support
            logits = self.head(query_vecs, support)                    # [N, K]
            return torch.softmax(logits, dim=1)

    def from_probs(self, probs: torch.Tensor, vote: str, *,
                   weights: Optional[torch.Tensor] = None,
                   tie_rule: str = 'lower') -> PrototypeEstMppResult:
        '''The other half: one `FoVVote` rule over `patch_probs`, then the
        snap to this WSI's pyramid.'''
        predicted_idx, vote_extra = fov_vote(vote, probs, rungs=self.level_ds,
                                             weights=weights, tie_rule=tie_rule)

        base_mpp = self.wsi.base_mpp
        estimated_ds = self.level_ds[predicted_idx]
        estimated_mpp = base_mpp * estimated_ds

        chosen_level = self.wsi.coarser_level_for_downsample(estimated_ds)
        chosen_ds = float(self.wsi.level_downsamples[chosen_level])
        chosen_mpp = base_mpp * chosen_ds

        return PrototypeEstMppResult(
            estimated_ds=estimated_ds, estimated_mpp=estimated_mpp,
            chosen_ds=chosen_ds, chosen_mpp=chosen_mpp,
            chosen_level=chosen_level,
            predicted_level=self.level_order[predicted_idx],
            vote_extra=vote_extra)
