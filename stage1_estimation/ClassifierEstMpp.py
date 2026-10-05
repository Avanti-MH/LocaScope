'''Stage 1 (mpp estimation) by a trained `MppRoutingHead` classifier, in the
shape `StageInterface.MppEstimator` describes.

    cfg = ClassifierEstMppConfig.from_checkpoint(
        '/work/u26130998/result/MppRoutingHead/weights/'
        'gigapath_frozen_arcface_best.pt')
    est = ClassifierEstMpp(cfg).build(wsi)
    result = est.estimate(query_img)   # query_img: RGB uint8 np.ndarray

Loading the model (`__init__`) and binding a WSI (`build`) are SEPARATE and
have very different costs, unlike `KnnEstMpp` where `build_samples`/
`build_ref_features` -- sampling and encoding a reference bank FROM the target
WSI -- are the expensive step. This classifier was trained once, offline, on
a different corpus entirely; binding a WSI here costs one property read
(`wsi.base_mpp`) because nothing about the trained model depends on which
slide a query is being routed against.

WHY `build(wsi)` IS STILL A REAL STEP AND NOT A NO-OP. The classifier's label
space (`RUNGS`, read off the checkpoint -- see below) is a RELATIVE ds
multiplier, deliberately: `training/MppRoutingHead/Datasets.py`'s own
docstring is "rung VALUE, not a per-WSI level index, so the class id means
the same thing on BRACS (4x pyramid) and Ki67 (2x pyramid)". A query photo
carries no `base_mpp` of its own -- it is a photograph, not a slide -- so
turning "this looks like a rung-4 image" into an absolute
`estimated_mpp` needs the ONE `base_mpp` number that the relative rung is
relative TO. That is the target WSI's, and it is the only thing `build(wsi)`
supplies.

PROCEDURE (`estimate`)
-----------------------
1. Cut `query` into `cfg.tile_size` patches (`PatchingLib.QueryPatchContainer`,
   `overlap=True` for full coverage, scored on MAIN patches only -- same
   convention `KnnEstMpp.estimate` uses, so a query photo is tiled the same
   way for either estimator).
2. Run every patch through the loaded encoder + head (`torch.no_grad()`: this
   is inference, no backward pass is ever taken here) and take each patch's
   own SOFTMAX probability distribution over classes -- not just its argmax.
3. AGGREGATE every patch's own softmax distribution into one FoV-level
   class via `FoVVote.vote(cfg.vote, probs, ...)` (`cfg.vote` defaults to
   `'mean_probability'`, this file's original and only method before
   `FoVVote.py` existed -- see that module and `FoV_Vote.md`, this
   directory, for the five other choices and the reasoning behind each).
4. `estimated_ds = rungs[predicted_class]`, `estimated_mpp = wsi.base_mpp *
   estimated_ds` -- the RELATIVE-to-ABSOLUTE step `build(wsi)` exists for.
5. Snap to a level this WSI actually has:
   `wsi.coarser_level_for_downsample(estimated_ds)` -- see `StageInterface`'s
   own docstring for why that call is inlined here rather than wrapped in a
   function of its own.
'''
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

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
from PatchingLib import QueryPatchContainer                              # noqa: E402
from SafeSlide import SafeSlide                                         # noqa: E402
from Checkpoints import build_from_checkpoint                          # noqa: E402
from Features import encode_raw, trunk_raw                              # noqa: E402

from stage1_estimation.StageInterface import EstMppResult                                 # noqa: E402
from stage1_estimation.FoVVote import vote as fov_vote                                     # noqa: E402


# ── config ───────────────────────────────────────────────────────────────────

@register('classifier_est_mpp')
@dataclass(frozen=True)
class ClassifierEstMppConfig(IdentifiedConfig):
    '''Which trained encoder + head to run, and at what tile size.

    `encoder` is `TileEncoderFunc`'s own registered name (`'gigapath'`,
    `'convnext_v2'`, ...); `classifier` is `Heads.py`'s registered name
    (`'linear'`, `'mlp'`, `'arcface'`) -- see `Heads.classifier_class`'s own
    docstring for why a registered name and not `type(...).__name__`.
    `reduction` (`common.Head.REDUCTIONS`: `'fixed'`/`'attn'`/`'clsattn'`) is
    the OTHER axis `common.Head.Head` composes a classifier with, and is not
    itself a registry -- none of its values is ever instantiated by name.

    Which encoder blocks a head mixes (`HeadConfig.encoder_layers`) is NOT a
    field here. It travels inside the checkpoint's `head_cfg`, the head is
    rebuilt from it, and `encode_raw` is asked for exactly those blocks -- so a
    config cannot disagree with it, and `weights_id` already tells two
    checkpoints apart. A field would also add `encoder_layers=()` to the
    identity of every config built before it, moving all their ids.

    All four (plus `tile_size`) are explicit fields rather than left for
    whoever opens `weights` to notice, for two reasons: a caller comparing
    several configs (a pipeline sweeping checkpoints) wants "gigapath + attn +
    arcface + 256" to read straight out of `identity_id`/`identity_parts`, and
    a config built by hand that DISAGREES with what its own `weights` file
    actually contains is exactly the mismatch `ClassifierEstMpp.__init__`
    checks for -- see there. `from_checkpoint` is the usual way to get one: it
    reads every field off the checkpoint itself, so a sweep names a PATH, not
    four redundant facts that could drift from what the file holds.

    `weights` is NOT_IDENTITY. `IdentifiedBuild.weights_id` (via
    `ClassifierEstMpp.model`) hashes the loaded state dicts directly, which is
    what actually answers "did the trained numbers change" -- the same
    weights loaded from a moved or renamed file must hash identically, and a
    stale path only costs a `FileNotFoundError`, never a silent wrong answer
    (`ConfigIdentity.ModelConfig.weights` is the same call for the same
    reason).
    '''
    encoder: str
    classifier: str
    reduction: str
    tile_size: int
    weights: str
    #: `FoVVote.VOTE_CHOICES` key -- an INFERENCE-time choice, never
    #: recorded on the checkpoint (training never sees a whole FoV, only
    #: individual tiles, so there is nothing for the checkpoint to have
    #: recorded). Part of `identity_id` like every field here except
    #: `weights`: two runs differing only in `vote` are two different
    #: experiments, not the same one re-labelled.
    vote: str = 'mean_probability'

    NOT_IDENTITY = ('weights',)

    @classmethod
    def from_checkpoint(cls, weights: str,
                        vote: str = 'mean_probability') -> 'ClassifierEstMppConfig':
        ckpt = torch.load(weights, map_location='cpu')
        return cls(encoder=ckpt['encoder'], classifier=ckpt['classifier'],
                   reduction=ckpt['reduction'],
                   tile_size=int(ckpt['extra']['tile_size']), weights=weights,
                   vote=vote)


# ── result ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassifierEstMppResult(EstMppResult):
    '''`EstMppResult` plus what is specific to how a classifier arrived at its
    answer.

    NOT `tile_size`/`query_patch_count`, even though the old
    `GigapathKnnEstiMppResult` this was drafted against carried both. Copying
    that Result's fields without re-deriving why each one is there was the
    mistake this one corrects: `tile_size` is already on
    `cfg.tile_size` and `query_patch_count` is already knowable from the input
    query at the call site -- neither is information this Result would be the
    only place to find, so storing either here is a copy able to go stale
    against the thing it copied. (`samples_per_level`/`k`, the KNN Result's
    other two fields, do not apply here at all -- they are that method's own
    build-time hyperparameters, not a property of an estimate.)

    `predicted_class` IS worth keeping despite being a deterministic function
    of `estimated_ds` and `cfg.rungs` (`estimated_ds = cfg.rungs[predicted_class]`):
    recovering it from `estimated_ds` needs a reverse float lookup into
    `cfg.rungs`, and a caller that wants the exact class index -- to score
    against a ground-truth rung label the way `training/MppRoutingHead/
    Runtime.score` does -- gets it directly rather than re-deriving it with a
    tolerance it would have to pick itself.

    `vote_extra` is whichever `FoVVote` function `cfg.vote` names filled
    in -- shape depends on the method (`mean_probability` gives
    `confidence`, `quality_weighted` also gives `effective_sample_size`,
    ...), see `FoVVote.py`'s own docstring for why forcing one common
    shape here across every vote choice would be the wrong move. NOT a
    dataclass field of its own kind per vote method -- `Dict[str, float]`
    keeps `ClassifierEstMppResult` ONE class regardless of `cfg.vote`,
    rather than a subclass per vote choice a caller would have to
    `isinstance` against.
    '''
    predicted_class: int
    vote_extra: Dict[str, float]


# ── estimator ────────────────────────────────────────────────────────────────

class ClassifierEstMpp(IdentifiedBuild):
    '''Loads once (`__init__`), binds a WSI once per slide (`build`), answers
    per query (`estimate`) -- see this module's docstring for the procedure
    and for why `build` is cheap here specifically.
    '''

    #: Append-only zero point for `identity_id`/`identity_parts` -- empty
    #: until a specific (encoder, classifier, reduction, tile_size)
    #: combination is adopted as the project's own default, per
    #: `ConfigIdentity`'s rule 1.
    BASELINE: Dict[str, Any] = {}

    def __init__(self, cfg: ClassifierEstMppConfig,
                device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
                encode_batch: int = 256):
        self.cfg = cfg
        self.device = torch.device(device)
        self.head, self.encoder, self._ckpt = build_from_checkpoint(
            cfg.weights, self.device)

        # A config built BY HAND (not via from_checkpoint) can name an
        # (encoder, classifier, reduction, tile_size) that disagrees with what
        # `weights` actually holds -- caught here rather than a mismatched
        # number surfacing later as a wrong estimate with no error at all.
        actual_tile = int(self._ckpt['extra']['tile_size'])
        mismatches = [
            (name, want, got) for name, want, got in
            (('encoder', cfg.encoder, self._ckpt['encoder']),
             ('classifier', cfg.classifier, self._ckpt['classifier']),
             ('reduction', cfg.reduction, self._ckpt['reduction']),
             ('tile_size', cfg.tile_size, actual_tile))
            if want != got]
        if mismatches:
            detail = '; '.join(f'{n}: cfg={w!r} checkpoint={g!r}'
                               for n, w, g in mismatches)
            raise ValueError(
                f'{cfg.weights} disagrees with cfg: {detail}. Build the '
                f'config with ClassifierEstMppConfig.from_checkpoint(weights) '
                f'rather than by hand if this was not deliberate')

        # The label space this head classifies into -- a ds multiplier per
        # class, RELATIVE to whichever WSI a query is eventually routed
        # against. Travels with the checkpoint (`Checkpoints.py`'s `extra`),
        # not re-imported from `training/MppRoutingHead/Datasets.RUNGS`, so
        # this stays correct even if that package's own default rung ladder
        # is later changed for a different training run.
        self.rungs = tuple(float(r) for r in self._ckpt['extra']['rungs'])

        # Combined so `IdentifiedBuild.weights_id` hashes both the head's and
        # (for a fine-tuned checkpoint) the trunk's actual parameters in one
        # pass -- a config naming the same encoder+head but loaded from a
        # DIFFERENT trained checkpoint must not collide with this one.
        self.model = torch.nn.ModuleDict(
            {'encoder': self.encoder.model, 'head': self.head})

        frozen = bool(self._ckpt['frozen'])
        # Same derivation `cli/evaluate.py` uses: a fine-tuned trunk's
        # spatial exit has already dropped its prefix tokens (`Features.
        # trunk_raw`'s own docstring), so num_prefix is 0 there regardless of
        # what the encoder's own model_spec reports.
        self._num_prefix = (int(self.encoder.model_spec.num_prefix) if frozen
                           else 0)
        self._encode_batch = encode_batch
        self._raw_of = (
            (lambda p: encode_raw(self.encoder, p, self._encode_batch, self.device,
                                  layers=self.head.layers))
            if frozen else (lambda p: trunk_raw(self.encoder, p, self.device)))

        self.wsi = None

    def build(self, wsi: Union[openslide.OpenSlide, str]) -> 'ClassifierEstMpp':
        '''Bind the WSI queries will be routed against. Cheap -- see this
        module's docstring for why, in contrast to a KNN estimator's own
        `build_samples`/`build_ref_features`.'''
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi = wsi
        return self

    def estimate(self, query: np.ndarray) -> ClassifierEstMppResult:
        return self.from_probs(self.patch_probs(query), self.cfg.vote)

    @property
    def classes_ds(self) -> Tuple[float, ...]:
        '''The ds each class of `patch_probs` stands for, in column order.'''
        return self.rungs

    def query_patches(self, query: np.ndarray) -> list:
        '''The patches the vote is over: the query's MAIN tiles only -- the
        overlap corners exist for coverage, not for this vote. Same
        convention as `KnnEstMpp.estimate`. Public so a caller weighting
        patches (FoVVote.QUALITY_SIGNALS) weights these very tiles.'''
        qc = QueryPatchContainer(query)
        qc.extract_all(self.cfg.tile_size, overlap=True)
        return list(qc.iter_main())

    def patch_probs(self, query: np.ndarray) -> torch.Tensor:
        '''`[M, num_classes]`: each main patch's softmax over `classes_ds`.
        The half of `estimate` that runs the model -- a caller comparing vote
        rules computes this once and hands it to `from_probs` per rule.'''
        if self.wsi is None:
            raise RuntimeError(
                'call build(wsi) before estimate() -- chosen_ds/chosen_mpp '
                'need the target WSI\'s own pyramid and base_mpp')

        patches = torch.from_numpy(np.stack(self.query_patches(query)))

        with torch.no_grad():
            raw = self._raw_of(patches)
            logits = self.head(raw, self._num_prefix)      # [M, num_classes]
            return torch.softmax(logits, dim=1)              # [M, num_classes]

    def from_probs(self, probs: torch.Tensor, vote: str, *,
                   weights: Optional[torch.Tensor] = None,
                   tie_rule: str = 'lower') -> ClassifierEstMppResult:
        '''The other half: one `FoVVote` rule over `patch_probs`, then the
        snap to this WSI's pyramid.'''
        predicted_class, vote_extra = fov_vote(vote, probs, rungs=self.rungs,
                                               weights=weights, tie_rule=tie_rule)

        base_mpp = self.wsi.base_mpp
        estimated_ds = self.rungs[predicted_class]
        estimated_mpp = base_mpp * estimated_ds

        chosen_level = self.wsi.coarser_level_for_downsample(estimated_ds)
        chosen_ds = float(self.wsi.level_downsamples[chosen_level])
        chosen_mpp = base_mpp * chosen_ds

        return ClassifierEstMppResult(
            estimated_ds=estimated_ds, estimated_mpp=estimated_mpp,
            chosen_ds=chosen_ds, chosen_mpp=chosen_mpp,
            chosen_level=chosen_level,
            predicted_class=predicted_class, vote_extra=vote_extra)
