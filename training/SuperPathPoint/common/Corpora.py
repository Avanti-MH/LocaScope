"""The pre-tile corpora SuperPathPoint reads, by name, as values.

    corpus = corpus_of('stageA', root, mask_cfg, tile=256)
    for folder in corpus.rung_dirs(slide): ...

A corpus is one extraction: a mask, a sampler config, a rung plan and a context
factor (`Store.PreTileCorpus`). Every one of those is known before a slide is
opened, so a reader computes the directory it needs instead of searching a root
for something that looks right: a root holds stage A and stage B of the same
slides on purpose, and a lookup by (slide, ds) would read the union or pick
whichever sorted first.

THE ONE DEFINITION OF EACH CORPUS'S KNOBS. `extract_pretiles` samples with
`sampler_config`, `prepare_chain_stack` names its three axes' corpora through
`RECIPES`, and every reader addresses a corpus through `corpus_of` -- so a
number written here is written nowhere else, and changing one moves every
reader to the new directory at once rather than leaving one behind on the old.

THE RUNGS ARE PART OF THE ADDRESS. A chain is only a chain over the rungs that
were sampled together (`_choose_centres` runs once over all of them), so a
corpus cut over (1..16) and one cut over (1..32) are two corpora even when
every other knob agrees. That is why `corpus_of` takes the rungs rather than
carrying a default per recipe: the caller that extracted knows them, and the
caller that reads passes the same value.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from DsLadder import DEFAULT_RUNGS
from ReadGeometry import ReadSpec
from Store import PreTileCorpus
from TileSampler import (PRE_TILE_FACTOR, SAMPLER_RECIPES, InheritConfig,
                         OverlapConfig,
                         PlanSpec, RichnessConfig, SamplerConfig, centre_margin)

#: A flat rejection budget per cell; `sampler_config` divides it by n.
MAX_TRIES = 2500


def pretile_spec(tile: int, factor: int = PRE_TILE_FACTOR) -> ReadSpec:
    """What a pre-tile extraction reads: a `tile` px tile and the context
    around it out to `tile * factor`, not rotated. The sampler reserves that
    read (`ReadSpec.place`), so the lattice never offers a position whose
    pre-tile runs off the scanned area."""
    return ReadSpec(int(tile), int(tile), rotates=False,
                      margin_out=centre_margin(int(tile), int(factor)))


def sampler_config(*, n: int, seed: int = 0,
                   candidates: str = 'lattice', max_tries: int = MAX_TRIES,
                   step: float = 1.0, max_overlap: float = 0.0,
                   overlapping_share: float = 0.0,
                   bucket_frame: str = 'per_rung', inherit_share: float = 0.0,
                   inherit_source_rung: Optional[float] = None) -> SamplerConfig:
    """One config for every rung, which is what inheritance requires.

    ONE SAMPLER OVER ALL RUNGS, NOT ONE PER RUNG. `_choose_centres` runs once,
    before any rung is filled, and `_place_inherited` then validates each
    centre at each rung; a sampler per rung would choose its own centres and no
    two rungs would share one.

    `stack_kind` is always 'F': every plan an extraction builds comes from
    DsLadder, which tags itself 'F' (TileSampler's native_plans); 'R' cannot
    occur here.
    """
    lattice = SAMPLER_RECIPES['lattice']
    return replace(
        lattice, n_per_rung=int(n), seed=int(seed),
        candidates=candidates,
        max_tries_per_tile=max(1, int(max_tries) // max(int(n), 1)),
        overlap=replace(lattice.overlap, step=float(step),
                        max_overlap_ratio=float(max_overlap),
                        overlapping_share=float(overlapping_share)),
        richness=replace(lattice.richness, bucket_frame=bucket_frame),
        inherit=replace(lattice.inherit, stack_kind='F',
                        share=float(inherit_share),
                        source_rung=inherit_source_rung))


#: The named corpora, every field written out (test_config_identity's recipe
#: lint). `stageA` is the training corpus (ExtractPreTiles.sh's defaults); the
#: other two are F's and C's own tiles for the survival analysis
#: (prepare_chain_stack.py). All three share RichnessConfig's floors and caps
#: and differ in the rest. `max_tries_per_tile` is MAX_TRIES spread over n.
RECIPES = {
    'stageA': SamplerConfig(
        n_per_rung=100, seed=0,
        richness=RichnessConfig(
            scorer='background', edges=(0.15, 0.30, 0.50, 0.70, 0.85, 0.95),
            floors=(0.05, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0),
            caps=(0.15, 0.25, 0.60, 0.20, 0.20, 0.0, 0.0),
            bucket_frame='per_rung', floor_frame='ask'),
        overlap=OverlapConfig(
            step=1.0, max_overlap_ratio=0.0, overlapping_share=0.0,
            jitter_offsets=((0.25, 1.0), (1.0, 0.25), (0.75, 1.0),
                            (1.0, 0.75), (1.25, 1.25)),
            jitter_cap=0.0),
        inherit=InheritConfig(stack_kind='F', share=0.0, source_rung=None,
                              on_incomplete='drop'),
        candidates='lattice', max_tries_per_tile=25),
    'stageB-fOwn': SamplerConfig(
        n_per_rung=200, seed=0,
        richness=RichnessConfig(
            scorer='background', edges=(0.15, 0.30, 0.50, 0.70, 0.85, 0.95),
            floors=(0.05, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0),
            caps=(0.15, 0.25, 0.60, 0.20, 0.20, 0.0, 0.0),
            bucket_frame='at_inherit', floor_frame='ask'),
        overlap=OverlapConfig(
            step=0.5, max_overlap_ratio=0.5, overlapping_share=1.0,
            jitter_offsets=((0.25, 1.0), (1.0, 0.25), (0.75, 1.0),
                            (1.0, 0.75), (1.25, 1.25)),
            jitter_cap=0.0),
        inherit=InheritConfig(stack_kind='F', share=1.0, source_rung=16.0,
                              on_incomplete='drop'),
        candidates='lattice', max_tries_per_tile=12),
    'stageB-cOwn': SamplerConfig(
        n_per_rung=10, seed=0,
        richness=RichnessConfig(
            scorer='background', edges=(0.15, 0.30, 0.50, 0.70, 0.85, 0.95),
            floors=(0.05, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0),
            caps=(0.15, 0.25, 0.60, 0.20, 0.20, 0.0, 0.0),
            bucket_frame='per_rung', floor_frame='ask'),
        overlap=OverlapConfig(
            step=1.0, max_overlap_ratio=0.0, overlapping_share=0.0,
            jitter_offsets=((0.25, 1.0), (1.0, 0.25), (0.75, 1.0),
                            (1.0, 0.75), (1.25, 1.25)),
            jitter_cap=0.0),
        inherit=InheritConfig(stack_kind='F', share=0.0, source_rung=None,
                              on_incomplete='drop'),
        candidates='lattice', max_tries_per_tile=250),
}

#: Which recipe is each survival axis's own corpus. R reads stageA and never
#: extracts it (prepare_chain_stack.ensure_corpus says why).
AXIS_RECIPE = {'F': 'stageB-fOwn', 'R': 'stageA', 'C': 'stageB-cOwn'}


def recipe_config(name: str) -> SamplerConfig:
    """`RECIPES[name]`, the SamplerConfig it samples with."""
    if name not in RECIPES:
        raise KeyError(f'no corpus recipe {name!r}; known: {sorted(RECIPES)}')
    return RECIPES[name]


def ladder(rungs: Optional[Sequence[float]], tile: int,
           factor: int = PRE_TILE_FACTOR) -> PlanSpec:
    """The rung plan an extraction was cut over, for the pre-tile camera. None
    rungs = DsLadder's default, which is what extract_pretiles cuts when no
    --rungs is given."""
    return PlanSpec('ladder', tuple(rungs) if rungs else tuple(DEFAULT_RUNGS),
                    camera=pretile_spec(tile, factor))


def corpus_of(name: str, root, mask_cfg, *, tile: int,
              rungs: Optional[Sequence[float]] = None,
              factor: int = PRE_TILE_FACTOR) -> PreTileCorpus:
    """The directory recipe `name` lands in under `root`, cut over `rungs`."""
    return PreTileCorpus.of(root, mask_cfg, recipe_config(name),
                            ladder(rungs, tile, factor), factor)
