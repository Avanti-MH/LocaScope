"""The pre-tile corpora SuperPathPoint reads, by name, as values.

    corpus = corpus_of('stageA', root, mask_cfg, tile=256)
    for folder in corpus.rung_dirs(slide): ...

A corpus is one extraction: a mask, a sampler config, a rung plan and a context
factor (`Store.PreTileCorpus`). Every one of those is known before a slide is
opened, so a reader computes the directory it needs instead of searching a root
for something that looks right. The search is what used to go wrong -- a root
holds stage A and stage B of the same slides on purpose, and a lookup by
(slide, ds) either read the union or picked whichever sorted first.

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

from typing import Optional, Sequence

from DsLadder import DEFAULT_RUNGS
from Store import PreTileCorpus
from TileSampler import (PRE_TILE_FACTOR, InheritConfig, OverlapConfig,
                         PlanSpec, RichnessConfig, SamplerConfig)

#: A flat rejection budget per cell; `sampler_config` divides it by n.
MAX_TRIES = 2500


def sampler_config(*, tile: int, n: int, seed: int = 0,
                   candidates: str = 'lattice', max_tries: int = MAX_TRIES,
                   grid_step: int = 0, max_overlap: float = 0.0,
                   overlapping_share: float = 0.0,
                   bucket_frame: str = 'per_rung', inherit_share: float = 0.0,
                   inherit_source_rung: Optional[float] = None) -> SamplerConfig:
    """One config for every rung, which is what inheritance requires.

    ONE SAMPLER OVER ALL RUNGS, NOT ONE PER RUNG. `_choose_centres` runs once,
    before any rung is filled, and `_place_inherited` then validates each
    centre at each rung; a sampler per rung would choose its own centres and no
    two rungs would share one. The corpus of 2026-08-27 has `inherit_id = -1`
    on all 6,388 rows for exactly that reason.

    `stack_kind` is always 'F': every plan an extraction builds comes from
    DsLadder, which tags itself 'F' (TileSampler's native_plans); 'R' cannot
    occur here.
    """
    return SamplerConfig(
        tile=int(tile), n_per_rung=int(n), seed=int(seed),
        candidates=candidates,
        max_tries_per_tile=max(1, int(max_tries) // max(int(n), 1)),
        overlap=OverlapConfig(grid_step=int(grid_step),
                              max_overlap_ratio=float(max_overlap),
                              overlapping_share=float(overlapping_share)),
        richness=RichnessConfig(bucket_frame=bucket_frame),
        inherit=InheritConfig(stack_kind='F', share=float(inherit_share),
                              source_rung=inherit_source_rung))


#: The named corpora. `stageA` is the 2026-08-27 training corpus
#: (ExtractPreTiles.sh's defaults); the other two are F's and C's own tiles for
#: the survival analysis (prepare_chain_stack.py). All three take
#: RichnessConfig's default floors and caps and differ in the rest.
RECIPES = {
    'stageA': dict(n=100, inherit_share=0.0, inherit_source_rung=None,
                   bucket_frame='per_rung',
                   grid_step=0, max_overlap=0.0, overlapping_share=0.0),
    'stageB-fOwn': dict(n=200, inherit_share=1.0, inherit_source_rung=16.0,
                        bucket_frame='at_inherit',
                        grid_step=128, max_overlap=0.5, overlapping_share=1.0),
    'stageB-cOwn': dict(n=10, inherit_share=0.0, inherit_source_rung=None,
                        bucket_frame='per_rung',
                        grid_step=0, max_overlap=0.0, overlapping_share=0.0),
}

#: Which recipe is each survival axis's own corpus. R reads stageA and never
#: extracts it (prepare_chain_stack.ensure_corpus says why).
AXIS_RECIPE = {'F': 'stageB-fOwn', 'R': 'stageA', 'C': 'stageB-cOwn'}


def recipe_config(name: str, tile: int) -> SamplerConfig:
    """`RECIPES[name]` as the SamplerConfig it samples with."""
    if name not in RECIPES:
        raise KeyError(f'no corpus recipe {name!r}; known: {sorted(RECIPES)}')
    return sampler_config(tile=tile, **RECIPES[name])


def ladder(rungs: Optional[Sequence[float]] = None) -> PlanSpec:
    """The rung plan an extraction was cut over. None = DsLadder's default,
    which is what extract_pretiles cuts when no --rungs is given."""
    return PlanSpec('ladder', tuple(rungs) if rungs else tuple(DEFAULT_RUNGS))


def corpus_of(name: str, root, mask_cfg, *, tile: int,
              rungs: Optional[Sequence[float]] = None,
              factor: int = PRE_TILE_FACTOR) -> PreTileCorpus:
    """The directory recipe `name` lands in under `root`, cut over `rungs`."""
    return PreTileCorpus.of(root, mask_cfg, recipe_config(name, tile),
                            ladder(rungs), factor)
