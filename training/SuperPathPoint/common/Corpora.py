"""The corpora SuperPathPoint reads, by name, as values.

    corpus = corpus_of('stageA', mask_cfg, tile=256, job=..., mask_job=...)
    for ds, tiles in corpus.tiles(slide).items():
        pre = corpus.read(tiles[0])          # the pre-tile, read now

A corpus is one draw: a mask, a sampler config and a rung plan whose camera is
the pre-tile (`pretile_spec`). `TileSampler.cached` makes it on first use and
reads it back after, so there is no extraction step: the positions are the
draw's index, and the pixels are read from the slide when they are asked for
(`Corpus.read`), the same read for every consumer. Every knob is known before a
slide is opened, so a reader computes the address it needs instead of
searching a root for something that looks right: stage A and stage B of the
same slides sit side by side on purpose, and a lookup by (slide, ds) would read
the union or pick whichever sorted first.

THE ONE DEFINITION OF EACH CORPUS'S KNOBS. `RECIPES` names them,
`prepare_chain_stack` picks its three axes' corpora from them, and every reader
addresses a corpus through `corpus_of` -- so a number written here is written
nowhere else, and changing one moves every reader to the new draw at once.

THE RUNGS ARE PART OF THE ADDRESS. A chain is only a chain over the rungs that
were sampled together (`_choose_centres` runs once over all of them), so a
corpus drawn over (1..16) and one over (1..32) are two corpora even when every
other knob agrees. That is why `corpus_of` takes the rungs rather than carrying
a default per recipe: the caller that drew knows them, and the caller that
reads passes the same value.

PIXELS IN THE PROCESS THAT READS THEM. `Corpus.read` opens one `SlideReader`
per slide per process, on first use: a DataLoader worker forked before any read
opens its own handle, and no openslide handle crosses a fork.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Cache import Address, wsi_stem_of
from DsLadder import DEFAULT_RUNGS
from ReadGeometry import ReadSpec
from TileSampler import (PRE_TILE_FACTOR, SAMPLER_RECIPES, InheritConfig,
                         OverlapConfig, PlanSpec, RichnessConfig, SampleMeta,
                         SamplerConfig, TileSampler, centre_margin)

def pretile_spec(tile: int, factor: int = PRE_TILE_FACTOR) -> ReadSpec:
    """What a corpus reads for one position: a `tile` px tile and the context
    around it out to `tile * factor`, not rotated. The sampler reserves that
    read (`ReadSpec.place`), so the lattice never offers a position whose
    pre-tile runs off the scanned area.

    WHY THE CONTEXT. `warp_image` fills anything sampled from outside its input
    with pure black, and a production homography needs 1.78x the source it is
    given (spec.md 6.6, `valid 67.8%`). Black is not a blank: it is a straight
    maximum-contrast edge with two right angles, which is what a corner
    detector fires on. A WSI has an answer a photograph does not -- tissue
    continues past the tile -- so the warp runs on the pre-tile and the tile
    is its centre crop (`TileSampler.centre_crop`), everywhere: every output
    pixel is real tissue. Nothing slides a pre-tile inward at a slide edge."""
    return ReadSpec(int(tile), int(tile), rotates=False,
                      margin_out=centre_margin(int(tile), int(factor)))


#: The named corpora, every field written out (test_config_identity's recipe
#: lint). `stageA` is the training corpus; the other two are F's and C's own
#: tiles for the survival analysis (prepare_chain_stack.py). All three share RichnessConfig's floors and caps
#: and differ in the rest. `max_tries_per_tile` is a budget of 2500 tries a
#: rung spread over n.
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

#: Which recipe is each survival axis's own corpus. R reads stageA's tiles.
AXIS_RECIPE = {'F': 'stageB-fOwn', 'R': 'stageA', 'C': 'stageB-cOwn'}


def recipe_config(name: str) -> SamplerConfig:
    """`RECIPES[name]`, the SamplerConfig it samples with."""
    if name not in RECIPES:
        raise KeyError(f'no corpus recipe {name!r}; known: {sorted(RECIPES)}')
    return RECIPES[name]


def ladder(rungs: Optional[Sequence[float]], tile: int,
           factor: int = PRE_TILE_FACTOR) -> PlanSpec:
    """The rung plan a corpus is drawn over, for the pre-tile camera. None
    rungs = DsLadder's default."""
    return PlanSpec('ladder', tuple(rungs) if rungs else tuple(DEFAULT_RUNGS),
                    camera=pretile_spec(tile, factor))


@dataclass(frozen=True)
class Tile:
    """One position of a corpus: the slide file, its row in the draw and the
    sampler's meta. Plain data -- it crosses into a DataLoader worker."""
    path: str
    index: int
    meta: SampleMeta

    @property
    def centre_l0(self) -> Tuple[float, float]:
        """The tile's level-0 centre -- what the rungs of a chain share. The
        centre, not a corner: under rotation corners are different corners of
        one footprint."""
        half = float(self.meta.footprint_l0) / 2.0
        return (self.meta.x + half, self.meta.y + half)


#: (pid, slide path) -> SlideReader. Per process, so a forked worker never
#: uses its parent's handle (module docstring).
_READERS: Dict[Tuple[int, str], object] = {}


@dataclass(frozen=True)
class Corpus:
    """One corpus: recipe `name` drawn through `mask_cfg` over `rungs`, its
    draws in job `job`'s cache, its masks in `mask_job`'s.

        slide=<s>/seg=<seg_id>/region=<region_id>/plan=<plan>/draw/
            index_<sampler_id>.csv ...            the positions
        .../draw=<sampler_id>/labels/             keypoint labels made on them

    `job` and `mask_job` say where, not what: they are not in `key`."""
    name: str
    mask_cfg: object
    sampler_cfg: SamplerConfig
    rungs: Tuple[float, ...]
    tile: int
    factor: int
    job: str
    mask_job: str

    @property
    def plan(self) -> PlanSpec:
        return ladder(self.rungs, self.tile, self.factor)

    @property
    def sampler_id(self) -> str:
        return self.sampler_cfg.identity_id()

    @property
    def key(self) -> str:
        """`<seg_id>/<region_id>/<plan>/<sampler_id>` -- what a label set and a
        checkpoint record as the corpus they were made from. An identity, not
        a path."""
        return (f'{self.mask_cfg.seg_id()}/{self.mask_cfg.region_id()}/'
                f'{self.plan.key()}/{self.sampler_id}')

    def address(self, slide: str, job: Optional[str] = None) -> Address:
        """The `draw=<sampler_id>` level of `slide` in `job`'s tree (default
        the draw's own) -- where what is made from these tiles is addressed."""
        return TileSampler.draw_address(job or self.job, wsi_stem_of(slide),
                                        self.mask_cfg, self.plan
                                        ).at(draw=self.sampler_id)

    @staticmethod
    def path_of(slide: str) -> str:
        """A slide file: `slide` itself when it is one, else the AccessDatasets
        entry of that name."""
        if os.path.exists(str(slide)):
            return str(slide)
        from AccessDatasets import locate                           # noqa: PLC0415
        return str(locate(str(slide)).path)

    def draw(self, slide: str) -> TileSampler:
        """The draw of `slide` -- read back when it is in `job`'s cache, made
        (the mask included, through `mask_job`'s cache) when it is not."""
        from TissueMaskConfig import MaskMaker                      # noqa: PLC0415
        import torch                                                # noqa: PLC0415
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with MaskMaker(self.mask_cfg, self.mask_job, device) as masks:
            return TileSampler.cached(self.path_of(slide), self.sampler_cfg,
                                      self.plan, self.job, masks=masks)

    def tiles(self, slide: str, rungs: Optional[Sequence[float]] = None
              ) -> Dict[float, List[Tile]]:
        """`{ds: [Tile]}` of `slide`, finest rung first, each rung in draw
        order; only `rungs` when given."""
        path = self.path_of(slide)
        keep = None if rungs is None else {float(r) for r in rungs}
        out: Dict[float, List[Tile]] = {}
        for i, sample in enumerate(self.draw(path)):
            ds = float(sample.meta.ds)
            if keep is None or ds in keep:
                out.setdefault(ds, []).append(Tile(path, i, sample.meta))
        return dict(sorted(out.items()))

    def read(self, tile: Tile) -> np.ndarray:
        """The pre-tile of `tile`, `tile * factor` px RGB uint8: the tile and
        the context around it, the read the sampler reserved
        (`pretile_spec`), with the stack the draw recorded. The tile is its
        centre crop (`TileSampler.centre_crop`)."""
        key = (os.getpid(), tile.path)
        reader = _READERS.get(key)
        if reader is None:
            from SafeSlide import SafeSlide                         # noqa: PLC0415
            from SlideReader import SlideReader                     # noqa: PLC0415
            reader = _READERS[key] = SlideReader(SafeSlide(tile.path),
                                                 resize='area')
        m = tile.meta
        image = reader.read(m.x, m.y, pretile_spec(self.tile, self.factor),
                            m.ds, stack=m.stack_kind)
        if image is None:
            raise ValueError(f'{Path(tile.path).stem} ds {m.ds:g} ({m.x}, {m.y}): '
                             f'the pre-tile reads off the slide, though the '
                             f'sampler reserved it')
        return image


def corpus_of(name: str, mask_cfg, *, tile: int, job: str, mask_job: str,
              rungs: Optional[Sequence[float]] = None,
              factor: int = PRE_TILE_FACTOR) -> Corpus:
    """Recipe `name` drawn through `mask_cfg` over `rungs` (None: DsLadder's)."""
    return Corpus(name, mask_cfg, recipe_config(name),
                  tuple(float(r) for r in (rungs or DEFAULT_RUNGS)), int(tile),
                  int(factor), str(job), str(mask_job))
