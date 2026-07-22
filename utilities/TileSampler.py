"""
TileSampler — sample tiles from WSI levels within tissue regions.

Usage:
    from TissuesRegionsMask import TissuesRegionsMask
    from TileSampler import TileSampler
    import openslide

    wsi = openslide.OpenSlide('slide.svs')
    mask = TissuesRegionsMask.from_wsi(wsi)
    sampler = TileSampler(wsi, mask, tile_size=256, seed=42)

    # Sample 100 tiles from each level
    sampler.sample(n=100)

    # Sample 50 tiles from level 1 only
    sampler.sample(n=50, level=1)

    # Access sampled tiles
    info = sampler[0]
    img = sampler.read(0)

    # Batch read for model inference
    for batch in sampler.iter_batches(batch_size=32):
        feats = model(batch)

    # Save / reload coordinates
    sampler.save('tiles.json')
    sampler2 = TileSampler.from_json(wsi, mask, 'tiles.json')
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Optional, Union
import json
import sys

import numpy as np
import openslide
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from TissuesRegionsMask import TissuesRegionsMask


@dataclass
class TileInfo:
    level: int      # WSI pyramid level
    x: int          # top-left x in level-0 coordinates
    y: int          # top-left y in level-0 coordinates
    tile_size: int  # tile width/height in level pixels
    mpp: float      # µm/px at this level


class TileSampler:
    def __init__(
        self,
        wsi: openslide.OpenSlide,
        tissue_mask: TissuesRegionsMask,
        tile_size: int = 256,
        seed: Optional[int] = None,
    ):
        self.wsi = wsi
        self.mask = tissue_mask
        self.tile_size = tile_size
        self.rng = np.random.default_rng(seed)
        self.tiles: List[TileInfo] = []

        base_mpp = float(wsi.properties.get('openslide.mpp-x', 0))
        self.level_mpps = [
            base_mpp * wsi.level_downsamples[lv]
            for lv in range(wsi.level_count)
        ]

    # ── Container API ─────────────────────────────────────────────────────────

    def __getitem__(self, index: int) -> TileInfo:
        return self.tiles[index]

    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self) -> Iterator[TileInfo]:
        return iter(self.tiles)

    def tiles_at_level(self, level: int) -> List[TileInfo]:
        """Return a copy of sampled tiles belonging to one pyramid level."""
        return [tile for tile in self.tiles if tile.level == level]

    # ── Core sampler ──────────────────────────────────────────────────────────

    def _sample_level(
        self,
        level: int,
        n: int,
        tissue_ratio: float = 0.5,
        max_tries: Optional[int] = None,
    ) -> List[TileInfo]:
        """Sample n tiles from one WSI level via region-first sampling.

        Applies filter_regions -> merge_overlapping -> filter_patchable to
        self.mask for THIS level, samples tiles from the surviving regions,
        then undoes the three mutations via self.mask.regions_undo() so the
        next level starts from the same base regions (no cross-level
        pollution).
        """
        if max_tries is None:
            max_tries = n * 5   # region-first hit rate is high

        ds = self.wsi.level_downsamples[level]
        tile_size_l0 = int(self.tile_size * ds)

        # Prep regions for THIS level (undone in the finally block)
        self.mask.filter_regions(min_ratio=0.01)
        self.mask.merge_overlapping()
        self.mask.filter_patchable(tile_size=self.tile_size, ds=ds)

        try:
            if not self.mask.tissue_regions:
                print(f'  [SKIP] Level {level}: no region fits '
                      f'tile_size={self.tile_size}')
                return []

            tiles: List[TileInfo] = []
            tries = 0

            while len(tiles) < n and tries < max_tries:
                tries += 1
                region = self.rng.choice(self.mask.tissue_regions)
                x0_picked = int(self.rng.integers(
                    region.x, region.x + region.w - tile_size_l0 + 1))
                y0_picked = int(self.rng.integers(
                    region.y, region.y + region.h - tile_size_l0 + 1))

                if self.mask.has_tissue_l0(x0_picked, y0_picked,
                                            tile_size_l0, tile_size_l0,
                                            tissue_ratio):
                    tiles.append(TileInfo(
                        level=level,
                        x=x0_picked,       # using when read_tile (read_region)
                        y=y0_picked,       # using when read_tile (read_region)
                        tile_size=self.tile_size,
                        mpp=self.level_mpps[level],
                    ))

            if len(tiles) < n:
                print(f'  [WARN] Level {level}: only sampled {len(tiles)}/{n} '
                      f'after {tries} tries')

            return tiles
        finally:
            # Undo the three mutations so the next level starts from the
            # same base tissue_regions.
            self.mask.regions_undo()  # filter_patchable
            self.mask.regions_undo()  # merge_overlapping
            self.mask.regions_undo()  # filter_regions

    def sample(
        self,
        n: int,
        level: Optional[int] = None,
        tissue_ratio: float = 0.5,
        max_tries: Optional[int] = None,
    ) -> TileSampler:
        """
        Sample tiles within tissue regions and store them in self.tiles.

        Args:
            n: number of tiles per level (or total when level is given)
            level: if None -> sample n tiles from each level
                   if int  -> sample n tiles from that level only
            tissue_ratio: minimum tissue fraction required per tile
            max_tries: rejection-sampling budget per level
        """
        if level is not None:
            self.tiles = self._sample_level(
                level, n, tissue_ratio, max_tries=max_tries
            )
            return self

        all_tiles: List[TileInfo] = []
        for lv in range(self.wsi.level_count):
            lv_tiles = self._sample_level(
                lv, n, tissue_ratio, max_tries=max_tries
            )
            all_tiles.extend(lv_tiles)
            mpp = self.level_mpps[lv]
            print(f'  Level {lv}  MPP={mpp:.3f}  sampled {len(lv_tiles)}/{n}')

        self.tiles = all_tiles
        return self

    # ── Read ─────────────────────────────────────────────────────────────────

    def read_tile(self, info: TileInfo) -> Image.Image:
        """Read one tile and return a PIL RGB image."""
        return self.wsi.read_region(
            (info.x, info.y), info.level, (info.tile_size, info.tile_size)
        ).convert('RGB')

    def read(self, index: int) -> Image.Image:
        """Read one sampled tile by index."""
        return self.read_tile(self.tiles[index])

    def read_all(self) -> List[Image.Image]:
        """Read all sampled tiles."""
        return [self.read(i) for i in range(len(self.tiles))]

    def iter_batches(self, batch_size: int = 32) -> Iterator[List[Image.Image]]:
        """Yield sampled tiles in batches for downstream model inference."""
        for start in range(0, len(self.tiles), batch_size):
            end = min(start + batch_size, len(self.tiles))
            yield [self.read(i) for i in range(start, end)]

    # ── Save / Load ─────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> TileSampler:
        """Save sampled tile coordinates to JSON."""
        self.save_tiles(self.tiles, path)
        return self

    @classmethod
    def from_json(
        cls,
        wsi: openslide.OpenSlide,
        tissue_mask: TissuesRegionsMask,
        path: Union[str, Path],
        tile_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> TileSampler:
        """Load tile coordinates from JSON into a new sampler instance."""
        tiles = cls.load_tiles(path)
        inferred_size = tiles[0].tile_size if tiles else 256
        sampler = cls(
            wsi,
            tissue_mask,
            tile_size=tile_size or inferred_size,
            seed=seed,
        )
        sampler.tiles = tiles
        return sampler

    @staticmethod
    def save_tiles(tiles: List[TileInfo], path: Union[str, Path]):
        """Save a tile list to JSON for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump([asdict(t) for t in tiles], f, indent=2)

    @staticmethod
    def load_tiles(path: Union[str, Path]) -> List[TileInfo]:
        """Load a tile list from JSON."""
        with open(path) as f:
            return [TileInfo(**d) for d in json.load(f)]

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> TileSampler:
        print(f'WSI levels : {self.wsi.level_count}')
        print(f'Tile size  : {self.tile_size}')
        print(f'Sampled    : {len(self.tiles)} tiles')
        print(f'Tissue frac: {self.mask.tissue_fraction()*100:.1f}%')
        print(f'Mask ds_x={self.mask.mask_ds_x:.1f}  ds_y={self.mask.mask_ds_y:.1f}')
        print(f'{"Level":>5}  {"MPP":>8}  {"W":>8}  {"H":>8}  {"Downsample":>10}  {"Tiles":>6}')
        for lv in range(self.wsi.level_count):
            W, H = self.wsi.level_dimensions[lv]
            ds = self.wsi.level_downsamples[lv]
            mpp = self.level_mpps[lv]
            n_lv = len(self.tiles_at_level(lv))
            print(f'{lv:>5}  {mpp:>8.3f}  {W:>8}  {H:>8}  {ds:>10.1f}  {n_lv:>6}')
        return self
