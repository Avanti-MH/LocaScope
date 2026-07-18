import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import openslide
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'utilities'))
sys.path.insert(0, str(ROOT / 'aiNNModel'))

from PatchingLib import QueryPatchContainer, FeaturesMap
from TissuesRegionsMask import TissuesRegionsMask
from TileSampler import TileSampler
from GigaPathFunc import gigapath_model, gigapath_encode


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GigapathKnnEstiMppResult:
    estimated_mpp: float
    base_mpp: float
    tile_size: int
    samples_per_level: int
    k: int
    query_patch_count: int


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


# ── Main class ────────────────────────────────────────────────────────────────

class GigaPathKnnEstiMpp:
    '''
    Staged GigaPath KNN MPP estimator.

    Stages (in order):
        1. build_samples()          — sample reference tiles from WSI pyramid
        2. build_ref_features()     — encode reference tiles → KnnClassifier
        3. build_query_features()   — encode query patches → FeaturesMap
        4. estimate()               — KNN vote → GigapathKnnEstiMppResult

    All intermediate state is stored on self for debugging and visualization.
    Stages that depend on earlier ones are built automatically if not called yet.
    '''

    def __init__(
        self,
        wsi: Union[openslide.OpenSlide, str],
        encoder: Callable,
        mask: Optional[TissuesRegionsMask] = None,
        tile_size: int = 256,
        samples_per_level: int = 40,
        k: int = 5,
    ):
        if isinstance(wsi, str):
            wsi = openslide.OpenSlide(wsi)
        self.wsi = wsi
        self.encoder = encoder
        self.mask = mask
        self.tile_size = tile_size
        self.samples_per_level = samples_per_level
        self.k = k

        # Intermediate state — inspect these at any stage for debugging
        self.sampler: Optional[TileSampler] = None
        self.ref_feats: Optional[torch.Tensor] = None    # [N, D]
        self.ref_mpps: Optional[List[float]] = None
        self.knn: Optional[KnnClassifier] = None
        self.qc: Optional[QueryPatchContainer] = None
        self.qfm: Optional[FeaturesMap] = None           # query FeaturesMap
        self.query_feats: Optional[torch.Tensor] = None  # [M, D] main patches only
        self.result: Optional[GigapathKnnEstiMppResult] = None

    # ── Stage 1: sample reference tiles ──────────────────────────────────────

    def build_samples(self) -> TileSampler:
        '''Sample n tiles per WSI level within tissue regions.'''
        if self.mask is None:
            self.mask = TissuesRegionsMask.from_wsi(self.wsi)
        self.sampler = TileSampler(self.wsi, self.mask, tile_size=self.tile_size)
        self.sampler.sample(n=self.samples_per_level)
        return self.sampler

    # ── Stage 2: encode reference tiles ──────────────────────────────────────

    def build_ref_features(self) -> torch.Tensor:
        '''Encode all sampled tiles; build the KnnClassifier.'''
        if self.sampler is None:
            self.build_samples()
        images = [self.sampler.read_tile(info) for info in self.sampler]
        self.ref_feats = self.encoder(images)               # [N, D]
        self.ref_mpps = [info.mpp for info in self.sampler]
        self.knn = KnnClassifier(
            self.ref_feats, np.array(self.ref_mpps), k=self.k
        )
        return self.ref_feats

    # ── Stage 3: encode query patches ────────────────────────────────────────

    def build_query_features(
        self,
        query: Union[QueryPatchContainer, Image.Image, np.ndarray],
        overlap: bool = True,
    ) -> FeaturesMap:
        '''Encode query patches into a FeaturesMap.'''
        if isinstance(query, (Image.Image, np.ndarray)):
            query = QueryPatchContainer(query)
        if query.grid is None:
            query.extract_all(self.tile_size, overlap=overlap)
        self.qc = query
        self.qfm = self.qc.to_features(self.encoder)
        self.query_feats = torch.stack(list(self.qfm.iter_main_features()))  # [M, D]
        return self.qfm

    # ── Stage 4: KNN estimate ─────────────────────────────────────────────────

    def estimate(
        self,
        query: Union[QueryPatchContainer, Image.Image, np.ndarray, None] = None,
        overlap: bool = True,
    ) -> GigapathKnnEstiMppResult:
        '''
        Run KNN and return estimated MPP.

        Pass query here or call build_query_features() beforehand.
        Missing earlier stages are built automatically.
        '''
        if query is not None:
            self.build_query_features(query, overlap=overlap)
        if self.ref_feats is None:
            self.build_ref_features()
        if self.query_feats is None:
            raise RuntimeError(
                'No query features — call build_query_features() or pass query to estimate()'
            )

        base_mpp = float(self.wsi.properties.get('openslide.mpp-x', 0))
        estimated_mpp = self.knn.predict(self.query_feats)

        self.result = GigapathKnnEstiMppResult(
            estimated_mpp=float(estimated_mpp),
            base_mpp=base_mpp,
            tile_size=self.tile_size,
            samples_per_level=self.samples_per_level,
            k=self.k,
            query_patch_count=int(self.query_feats.shape[0]),
        )
        return self.result
