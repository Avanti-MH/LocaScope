"""
估計 query 顯微照片的 MPP —— 用「倍率指紋」比對 WSI 各層(跟位置無關)。

    est = ClassicEstMpp(ClassicEstMppConfig()).build(wsi, mask=mask)
    result = est.estimate(query_img)   # RGB uint8 np.ndarray -> ClassicEstMppResult

原理:組織在某倍率下的「結構大小(像素)」全片一致,所以
  - 頻率重心(spectral centroid):倍率越高 → 結構越大 → 重心頻率越低
  - 自相關長度(autocorr length):倍率越高 → 特徵長度越大
這兩個量「只看倍率、不看是哪一塊組織」,所以稀疏 sample 幾塊就能估,
不必命中 query 的真實位置。

流程:每個原生層 sample 幾塊 tile → 算兩個指紋各取中位數 → 兩維 z-score 後
      對 query 指紋做距離加權 KNN,在 log-mpp 上平均出估計值。

baseline:不用任何深度學習表徵,存在的價值是拿來比較(README 的狀態標籤)。

2026-10-05 起是 `EstMppResult` 介面的估計器,讓 bench_stage1_mpp 能和其他方法
在同一批 FoV 上比較。和原本腳本版的差別:
  - 參考 tile 的位置由 TileSampler 在 tissue mask 內放、讀圖走 SlideReader,
    不再在整層上無種子地隨機抽、自己 read_region。原本只靠 `min_std` 擋背景,
    隨機位置大多落在玻片空白處;現在位置在組織內,`min_std` 只剩擋極淡的 tile。
    seed 固定,同一張片兩次 build 得到同一張參考表。
  - 只保留兩個指紋融合的 KNN 版本(原 `estimate_knn`);單一指紋最近層 + 內插
    的版本只會 print、不回傳數字,沒有呼叫端,刪除。
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import openslide

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'utilities'))
from _paths import setup_import_paths                                   # noqa: E402
setup_import_paths()

from ReadGeometry import ReadSpec                                        # noqa: E402
from SafeSlide import SafeSlide                                         # noqa: E402
from SlideReader import SlideReader                                     # noqa: E402
from stage1_estimation.StageInterface import EstMppResult                                 # noqa: E402
from TileSampler import (OverlapConfig, RichnessConfig, SamplerConfig,  # noqa: E402
                         TileSampler, native_plans)
from TissueMaskConfig import MASK_RECIPES                                # noqa: E402


def _windowed(gray):
    g = gray.astype(np.float32)
    g = g - g.mean()
    win = np.hanning(g.shape[0])[:, None] * np.hanning(g.shape[1])[None, :]
    return g * win


def spectral_centroid(gray):
    """能量重心頻率;倍率越高越小。"""
    g = _windowed(gray)
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    y, x = np.indices((h, w))
    r = np.hypot(x - w // 2, y - h // 2).astype(int)
    prof = np.bincount(r.ravel(), power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    prof[0] = 0.0                       # 去 DC
    f = np.arange(len(prof))
    return float((f * prof).sum() / (prof.sum() + 1e-9))


def autocorr_length(gray):
    """自相關掉到 0.5 的距離(像素);倍率越高越大。"""
    g = gray.astype(np.float32) - float(gray.mean())
    F = np.fft.fft2(g)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    if ac.max() <= 0:
        return 0.0
    ac /= ac.max()
    line = ac[g.shape[0] // 2, g.shape[1] // 2:]
    below = np.where(line < 0.5)[0]
    return float(below[0]) if len(below) else float(len(line))


def fingerprint(rgb: np.ndarray) -> np.ndarray:
    """(spectral_centroid, autocorr_length) of one RGB tile, on BT.601 luma."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.array([spectral_centroid(gray), autocorr_length(gray)])


# ── config / result ──────────────────────────────────────────────────────────

#: The richness shape the other reference banks use (PrototypeEstMpp,
#: KnnEstMpp): the bottom four buckets -- mostly background -- excluded.
_REFERENCE_BANK_RICHNESS = RichnessConfig(
    floors=(0.0,) * 7, caps=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class ClassicEstMppConfig:
    tile: int = 256          # 參考 tile 與 query 中央裁切的邊長(不縮放)
    samples: int = 40        # 每層參考 tile 數
    k: int = 3
    min_std: float = 6.0     # 灰階標準差低於此的 tile 視為空白,不計
    seed: int = 0
    seg: str = 'hest'        # build(wsi) 沒給 mask 時用的 MASK_RECIPES 名稱
    sampler_cfg: Optional[SamplerConfig] = field(default=None)

    def sampler(self) -> SamplerConfig:
        if self.sampler_cfg is not None:
            return self.sampler_cfg
        return SamplerConfig(n_per_rung=self.samples, seed=self.seed,
                             richness=_REFERENCE_BANK_RICHNESS,
                             overlap=OverlapConfig())


@dataclass(frozen=True)
class ClassicEstMppResult(EstMppResult):
    '''`query_fingerprint` is (spectral_centroid, autocorr_length) of the
    query's centre crop; `neighbour_levels` the k reference levels nearest
    it, nearest first.'''
    query_fingerprint: List[float]
    neighbour_levels: List[int]


# ── estimator ────────────────────────────────────────────────────────────────

class ClassicEstMpp:

    def __init__(self, cfg: Optional[ClassicEstMppConfig] = None, device=None):
        self.cfg = cfg or ClassicEstMppConfig()
        self.device = device
        self.wsi = None
        self.ref_levels: List[int] = []
        self.ref_mpps: Optional[np.ndarray] = None
        self.ref_feats: Optional[np.ndarray] = None

    def build(self, wsi: Union[openslide.OpenSlide, str],
              mask=None) -> 'ClassicEstMpp':
        '''One fingerprint per native level: the median over that level's
        reference tiles that pass `min_std`. A level with none is left out.'''
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi = wsi
        if mask is None:
            mask = MASK_RECIPES[self.cfg.seg].build(wsi, self.device)

        sampler = TileSampler(wsi, mask, self.cfg.sampler())
        sampler.sample(native_plans(wsi, self.cfg.tile))
        images = SlideReader(wsi, resize='area').read_samples(
            sampler, ReadSpec(self.cfg.tile, self.cfg.tile))

        by_level: Dict[int, List[np.ndarray]] = {}
        for sample, rgb in zip(sampler, images):
            if cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).std() < self.cfg.min_std:
                continue
            by_level.setdefault(sample.meta.level, []).append(fingerprint(rgb))

        base_mpp = float(wsi.base_mpp)
        self.ref_levels = sorted(by_level)
        if len(self.ref_levels) < 1:
            raise RuntimeError('no level kept a reference tile above min_std')
        self.ref_mpps = np.array([base_mpp * float(wsi.level_downsamples[lv])
                                  for lv in self.ref_levels])
        self.ref_feats = np.array([np.median(np.stack(by_level[lv]), axis=0)
                                   for lv in self.ref_levels])
        return self

    def estimate(self, query: np.ndarray) -> ClassicEstMppResult:
        if self.wsi is None:
            raise RuntimeError('call build(wsi) before estimate()')
        h, w = query.shape[:2]
        t = min(h, w, self.cfg.tile)
        cy, cx = h // 2, w // 2
        crop = query[cy - t // 2:cy - t // 2 + t, cx - t // 2:cx - t // 2 + t]   # 中央裁切,不 resize
        q_feat = fingerprint(np.ascontiguousarray(crop))

        mean = self.ref_feats.mean(axis=0)
        std = self.ref_feats.std(axis=0) + 1e-9
        dists = np.linalg.norm((self.ref_feats - mean) / std - (q_feat - mean) / std,
                               axis=1)
        k = min(self.cfg.k, len(dists))
        idx = np.argsort(dists)[:k]
        if dists[idx[0]] < 1e-9:                                   # exact hit
            estimated_mpp = float(self.ref_mpps[idx[0]])
        else:
            wts = 1.0 / (dists[idx] + 1e-9)
            wts /= wts.sum()
            estimated_mpp = float(np.exp((wts * np.log(self.ref_mpps[idx])).sum()))

        base_mpp = float(self.wsi.base_mpp)
        estimated_ds = estimated_mpp / base_mpp
        chosen_level = self.wsi.coarser_level_for_downsample(estimated_ds)
        chosen_ds = float(self.wsi.level_downsamples[chosen_level])
        return ClassicEstMppResult(
            estimated_ds=estimated_ds, estimated_mpp=estimated_mpp,
            chosen_ds=chosen_ds, chosen_mpp=base_mpp * chosen_ds,
            chosen_level=chosen_level,
            query_fingerprint=[float(v) for v in q_feat],
            neighbour_levels=[self.ref_levels[i] for i in idx])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wsi", help="WSI 檔(.svs/.ndpi/...)")
    ap.add_argument("query", help="query 顯微照片")
    ap.add_argument("--tile", type=int, default=256, help="比對視窗大小")
    ap.add_argument("--samples", type=int, default=40, help="每層 sample 幾塊")
    ap.add_argument("--k", type=int, default=3, help="KNN 的 K 值")
    ap.add_argument("--seg", default='hest', choices=sorted(MASK_RECIPES))
    ap.add_argument("--device", default='cuda')
    args = ap.parse_args()
    import torch
    query = cv2.cvtColor(cv2.imread(args.query), cv2.COLOR_BGR2RGB)
    est = ClassicEstMpp(ClassicEstMppConfig(tile=args.tile, samples=args.samples,
                                            k=args.k, seg=args.seg),
                        torch.device(args.device)).build(args.wsi)
    r = est.estimate(query)
    print(f"query 指紋 = sc:{r.query_fingerprint[0]:.3f}  ac:{r.query_fingerprint[1]:.3f}")
    print(f"最近層 {r.neighbour_levels}   估計 MPP ≈ {r.estimated_mpp:.4f}   "
          f"chosen level {r.chosen_level}")
