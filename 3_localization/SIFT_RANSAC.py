import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'utilities'))
sys.path.insert(0, str(ROOT / '2_retrieval'))

from PatchingLib import QueryPatchContainer, WsiTissuesContainer
from GigaPathSlideWinSim import SlideWinSimResult


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SiftRansacResult:
    x: int               # top-left X @ level-n in WSI global space
    y: int               # top-left Y @ level-n
    x0: int              # top-left X @ level-0
    y0: int              # top-left Y @ level-0
    H: Optional[np.ndarray]  # 3×3 homography (query px → wsi_crop px), None if failed
    inlier_count: int
    match_count: int
    success: bool
    region_index: int
    ds: float
    level: int


# ── Localizer class ───────────────────────────────────────────────────────────

class SiftRansacLocalizer:
    '''
    Staged SIFT+RANSAC sub-tile localizer.

    Sliding window retrieval 給的是 tile 級精度（best_x/y 對齊到 tile grid），
    SIFT+RANSAC 把它細化到 sub-pixel 級：

    Stage 1  read_wsi_crop
      ┌─────────────────────────────────────────────┐
      │  tpc.img裡, 以 best_x/y 為中心               │
      │  取 ±padding tiles 的一塊 wsi_crop           │
      │  記錄 crop_origin_x/y (wsi_crop[0,0] 的      │
      │  level-n global 座標)                        │
      └─────────────────────────────────────────────┘
               ↓
    Stage 2  detect_and_match
      ┌─────────────────────────────────────────────┐
      │  query_img → SIFT keypoints + descriptors   │
      │  wsi_crop  → SIFT keypoints + descriptors   │
      │  BFMatcher knnMatch(k=2) + Lowe ratio 0.75  │
      └─────────────────────────────────────────────┘
               ↓
    Stage 3  estimate_homography
      ┌─────────────────────────────────────────────┐
      │  good_matches 裡每對點：                     │
      │    src_pts[i] = query keypoint (query px)   │
      │    dst_pts[i] = crop  keypoint (crop px)    │
      │                                             │
      │  H, mask = findHomography(                  │
      │      src_pts, dst_pts, RANSAC, 5.0)         │
      │  # H: query px → wsi_crop px                │
      │                                             │
      │  tl_in_crop = perspectiveTransform(         │
      │      [[0, 0]], H)  →  (dx, dy)              │
      │                                             │
      │  x_ln = crop_origin_x + dx  # level-n       │
      │  y_ln = crop_origin_y + dy                  │
      │  x0   = x_ln * ds           # level-0       │
      └─────────────────────────────────────────────┘

    若 RANSAC 失敗或 inliers < min_inliers，fallback 到 retrieval 的 best_x/y。

    精度提升的關鍵：retrieval 只能找到「哪個 256px tile 最像」，誤差 ≤ 1 tile。
    SIFT keypoint 精確到 subpixel，H 把 query(0,0) 映射到 crop 的精確位置，
    誤差理論上降到 keypoint 定位精度（~1–3 px @ level-n）。

    All intermediate state is stored on self for debugging and visualization.
    Stages that depend on earlier ones are built automatically if not called yet.
    '''

    def __init__(
        self,
        wsi_container: WsiTissuesContainer,
        query: QueryPatchContainer,
        location: SlideWinSimResult,
        min_inliers: int = 10,
        padding: int = 2,
    ):
        self.wsi_container = wsi_container
        self.query = query
        self.location = location
        self.min_inliers = min_inliers
        self.padding = padding

        # Intermediate state
        self.wsi_crop: Optional[np.ndarray] = None
        self.crop_origin_x: Optional[int] = None   # level-n global x of wsi_crop[0,0]
        self.crop_origin_y: Optional[int] = None   # level-n global y of wsi_crop[0,0]
        self.query_kps = None
        self.query_descs: Optional[np.ndarray] = None
        self.crop_kps = None
        self.crop_descs: Optional[np.ndarray] = None
        self.good_matches: Optional[list] = None
        self.result: Optional[SiftRansacResult] = None

    # ── Stage 1 ──────────────────────────────────────────────────────────────

    def read_wsi_crop(self, padding: Optional[int] = None) -> np.ndarray:
        '''Crop WSI image around the retrieval best match, ± padding tiles.'''
        pad = padding if padding is not None else self.padding
        tpc = self.wsi_container[self.location.best_region_index]
        ts = self.wsi_container.tile_size

        # Best match top-left in level-n global coords → local image coords
        local_x = self.location.best_x - tpc.img_origin_x
        local_y = self.location.best_y - tpc.img_origin_y

        # Window covers query size rounded up to tile boundary
        win_w = int(np.ceil(self.query.width  / ts)) * ts
        win_h = int(np.ceil(self.query.height / ts)) * ts

        x0 = max(0, local_x - pad * ts)
        y0 = max(0, local_y - pad * ts)
        x1 = min(tpc.img.shape[1], local_x + win_w + pad * ts)
        y1 = min(tpc.img.shape[0], local_y + win_h + pad * ts)

        self.wsi_crop = tpc.img[y0:y1, x0:x1].copy()
        self.crop_origin_x = tpc.img_origin_x + x0   # level-n global
        self.crop_origin_y = tpc.img_origin_y + y0
        return self.wsi_crop

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    def detect_and_match(self) -> list:
        '''SIFT detect on query + wsi_crop, then BFMatcher with Lowe ratio test.'''
        if self.wsi_crop is None:
            self.read_wsi_crop()

        sift = cv2.SIFT_create()
        q_gray = cv2.cvtColor(self.query.img, cv2.COLOR_RGB2GRAY)
        c_gray = cv2.cvtColor(self.wsi_crop,  cv2.COLOR_RGB2GRAY)

        self.query_kps, self.query_descs = sift.detectAndCompute(q_gray, None)
        self.crop_kps,  self.crop_descs  = sift.detectAndCompute(c_gray, None)

        if self.query_descs is None or self.crop_descs is None:
            self.good_matches = []
            return self.good_matches

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(self.query_descs, self.crop_descs, k=2)
        self.good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return self.good_matches

    # ── Stage 3 ──────────────────────────────────────────────────────────────

    def estimate_homography(self) -> SiftRansacResult:
        '''RANSAC homography → map query top-left to WSI level-n/level-0 coordinates.'''
        if self.good_matches is None:
            self.detect_and_match()

        ds = self.location.ds
        region_idx = self.location.best_region_index
        level = self.wsi_container.level
        n_matches = len(self.good_matches)

        H = None
        inliers = 0
        success = False
        x_ln = self.location.best_x
        y_ln = self.location.best_y

        if n_matches >= 4:
            src_pts = np.float32(
                [self.query_kps[m.queryIdx].pt for m in self.good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [self.crop_kps[m.trainIdx].pt for m in self.good_matches]
            ).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                inliers = int(mask.sum())
                success = inliers >= self.min_inliers
                if success:
                    # Map query top-left corner (0,0) through H → position in wsi_crop
                    tl_in_crop = cv2.perspectiveTransform(
                        np.array([[[0.0, 0.0]]], dtype=np.float32), H
                    )[0, 0]
                    x_ln = int(self.crop_origin_x + tl_in_crop[0])
                    y_ln = int(self.crop_origin_y + tl_in_crop[1])

        self.result = SiftRansacResult(
            x=x_ln,  y=y_ln,
            x0=int(x_ln * ds), y0=int(y_ln * ds),
            H=H,
            inlier_count=inliers,
            match_count=n_matches,
            success=success,
            region_index=region_idx,
            ds=ds,
            level=level,
        )
        return self.result
