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
    # Query CENTRE mapped through H. Rotation-invariant anchor: the query is
    # rotated about its own centre, so this stays comparable to a ground-truth
    # centre no matter which 90-degree step the query came in at. Prefer these
    # over x/y when the query orientation is unknown.
    center_x:  int = 0   # centre X @ level-n
    center_y:  int = 0   # centre Y @ level-n
    center_x0: int = 0   # centre X @ level-0
    center_y0: int = 0   # centre Y @ level-0


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

        if x1 <= x0 or y1 <= y0:
            # best_x/best_y landed outside this region's image. That means the
            # retrieval handed over a position it never actually scored — e.g.
            # its (0, 0) sentinel when no window fit anywhere. Fail loudly here
            # rather than letting cv2 report an empty-Mat assertion later.
            raise ValueError(
                f'empty WSI crop for region {self.location.best_region_index}: '
                f'best=({self.location.best_x}, {self.location.best_y}) maps to '
                f'local=({local_x}, {local_y}) in a {tpc.img.shape[1]}x'
                f'{tpc.img.shape[0]} region image; window {win_w}x{win_h} '
                f'+{pad} tiles gives x[{x0}:{x1}] y[{y0}:{y1}]'
            )

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

        # Query footprint at level-n. When the retrieval reports a rotation,
        # the matched window is the ROTATED query, so width/height swap for
        # the 90/270 steps — used for the fallback centre below.
        h_q, w_q = self.query.img.shape[:2]
        rot = getattr(self.location, 'best_rotation', 0)
        w_eff, h_eff = (h_q, w_q) if rot in (90, 270) else (w_q, h_q)
        cx_ln = x_ln + w_eff // 2
        cy_ln = y_ln + h_eff // 2

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
                    # Map query top-left (0,0) and centre through H → wsi_crop px
                    pts = np.array(
                        [[[0.0, 0.0]], [[w_q / 2.0, h_q / 2.0]]], dtype=np.float32,
                    )
                    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                    x_ln  = int(self.crop_origin_x + mapped[0][0])
                    y_ln  = int(self.crop_origin_y + mapped[0][1])
                    cx_ln = int(self.crop_origin_x + mapped[1][0])
                    cy_ln = int(self.crop_origin_y + mapped[1][1])

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
            center_x=cx_ln, center_y=cy_ln,
            center_x0=int(cx_ln * ds), center_y0=int(cy_ln * ds),
        )
        return self.result
