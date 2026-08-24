#!/usr/bin/env python3
"""Locate a photo in a WSI by sliding a SIFT match across the whole slide.

Standalone. This shares nothing with LocaScopePipeline -- no encoder, no KNN, no
reference bank, no tissue mask, no mpp estimation -- and imports nothing from it.
A slide handle and OpenCV are the whole dependency list. It exists to answer one
question on its own terms: what does brute force cost, and what does it get?

    window     = the query image's own size
    stride     = half the query, in both axes  (step_frac)
    operator   = SIFT detect + BFMatcher + Lowe ratio + RANSAC homography
    score      = inlier count
    result     = the top-k windows by inlier count

The stride is half a frame because a smaller step buys nothing: SIFT is not
translation-quantised, so a window containing most of the query already yields
the exact position through the homography. Half a frame guarantees every point
on the slide falls at least 1/4-frame inside some window, which is enough
overlap for the match to survive.

`limit_bounds` is the one thing that is skipped rather than scanned, and it is
not a tissue judgement: openslide.bounds-* is the rectangle the scanner actually
covered, and on a MIRAX the surrounding canvas is stage travel range holding no
image data at all. Windows over it would be reading nothing.

What the top-k are
------------------
Literally the k windows with the highest inlier counts. Nothing is merged or
deduplicated by default, so expect the top few to be the SAME place seen by
overlapping neighbours -- with a half-frame stride a genuine match falls inside
up to four windows and all four score high. That is information, not noise: four
independent windows agreeing is what a real match looks like, and one window
alone scoring high is what a coincidence looks like.

`nms_frac > 0` switches to k distinct places instead, dropping any window whose
predicted centre repeats one already kept. Off by default because it throws away
exactly the agreement signal above.

Cost
----
Measured on S1103037 (Ki67 MRXS), 1440x1024 query, stride half a frame, tissue
extent 94000 x 208316 at level-0:

    L0  130 x 406 = 52780 windows    77.8 Gpx read
    L1   65 x 203 = 13195            19.5
    L2   32 x 101 =  3232             4.8
    L3   16 x  50 =   800             1.2

Every window pays a read AND a SIFT detect on 1.5 MP, and the halved stride makes
the read cover the slide four times over. That is the shape of the method, not an
inefficiency to be optimised away -- which is why `SlideWinSiftResult` breaks the
elapsed time into read / detect / match / ransac rather than reporting one total.
Knowing which of the four dominates is the point of running this at all.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from SafeSlide import SafeSlide

# The one exception to the dependency list above. is_invertible lives with the
# other RANSAC caller, and reaching it pulls 3_localization -> 2_retrieval ->
# torch and the encoder config into this file. Taken deliberately: this module
# has exactly one importer, cli/slide_win_sift.py:80, and it is the brute-force
# baseline rather than anything the pipeline runs, so the import cost buys
# convenience at no risk to production. 3_localization is not on sys.path when
# that CLI loads us -- it inserts utilities/ only -- so put it there here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '3_localization'))
from SIFT_RANSAC import is_invertible          # noqa: E402


# A homography needs 4 correspondences. Below this, findHomography either
# raises or returns something RANSAC never had the freedom to reject.
_MIN_MATCHES_FOR_H = 4


@dataclass
class SiftWindowHit:
    """One window that matched, and where inside it the photo landed."""
    window_index: int
    win_x0: int             # level-0 top-left of the WINDOW
    win_y0: int
    win_w_ln: int           # window size at level-n (= the query's size)
    win_h_ln: int
    x0: int                 # level-0 top-left of the located PHOTO
    y0: int
    center_x0: int          # level-0 centre of the located photo
    center_y0: int
    inlier_count: int
    match_count: int
    theta_deg: float        # rotation the homography carries
    scale: float
    H: Optional[np.ndarray] = None          # query px -> window px
    crop: Optional[np.ndarray] = None       # the window's RGB, kept for figures
    crop_kps: Optional[tuple] = None
    good_matches: Optional[list] = None

    def drop_heavy(self) -> None:
        """Release the pixels but keep the numbers."""
        self.crop = None
        self.crop_kps = None
        self.good_matches = None


@dataclass
class SlideWinSiftResult:
    hits: List[SiftWindowHit] = field(default_factory=list)
    level: int = 0
    ds: float = 1.0
    plane_w: int = 0
    plane_h: int = 0
    n_windows: int = 0          # windows the grid contains
    n_scanned: int = 0          # windows actually read and SIFTed
    n_skipped_flat: int = 0
    n_with_homography: int = 0
    n_hits_raw: int = 0         # before duplicate suppression
    query_kp_n: int = 0
    query_kps: Optional[tuple] = None    # kept so a caller can draw match lines
    stopped_early: bool = False
    t_total_s: float = 0.0
    t_read_s: float = 0.0
    t_detect_s: float = 0.0
    t_match_s: float = 0.0
    t_ransac_s: float = 0.0

    @property
    def best(self) -> Optional[SiftWindowHit]:
        return self.hits[0] if self.hits else None

    def timing_line(self) -> str:
        t = self.t_total_s
        other = t - (self.t_read_s + self.t_detect_s + self.t_match_s + self.t_ransac_s)
        per = (f'  ({t / self.n_scanned * 1000:.0f} ms/window)') if self.n_scanned else ''
        return (f'{t:.1f}s total  read {self.t_read_s:.1f}s  '
                f'detect {self.t_detect_s:.1f}s  match {self.t_match_s:.1f}s  '
                f'ransac {self.t_ransac_s:.1f}s  other {other:.1f}s   '
                f'{self.n_scanned}/{self.n_windows} windows{per}')


class SlideWinSift:
    """Build once per WSI, `run(img)` once per photo.

    Nothing here is per-photo except the scan itself: the slide handle and the
    window grid are fixed by `build()`, so a folder of photos from one slide
    pays for them once.
    """

    def __init__(self,
                 wsi_path: str,
                 level: int = 0,
                 limit_bounds: bool = True,
                 step_frac: float = 0.5,
                 min_inliers: int = 10,
                 top_k: int = 5,
                 ratio: float = 0.75,
                 ransac_thresh: float = 5.0,
                 nms_frac: float = 0.0,
                 sift_nfeatures: int = 0,
                 min_std: float = 0.0,
                 stop_at_inliers: int = 0,
                 sample_windows: int = 0,
                 keep_crops: Optional[int] = None):
        """
        Args:
            level:        WSI level to scan. The query is assumed to already be
                          at this level's mpp -- nothing here rescales it.
            limit_bounds: Scan only openslide.bounds-* (the scanned rectangle).
                          See the module docstring: this is about where image
                          data exists, not about where tissue is.
            step_frac:    Stride as a fraction of the query. 0.5 = half a frame.
            min_inliers:  Below this a window is not recorded as a hit at all.
            nms_frac:     0.0 (default) = the top-k are simply the k best
                          windows, duplicates and all. Above 0, two hits closer
                          than this many frames count as the same place and the
                          weaker is dropped, giving k distinct locations
                          instead. See the module docstring for why the default
                          is off.
            min_std:      Skip SIFT on a window whose grayscale std is below
                          this. The window is still read, so it saves detect and
                          match time but no IO. Slide-dependent, so the default
                          is 0.0 (off) rather than a guessed value.
            stop_at_inliers: Stop the scan the moment a window beats this. 0 =
                          scan everything. Use for "find it fast", not for
                          measuring, since it makes the timing meaningless.
            sample_windows: Scan an evenly spaced subset of this many windows
                          instead of all of them. 0 = all. This is the
                          calibration lever: a full level-0 scan is tens of
                          thousands of windows, and per-window cost is what
                          decides whether it is worth starting.

                          EVENLY SPACED, not the first N, and that is the whole
                          point. The first N windows are the top-left corner of
                          the slide, which is blank glass -- SIFT finds almost
                          no keypoints there, so detect is quick and the match
                          against them is quicker still. Timing the corner would
                          understate the real cost by whatever the tissue
                          fraction happens to be. A spread sample pays the
                          slide's actual mix of glass and tissue.

                          A sampled run says nothing about accuracy: it steps
                          over most of the slide, so the truth is almost
                          certainly in a window it never looked at.
            keep_crops:   How many hits keep their pixels for figures. Defaults
                          to 4 * top_k, which bounds memory at a handful of
                          window-sized arrays however many windows match.
        """
        self.wsi_path = wsi_path
        self.level = level
        self.limit_bounds = limit_bounds
        self.step_frac = float(step_frac)
        self.min_inliers = int(min_inliers)
        self.top_k = int(top_k)
        self.ratio = float(ratio)
        self.ransac_thresh = float(ransac_thresh)
        self.nms_frac = float(nms_frac)
        self.sift_nfeatures = int(sift_nfeatures)
        self.min_std = float(min_std)
        self.stop_at_inliers = int(stop_at_inliers)
        self.sample_windows = int(sample_windows)
        self.keep_crops = int(keep_crops) if keep_crops else 4 * self.top_k

        self.wsi: Optional[SafeSlide] = None
        self.lv = level
        self.ds_lv = 1.0
        self.origin_x = self.origin_y = 0    # level-0 top-left of the scan rect
        self.span_w = self.span_h = 0        # its size, level-0
        self.plane_w = self.plane_h = 0      # its size at level lv
        self._sift = None

    # ── build ────────────────────────────────────────────────────────────────

    def build(self) -> 'SlideWinSift':
        """Open the slide and resolve the rectangle to scan.

        Three coordinate systems meet here and are deliberately kept apart:
        origin_*/span_* are LEVEL-0 absolute, plane_* is that same rectangle
        counted in level-lv pixels, and every window position below is a
        level-lv offset inside it. read_region wants a level-0 location and a
        level-lv size, which is exactly the mix that makes this worth naming.
        """
        self.wsi = SafeSlide(self.wsi_path)
        p = self.wsi.properties
        w0, h0 = self.wsi.level_dimensions[0]

        n_levels = len(self.wsi.level_dimensions)
        self.lv = self.level if self.level >= 0 else n_levels + self.level
        if not 0 <= self.lv < n_levels:
            raise ValueError(f'level {self.level} out of range for a '
                             f'{n_levels}-level slide')

        if self.limit_bounds:
            self.origin_x = int(p.get('openslide.bounds-x', 0))
            self.origin_y = int(p.get('openslide.bounds-y', 0))
            self.span_w = int(p.get('openslide.bounds-width', w0))
            self.span_h = int(p.get('openslide.bounds-height', h0))
        else:
            self.origin_x = self.origin_y = 0
            self.span_w, self.span_h = w0, h0

        self.ds_lv = self.wsi.level_downsamples[self.lv]
        self.plane_w = max(1, int(self.span_w / self.ds_lv))
        self.plane_h = max(1, int(self.span_h / self.ds_lv))

        self._sift = cv2.SIFT_create(nfeatures=self.sift_nfeatures)
        return self

    def close(self) -> None:
        if self.wsi is not None:
            self.wsi.close()
            self.wsi = None

    # ── the window grid ──────────────────────────────────────────────────────

    def _axis_positions(self, plane: int, win: int, step: int) -> List[int]:
        """Start offsets along one axis, with the last window flush to the edge.

        range() alone leaves a strip at the far edge covered by no window when
        the plane is not a whole number of steps past the first one; the flush
        tail is what stops a photo taken at the edge of the slide from being
        unfindable in principle.
        """
        if plane <= win:
            return [0]
        xs = list(range(0, plane - win + 1, step))
        if xs[-1] != plane - win:
            xs.append(plane - win)
        return xs

    def window_grid(self, qw: int, qh: int) -> tuple:
        """(xs, ys, win_w, win_h) at level lv. Pure, so a CLI can cost a run."""
        win_w = min(qw, self.plane_w)
        win_h = min(qh, self.plane_h)
        step_x = max(1, int(round(qw * self.step_frac)))
        step_y = max(1, int(round(qh * self.step_frac)))
        xs = self._axis_positions(self.plane_w, win_w, step_x)
        ys = self._axis_positions(self.plane_h, win_h, step_y)
        return xs, ys, win_w, win_h

    def _to_level0(self, x_ln: float, y_ln: float) -> tuple:
        """Level-lv position inside the scanned rectangle -> absolute level-0."""
        return (int(round(self.origin_x + x_ln * self.ds_lv)),
                int(round(self.origin_y + y_ln * self.ds_lv)))

    # ── the scan ─────────────────────────────────────────────────────────────

    def run(self, img: np.ndarray,
            progress: Optional[Callable] = None,
            progress_every: int = 200) -> SlideWinSiftResult:
        """Slide the query over the whole level and return the best k placements."""
        if self.wsi is None:
            self.build()

        qh, qw = img.shape[:2]
        res = SlideWinSiftResult(level=self.lv, ds=self.ds_lv,
                                 plane_w=self.plane_w, plane_h=self.plane_h)
        t0 = time.time()

        q_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        q_kps, q_descs = self._sift.detectAndCompute(q_gray, None)
        res.query_kp_n = 0 if q_kps is None else len(q_kps)
        res.query_kps = q_kps
        res.t_detect_s += time.time() - t0
        if q_descs is None or len(q_descs) < _MIN_MATCHES_FOR_H:
            res.t_total_s = time.time() - t0
            return res

        bf = cv2.BFMatcher(cv2.NORM_L2)
        xs, ys, win_w, win_h = self.window_grid(qw, qh)
        res.n_windows = len(xs) * len(ys)

        # Corners and centre of the query, mapped through H once a match lands.
        q_corners = np.float32([[0, 0], [qw, 0], [qw, qh], [0, qh]]).reshape(-1, 1, 2)
        q_centre = np.float32([[qw / 2.0, qh / 2.0]]).reshape(-1, 1, 2)
        need = max(_MIN_MATCHES_FOR_H, self.min_inliers)

        # The grid as an explicit list so sampling is a stride over it. Raster
        # order, x fastest, so window_index reads as "the nth window" either way.
        grid = [(i, x, y) for i, (y, x) in
                enumerate((y, x) for y in ys for x in xs)]
        if self.sample_windows and self.sample_windows < len(grid):
            step = len(grid) / float(self.sample_windows)
            grid = [grid[int(i * step)] for i in range(self.sample_windows)]
            res.stopped_early = True

        raw: List[SiftWindowHit] = []
        for n_done, (idx, x_ln, y_ln) in enumerate(grid, 1):
            # Before the window is touched, not after it is recorded. Every exit
            # below is a `continue`, and a progress call placed past them only
            # fires when the counter lands on a window that happened to score a
            # hit -- which on a first run printed 1000, 2200, 3000, 7200 and
            # read as an erratic step size rather than as a filter.
            if progress and progress_every and n_done % progress_every == 0:
                progress(n_done, len(grid),
                         max(raw, key=lambda h: h.inlier_count) if raw else None)

            win_x0, win_y0 = self._to_level0(x_ln, y_ln)

            t = time.time()
            crop = self.wsi.read_region_rgb((win_x0, win_y0), self.lv,
                                            (win_w, win_h))
            res.t_read_s += time.time() - t

            c_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            if self.min_std > 0 and float(c_gray.std()) < self.min_std:
                res.n_skipped_flat += 1
                continue
            res.n_scanned += 1

            t = time.time()
            c_kps, c_descs = self._sift.detectAndCompute(c_gray, None)
            res.t_detect_s += time.time() - t
            if c_descs is None or len(c_descs) < 2:
                continue

            t = time.time()
            pairs = bf.knnMatch(q_descs, c_descs, k=2)
            good = [a for a, b in pairs if a.distance < self.ratio * b.distance]
            res.t_match_s += time.time() - t
            if len(good) < need:
                continue

            t = time.time()
            src = np.float32([q_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([c_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
            res.t_ransac_s += time.time() - t
            if H is None or inl is None:
                continue
            if not is_invertible(H):
                # Rank-deficient: it collapses the plane onto a line, so it has
                # no inverse and maps the query's corners nowhere useful.
                #
                # Not a rare case here. log/TODO.log:476 records a degenerate H
                # as this scanner's failure mode B, at 233-278 inliers with a
                # ratio near 1 -- every match "passes" a geometric check made
                # with the broken H itself. Those are near-singular rather than
                # singular and still get through this line; what it does catch
                # is the exact collapse, which would otherwise reach the figure
                # and be inverted there.
                continue
            res.n_with_homography += 1
            n_in = int(inl.sum())
            if n_in < self.min_inliers:
                continue

            raw.append(self._make_hit(idx, win_x0, win_y0, win_w, win_h,
                                      H, n_in, len(good), q_corners, q_centre,
                                      crop, c_kps, good))
            self._prune(raw)

            if self.stop_at_inliers and n_in >= self.stop_at_inliers:
                res.stopped_early = True
                break

        res.n_hits_raw = len(raw)
        kept = self._suppress(raw)
        res.hits = kept[:self.top_k]
        for h in kept[self.top_k:]:
            h.drop_heavy()
        res.t_total_s = time.time() - t0
        return res

    # ── hit bookkeeping ──────────────────────────────────────────────────────

    def _make_hit(self, idx, win_x0, win_y0, win_w, win_h,
                  H, n_in, n_good, q_corners, q_centre,
                  crop, c_kps, good) -> SiftWindowHit:
        tl = cv2.perspectiveTransform(q_corners, H).reshape(-1, 2)[0]
        ct = cv2.perspectiveTransform(q_centre, H).reshape(-1, 2)[0]
        # H maps query px -> window px, both at level-lv, so a level-lv offset
        # scales by ds_lv on its way to level-0 while win_x0 -- already level-0
        # -- does not. Mixing the two is the bug this file most wants to avoid.
        return SiftWindowHit(
            window_index=idx, win_x0=win_x0, win_y0=win_y0,
            win_w_ln=win_w, win_h_ln=win_h,
            x0=win_x0 + int(round(float(tl[0]) * self.ds_lv)),
            y0=win_y0 + int(round(float(tl[1]) * self.ds_lv)),
            center_x0=win_x0 + int(round(float(ct[0]) * self.ds_lv)),
            center_y0=win_y0 + int(round(float(ct[1]) * self.ds_lv)),
            inlier_count=n_in, match_count=n_good,
            theta_deg=math.degrees(math.atan2(H[1, 0], H[0, 0])),
            scale=math.hypot(H[0, 0], H[1, 0]),
            H=H, crop=crop, crop_kps=c_kps, good_matches=good)

    def _prune(self, raw: List[SiftWindowHit]) -> None:
        """Keep every hit's numbers, but only the best `keep_crops` hits' pixels.

        Without this a slide that matches weakly in a thousand places would hold
        a thousand window-sized arrays; the cap makes memory independent of how
        cooperative the slide is.
        """
        if len(raw) <= self.keep_crops:
            return
        for h in sorted(raw, key=lambda h: h.inlier_count,
                        reverse=True)[self.keep_crops:]:
            h.drop_heavy()

    def _suppress(self, raw: List[SiftWindowHit]) -> List[SiftWindowHit]:
        """Best window first. With nms_frac > 0, also drop repeated places."""
        if self.nms_frac <= 0:
            return sorted(raw, key=lambda h: h.inlier_count, reverse=True)
        kept: List[SiftWindowHit] = []
        for h in sorted(raw, key=lambda h: h.inlier_count, reverse=True):
            min_d = self.nms_frac * min(h.win_w_ln, h.win_h_ln) * self.ds_lv
            if any(math.hypot(h.center_x0 - k.center_x0,
                              h.center_y0 - k.center_y0) < min_d for k in kept):
                continue
            kept.append(h)
        return kept
